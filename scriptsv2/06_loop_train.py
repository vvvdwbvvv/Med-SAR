# Sample usage:
# python scriptsv2/06_loop_train.py \
#   --train data/processed/m23k_v2/m23k_train.jsonl \
#   --dev data/processed/m23k_v2/m23k_val.jsonl \
#   --base_model meta-llama/Llama-3.2-3B-Instruct \
#   --out_dir outputs/loop_train_v2 \
#   --calibration outputs/calibration.json \
#   --guard_spec outputs/fact_guard_spec.yaml

from __future__ import annotations

import argparse
import itertools
import json
import random
import subprocess
from pathlib import Path
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from med_sar.guard.rules import GuardConfig, check_guard
from med_sar.operators.chain import apply_chain
from med_sar.operators.calibration import load_calibration, level_for_t
from med_sar.operators.library import operator_names
from utils.io import read_jsonl


def _load_guard_config(path: str | None) -> GuardConfig:
    if not path:
        return GuardConfig()
    import yaml

    data = yaml.safe_load(Path(path).read_text()) or {}
    return GuardConfig(
        min_entity_jaccard=float(data.get("min_entity_jaccard", 0.5)),
        min_length_ratio=float(data.get("min_length_ratio", 0.5)),
        max_length_ratio=float(data.get("max_length_ratio", 1.8)),
        allow_unit_changes=bool(data.get("allow_unit_changes", False)),
    )


def _candidate_policies(t_values: List[float]) -> List[Tuple[List[str], float]]:
    ops = operator_names()
    chains = [list(pair) for pair in itertools.combinations(ops, 2)]
    return [(chain, t) for chain in chains for t in t_values]


def _policy_score(
    sample_rows: List[Dict],
    ops: List[str],
    t: float,
    seed: int,
    guard_cfg: GuardConfig,
    calibration,
) -> float:
    scores = []
    for i, r in enumerate(sample_rows):
        clean = r.get("x_wrapped") or r.get("x_raw") or ""
        adv = apply_chain(
            clean,
            ops,
            t=t,
            seed=seed + i,
            level_fn=lambda op, t_val: level_for_t(op, t_val, calibration),
        )
        guard = check_guard(clean, adv, guard_cfg)
        if guard.passed:
            scores.append(1.0 - guard.metrics.get("entity_jaccard", 1.0))
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, required=True)
    ap.add_argument("--dev", type=str, required=True)
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--calibration", type=str, default=None)
    ap.add_argument("--guard_spec", type=str, default=None)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--t_values", type=float, nargs="+", default=[0.0, 0.3, 0.6, 1.0])
    ap.add_argument("--policy_eval_samples", type=int, default=200)
    ap.add_argument("--num_candidates", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_train", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "policy_selection_logs.jsonl"

    train_rows = list(read_jsonl(args.train))
    rng = random.Random(args.seed)

    calibration = load_calibration(args.calibration)
    guard_cfg = _load_guard_config(args.guard_spec)

    candidates = _candidate_policies(args.t_values)

    with log_path.open("w", encoding="utf-8") as log_f:
        for round_idx in range(1, args.rounds + 1):
            sample = rng.sample(
                train_rows, k=min(args.policy_eval_samples, len(train_rows))
            )
            rng.shuffle(candidates)
            shortlist = candidates[: min(args.num_candidates, len(candidates))]

            scored = []
            for ops, t in shortlist:
                score = _policy_score(
                    sample,
                    ops,
                    t,
                    seed=args.seed + round_idx * 1000,
                    guard_cfg=guard_cfg,
                    calibration=calibration,
                )
                scored.append((score, ops, t))

            scored.sort(reverse=True, key=lambda x: x[0])
            best_score, best_ops, best_t = scored[0]

            log_f.write(
                json.dumps(
                    {
                        "round": round_idx,
                        "best_ops": best_ops,
                        "best_t": best_t,
                        "best_score": best_score,
                        "candidates": [
                            {"ops": ops, "t": t, "score": score}
                            for score, ops, t in scored
                        ],
                    }
                )
                + "\n"
            )
            log_f.flush()

            adv_path = out_dir / f"adv_round_{round_idx}.jsonl"
            stats_path = out_dir / f"guard_stats_round_{round_idx}.jsonl"

            subprocess.check_call(
                [
                    "python",
                    "scriptsv2/05_generate_adv_batch.py",
                    "--m23k",
                    args.train,
                    "--calibration",
                    args.calibration or "",
                    "--guard_spec",
                    args.guard_spec or "",
                    "--out",
                    str(adv_path),
                    "--stats_out",
                    str(stats_path),
                    "--t_values",
                    str(best_t),
                    "--ops",
                    ",".join(best_ops),
                ]
            )

            if not args.skip_train:
                ckpt_dir = out_dir / f"doctor_round_{round_idx}"
                subprocess.check_call(
                    [
                        "python",
                        "scriptsv2/04_sft_doctor.py",
                        "--train",
                        args.train,
                        "--dev",
                        args.dev,
                        "--adv_train",
                        str(adv_path),
                        "--invariance_aug",
                        "--base",
                        args.base_model,
                        "--out",
                        str(ckpt_dir),
                        "--input_field",
                        "x_wrapped",
                        "--answer_field",
                        "y",
                    ]
                )

    print(f"wrote policy logs to {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
