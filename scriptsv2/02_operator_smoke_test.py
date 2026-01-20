# Sample usage:
# python scriptsv2/02_operator_smoke_test.py \
#   --m23k data/processed/m23k_v2/m23k_val.jsonl \
#   --calibration outputs/calibration.json \
#   --guard_spec outputs/fact_guard_spec.yaml \
#   --out outputs/smoke/operator_samples.jsonl

from __future__ import annotations

import argparse
import difflib
import json
import random
from pathlib import Path
import sys
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from med_sar.guard.rules import GuardConfig, check_guard
from med_sar.operators.chain import apply_chain, parse_ops, sample_chain
from med_sar.operators.calibration import load_calibration, level_for_t
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


def _print_diff(a: str, b: str) -> str:
    diff = difflib.unified_diff(
        a.splitlines(), b.splitlines(), lineterm="", fromfile="clean", tofile="adv"
    )
    return "\n".join(diff)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m23k", type=str, required=True)
    ap.add_argument("--calibration", type=str, default=None)
    ap.add_argument("--guard_spec", type=str, default=None)
    ap.add_argument("--out", type=str, default="runs/smoke/operator_smoke.jsonl")
    ap.add_argument("--diff_out", type=str, default="runs/smoke/samples_diff.txt")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t_values", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    ap.add_argument("--ops", type=str, default=None)
    ap.add_argument("--max_previews", type=int, default=2)
    ap.add_argument("--max_resample", type=int, default=3)
    args = ap.parse_args()

    rows = list(read_jsonl(args.m23k))
    rng = random.Random(args.seed)
    sample = rng.sample(rows, k=min(args.n, len(rows)))

    calibration = load_calibration(args.calibration)
    guard_cfg = _load_guard_config(args.guard_spec)

    out_rows: List[Dict] = []
    previewed = 0
    accept = 0
    reject = 0
    accept_by_t: Dict[str, int] = {}
    total_by_t: Dict[str, int] = {}
    reject_reasons: Dict[str, int] = {}

    diff_path = Path(args.diff_out)
    diff_path.parent.mkdir(parents=True, exist_ok=True)
    diff_f = diff_path.open("w", encoding="utf-8")

    for i, r in enumerate(sample):
        clean = r.get("x_wrapped") or r.get("x_raw") or ""
        for t in args.t_values:
            tries = 0
            guard = None
            adv = clean
            ops = []
            while tries < args.max_resample:
                tries += 1
                if args.ops:
                    ops = parse_ops(args.ops)
                else:
                    ops = sample_chain(rng)
                adv = apply_chain(
                    clean,
                    ops,
                    t=t,
                    seed=args.seed + i + tries,
                    level_fn=lambda op, t_val: level_for_t(op, t_val, calibration),
                )
                guard = check_guard(clean, adv, guard_cfg)
                if guard.passed:
                    break
            if guard is None:
                continue

            total_by_t[f"{t:.2f}"] = total_by_t.get(f"{t:.2f}", 0) + 1
            if guard.passed:
                accept += 1
                accept_by_t[f"{t:.2f}"] = accept_by_t.get(f"{t:.2f}", 0) + 1
            else:
                reject += 1
                for reason in guard.reasons:
                    reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
                if previewed < args.max_previews:
                    diff_f.write("\n--- diff preview ---\n")
                    diff_f.write(_print_diff(clean, adv) + "\n")
                    diff_f.write(f"reasons: {guard.reasons}\n")
                    previewed += 1

            f_vec = {
                "F1": int("number_mismatch" in guard.reasons),
                "F2": int("negation_flip" in guard.reasons),
                "F3": int("unit_mismatch" in guard.reasons),
                "F4": int("entity_drop" in guard.reasons),
                "F5": int("length_ratio" in guard.reasons),
            }
            out_rows.append(
                {
                    "id": r.get("id"),
                    "t": t,
                    "ops_chain": ops,
                    "tries": tries,
                    "accepted": guard.passed,
                    "reject_reason": guard.reasons[0] if guard.reasons else None,
                    "fact_violation_vec": f_vec,
                }
            )

    diff_f.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = accept + reject
    print(f"accept={accept} reject={reject} total={total}")
    print(f"wrote {len(out_rows)} to {args.out}")

    summary = {
        "accept_rate_by_t": {
            t: accept_by_t.get(t, 0) / max(1, total_by_t.get(t, 0)) for t in total_by_t
        },
        "top_reject_reasons": sorted(
            reject_reasons.items(), key=lambda kv: kv[1], reverse=True
        )[:5],
    }
    summary_path = out_path.parent / "operator_smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
