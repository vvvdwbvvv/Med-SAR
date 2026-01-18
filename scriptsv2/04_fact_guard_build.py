# Sample usage:
# python scriptsv2/04_fact_guard_build.py \
#   --m23k data/processed/m23k_v2/m23k_train.jsonl \
#   --calibration outputs/calibration.json \
#   --out_dir outputs/fact_guard

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
import sys
from typing import Dict

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from med_sar.guard.rules import GuardConfig, check_guard
from med_sar.operators.chain import apply_chain, sample_chain
from med_sar.operators.calibration import load_calibration, level_for_t
from utils.io import read_jsonl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m23k", type=str, required=True)
    ap.add_argument("--calibration", type=str, default=None)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t_values", type=float, nargs="+", default=[0.0, 0.3, 0.6, 1.0])
    ap.add_argument("--min_entity_jaccard", type=float, default=0.5)
    ap.add_argument("--min_length_ratio", type=float, default=0.5)
    ap.add_argument("--max_length_ratio", type=float, default=1.8)
    ap.add_argument("--allow_unit_changes", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = list(read_jsonl(args.m23k))
    sample = rng.sample(rows, k=min(args.n, len(rows)))

    guard_cfg = GuardConfig(
        min_entity_jaccard=args.min_entity_jaccard,
        min_length_ratio=args.min_length_ratio,
        max_length_ratio=args.max_length_ratio,
        allow_unit_changes=args.allow_unit_changes,
    )
    calibration = load_calibration(args.calibration)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec_path = out_dir / "fact_guard_spec.yaml"
    spec_payload = {
        "min_entity_jaccard": guard_cfg.min_entity_jaccard,
        "min_length_ratio": guard_cfg.min_length_ratio,
        "max_length_ratio": guard_cfg.max_length_ratio,
        "allow_unit_changes": guard_cfg.allow_unit_changes,
    }
    spec_path.write_text(
        yaml.safe_dump(spec_payload, sort_keys=False), encoding="utf-8"
    )

    eval_path = out_dir / "guard_eval_samples.csv"
    table2_path = out_dir / "table2_guard_precision.json"

    reason_counts: Dict[str, int] = {}
    total = 0
    passed = 0

    with eval_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "t",
                "ops",
                "guard_pass",
                "guard_reasons",
                "x_clean",
                "x_adv",
            ],
        )
        w.writeheader()
        for i, r in enumerate(sample):
            clean = r.get("x_wrapped") or r.get("x_raw") or ""
            t = rng.choice(args.t_values)
            ops = sample_chain(rng)
            adv = apply_chain(
                clean,
                ops,
                t=t,
                seed=args.seed + i,
                level_fn=lambda op, t_val: level_for_t(op, t_val, calibration),
            )
            guard = check_guard(clean, adv, guard_cfg)
            total += 1
            if guard.passed:
                passed += 1
            for reason in guard.reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            w.writerow(
                {
                    "id": r.get("id"),
                    "t": t,
                    "ops": ",".join(ops),
                    "guard_pass": guard.passed,
                    "guard_reasons": ";".join(guard.reasons),
                    "x_clean": clean,
                    "x_adv": adv,
                }
            )

    table2_path.write_text(
        json.dumps(
            {
                "total_samples": total,
                "passed": passed,
                "pass_rate": passed / max(1, total),
                "reject_reasons": reason_counts,
                "note": "Placeholder precision; replace with manual labels for Table 2.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"wrote {spec_path}")
    print(f"wrote {eval_path}")
    print(f"wrote {table2_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
