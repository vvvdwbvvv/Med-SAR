# Sample usage:
# python scriptsv2/05_generate_adv_batch.py \
#   --m23k data/processed/m23k_v2/m23k_train.jsonl \
#   --calibration outputs/calibration.json \
#   --guard_spec outputs/fact_guard_spec.yaml \
#   --out outputs/adv_train.jsonl \
#   --stats_out outputs/guard_stats.jsonl

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
from typing import Dict, List, Optional

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from med_sar.guard.rules import GuardConfig, check_guard
from med_sar.operators.chain import apply_chain, parse_ops, sample_chain
from med_sar.operators.calibration import load_calibration, level_for_t
from med_sar.operators.proxies import compute_proxies
from utils.io import read_jsonl


def _load_guard_config(path: str | None) -> GuardConfig:
    if not path:
        return GuardConfig()
    data = yaml.safe_load(Path(path).read_text()) or {}
    return GuardConfig(
        min_entity_jaccard=float(data.get("min_entity_jaccard", 0.5)),
        min_length_ratio=float(data.get("min_length_ratio", 0.5)),
        max_length_ratio=float(data.get("max_length_ratio", 1.8)),
        allow_unit_changes=bool(data.get("allow_unit_changes", False)),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m23k", type=str, required=True)
    ap.add_argument("--calibration", type=str, default=None)
    ap.add_argument("--guard_spec", type=str, default=None)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--stats_out", type=str, required=True)
    ap.add_argument("--t_values", type=float, nargs="+", default=[i / 10 for i in range(11)])
    ap.add_argument("--ops", type=str, default=None, help="Comma-separated ops chain")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_resample", type=int, default=3)
    ap.add_argument("--max_records", type=int, default=None)
    args = ap.parse_args()

    rows = list(read_jsonl(args.m23k))
    if args.max_records:
        rows = rows[: args.max_records]

    rng = random.Random(args.seed)
    calibration = load_calibration(args.calibration)
    guard_cfg = _load_guard_config(args.guard_spec)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path = Path(args.stats_out)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    out_rows: List[Dict] = []
    stat_rows: List[Dict] = []

    for i, r in enumerate(rows):
        clean = r.get("x_wrapped") or r.get("x_raw") or ""
        y = r.get("y")
        t = rng.choice(args.t_values)
        tries = 0
        guard_result = None
        adv = clean
        ops_used: Optional[List[str]] = None
        while tries < args.max_resample:
            tries += 1
            if args.ops:
                ops = parse_ops(args.ops)
            else:
                ops = sample_chain(rng)
            candidate = apply_chain(
                clean,
                ops,
                t=t,
                seed=args.seed + i + tries,
                level_fn=lambda op, t_val: level_for_t(op, t_val, calibration),
            )
            guard_result = check_guard(clean, candidate, guard_cfg)
            if guard_result.passed:
                adv = candidate
                ops_used = ops
                break
            adv = candidate
            ops_used = ops

        if guard_result is None:
            continue

        proxies = compute_proxies(adv)
        out_rows.append(
            {
                "id": r.get("id"),
                "x_adv": adv,
                "y": y,
                "meta": {
                    "t": t,
                    "ops": ops_used,
                    "tries": tries,
                    "guard_pass": guard_result.passed,
                    "guard_reasons": guard_result.reasons,
                },
            }
        )
        stat_rows.append(
            {
                "id": r.get("id"),
                "t": t,
                "ops": ops_used,
                "tries": tries,
                "guard_pass": guard_result.passed,
                "guard_reasons": guard_result.reasons,
                "guard_primary": guard_result.primary_reason,
                **guard_result.metrics,
                **proxies,
            }
        )

    with out_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with stats_path.open("w", encoding="utf-8") as f:
        for row in stat_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(out_rows)} to {out_path}")
    print(f"wrote {len(stat_rows)} to {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
