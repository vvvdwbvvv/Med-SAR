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

import pandas as pd
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


def _require_parquet() -> None:
    try:
        import pyarrow  # noqa: F401
    except Exception as exc:  # pragma: no cover - runtime check
        raise SystemExit(
            "pyarrow is required for parquet outputs. Install it and retry."
        ) from exc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m23k", type=str, required=True)
    ap.add_argument("--calibration", type=str, default=None)
    ap.add_argument("--guard_spec", type=str, default=None)
    ap.add_argument("--out", type=str, default="data/processed/adv/adv_train.jsonl")
    ap.add_argument("--guard_log_out", type=str, default="runs/adv_gen/guard_log.jsonl")
    ap.add_argument(
        "--proxies_out",
        type=str,
        default="runs/adv_gen/post_guard_proxies.parquet",
    )
    ap.add_argument(
        "--stats_out",
        type=str,
        default=None,
        help="Legacy path for guard stats (jsonl).",
    )
    ap.add_argument(
        "--t_values", type=float, nargs="+", default=[i / 10 for i in range(11)]
    )
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
    guard_log_path = Path(args.guard_log_out)
    guard_log_path.parent.mkdir(parents=True, exist_ok=True)
    proxies_path = Path(args.proxies_out)
    proxies_path.parent.mkdir(parents=True, exist_ok=True)
    if args.stats_out:
        stats_path = Path(args.stats_out)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        stats_path = None

    out_rows: List[Dict] = []
    guard_rows: List[Dict] = []
    proxy_rows: List[Dict] = []

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
        proxies_prefixed = {f"proxy_{k}": v for k, v in proxies.items()}
        f_vec = {
            "F1": int("number_mismatch" in guard_result.reasons),
            "F2": int("negation_flip" in guard_result.reasons),
            "F3": int("unit_mismatch" in guard_result.reasons),
            "F4": int("entity_drop" in guard_result.reasons),
            "F5": int("length_ratio" in guard_result.reasons),
        }
        out_rows.append(
            {
                "id": r.get("id"),
                "x_adv": adv,
                "y": y,
                "meta": {
                    "t": t,
                    "ops_chain": ops_used,
                    "tries": tries,
                    "guard": {
                        "accepted": guard_result.passed,
                        "violations": f_vec,
                    },
                },
            }
        )
        guard_rows.append(
            {
                "id": r.get("id"),
                "t": t,
                "ops_chain": ops_used,
                "tries": tries,
                "accepted": guard_result.passed,
                "guard_reasons": guard_result.reasons,
                "guard_primary": guard_result.primary_reason,
                **f_vec,
                **guard_result.metrics,
                **proxies_prefixed,
            }
        )
        if guard_result.passed:
            proxy_rows.append(
                {
                    "id": r.get("id"),
                    "t": t,
                    **proxies_prefixed,
                }
            )

    with out_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with guard_log_path.open("w", encoding="utf-8") as f:
        for row in guard_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if stats_path is not None:
        with stats_path.open("w", encoding="utf-8") as f:
            for row in guard_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    _require_parquet()
    pd.DataFrame(proxy_rows).to_parquet(proxies_path, index=False)

    summary = {}
    if guard_rows:
        df = pd.DataFrame(guard_rows)
        summary["mean_retries_by_t"] = (
            df.groupby("t")["tries"].mean().dropna().to_dict()
        )
        summary["reject_reason_dist"] = (
            df.explode("guard_reasons")["guard_reasons"]
            .value_counts()
            .head(10)
            .to_dict()
        )
    summary_path = guard_log_path.parent / "guard_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"wrote {len(out_rows)} to {out_path}")
    print(f"wrote {len(guard_rows)} to {guard_log_path}")
    print(f"wrote {proxies_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
