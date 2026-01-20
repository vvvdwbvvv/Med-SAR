# Sample usage:
# python scriptsv2/10_selfplay_evolving_effect.py \
#   --policy_logs runs/selfplay/policy_selection_log.jsonl \
#   --out_dir runs/selfplay_evolving \
#   --baseline_dir runs/baseline_benchmark \
#   --selfplay_dir runs/selfplay_benchmark

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load_parquet(path: Path):
    import pandas as pd

    return pd.read_parquet(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--policy_logs", type=str, default="runs/selfplay/policy_selection_log.jsonl"
    )
    ap.add_argument("--out_dir", type=str, default="runs/selfplay_evolving")
    ap.add_argument("--baseline_dir", type=str, default=None)
    ap.add_argument("--selfplay_dir", type=str, default=None)
    args = ap.parse_args()

    rows = []
    with Path(args.policy_logs).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))

    summary = []
    for r in rows:
        chosen = r.get("chosen", {})
        best_t = chosen.get("t", r.get("best_t"))
        best_ops = chosen.get("ops", r.get("best_ops", []))
        best_score = chosen.get("loss", r.get("best_score"))
        summary.append(
            {
                "round": r.get("round"),
                "best_t": best_t,
                "best_ops": ",".join(best_ops) if best_ops else "",
                "best_score": best_score,
            }
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "selfplay_policy_summary.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["round", "best_t", "best_ops", "best_score"])
        w.writeheader()
        w.writerows(summary)

    out_json = out_dir / "selfplay_policy_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    evolving_summary = {}
    if args.baseline_dir and args.selfplay_dir:
        baseline_dir = Path(args.baseline_dir)
        selfplay_dir = Path(args.selfplay_dir)
        base_metrics = _load_parquet(baseline_dir / "slice_metrics.parquet")
        sp_metrics = _load_parquet(selfplay_dir / "slice_metrics.parquet")

        def _agg(df):
            cols = {
                "P": "P" if "P" in df.columns else "accuracy",
                "S": "S" if "S" in df.columns else "stability",
                "F_overall": "F_overall" if "F_overall" in df.columns else "fact_error",
            }
            tmp = df.groupby("t")[list(cols.values())].mean().reset_index()
            tmp = tmp.rename(columns={v: k for k, v in cols.items()})
            return tmp

        base_t = _agg(base_metrics)
        sp_t = _agg(sp_metrics)
        merged = base_t.merge(sp_t, on="t", suffixes=("_base", "_sp"))
        if not merged.empty:
            evolving_summary["matched_t_deltas"] = {
                "S": float((merged["S_sp"] - merged["S_base"]).mean()),
                "F_overall": float(
                    (merged["F_overall_sp"] - merged["F_overall_base"]).mean()
                ),
            }

        base_bp = _load_parquet(baseline_dir / "breakpoints.parquet")
        sp_bp = _load_parquet(selfplay_dir / "breakpoints.parquet")
        bp = base_bp.merge(sp_bp, on=["slice_id", "metric"], suffixes=("_base", "_sp"))
        if not bp.empty:
            bp["delta_t_star"] = bp["t_star_sp"] - bp["t_star_base"]
            evolving_summary["breakpoint_shift"] = {
                m: float(bp[bp["metric"] == m]["delta_t_star"].mean())
                for m in bp["metric"].unique()
            }

    if evolving_summary:
        (out_dir / "selfplay_evolving_summary.json").write_text(
            json.dumps(evolving_summary, indent=2),
            encoding="utf-8",
        )

    print(f"wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
