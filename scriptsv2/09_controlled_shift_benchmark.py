# Sample usage:
# python scriptsv2/09_controlled_shift_benchmark.py \
#   --preds outputs/preds.parquet \
#   --manifest data/processed/mimic_v2/mimic_manifest.parquet \
#   --out_dir outputs/controlled_shift_v2

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from med_sar.protocol.n10_slicing import (
    assign_length_buckets,
    assign_time_buckets,
    build_slice_index,
    clean_note_type,
)
from med_sar.protocol.frontier import pareto_frontier, compute_breakpoints


def _load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix in {".csv", ".tsv"}:
        return pd.read_csv(p)
    if p.suffix == ".jsonl":
        return pd.read_json(p, lines=True)
    raise ValueError(f"Unsupported file format: {p}")


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if not mask.any():
            continue
        acc = (y_true[mask] == 1).mean()
        conf = y_prob[mask].mean()
        ece += abs(acc - conf) * mask.mean()
    return float(ece)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=str, required=True)
    ap.add_argument("--manifest", type=str, default=None)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--id_col", type=str, default="id")
    ap.add_argument("--t_col", type=str, default="t")
    ap.add_argument("--y_true_col", type=str, default="y_true")
    ap.add_argument("--y_pred_col", type=str, default="y_pred")
    ap.add_argument("--y_prob_col", type=str, default="y_prob")
    ap.add_argument("--guard_pass_col", type=str, default="guard_pass")
    ap.add_argument("--min_slice_size", type=int, default=200)
    ap.add_argument("--time_buckets", type=int, default=3)
    ap.add_argument("--length_buckets", type=int, default=3)
    ap.add_argument("--perf_drop", type=float, default=0.05)
    ap.add_argument("--stability_drop", type=float, default=0.05)
    ap.add_argument("--fact_increase", type=float, default=0.05)
    ap.add_argument("--calib_increase", type=float, default=0.05)
    args = ap.parse_args()

    df = _load_table(args.preds)

    if args.manifest:
        manifest = _load_table(args.manifest)
        if args.id_col in manifest.columns and args.id_col in df.columns:
            df = df.merge(manifest, on=args.id_col, how="left")

    if "note_type_clean" in df.columns:
        df["note_type_clean"] = df["note_type_clean"].apply(clean_note_type)
    else:
        df["note_type_clean"] = "unknown"

    if "time_bucket" not in df.columns and "admission_time_bucket" in df.columns:
        df["time_bucket"] = df["admission_time_bucket"]
    if "time_bucket" not in df.columns and "charttime" in df.columns:
        df["time_bucket"] = assign_time_buckets(
            df, time_col="charttime", k=args.time_buckets
        )
    if "length_bucket" not in df.columns and "length_tokens" in df.columns:
        df["length_bucket"] = assign_length_buckets(
            df, length_col="length_tokens", k=args.length_buckets
        )

    df = build_slice_index(df)

    t_values = (
        sorted(df[args.t_col].dropna().unique().tolist())
        if args.t_col in df.columns
        else [None]
    )
    base_t = min(t_values) if t_values else None

    if base_t is not None:
        base_counts = (
            df[df[args.t_col] == base_t].groupby("slice_id")[args.id_col].nunique()
        )
        keep_slices = base_counts[base_counts >= args.min_slice_size].index.tolist()
        df = df[df["slice_id"].isin(keep_slices)]

    slice_meta = (
        df.groupby(["slice_id", "time_bucket", "note_type_clean", "length_bucket"])[
            args.id_col
        ]
        .nunique()
        .reset_index()
        .rename(columns={args.id_col: "n"})
    )

    base_preds = None
    if base_t is not None and args.id_col in df.columns:
        base_df = df[df[args.t_col] == base_t]
        base_preds = base_df.set_index(args.id_col)[args.y_pred_col].to_dict()

    rows: List[Dict] = []
    for slice_id, slice_df in df.groupby("slice_id"):
        for t in t_values:
            t_df = slice_df if t is None else slice_df[slice_df[args.t_col] == t]
            if t_df.empty:
                continue
            y_true = t_df[args.y_true_col].to_numpy()
            y_pred = t_df[args.y_pred_col].to_numpy()
            acc = float((y_true == y_pred).mean())

            stability = None
            if base_preds is not None:
                same = 0
                total = 0
                for _, row in t_df.iterrows():
                    base_pred = base_preds.get(row[args.id_col])
                    if base_pred is None:
                        continue
                    total += 1
                    if base_pred == row[args.y_pred_col]:
                        same += 1
                stability = float(same / max(1, total))

            fact_error = None
            if args.guard_pass_col in t_df.columns:
                fact_error = float(1.0 - t_df[args.guard_pass_col].mean())
            elif "fact_error" in t_df.columns:
                fact_error = float(t_df["fact_error"].mean())

            calibration = None
            if args.y_prob_col in t_df.columns:
                y_prob = t_df[args.y_prob_col].to_numpy()
                calibration = _ece((y_true == y_pred).astype(int), y_prob)

            rows.append(
                {
                    "slice_id": slice_id,
                    "t": t,
                    "accuracy": acc,
                    "stability": stability,
                    "fact_error": fact_error,
                    "calibration": calibration,
                    "n": int(len(t_df)),
                }
            )

    metrics_df = pd.DataFrame(rows)

    metric_cols = [
        col
        for col in ["accuracy", "stability", "fact_error", "calibration"]
        if col in metrics_df.columns and metrics_df[col].notna().any()
    ]
    maximize = {
        "accuracy": True,
        "stability": True,
        "fact_error": False,
        "calibration": False,
    }
    thresholds = {
        "accuracy": args.perf_drop,
        "stability": args.stability_drop,
        "fact_error": args.fact_increase,
        "calibration": args.calib_increase,
    }

    breakpoints = compute_breakpoints(
        metrics_df,
        group_cols=["slice_id"],
        t_col="t",
        metrics=metric_cols,
        maximize=maximize,
        thresholds=thresholds,
    )

    frontier = metrics_df.groupby("t")[metric_cols].mean().reset_index()
    frontier = frontier.dropna(subset=metric_cols)
    frontier["on_frontier"] = pareto_frontier(
        frontier, metrics=metric_cols, maximize=maximize
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slice_meta.to_parquet(out_dir / "n10_slices.parquet", index=False)
    metrics_df.to_parquet(out_dir / "slice_metrics.parquet", index=False)
    frontier.to_parquet(out_dir / "frontier_points.parquet", index=False)
    breakpoints.to_parquet(out_dir / "breakpoints.parquet", index=False)

    print(f"wrote {out_dir / 'n10_slices.parquet'}")
    print(f"wrote {out_dir / 'slice_metrics.parquet'}")
    print(f"wrote {out_dir / 'frontier_points.parquet'}")
    print(f"wrote {out_dir / 'breakpoints.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
