# Sample usage:
# python scriptsv2/07_eval_baseline.py \
#   --preds outputs/preds.parquet \
#   --out outputs/metrics.json

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


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
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--id_col", type=str, default="id")
    ap.add_argument("--t_col", type=str, default="t")
    ap.add_argument("--y_true_col", type=str, default="y_true")
    ap.add_argument("--y_pred_col", type=str, default="y_pred")
    ap.add_argument("--y_prob_col", type=str, default="y_prob")
    ap.add_argument("--guard_pass_col", type=str, default="guard_pass")
    args = ap.parse_args()

    df = _load_table(args.preds)

    metrics: Dict[str, Dict[str, float]] = {}
    t_values = (
        sorted(df[args.t_col].dropna().unique().tolist())
        if args.t_col in df.columns
        else [None]
    )

    base_preds = None
    if args.t_col in df.columns and args.id_col in df.columns:
        base = df[df[args.t_col] == min(t_values)]
        base_preds = (
            base.set_index(args.id_col)[args.y_pred_col].to_dict()
            if not base.empty
            else None
        )

    for t in t_values:
        slice_df = df if t is None else df[df[args.t_col] == t]
        if slice_df.empty:
            continue
        y_true = slice_df[args.y_true_col].to_numpy()
        y_pred = slice_df[args.y_pred_col].to_numpy()
        acc = float((y_true == y_pred).mean())

        stability = None
        if base_preds and args.id_col in slice_df.columns:
            same = 0
            total = 0
            for _, row in slice_df.iterrows():
                base_pred = base_preds.get(row[args.id_col])
                if base_pred is None:
                    continue
                total += 1
                if base_pred == row[args.y_pred_col]:
                    same += 1
            stability = float(same / max(1, total))

        fact_error = None
        if args.guard_pass_col in slice_df.columns:
            fact_error = float(1.0 - slice_df[args.guard_pass_col].mean())
        elif "fact_error" in slice_df.columns:
            fact_error = float(slice_df["fact_error"].mean())

        calibration = None
        if args.y_prob_col in slice_df.columns:
            y_prob = slice_df[args.y_prob_col].to_numpy()
            calibration = _ece((y_true == y_pred).astype(int), y_prob)

        metrics[str(t)] = {
            "accuracy": acc,
            "stability": stability,
            "fact_error": fact_error,
            "calibration": calibration,
            "n": int(len(slice_df)),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"wrote metrics to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
