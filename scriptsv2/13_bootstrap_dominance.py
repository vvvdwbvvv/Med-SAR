# Sample usage:
# python scriptsv2/13_bootstrap_dominance.py \
#   --preds outputs/preds.parquet \
#   --method_a goc \
#   --method_b baseline \
#   --out outputs/dominance.json

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from med_sar.protocol.bootstrap import bootstrap_dominance


def _load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix in {".csv", ".tsv"}:
        return pd.read_csv(p)
    if p.suffix == ".jsonl":
        return pd.read_json(p, lines=True)
    raise ValueError(f"Unsupported file format: {p}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=str, required=True)
    ap.add_argument("--method_a", type=str, required=True)
    ap.add_argument("--method_b", type=str, required=True)
    ap.add_argument("--method_col", type=str, default="method")
    ap.add_argument("--patient_col", type=str, default="patient_id")
    ap.add_argument("--y_true_col", type=str, default="y_true")
    ap.add_argument("--y_pred_col", type=str, default="y_pred")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = _load_table(args.preds)

    if "accuracy" not in df.columns and args.y_true_col in df.columns:
        df["accuracy"] = (df[args.y_true_col] == df[args.y_pred_col]).astype(float)

    metric_cols: List[str] = ["accuracy"]
    if "stability" in df.columns:
        metric_cols.append("stability")
    if "fact_error" in df.columns:
        metric_cols.append("fact_error")
    if "calibration" in df.columns:
        metric_cols.append("calibration")

    maximize = {
        "accuracy": True,
        "stability": True,
        "fact_error": False,
        "calibration": False,
    }

    result = bootstrap_dominance(
        df,
        method_col=args.method_col,
        patient_col=args.patient_col,
        metrics=metric_cols,
        maximize=maximize,
        method_a=args.method_a,
        method_b=args.method_b,
        n_boot=args.n_boot,
        seed=args.seed,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "method_a": args.method_a,
                "method_b": args.method_b,
                "metrics": metric_cols,
                "dominance_rate": result.dominance_rate,
                "ci_low": result.ci_low,
                "ci_high": result.ci_high,
                "n_boot": result.n_boot,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
