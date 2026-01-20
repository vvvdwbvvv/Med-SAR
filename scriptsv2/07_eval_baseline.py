# Sample usage:
# python scriptsv2/07_eval_baseline.py \
#   --preds outputs/preds.parquet \
#   --out outputs/metrics.json

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

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


def _parse_reasons(val) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(v) for v in val]
    if isinstance(val, str):
        if ";" in val:
            return [v for v in val.split(";") if v]
        return [val] if val else []
    return []


_F_MAP = {
    "number_mismatch": "F1",
    "negation_flip": "F2",
    "unit_mismatch": "F3",
    "entity_drop": "F4",
    "length_ratio": "F5",
}


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

    def compute_metrics(slice_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        y_true = slice_df[args.y_true_col].to_numpy()
        y_pred = slice_df[args.y_pred_col].to_numpy()
        perf_f1 = None
        perf_auc = None
        try:
            perf_f1 = float(f1_score(y_true, y_pred, average="macro"))
        except Exception:
            perf_f1 = None
        if args.y_prob_col in slice_df.columns:
            try:
                perf_auc = float(
                    roc_auc_score(
                        y_true.astype(int), slice_df[args.y_prob_col].to_numpy()
                    )
                )
            except Exception:
                perf_auc = None

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

        fact_overall = None
        f_vec = {f"F{i}": None for i in range(1, 6)}
        if args.guard_pass_col in slice_df.columns:
            fact_overall = float(1.0 - slice_df[args.guard_pass_col].mean())
        elif "fact_error" in slice_df.columns:
            fact_overall = float(slice_df["fact_error"].mean())

        for i in range(1, 6):
            col = f"F{i}"
            if col in slice_df.columns:
                f_vec[col] = float(slice_df[col].mean())

        if (
            any(v is None for v in f_vec.values())
            and "guard_reasons" in slice_df.columns
        ):
            rows = slice_df["guard_reasons"].apply(_parse_reasons)
            rows = rows.apply(lambda rs: [_F_MAP.get(r, r) for r in rs])
            for i in range(1, 6):
                key = f"F{i}"
                if f_vec[key] is None:
                    tag = f"F{i}"
                    f_vec[key] = float(rows.apply(lambda r: tag in r).mean())

        brier = None
        ece = None
        if args.y_prob_col in slice_df.columns:
            y_prob = slice_df[args.y_prob_col].to_numpy()
            brier = float(np.mean((y_prob - y_true) ** 2))
            ece = _ece((y_true == y_pred).astype(int), y_prob)

        return {
            "performance": {"f1": perf_f1, "auroc": perf_auc},
            "stability": {"M": 5, "topk": 5, "jaccard": stability},
            "fact_error": {"overall": fact_overall, **f_vec},
            "calibration": {"brier_micro": brier, "ece_optional": ece},
            "n": int(len(slice_df)),
        }

    by_t: Dict[str, Dict[str, Dict[str, float]]] = {}
    for t in t_values:
        slice_df = df if t is None else df[df[args.t_col] == t]
        if slice_df.empty:
            continue
        by_t[str(t)] = compute_metrics(slice_df)

    base_metrics = None
    if t_values:
        base_key = str(t_values[0])
        if base_key in by_t:
            base_metrics = by_t[base_key]
    if base_metrics is None and by_t:
        base_metrics = list(by_t.values())[0]
    else:
        base_metrics = compute_metrics(df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = base_metrics or {}
    if len(by_t) > 1:
        payload["by_t"] = by_t
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote metrics to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
