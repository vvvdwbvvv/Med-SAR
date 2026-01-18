from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DominanceResult:
    dominance_rate: float
    ci_low: float
    ci_high: float
    n_boot: int


def _dominates(a: Dict[str, float], b: Dict[str, float], maximize: Dict[str, bool]) -> bool:
    better_or_equal = True
    strictly_better = False
    for m, a_val in a.items():
        b_val = b.get(m)
        if b_val is None:
            continue
        if maximize.get(m, True):
            if a_val < b_val:
                better_or_equal = False
                break
            if a_val > b_val:
                strictly_better = True
        else:
            if a_val > b_val:
                better_or_equal = False
                break
            if a_val < b_val:
                strictly_better = True
    return better_or_equal and strictly_better


def bootstrap_dominance(
    df: pd.DataFrame,
    *,
    method_col: str,
    patient_col: str,
    metrics: Sequence[str],
    maximize: Dict[str, bool],
    method_a: str,
    method_b: str,
    n_boot: int = 200,
    seed: int = 0,
) -> DominanceResult:
    rng = np.random.default_rng(seed)
    patients = df[patient_col].dropna().unique().tolist()
    if not patients:
        return DominanceResult(0.0, 0.0, 0.0, n_boot)

    rates = []
    for _ in range(n_boot):
        sample = rng.choice(patients, size=len(patients), replace=True)
        sample_df = df[df[patient_col].isin(sample)]
        agg = (
            sample_df.groupby([method_col, patient_col])[list(metrics)]
            .mean()
            .reset_index()
            .groupby(method_col)[list(metrics)]
            .mean()
        )
        if method_a not in agg.index or method_b not in agg.index:
            rates.append(0.0)
            continue
        a_vals = agg.loc[method_a].to_dict()
        b_vals = agg.loc[method_b].to_dict()
        rates.append(1.0 if _dominates(a_vals, b_vals, maximize) else 0.0)

    rates_arr = np.array(rates)
    dominance_rate = float(rates_arr.mean())
    ci_low = float(np.percentile(rates_arr, 2.5))
    ci_high = float(np.percentile(rates_arr, 97.5))
    return DominanceResult(dominance_rate, ci_low, ci_high, n_boot)
