from __future__ import annotations

from typing import Dict, Sequence

import pandas as pd


def pareto_frontier(
    df: pd.DataFrame,
    *,
    metrics: Sequence[str],
    maximize: Dict[str, bool],
) -> pd.Series:
    values = df[metrics].to_numpy()
    n = len(df)
    is_frontier = [True] * n
    for i in range(n):
        if not is_frontier[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            better_or_equal = True
            strictly_better = False
            for k, m in enumerate(metrics):
                if maximize.get(m, True):
                    if values[j, k] < values[i, k]:
                        better_or_equal = False
                        break
                    if values[j, k] > values[i, k]:
                        strictly_better = True
                else:
                    if values[j, k] > values[i, k]:
                        better_or_equal = False
                        break
                    if values[j, k] < values[i, k]:
                        strictly_better = True
            if better_or_equal and strictly_better:
                is_frontier[i] = False
                break
    return pd.Series(is_frontier, index=df.index)


def compute_breakpoints(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    t_col: str,
    metrics: Sequence[str],
    maximize: Dict[str, bool],
    thresholds: Dict[str, float],
) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(list(group_cols))
    for group_keys, group_df in grouped:
        group_df = group_df.sort_values(t_col)
        if group_df.empty:
            continue
        t0 = group_df[t_col].min()
        base = group_df[group_df[t_col] == t0].iloc[0]
        for m in metrics:
            base_val = float(base[m])
            threshold = thresholds.get(m, 0.05)
            t_star = None
            for _, row in group_df.iterrows():
                val = float(row[m])
                if maximize.get(m, True):
                    if val < base_val * (1 - threshold):
                        t_star = float(row[t_col])
                        break
                else:
                    if val > base_val * (1 + threshold):
                        t_star = float(row[t_col])
                        break
            rows.append(
                {
                    **{
                        col: key
                        for col, key in zip(
                            group_cols,
                            group_keys
                            if isinstance(group_keys, tuple)
                            else (group_keys,),
                        )
                    },
                    "metric": m,
                    "t_star": t_star,
                    "baseline": base_val,
                    "threshold": threshold,
                }
            )
    return pd.DataFrame(rows)
