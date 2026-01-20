from __future__ import annotations

import re

import pandas as pd


def clean_note_type(val: str | None) -> str:
    if not val:
        return "unknown"
    text = str(val).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def assign_time_buckets(df: pd.DataFrame, *, time_col: str, k: int = 3) -> pd.Series:
    """
    Deterministic K-bucket slicing over time with missing-safe behavior.

    - Uses ONLY non-null timestamps to compute quantile (qcut) buckets.
    - Missing / unparsable timestamps get None.
    - If there are <k valid timestamps, returns all None (prevents unstable qcut).
    """
    if time_col not in df.columns:
        return pd.Series([None] * len(df), index=df.index)

    ts = pd.to_datetime(df[time_col], errors="coerce")
    mask = ts.notna()
    n_valid = int(mask.sum())
    if n_valid < k:
        return pd.Series([None] * len(df), index=df.index)

    # int64 nanoseconds for valid entries only (avoid NaT sentinel values)
    values = ts.loc[mask].astype("int64")

    try:
        b = pd.qcut(values, q=k, labels=False, duplicates="drop")
    except ValueError:
        b = pd.cut(values, bins=k, labels=False)

    out = pd.Series([None] * len(df), index=df.index, dtype=object)
    out.loc[mask] = b.astype(int).astype(object)
    return out


def assign_length_buckets(
    df: pd.DataFrame, *, length_col: str = "length_tokens", k: int = 3
) -> pd.Series:
    """
    Deterministic K-bucket slicing over lengths with missing-safe behavior.

    - Uses ONLY non-null numeric values to compute quantile (qcut) buckets.
    - Missing / invalid values get None.
    - If there are <k valid values, returns all None.
    """
    if length_col not in df.columns:
        return pd.Series([None] * len(df), index=df.index)

    values = pd.to_numeric(df[length_col], errors="coerce")
    mask = values.notna()
    n_valid = int(mask.sum())
    if n_valid < k:
        return pd.Series([None] * len(df), index=df.index)

    v = values.loc[mask].astype(float)

    try:
        b = pd.qcut(v, q=k, labels=False, duplicates="drop")
    except ValueError:
        b = pd.cut(v, bins=k, labels=False)

    out = pd.Series([None] * len(df), index=df.index, dtype=object)
    out.loc[mask] = b.astype(int).astype(object)
    return out


def build_slice_index(
    df: pd.DataFrame,
    *,
    time_bucket_col: str = "time_bucket",
    note_type_col: str = "note_type_clean",
    length_bucket_col: str = "length_bucket",
) -> pd.DataFrame:
    """
    Build deterministic slice_id for N10 / LODO.

    slice_id format:
      t{time_bucket}|type{note_type_clean}|len{length_bucket}

    Missing buckets become 'na' (not 'None' or '<NA>') to avoid unstable stringification.
    Note type is always normalized via clean_note_type; missing becomes 'unknown'.
    """
    out = df.copy()

    # note type normalization
    if note_type_col in out.columns:
        out[note_type_col] = (
            out[note_type_col].fillna("unknown").astype(str).apply(clean_note_type)
        )
    else:
        out[note_type_col] = "unknown"

    # ensure bucket columns exist
    if time_bucket_col not in out.columns:
        out[time_bucket_col] = pd.NA
    if length_bucket_col not in out.columns:
        out[length_bucket_col] = pd.NA

    # stable stringification for buckets: Int64 -> string with 'na' for missing
    t_str = (
        out[time_bucket_col]
        .astype("Int64")
        .astype(str)
        .replace({"<NA>": "na", "None": "na"})
    )
    l_str = (
        out[length_bucket_col]
        .astype("Int64")
        .astype(str)
        .replace({"<NA>": "na", "None": "na"})
    )
    type_str = (
        out[note_type_col].astype(str).replace({"<NA>": "unknown", "None": "unknown"})
    )

    out["slice_id"] = "t" + t_str + "|type" + type_str + "|len" + l_str
    return out
