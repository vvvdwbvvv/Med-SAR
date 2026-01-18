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
    if time_col not in df.columns:
        return pd.Series([None] * len(df), index=df.index)
    ts = pd.to_datetime(df[time_col], errors="coerce")
    if ts.notna().sum() < k:
        return pd.Series([None] * len(df), index=df.index)
    values = ts.view("int64")
    try:
        buckets = pd.qcut(values, q=k, labels=False, duplicates="drop")
    except ValueError:
        buckets = pd.cut(values, bins=k, labels=False)
    return buckets


def assign_length_buckets(
    df: pd.DataFrame, *, length_col: str = "length_tokens", k: int = 3
) -> pd.Series:
    if length_col not in df.columns:
        return pd.Series([None] * len(df), index=df.index)
    values = df[length_col].astype(float)
    if values.notna().sum() < k:
        return pd.Series([None] * len(df), index=df.index)
    try:
        buckets = pd.qcut(values, q=k, labels=False, duplicates="drop")
    except ValueError:
        buckets = pd.cut(values, bins=k, labels=False)
    return buckets


def build_slice_index(
    df: pd.DataFrame,
    *,
    time_bucket_col: str = "time_bucket",
    note_type_col: str = "note_type_clean",
    length_bucket_col: str = "length_bucket",
) -> pd.DataFrame:
    df = df.copy()
    if note_type_col in df.columns:
        df[note_type_col] = df[note_type_col].apply(clean_note_type)
    else:
        df[note_type_col] = "unknown"

    if time_bucket_col not in df.columns:
        df[time_bucket_col] = None
    if length_bucket_col not in df.columns:
        df[length_bucket_col] = None

    df["slice_id"] = (
        "t"
        + df[time_bucket_col].astype("Int64").astype(str)
        + "|type"
        + df[note_type_col].astype(str)
        + "|len"
        + df[length_bucket_col].astype("Int64").astype(str)
    )
    return df
