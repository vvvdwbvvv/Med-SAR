#!/usr/bin/env python3
# Sample usage:
# python scriptsv2/01_build_mimic_corpus.py \
#   --input data/raw/NOTEEVENTS.csv \
#   --output_dir data/processed/mimic_v2 \
#   --text_col TEXT \
#   --min_note_chars 40

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from med_sar.data.mimic import iter_mimic_notes
from med_sar.operators.proxies import compute_proxies
from med_sar.protocol.n10_slicing import (
    assign_time_buckets,
    assign_length_buckets,
    clean_note_type,
    build_slice_index,
)


def _require_parquet() -> None:
    try:
        import pyarrow  # noqa: F401
    except Exception as exc:  # pragma: no cover - runtime check
        raise SystemExit(
            "pyarrow is required for parquet outputs. Install it and retry."
        ) from exc


def _token_count(text: str) -> int:
    return len(text.split())


def _sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _parse_dt(v: Any) -> Optional[pd.Timestamp]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    ts = pd.to_datetime(s, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.floor("s")


def _pick_time_key(note: Dict[str, Any]) -> Optional[str]:
    """
    Choose a stable time key for slicing fallback.
    Priority:
      1) chartdate (often populated in NOTEEVENTS)
      2) charttime
      3) discharge_date
      4) admit_date
    """
    for k in ("chartdate", "charttime", "discharge_date", "admit_date"):
        v = note.get(k)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return None


def _resolve_note_time_dt(
    note: Dict[str, Any], *, time_key: Optional[str]
) -> Optional[pd.Timestamp]:
    """
    Resolve a single datetime-like value used for deterministic within-admission sorting.
    Priority:
      1) charttime
      2) chartdate
      3) time_key (already resolved fallback)
      4) discharge_date
      5) admit_date
    """
    for v in (
        note.get("charttime"),
        note.get("chartdate"),
        time_key,
        note.get("discharge_date"),
        note.get("admit_date"),
    ):
        ts = _parse_dt(v)
        if ts is not None:
            return ts
    return None


def _resolve_admit_time_dt(note: Dict[str, Any]) -> Optional[pd.Timestamp]:
    """
    Admission-time proxy extracted from note header (via iter_mimic_notes).
    May be missing or noisy; treat as a proxy.
    """
    return _parse_dt(note.get("admit_date"))


def _resolve_bucket_time_dt(
    *, admit_time_dt: Optional[pd.Timestamp], note_time_dt: Optional[pd.Timestamp]
) -> Optional[pd.Timestamp]:
    """
    Temporal bucketing axis used by N10 protocol.
    Prefer admit_time proxy when available; otherwise fall back to note_time proxy.
    """
    if admit_time_dt is not None:
        return admit_time_dt
    return note_time_dt


def build_manifest(
    *,
    input_paths: List[str],
    text_col: Optional[str],
    note_id_col: Optional[str],
    subject_id_col: Optional[str],
    hadm_id_col: Optional[str],
    stay_id_col: Optional[str],
    # IMPORTANT: mimic.py now distinguishes chartdate vs charttime
    chartdate_col: Optional[str],
    charttime_col: Optional[str],
    category_col: Optional[str],
    description_col: Optional[str],
    min_note_chars: int,
    max_notes: Optional[int],
    # Pass-through extraction controls (match mimic.py defaults)
    extract_admit_discharge: bool = True,
    extract_dictation: bool = False,
    header_window_chars: int = 2000,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    notes_seen = 0

    note_iter = iter_mimic_notes(
        input_paths,
        text_col=text_col,
        note_id_col=note_id_col,
        subject_id_col=subject_id_col,
        hadm_id_col=hadm_id_col,
        stay_id_col=stay_id_col,
        chartdate_col=chartdate_col,
        charttime_col=charttime_col,
        category_col=category_col,
        description_col=description_col,
        min_chars=min_note_chars,
        extract_admit_discharge=extract_admit_discharge,
        extract_dictation=extract_dictation,
        header_window_chars=header_window_chars,
    )

    proxy_cols = [
        "newline_ratio",
        "colon_ratio",
        "digit_ratio",
        "header_density",
        "abbrev_ratio",
    ]
    proxy_cols_prefixed = [f"proxy_{name}" for name in proxy_cols]

    pbar = tqdm(note_iter, total=max_notes, desc="Processing notes", unit="note")
    for note in pbar:
        notes_seen += 1
        if max_notes and notes_seen > max_notes:
            break

        text = note["text"]
        proxies = compute_proxies(text)
        proxies_prefixed = {f"proxy_{k}": v for k, v in proxies.items()}

        # time resolution
        time_key = _pick_time_key(note)
        note_time_dt = _resolve_note_time_dt(note, time_key=time_key)
        admit_time_dt = _resolve_admit_time_dt(note)
        bucket_time_dt = _resolve_bucket_time_dt(
            admit_time_dt=admit_time_dt, note_time_dt=note_time_dt
        )

        note_time = (
            note_time_dt.isoformat(sep=" ") if note_time_dt is not None else None
        )
        admit_time = (
            admit_time_dt.isoformat(sep=" ") if admit_time_dt is not None else None
        )
        bucket_time = (
            bucket_time_dt.isoformat(sep=" ") if bucket_time_dt is not None else None
        )
        bucket_time_basis = (
            "admit_date_header" if admit_time_dt is not None else "note_time_fallback"
        )

        note_type_raw = note.get("category") or note.get("description") or ""
        note_type_clean = clean_note_type(note_type_raw)
        note_type_is_main = note_type_clean in {"discharge", "nursing", "radiology"}

        patient_id = note.get("subject_id")
        hadm_id = note.get("hadm_id")

        note_day_offset = None
        if note_time_dt is not None and admit_time_dt is not None:
            try:
                note_day_offset = int((note_time_dt - admit_time_dt).days)
            except Exception:
                note_day_offset = None

        rows.append(
            {
                # stable split/group keys (N10 / LODO reproducibility)
                "patient_id": patient_id,
                "patient_key": f"patient:{patient_id}" if patient_id else None,
                "hadm_id": hadm_id,
                "admission_id": hadm_id,  # alias for consistency across tables
                "admission_key": f"hadm:{hadm_id}" if hadm_id else None,
                "stay_id": note.get("stay_id"),
                "note_id": note.get("note_id"),
                # original time fields
                "chartdate": note.get("chartdate"),
                "charttime": note.get("charttime"),
                # extracted from TEXT header (raw strings as parsed upstream)
                "admit_date": note.get("admit_date"),
                "discharge_date": note.get("discharge_date"),
                # resolved time fields (ISO strings, stable for parquet)
                "time_key": time_key,  # legacy fallback trace
                "note_time": note_time,  # ordering proxy
                "admit_time": admit_time,  # header-derived admission proxy
                "bucket_time": bucket_time,  # main bucketing axis (admit_time else note_time)
                "bucket_time_basis": bucket_time_basis,
                "note_day_offset": note_day_offset,  # diagnostics for header parsing sanity
                # optional extra timestamps (if enabled)
                "dictated_at": note.get("dictated_at"),
                "transcribed_at": note.get("transcribed_at"),
                # note type audit fields (defang unknown/other concerns)
                "note_type_raw": note_type_raw,
                "note_type_clean": note_type_clean,
                "note_type_is_main": note_type_is_main,
                # length features
                "length_tokens": _token_count(text),
                # proxies (exactly-5) + proxy anchoring vector/version tag
                **proxies,
                **proxies_prefixed,
                "proxy_vector": {
                    k: proxies_prefixed.get(k) for k in proxy_cols_prefixed
                },
                "proxy_set_version": "proxy5_v1",
                # reproducibility/debug joins (dedup + join w/o carrying text)
                "text_sha1": _sha1_hex(text),
                # Fact Guard audit placeholders (filled later in perturbation stage)
                "fg_parse_ok": None,
                "fg_fact_signature": None,
                "fg_violation_mask": None,
                "fg_accept": None,
                "fg_reject_reason": None,
                "resample_tries": None,
                # payload
                "text": text,
            }
        )
        pbar.set_postfix({"notes": notes_seen})

    pbar.close()
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True)
    ap.add_argument("--output_dir", type=str, default="data/processed/mimic")
    ap.add_argument("--configs_dir", type=str, default="configs")
    ap.add_argument("--text_col", type=str, default="TEXT")
    ap.add_argument("--note_id_col", type=str, default=None)
    ap.add_argument("--subject_id_col", type=str, default=None)
    ap.add_argument("--hadm_id_col", type=str, default=None)
    ap.add_argument("--stay_id_col", type=str, default=None)

    # distinguish chartdate vs charttime
    ap.add_argument("--chartdate_col", type=str, default=None)
    ap.add_argument("--charttime_col", type=str, default=None)

    ap.add_argument("--category_col", type=str, default=None)
    ap.add_argument("--description_col", type=str, default=None)
    ap.add_argument("--min_note_chars", type=int, default=40)
    ap.add_argument("--max_notes", type=int, default=None)
    ap.add_argument("--time_buckets", type=int, default=3)
    ap.add_argument("--length_buckets", type=int, default=3)

    # Pass-through extraction knobs (optional; keep defaults aligned with mimic.py)
    ap.add_argument("--extract_admit_discharge", action="store_true", default=True)
    ap.add_argument(
        "--no_extract_admit_discharge",
        action="store_false",
        dest="extract_admit_discharge",
    )
    ap.add_argument("--extract_dictation", action="store_true", default=False)
    ap.add_argument("--header_window_chars", type=int, default=2000)

    args = ap.parse_args()

    _require_parquet()

    df = build_manifest(
        input_paths=args.input,
        text_col=args.text_col,
        note_id_col=args.note_id_col,
        subject_id_col=args.subject_id_col,
        hadm_id_col=args.hadm_id_col,
        stay_id_col=args.stay_id_col,
        chartdate_col=args.chartdate_col,
        charttime_col=args.charttime_col,
        category_col=args.category_col,
        description_col=args.description_col,
        min_note_chars=args.min_note_chars,
        max_notes=args.max_notes,
        extract_admit_discharge=args.extract_admit_discharge,
        extract_dictation=args.extract_dictation,
        header_window_chars=args.header_window_chars,
    )

    # ---- N10 temporal slicing: bucket on bucket_time (admit_time proxy else note_time fallback) ----
    df["bucket_time_dt"] = pd.to_datetime(df["bucket_time"], errors="coerce")

    # If some notes still have no time, they will get NaT; keep them but they may map to a fallback bucket.
    # assign_time_buckets should handle NaT/NaN deterministically; if not, handle before calling.
    df["time_bucket"] = assign_time_buckets(
        df, time_col="bucket_time_dt", k=args.time_buckets
    )
    df["time_bucket_method"] = f"bucket_time_k{int(args.time_buckets)}"

    # DO NOT name this "admission_time_bucket" because it can fall back to note_time.
    df["temporal_bucket"] = df["time_bucket"]
    df["temporal_bucket_method"] = df["time_bucket_method"]

    # Optional: admission-only bucket (only where admit_time exists), for appendix audits
    df["admit_time_dt"] = pd.to_datetime(df["admit_time"], errors="coerce")
    df["admit_time_bucket"] = None
    if df["admit_time_dt"].notna().any():
        tmp = df[df["admit_time_dt"].notna()].copy()
        tmp["admit_time_bucket"] = assign_time_buckets(
            tmp, time_col="admit_time_dt", k=args.time_buckets
        )
        df.loc[tmp.index, "admit_time_bucket"] = tmp["admit_time_bucket"]
    df["admit_time_bucket_method"] = f"admit_time_k{int(args.time_buckets)}"

    # ---- length slicing ----
    df["length_bucket"] = assign_length_buckets(
        df, length_col="length_tokens", k=args.length_buckets
    )
    df["length_bucket_method"] = f"length_tokens_k{int(args.length_buckets)}"

    # ---- proxy stats (median/p90 anchors) ----
    proxy_cols = [
        "proxy_newline_ratio",
        "proxy_colon_ratio",
        "proxy_digit_ratio",
        "proxy_header_density",
        "proxy_abbrev_ratio",
    ]
    proxy_stats = {
        proxy: {
            "median": float(df[proxy].median()) if proxy in df.columns else 0.0,
            "p90": float(df[proxy].quantile(0.9)) if proxy in df.columns else 0.0,
        }
        for proxy in proxy_cols
    }

    # ---- time coverage audit (must-report to defang temporal slicing attacks) ----
    time_coverage = {
        "admit_time_present_rate": float(df["admit_time"].notna().mean()),
        "bucket_time_present_rate": float(df["bucket_time"].notna().mean()),
        "bucket_time_fallback_rate": float(
            (df["bucket_time_basis"] == "note_time_fallback").mean()
        ),
        "time_buckets_k": int(args.time_buckets),
        "length_buckets_k": int(args.length_buckets),
        "time_bucket_method": f"bucket_time_k{int(args.time_buckets)}",
        "length_bucket_method": f"length_tokens_k{int(args.length_buckets)}",
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir = Path(args.configs_dir)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    notes_path = out_dir / "mimic_notes.parquet"
    manifest_path = out_dir / "mimic_manifest.parquet"

    df.to_parquet(notes_path, index=False)
    df.drop(columns=["text"]).to_parquet(manifest_path, index=False)

    (out_dir / "proxy_stats.json").write_text(
        json.dumps(proxy_stats, indent=2), encoding="utf-8"
    )
    (out_dir / "time_coverage.json").write_text(
        json.dumps(time_coverage, indent=2), encoding="utf-8"
    )

    note_type_dist = (
        df["note_type_clean"].fillna("unknown").value_counts().to_dict()
        if "note_type_clean" in df.columns
        else {}
    )
    (out_dir / "note_type_distribution.json").write_text(
        json.dumps(note_type_dist, indent=2), encoding="utf-8"
    )

    df_slices = build_slice_index(df)
    slice_counts = (
        df_slices.groupby("slice_id")["note_id"].nunique().sort_values(ascending=False)
    )
    below_min = int((slice_counts < 200).sum())
    slice_audit = {
        "min_slice_size": 200,
        "total_slices": int(len(slice_counts)),
        "slices_below_min": below_min,
    }
    (out_dir / "slice_size_audit.json").write_text(
        json.dumps(slice_audit, indent=2), encoding="utf-8"
    )

    n10_protocol = {
        "time_buckets_k": int(args.time_buckets),
        "length_buckets_k": int(args.length_buckets),
        "note_types": sorted(note_type_dist.keys()),
        "min_slice_size": 200,
        "J": 5,
        "B": 1000,
        "tau_F": None,
        "tau_S": None,
    }
    (cfg_dir / "n10_protocol.yaml").write_text(
        yaml.safe_dump(n10_protocol, sort_keys=False), encoding="utf-8"
    )

    print(f"[mimic] notes={len(df)}")
    print(f"[mimic] wrote: {notes_path}")
    print(f"[mimic] wrote: {manifest_path}")
    print(f"[mimic] wrote: {out_dir / 'proxy_stats.json'}")
    print(f"[mimic] wrote: {out_dir / 'time_coverage.json'}")
    print(f"[mimic] wrote: {out_dir / 'note_type_distribution.json'}")
    print(f"[mimic] wrote: {out_dir / 'slice_size_audit.json'}")
    print(f"[mimic] wrote: {cfg_dir / 'n10_protocol.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
