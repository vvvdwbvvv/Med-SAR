#!/usr/bin/env python3
# Sample usage:
# python scriptsv2/01_build_mimic_corpus.py \
#   --input data/raw/NOTEEVENTS.csv \
#   --output_dir data/processed/mimic_v2 \
#   --text_col TEXT \
#   --min_note_chars 40

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from med_sar.data.mimic import iter_mimic_notes
from med_sar.operators.proxies import compute_proxies
from med_sar.protocol.n10_slicing import assign_time_buckets, assign_length_buckets, clean_note_type


def _require_parquet() -> None:
    try:
        import pyarrow  # noqa: F401
    except Exception as exc:  # pragma: no cover - runtime check
        raise SystemExit(
            "pyarrow is required for parquet outputs. Install it and retry."
        ) from exc


def _token_count(text: str) -> int:
    return len(text.split())


def build_manifest(
    *,
    input_paths: List[str],
    text_col: Optional[str],
    note_id_col: Optional[str],
    subject_id_col: Optional[str],
    hadm_id_col: Optional[str],
    stay_id_col: Optional[str],
    charttime_col: Optional[str],
    category_col: Optional[str],
    description_col: Optional[str],
    min_note_chars: int,
    max_notes: Optional[int],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    notes_seen = 0
    for note in iter_mimic_notes(
        input_paths,
        text_col=text_col,
        note_id_col=note_id_col,
        subject_id_col=subject_id_col,
        hadm_id_col=hadm_id_col,
        stay_id_col=stay_id_col,
        charttime_col=charttime_col,
        category_col=category_col,
        description_col=description_col,
        min_chars=min_note_chars,
    ):
        notes_seen += 1
        if max_notes and notes_seen > max_notes:
            break
        text = note["text"]
        proxies = compute_proxies(text)
        rows.append(
            {
                "note_id": note.get("note_id"),
                "patient_id": note.get("subject_id"),
                "hadm_id": note.get("hadm_id"),
                "stay_id": note.get("stay_id"),
                "charttime": note.get("charttime"),
                "note_type_clean": clean_note_type(
                    note.get("category") or note.get("description")
                ),
                "length_tokens": _token_count(text),
                **proxies,
                "text": text,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True)
    ap.add_argument("--output_dir", type=str, default="data/processed/mimic_v2")
    ap.add_argument("--text_col", type=str, default="TEXT")
    ap.add_argument("--note_id_col", type=str, default=None)
    ap.add_argument("--subject_id_col", type=str, default=None)
    ap.add_argument("--hadm_id_col", type=str, default=None)
    ap.add_argument("--stay_id_col", type=str, default=None)
    ap.add_argument("--charttime_col", type=str, default=None)
    ap.add_argument("--category_col", type=str, default=None)
    ap.add_argument("--description_col", type=str, default=None)
    ap.add_argument("--min_note_chars", type=int, default=40)
    ap.add_argument("--max_notes", type=int, default=None)
    ap.add_argument("--time_buckets", type=int, default=3)
    ap.add_argument("--length_buckets", type=int, default=3)
    args = ap.parse_args()

    _require_parquet()

    df = build_manifest(
        input_paths=args.input,
        text_col=args.text_col,
        note_id_col=args.note_id_col,
        subject_id_col=args.subject_id_col,
        hadm_id_col=args.hadm_id_col,
        stay_id_col=args.stay_id_col,
        charttime_col=args.charttime_col,
        category_col=args.category_col,
        description_col=args.description_col,
        min_note_chars=args.min_note_chars,
        max_notes=args.max_notes,
    )

    df["time_bucket"] = assign_time_buckets(df, time_col="charttime", k=args.time_buckets)
    df["admission_time_bucket"] = df["time_bucket"]
    df["length_bucket"] = assign_length_buckets(
        df, length_col="length_tokens", k=args.length_buckets
    )

    proxy_cols = [
        "newline_ratio",
        "colon_ratio",
        "digit_ratio",
        "header_density",
        "abbrev_ratio",
    ]
    proxy_stats = {
        proxy: {
            "median": float(df[proxy].median()) if proxy in df.columns else 0.0,
            "p90": float(df[proxy].quantile(0.9)) if proxy in df.columns else 0.0,
        }
        for proxy in proxy_cols
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    notes_path = out_dir / "mimic_notes.parquet"
    manifest_path = out_dir / "mimic_manifest.parquet"

    df.to_parquet(notes_path, index=False)
    df.drop(columns=["text"]).to_parquet(manifest_path, index=False)

    import json

    (out_dir / "proxy_stats.json").write_text(
        json.dumps(proxy_stats, indent=2), encoding="utf-8"
    )

    print(f"[mimic_v2] notes={len(df)}")
    print(f"[mimic_v2] wrote: {notes_path}")
    print(f"[mimic_v2] wrote: {manifest_path}")
    print(f"[mimic_v2] wrote: {out_dir / 'proxy_stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
