#!/usr/bin/env python3
"""
Build a normalized MIMIC clinical-notes corpus JSONL suitable for retrieval / training.

Expected input:
- A file produced by scripts/00_download_mimic_notes.sql, typically exported as CSV/TSV/JSONL/Parquet.
- Must contain at least a note text column; other common columns:
  subject_id, hadm_id, stay_id, note_id, charttime, category, description

Outputs:
- data/processed/mimic_corpus/corpus.jsonl
- data/processed/mimic_corpus/stats.json

Features:
- Deterministic ordering (stable uid).
- Optional de-identification via med_sar.privacy.deid if available.
- Sectionization heuristics for clinical notes.
- Atomic writes; no raw-text logging.

Example:
  python scripts/02_build_mimic_corpus.py \
    --input data/raw/mimic_notes.csv \
    --output_dir data/processed/mimic_corpus \
    --text_col text --id_col note_id --deid
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None


def _try_import_med_sar():
    """
    Optional integration points:
      - med_sar.utils.io.atomic_write_jsonl
      - med_sar.utils.text.normalize_text
      - med_sar.privacy.deid.deidentify (or similar)
    """
    atomic_write_jsonl = None
    normalize_text = None
    deid_fn = None

    try:
        from med_sar.utils.io import atomic_write_jsonl as _aw  # type: ignore
        atomic_write_jsonl = _aw
    except Exception:
        pass

    try:
        from med_sar.utils.text import normalize_text as _nt  # type: ignore
        normalize_text = _nt
    except Exception:
        pass

    # Be permissive: deidentify(), deid(), scrub(), etc.
    try:
        from med_sar.privacy.deid import deidentify as _deid  # type: ignore
        deid_fn = _deid
    except Exception:
        try:
            from med_sar.privacy.deid import deid as _deid2  # type: ignore
            deid_fn = _deid2
        except Exception:
            deid_fn = None

    return atomic_write_jsonl, normalize_text, deid_fn


atomic_write_jsonl, normalize_text_fn, deid_fn = _try_import_med_sar()


def normalize_text_local(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def atomic_write_jsonl_local(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _normalize(text: str) -> str:
    if normalize_text_fn:
        return normalize_text_fn(text)
    return normalize_text_local(text)


def _atomic_write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    if atomic_write_jsonl:
        atomic_write_jsonl(str(path), rows)
    else:
        atomic_write_jsonl_local(path, rows)


def _stable_uid(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\x1f")
    return h.hexdigest()[:24]


def _guess_columns(columns: List[str]) -> Dict[str, str]:
    cols_l = [c.lower() for c in columns]

    def pick(cands: List[str]) -> Optional[str]:
        for cand in cands:
            if cand in cols_l:
                return columns[cols_l.index(cand)]
        return None

    text_col = pick(["text", "note_text", "note", "content"])
    if not text_col:
        raise ValueError(f"Could not find a note text column among: {columns}. Pass --text_col explicitly.")

    return {
        "text": text_col,
        "note_id": pick(["note_id", "row_id", "id"]),
        "subject_id": pick(["subject_id", "patient_id"]),
        "hadm_id": pick(["hadm_id", "admission_id"]),
        "stay_id": pick(["stay_id", "icustay_id"]),
        "charttime": pick(["charttime", "chart_time", "time", "timestamp", "note_time"]),
        "category": pick(["category", "note_type", "type"]),
        "description": pick(["description", "title"]),
    }


SECTION_HEADER_RE = re.compile(
    r"(?m)^(?:\s*)([A-Z][A-Z0-9 /,-]{2,}):\s*$"
)

INLINE_HEADER_RE = re.compile(
    r"(?m)^(?:\s*)([A-Z][A-Z0-9 /,-]{2,}):\s*(.+?)\s*$"
)


def sectionize(text: str, max_sections: int = 80) -> List[Dict[str, Any]]:
    """
    Heuristic sectionizer:
      - recognizes lines like 'HISTORY OF PRESENT ILLNESS:' and optionally inline content.
      - returns [{title, start, end, text}]
    """
    text = text or ""
    lines = text.splitlines(True)  # keep line endings
    # Gather candidate headers with their character offsets
    offsets: List[Tuple[int, str, Optional[str]]] = []
    cursor = 0
    for ln in lines:
        m_inline = INLINE_HEADER_RE.match(ln)
        if m_inline:
            title = m_inline.group(1).strip()
            rest = m_inline.group(2).strip()
            offsets.append((cursor, title, rest))
        else:
            m = SECTION_HEADER_RE.match(ln)
            if m:
                title = m.group(1).strip()
                offsets.append((cursor, title, None))
        cursor += len(ln)

    if not offsets:
        return [{"title": "NOTE", "start": 0, "end": len(text), "text": text}]

    # Deduplicate near-duplicate headers
    dedup: List[Tuple[int, str, Optional[str]]] = []
    prev_pos = -10**9
    prev_title = ""
    for pos, title, rest in offsets:
        if pos - prev_pos < 3 and title == prev_title:
            continue
        dedup.append((pos, title, rest))
        prev_pos, prev_title = pos, title

    offsets = dedup[:max_sections]

    sections: List[Dict[str, Any]] = []
    for i, (pos, title, rest) in enumerate(offsets):
        start = pos
        end = offsets[i + 1][0] if i + 1 < len(offsets) else len(text)
        seg = text[start:end]
        # If inline header had rest, ensure it is included (already in seg).
        sections.append({"title": title, "start": start, "end": end, "text": seg})
    return sections


def _read_records(input_path: Path) -> List[Dict[str, Any]]:
    suffix = input_path.suffix.lower()

    if suffix == ".jsonl":
        out: List[Dict[str, Any]] = []
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out

    if suffix == ".json":
        with input_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
            return obj["data"]
        raise ValueError("JSON input must be a list or a dict with a top-level 'data' list.")

    if suffix in [".csv", ".tsv"]:
        dialect = "excel-tab" if suffix == ".tsv" else "excel"
        with input_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, dialect=dialect)
            return list(reader)

    if suffix == ".parquet":
        if pd is None:
            raise RuntimeError("pandas is required to read parquet; please install pandas.")
        df = pd.read_parquet(str(input_path))
        return df.to_dict(orient="records")

    raise ValueError(f"Unsupported input format: {input_path.name}")


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None or x == "":
            return None
        return int(float(x))
    except Exception:
        return None


def _maybe_deid(text: str, enable: bool) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (processed_text, deid_meta).
    Does not fail the pipeline if deid is unavailable; it will annotate status accordingly.
    """
    if not enable:
        return text, {"enabled": False}

    if deid_fn is None:
        return text, {"enabled": True, "applied": False, "reason": "deid_fn_unavailable"}

    try:
        out = deid_fn(text)  # type: ignore[misc]
        # allow either str or (str, meta)
        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], str) and isinstance(out[1], dict):
            return out[0], {"enabled": True, "applied": True, **out[1]}
        if isinstance(out, str):
            return out, {"enabled": True, "applied": True}
        return text, {"enabled": True, "applied": False, "reason": "unexpected_deid_return_type"}
    except Exception:
        # Do not log raw text; only signal failure.
        return text, {"enabled": True, "applied": False, "reason": "exception"}


@dataclass
class Note:
    uid: str
    text: str
    meta: Dict[str, Any]
    sections: List[Dict[str, Any]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="Path to extracted MIMIC notes (csv/tsv/jsonl/json/parquet).")
    ap.add_argument("--output_dir", type=str, default="data/processed/mimic_corpus")
    ap.add_argument("--text_col", type=str, default=None, help="Text column (default: auto-detect).")
    ap.add_argument("--id_col", type=str, default=None, help="Note id column (default: auto-detect).")
    ap.add_argument("--subject_col", type=str, default=None, help="subject_id column (default: auto-detect).")
    ap.add_argument("--hadm_col", type=str, default=None, help="hadm_id column (default: auto-detect).")
    ap.add_argument("--stay_col", type=str, default=None, help="stay_id/icustay_id column (default: auto-detect).")
    ap.add_argument("--charttime_col", type=str, default=None, help="charttime column (default: auto-detect).")
    ap.add_argument("--category_col", type=str, default=None, help="category column (default: auto-detect).")
    ap.add_argument("--description_col", type=str, default=None, help="description/title column (default: auto-detect).")
    ap.add_argument("--deid", action="store_true", help="Apply de-identification if available (recommended).")
    ap.add_argument("--min_chars", type=int, default=40, help="Drop notes shorter than this many characters after normalization.")
    ap.add_argument("--max_notes", type=int, default=0, help="Optional cap for debugging (0 = no cap).")
    args = ap.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)

    records = _read_records(input_path)
    if not records:
        raise RuntimeError("No records found in input.")

    cols = list(records[0].keys())
    guessed = _guess_columns(cols)

    text_col = args.text_col or guessed["text"]
    id_col = args.id_col or guessed["note_id"]
    subject_col = args.subject_col or guessed["subject_id"]
    hadm_col = args.hadm_col or guessed["hadm_id"]
    stay_col = args.stay_col or guessed["stay_id"]
    charttime_col = args.charttime_col or guessed["charttime"]
    category_col = args.category_col or guessed["category"]
    description_col = args.description_col or guessed["description"]

    notes: List[Note] = []
    n_seen = 0
    n_dropped_short = 0

    for r in records:
        n_seen += 1
        raw_text = r.get(text_col, "")
        if raw_text is None:
            continue
        text = _normalize(str(raw_text))
        if len(text) < args.min_chars:
            n_dropped_short += 1
            continue

        # De-identify if enabled
        text_deid, deid_meta = _maybe_deid(text, enable=args.deid)

        # Build uid
        raw_id = str(r.get(id_col, "")).strip() if (id_col and r.get(id_col) is not None) else ""
        subject_id = str(r.get(subject_col, "")).strip() if subject_col else ""
        hadm_id = str(r.get(hadm_col, "")).strip() if hadm_col else ""
        charttime = str(r.get(charttime_col, "")).strip() if charttime_col else ""

        uid = _stable_uid(raw_id, subject_id, hadm_id, charttime, text_deid[:200])

        meta: Dict[str, Any] = {
            "note_id": raw_id or None,
            "subject_id": _safe_int(r.get(subject_col)) if subject_col else None,
            "hadm_id": _safe_int(r.get(hadm_col)) if hadm_col else None,
            "stay_id": _safe_int(r.get(stay_col)) if stay_col else None,
            "charttime": charttime or None,
            "category": (str(r.get(category_col)).strip() if category_col and r.get(category_col) is not None else None),
            "description": (str(r.get(description_col)).strip() if description_col and r.get(description_col) is not None else None),
            "deid": deid_meta,
        }

        secs = sectionize(text_deid)

        notes.append(Note(uid=uid, text=text_deid, meta=meta, sections=secs))

        if args.max_notes and len(notes) >= args.max_notes:
            break

    # Deterministic order: sort by uid
    notes.sort(key=lambda n: n.uid)

    def rows() -> Iterable[Dict[str, Any]]:
        for n in notes:
            yield {
                "id": n.uid,
                "text": n.text,
                "meta": n.meta,
                "sections": n.sections,
            }

    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_jsonl(out_dir / "corpus.jsonl", rows())

    # Stats (no raw text)
    lengths = [len(n.text) for n in notes]
    lengths_sorted = sorted(lengths)
    def pct(p: float) -> Optional[int]:
        if not lengths_sorted:
            return None
        i = max(0, min(len(lengths_sorted) - 1, int(round(p * (len(lengths_sorted) - 1)))))
        return lengths_sorted[i]

    stats = {
        "built_at": datetime.utcnow().isoformat() + "Z",
        "input": str(input_path),
        "output_dir": str(out_dir),
        "counts": {
            "seen": n_seen,
            "kept": len(notes),
            "dropped_short": n_dropped_short,
        },
        "text_len_chars": {
            "min": lengths_sorted[0] if lengths_sorted else None,
            "p50": pct(0.50),
            "p95": pct(0.95),
            "max": lengths_sorted[-1] if lengths_sorted else None,
            "mean": (sum(lengths_sorted) / len(lengths_sorted)) if lengths_sorted else None,
        },
        "columns": {
            "text_col": text_col,
            "id_col": id_col,
            "subject_col": subject_col,
            "hadm_col": hadm_col,
            "stay_col": stay_col,
            "charttime_col": charttime_col,
            "category_col": category_col,
            "description_col": description_col,
        },
        "deid": {
            "requested": bool(args.deid),
            "available": bool(deid_fn is not None),
        },
    }

    tmp = out_dir / "stats.json.tmp"
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    tmp.replace(out_dir / "stats.json")

    print(f"[mimic] seen={n_seen} kept={len(notes)} dropped_short={n_dropped_short} deid_requested={bool(args.deid)} deid_available={bool(deid_fn is not None)}")
    print(f"[mimic] wrote: {out_dir}/(corpus.jsonl, stats.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
