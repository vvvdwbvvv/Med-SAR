from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

from utils.io import write_jsonl, read_jsonl, read_json, read_csv
from utils.text import normalize_text


def stable_uid(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8", errors="ignore"))
        h.update(b"\x1f")
    return h.hexdigest()[:24]


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    # pandas NaN (float) or numpy.nan
    try:
        if isinstance(v, float) and math.isnan(v):
            return True
    except Exception:
        pass
    if isinstance(v, str) and v.strip() == "":
        return True
    return False


def _clean_scalar(v: Any) -> Optional[str]:
    if _is_missing(v):
        return None
    s = str(v).strip()
    # defensive against pandas stringification
    if s.lower() in {"nan", "none", "null"}:
        return None
    return s


def _infer_col(columns: List[str], candidates: List[str]) -> Optional[str]:
    # exact match first
    colset = set(columns)
    for c in candidates:
        if c in colset:
            return c
    # case-insensitive fallback
    lower_map = {c.lower(): c for c in columns}
    for c in candidates:
        hit = lower_map.get(c.lower())
        if hit is not None:
            return hit
    return None


# -------------------------
# Date extraction utilities
# -------------------------

# Matches:
#   Admission Date:  [**2151-7-16**]
#   Admission Date:  2151-07-16
_ADMIT_RE = re.compile(
    r"\bAdmission\s*Date\s*:\s*(?:\[\*\*)?\s*(\d{4}-\d{1,2}-\d{1,2})\s*(?:\*\*\])?",
    flags=re.IGNORECASE,
)
_DISCH_RE = re.compile(
    r"\bDischarge\s*Date\s*:\s*(?:\[\*\*)?\s*(\d{4}-\d{1,2}-\d{1,2})\s*(?:\*\*\])?",
    flags=re.IGNORECASE,
)

# Optional: dictation/transcription timestamps often appear as:
#   D:  [**2151-8-5**]  12:11
#   T:  [**2151-8-5**]  12:21
_DICTATED_RE = re.compile(
    r"(?m)^\s*D\s*:\s*(?:\[\*\*)?\s*(\d{4}-\d{1,2}-\d{1,2})\s*(?:\*\*\])?\s*([0-2]?\d:[0-5]\d)?",
    flags=re.IGNORECASE,
)
_TRANSCRIBED_RE = re.compile(
    r"(?m)^\s*T\s*:\s*(?:\[\*\*)?\s*(\d{4}-\d{1,2}-\d{1,2})\s*(?:\*\*\])?\s*([0-2]?\d:[0-5]\d)?",
    flags=re.IGNORECASE,
)


def _normalize_ymd(date_s: str) -> Optional[str]:
    """
    Normalize YYYY-M-D or YYYY-MM-DD -> YYYY-MM-DD.
    Returns None if parsing fails.
    """
    if not date_s:
        return None
    s = date_s.strip()
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if not m:
        return None
    y, mo, d = m.groups()
    try:
        # Validate date
        dt = datetime(int(y), int(mo), int(d))
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def extract_admit_discharge_dates(
    text: str, *, window_chars: int = 2000
) -> Dict[str, Optional[str]]:
    """
    Extract Admission Date / Discharge Date from the header region of a MIMIC note.
    Restricts search to the first `window_chars` characters to avoid false matches.
    """
    if not text:
        return {"admit_date": None, "discharge_date": None}

    head = text[: max(0, window_chars)]

    admit = None
    discharge = None

    m1 = _ADMIT_RE.search(head)
    if m1:
        admit = _normalize_ymd(m1.group(1))

    m2 = _DISCH_RE.search(head)
    if m2:
        discharge = _normalize_ymd(m2.group(1))

    return {"admit_date": admit, "discharge_date": discharge}


def extract_dictation_times(
    text: str, *, window_chars: int = 3000
) -> Dict[str, Optional[str]]:
    """
    Optional: Extract dictation/transcription dates from header/footer.
    Returned as ISO date or date+time if present.
    """
    if not text:
        return {"dictated_at": None, "transcribed_at": None}

    # Dictation can appear near the end in some notes; but to avoid scanning huge notes,
    # we search a combined window: head + tail (best-effort).
    head = text[: max(0, window_chars)]
    tail = text[-max(0, window_chars) :] if len(text) > window_chars else ""
    blob = head + "\n" + tail

    dictated_at = None
    transcribed_at = None

    md = _DICTATED_RE.search(blob)
    if md:
        d = _normalize_ymd(md.group(1))
        t = (md.group(2) or "").strip()
        dictated_at = f"{d} {t}".strip() if d else None

    mt = _TRANSCRIBED_RE.search(blob)
    if mt:
        d = _normalize_ymd(mt.group(1))
        t = (mt.group(2) or "").strip()
        transcribed_at = f"{d} {t}".strip() if d else None

    return {"dictated_at": dictated_at, "transcribed_at": transcribed_at}


# -------------------------
# Column resolution
# -------------------------


def resolve_columns(
    columns: List[str],
    *,
    text_col: Optional[str],
    note_id_col: Optional[str],
    subject_id_col: Optional[str],
    hadm_id_col: Optional[str],
    stay_id_col: Optional[str],
    chartdate_col: Optional[str],
    charttime_col: Optional[str],
    category_col: Optional[str],
    description_col: Optional[str],
) -> Dict[str, Optional[str]]:
    resolved_text = text_col or _infer_col(columns, ["TEXT", "text", "note_text"])
    if not resolved_text:
        raise ValueError(f"text_col is required. Available columns: {columns}")

    resolved_note_id = note_id_col or _infer_col(
        columns, ["ROW_ID", "NOTE_ID", "note_id"]
    )
    resolved_subject_id = subject_id_col or _infer_col(
        columns, ["SUBJECT_ID", "subject_id", "PATIENT_ID"]
    )
    resolved_hadm_id = hadm_id_col or _infer_col(
        columns, ["HADM_ID", "hadm_id", "ADMISSION_ID"]
    )
    resolved_stay_id = stay_id_col or _infer_col(
        columns, ["stay_id", "ICUSTAY_ID", "icustay_id"]
    )

    # NOTE: split chartdate and charttime (do NOT merge into one field)
    resolved_chartdate = chartdate_col or _infer_col(
        columns, ["CHARTDATE", "chartdate"]
    )
    resolved_charttime = charttime_col or _infer_col(
        columns, ["CHARTTIME", "charttime"]
    )

    resolved_category = category_col or _infer_col(columns, ["CATEGORY", "category"])
    resolved_description = description_col or _infer_col(
        columns, ["DESCRIPTION", "description"]
    )

    return {
        "text": resolved_text,
        "note_id": resolved_note_id,
        "subject_id": resolved_subject_id,
        "hadm_id": resolved_hadm_id,
        "stay_id": resolved_stay_id,
        "chartdate": resolved_chartdate,
        "charttime": resolved_charttime,
        "category": resolved_category,
        "description": resolved_description,
    }


def iter_mimic_notes(
    input_paths: Union[str, Path, Sequence[Union[str, Path]]],
    *,
    text_col: Optional[str] = None,
    note_id_col: Optional[str] = None,
    subject_id_col: Optional[str] = None,
    hadm_id_col: Optional[str] = None,
    stay_id_col: Optional[str] = None,
    chartdate_col: Optional[str] = None,
    charttime_col: Optional[str] = None,
    category_col: Optional[str] = None,
    description_col: Optional[str] = None,
    min_chars: int = 40,
    # Extraction controls
    extract_admit_discharge: bool = True,
    extract_dictation: bool = False,
    header_window_chars: int = 2000,
) -> Iterator[Dict[str, Any]]:
    if isinstance(input_paths, (str, Path)):
        input_paths = [input_paths]

    for path_like in input_paths:
        path = Path(path_like)
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            records = read_jsonl(path)
        elif suffix == ".json":
            records = read_json(path)
        elif suffix in {".csv", ".tsv"}:
            records = read_csv(path)
        else:
            raise ValueError(f"Unsupported input format: {path.name}")

        resolved = None
        for row in records:
            if resolved is None:
                if text_col and text_col not in row:
                    raise ValueError(
                        f"text_col '{text_col}' not found. Available columns: {list(row.keys())}"
                    )
                resolved = resolve_columns(
                    list(row.keys()),
                    text_col=text_col,
                    note_id_col=note_id_col,
                    subject_id_col=subject_id_col,
                    hadm_id_col=hadm_id_col,
                    stay_id_col=stay_id_col,
                    chartdate_col=chartdate_col,
                    charttime_col=charttime_col,
                    category_col=category_col,
                    description_col=description_col,
                )

            raw_text = row.get(resolved["text"], "")
            raw_text = "" if _is_missing(raw_text) else raw_text
            text = normalize_text(raw_text)
            if min_chars and len(text) < min_chars:
                continue

            chartdate = (
                _clean_scalar(row.get(resolved["chartdate"]))
                if resolved["chartdate"]
                else None
            )
            charttime = (
                _clean_scalar(row.get(resolved["charttime"]))
                if resolved["charttime"]
                else None
            )

            # Normalize chartdate if it's in YYYY-MM-DD; if not, keep as-is.
            chartdate_norm = _normalize_ymd(chartdate) if chartdate else None

            admit_date = None
            discharge_date = None
            dictated_at = None
            transcribed_at = None

            if extract_admit_discharge:
                d = extract_admit_discharge_dates(
                    text, window_chars=header_window_chars
                )
                admit_date = d["admit_date"]
                discharge_date = d["discharge_date"]

            if extract_dictation:
                t = extract_dictation_times(
                    text, window_chars=max(3000, header_window_chars)
                )
                dictated_at = t["dictated_at"]
                transcribed_at = t["transcribed_at"]

            yield {
                "note_id": _clean_scalar(row.get(resolved["note_id"]))
                if resolved["note_id"]
                else None,
                "subject_id": _clean_scalar(row.get(resolved["subject_id"]))
                if resolved["subject_id"]
                else None,
                "hadm_id": _clean_scalar(row.get(resolved["hadm_id"]))
                if resolved["hadm_id"]
                else None,
                "stay_id": _clean_scalar(row.get(resolved["stay_id"]))
                if resolved["stay_id"]
                else None,
                # keep both chartdate/charttime to avoid semantic ambiguity
                "chartdate": chartdate_norm or chartdate,
                "charttime": charttime,
                # extracted from text header
                "admit_date": admit_date,
                "discharge_date": discharge_date,
                # optional extra timestamps (off by default)
                "dictated_at": dictated_at,
                "transcribed_at": transcribed_at,
                "category": _clean_scalar(row.get(resolved["category"]))
                if resolved["category"]
                else None,
                "description": _clean_scalar(row.get(resolved["description"]))
                if resolved["description"]
                else None,
                "text": text,
            }


def chunk_note(
    text: str,
    *,
    max_chars: int = 1200,
    overlap: int = 200,
    min_chunk_chars: int = 200,
) -> List[str]:
    text = normalize_text(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return [text] if text else []

    overlap = max(0, min(overlap, max_chars - 1))
    chunks: List[str] = []

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    current = ""
    for para in paragraphs:
        candidate = para if not current else f"{current}\n\n{para}"
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current and len(current) >= min_chunk_chars:
            chunks.append(current)
        current = ""

        if len(para) <= max_chars:
            current = para
            continue

        start = 0
        while start < len(para):
            end = min(len(para), start + max_chars)
            chunk = para[start:end].strip()
            if len(chunk) >= min_chunk_chars:
                chunks.append(chunk)
            if end >= len(para):
                break
            start = max(0, end - overlap)

    if current and len(current) >= min_chunk_chars:
        chunks.append(current)

    return chunks


def build_mimic_deid_corpus(
    *,
    input_paths: Union[str, Path, Sequence[Union[str, Path]]],
    output_jsonl: Union[str, Path],
    chunk_chars: int = 1200,
    chunk_overlap: int = 200,
    min_note_chars: int = 40,
    min_chunk_chars: int = 200,
    max_notes: Optional[int] = None,
    # Extraction controls (passed through)
    extract_admit_discharge: bool = True,
    extract_dictation: bool = False,
    header_window_chars: int = 2000,
) -> int:
    count = 0

    def iter_chunks() -> Iterator[Dict[str, Any]]:
        nonlocal count
        seen = 0
        for note in iter_mimic_notes(
            input_paths,
            min_chars=min_note_chars,
            extract_admit_discharge=extract_admit_discharge,
            extract_dictation=extract_dictation,
            header_window_chars=header_window_chars,
        ):
            seen += 1
            if max_notes is not None and seen > max_notes:
                break

            chunks = chunk_note(
                note["text"],
                max_chars=chunk_chars,
                overlap=chunk_overlap,
                min_chunk_chars=min_chunk_chars,
            )

            for i, chunk in enumerate(chunks):
                count += 1
                uid = stable_uid(
                    note.get("note_id") or "",
                    note.get("subject_id") or "",
                    note.get("hadm_id") or "",
                    # stable across date fields
                    note.get("chartdate") or "",
                    note.get("charttime") or "",
                    note.get("admit_date") or "",
                    note.get("discharge_date") or "",
                    str(i),
                    chunk[:200],
                )
                yield {
                    "id": uid,
                    "text": chunk,
                    "meta": {
                        "note_id": note.get("note_id"),
                        "subject_id": note.get("subject_id"),
                        "hadm_id": note.get("hadm_id"),
                        "stay_id": note.get("stay_id"),
                        "chartdate": note.get("chartdate"),
                        "charttime": note.get("charttime"),
                        "admit_date": note.get("admit_date"),
                        "discharge_date": note.get("discharge_date"),
                        "dictated_at": note.get("dictated_at"),
                        "transcribed_at": note.get("transcribed_at"),
                        "category": note.get("category"),
                        "description": note.get("description"),
                        "chunk_id": i,
                        "chunk_count": len(chunks),
                    },
                }

    return write_jsonl(output_jsonl, iter_chunks())
