from __future__ import annotations

import hashlib
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


def resolve_columns(
    columns: List[str],
    *,
    text_col: Optional[str],
    note_id_col: Optional[str],
    subject_id_col: Optional[str],
    hadm_id_col: Optional[str],
    stay_id_col: Optional[str],
    charttime_col: Optional[str],
    category_col: Optional[str],
    description_col: Optional[str],
) -> Dict[str, Optional[str]]:
    resolved_text = text_col
    if not resolved_text:
        raise ValueError(f"text_col is required. Available columns: {columns}")
    return {
        "text": resolved_text,
        "note_id": note_id_col,
        "subject_id": subject_id_col,
        "hadm_id": hadm_id_col,
        "stay_id": stay_id_col,
        "charttime": charttime_col,
        "category": category_col,
        "description": description_col,
    }


def iter_mimic_notes(
    input_paths: Union[str, Path, Sequence[Union[str, Path]]],
    *,
    text_col: Optional[str] = None,
    note_id_col: Optional[str] = None,
    subject_id_col: Optional[str] = None,
    hadm_id_col: Optional[str] = None,
    stay_id_col: Optional[str] = None,
    charttime_col: Optional[str] = None,
    category_col: Optional[str] = None,
    description_col: Optional[str] = None,
    min_chars: int = 40,
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
                    charttime_col=charttime_col,
                    category_col=category_col,
                    description_col=description_col,
                )

            raw_text = row.get(resolved["text"], "")
            text = normalize_text(raw_text)
            if min_chars and len(text) < min_chars:
                continue

            yield {
                "note_id": (
                    str(row.get(resolved["note_id"])).strip()
                    if resolved["note_id"] and row.get(resolved["note_id"]) is not None
                    else None
                ),
                "subject_id": (
                    str(row.get(resolved["subject_id"])).strip()
                    if resolved["subject_id"]
                    and row.get(resolved["subject_id"]) is not None
                    else None
                ),
                "hadm_id": (
                    str(row.get(resolved["hadm_id"])).strip()
                    if resolved["hadm_id"] and row.get(resolved["hadm_id"]) is not None
                    else None
                ),
                "stay_id": (
                    str(row.get(resolved["stay_id"])).strip()
                    if resolved["stay_id"] and row.get(resolved["stay_id"]) is not None
                    else None
                ),
                "charttime": (
                    str(row.get(resolved["charttime"])).strip()
                    if resolved["charttime"]
                    and row.get(resolved["charttime"]) is not None
                    else None
                ),
                "category": (
                    str(row.get(resolved["category"])).strip()
                    if resolved["category"]
                    and row.get(resolved["category"]) is not None
                    else None
                ),
                "description": (
                    str(row.get(resolved["description"])).strip()
                    if resolved["description"]
                    and row.get(resolved["description"]) is not None
                    else None
                ),
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
) -> int:
    count = 0

    def iter_chunks() -> Iterator[Dict[str, Any]]:
        nonlocal count
        seen = 0
        for note in iter_mimic_notes(input_paths, min_chars=min_note_chars):
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
                    note.get("charttime") or "",
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
                        "charttime": note.get("charttime"),
                        "category": note.get("category"),
                        "description": note.get("description"),
                        "chunk_id": i,
                        "chunk_count": len(chunks),
                    },
                }

    return write_jsonl(output_jsonl, iter_chunks())
