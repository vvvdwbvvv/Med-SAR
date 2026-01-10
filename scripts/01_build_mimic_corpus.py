#!/usr/bin/env python3
# Sample usage:
# python scripts/01_build_mimic_corpus.py \
#   --input data/raw/NOTEEVENTS.csv \
#   --output_dir data/processed \
#   --text_col TEXT \
#   --chunk_chars 1200 \
#   --chunk_overlap 200 \
#   --min_note_chars 40 \
#   --min_chunk_chars 200

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from med_sar.data.mimic import chunk_note, iter_mimic_notes, stable_uid
from utils.io import write_jsonl
from tqdm import tqdm


def build_corpus(
    *,
    input_paths: List[str],
    output_dir: Path,
    text_col: Optional[str],
    note_id_col: Optional[str],
    subject_id_col: Optional[str],
    hadm_id_col: Optional[str],
    stay_id_col: Optional[str],
    charttime_col: Optional[str],
    category_col: Optional[str],
    description_col: Optional[str],
    min_note_chars: int,
    min_chunk_chars: int,
    chunk_chars: int,
    chunk_overlap: int,
    max_notes: Optional[int],
) -> Dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = output_dir / "corpus.jsonl"

    note_count = 0
    chunk_count = 0

    def iter_rows() -> Iterable[Dict[str, Any]]:
        nonlocal note_count, chunk_count
        seen = 0
        note_iter = iter_mimic_notes(
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
        )
        note_total = max_notes if max_notes is not None and max_notes > 0 else None
        with (
            tqdm(note_iter, unit="notes", desc="notes", total=note_total) as note_bar,
            tqdm(unit="chunks", desc="chunks") as chunk_bar,
        ):
            for note in note_bar:
                seen += 1
                if max_notes is not None and seen > max_notes:
                    break

                note_text = note["text"]
                note_count += 1

                chunks = chunk_note(
                    note_text,
                    max_chars=chunk_chars,
                    overlap=chunk_overlap,
                    min_chunk_chars=min_chunk_chars,
                )
                for i, chunk in enumerate(chunks):
                    chunk_count += 1
                    chunk_bar.update(1)
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

    write_jsonl(output_jsonl, iter_rows())
    return {"notes": note_count, "chunks": chunk_count}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input", action="append", required=True, help="Path(s) to notes files."
    )
    ap.add_argument("--output_dir", type=str, default="data/processed")
    ap.add_argument("--text_col", type=str, default="TEXT")
    ap.add_argument("--note_id_col", type=str, default=None)
    ap.add_argument("--subject_id_col", type=str, default=None)
    ap.add_argument("--hadm_id_col", type=str, default=None)
    ap.add_argument("--stay_id_col", type=str, default=None)
    ap.add_argument("--charttime_col", type=str, default=None)
    ap.add_argument("--category_col", type=str, default=None)
    ap.add_argument("--description_col", type=str, default=None)
    ap.add_argument("--min_note_chars", type=int, default=40)
    ap.add_argument("--min_chunk_chars", type=int, default=200)
    ap.add_argument("--chunk_chars", type=int, default=1200)
    ap.add_argument("--chunk_overlap", type=int, default=200)
    ap.add_argument("--max_notes", type=int, default=None)
    args = ap.parse_args()

    counts = build_corpus(
        input_paths=args.input,
        output_dir=Path(args.output_dir),
        text_col=args.text_col,
        note_id_col=args.note_id_col,
        subject_id_col=args.subject_id_col,
        hadm_id_col=args.hadm_id_col,
        stay_id_col=args.stay_id_col,
        charttime_col=args.charttime_col,
        category_col=args.category_col,
        description_col=args.description_col,
        min_note_chars=args.min_note_chars,
        min_chunk_chars=args.min_chunk_chars,
        chunk_chars=args.chunk_chars,
        chunk_overlap=args.chunk_overlap,
        max_notes=args.max_notes,
    )

    print(f"[mimic] notes={counts['notes']} chunks={counts['chunks']}")
    print(f"[mimic] wrote: {args.output_dir}/corpus.jsonl")
    return 0


if __name__ == "__main__":
    main()
