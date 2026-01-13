# Sample usage:
# python scripts/00_build_m23k_json.py \
#   --output_dir data/processed/m23k \
#   --seed 42 \
#   --train_ratio 0.9 \
#   --val_ratio 0.05 \
#   --test_ratio 0.05

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

# Add parent directory to path to import utils and src for med_sar
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.io import write_jsonl, iter_upstream_records
from utils.seed import set_seed
from med_sar.data.m23k import normalize_m23k_record, load_m23k_raw


@dataclass(frozen=True)
class NormalizeConfig:
    split: str = "train"
    task: str = "mcq_rationale"
    # If True, also copy upstream `text` into input.context (when non-empty and not redundant).
    place_text_in_context: bool = True
    # If True, require at least one of answer_string/distilled_answer_string.
    require_answer_text: bool = True


def build_m23k_jsonl(
    *,
    input_paths: Union[str, Path, Sequence[Union[str, Path]]],
    output_jsonl: Union[str, Path],
    split: str = "train",
    max_records: Optional[int] = None,
    dedup: bool = True,
) -> None:
    cfg = NormalizeConfig(split=split)

    count = 0

    def iter_normalized() -> Iterator[Dict[str, Any]]:
        nonlocal count
        for row in iter_upstream_records(input_paths):
            count += 1
            if count > max_records:
                break
            yield normalize_m23k_record(row, cfg=cfg)

    normalized_iter: Iterable[Dict[str, Any]] = iter_normalized()
    if dedup:

        def iter_deduped() -> Iterator[Dict[str, Any]]:
            for rec in normalized_iter:
                key_payload = {
                    "prompt": rec.get("input", {}).get("prompt"),
                    "distilled": rec.get("output", {}).get("distilled_answer_string"),
                    "answer": rec.get("output", {}).get("answer_string"),
                    "answer_idx": rec.get("output", {}).get("answer_idx"),
                }
                yield rec

        normalized_iter = iter_deduped()

    write_jsonl(output_jsonl, normalized_iter)


def split(
    records: List[Dict[str, Any]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    set_seed(seed)
    idx = list(range(len(records)))
    random.shuffle(idx)

    n = len(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = [records[i] for i in idx[:n_train]]
    val = [records[i] for i in idx[n_train : n_train + n_val]]
    test = [records[i] for i in idx[n_train + n_val :]]
    return train, val, test


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/m23k",
        help="Output directory.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.9)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--test_ratio", type=float, default=0.05)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)

    records = list(load_m23k_raw())

    train, val, test = split(
        records, args.seed, args.train_ratio, args.val_ratio, args.test_ratio
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    # write_jsonl(out_dir / "m23k.jsonl", records)
    write_jsonl(out_dir / "m23k_train.jsonl", train)
    write_jsonl(out_dir / "m23k_val.jsonl", val)
    write_jsonl(out_dir / "m23k_test.jsonl", test)

    # Minimal stdout (no raw text)
    print(
        f"[m23k] records={len(records)} train={len(train)} val={len(val)} test={len(test)}"
    )
    print(f"[m23k] wrote: {out_dir}/(m23k.jsonl, train.jsonl, val.jsonl, test.jsonl)")
    return 0


if __name__ == "__main__":
    main()
