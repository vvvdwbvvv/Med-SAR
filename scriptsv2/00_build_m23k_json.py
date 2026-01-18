# Sample usage:
# python scriptsv2/00_build_m23k_json.py \
#   --output_dir data/processed/m23k_v2 \
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
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.io import write_jsonl
from utils.seed import set_seed
from med_sar.data.m23k import NormalizeConfig, normalize_m23k_record, load_m23k_raw

WRAPPER_ID = "V1"
WRAPPER_V1 = (
    "ED NOTE\n"
    "CHIEF COMPLAINT: {x}\n"
    "HISTORY OF PRESENT ILLNESS: [omitted]\n"
    "PAST MEDICAL HISTORY: [omitted]\n"
    "MEDICATIONS: [omitted]\n"
    "ALLERGIES: [omitted]\n"
    "PHYSICAL EXAM: [omitted]\n"
    "PLAN: [omitted]\n"
)


@dataclass(frozen=True)
class SplitConfig:
    seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float


def split_records(
    records: List[Dict[str, Any]], cfg: SplitConfig
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    set_seed(cfg.seed)
    idx = list(range(len(records)))
    random.shuffle(idx)

    n = len(idx)
    n_train = int(n * cfg.train_ratio)
    n_val = int(n * cfg.val_ratio)
    train = [records[i] for i in idx[:n_train]]
    val = [records[i] for i in idx[n_train : n_train + n_val]]
    test = [records[i] for i in idx[n_train + n_val :]]
    return train, val, test


def _pick_answer(output: Dict[str, Any]) -> Any:
    for key in (
        "distilled_answer_string",
        "answer_string",
        "answer_letter",
        "answer_idx",
    ):
        val = output.get(key)
        if val not in (None, ""):
            return val
    return None


def build_wrapped_record(raw_row: Dict[str, Any], split: str) -> Dict[str, Any]:
    norm = normalize_m23k_record(raw_row, cfg=NormalizeConfig(split=split))
    prompt = norm.get("input", {}).get("prompt", "")
    output = norm.get("output", {})
    answer = _pick_answer(output)
    return {
        "id": norm.get("id"),
        "x_raw": prompt,
        "x_wrapped": WRAPPER_V1.format(x=prompt),
        "y": answer,
        "meta": {
            "wrapper_id": WRAPPER_ID,
            "source_id": norm.get("id"),
            "source": norm.get("source"),
            "split": split,
            "answer": output,
            "upstream_metadata": norm.get("meta", {}).get("metadata"),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/m23k_v2",
        help="Output directory.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.9)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--test_ratio", type=float, default=0.05)
    ap.add_argument("--max_records", type=int, default=None)
    args = ap.parse_args()

    records = list(load_m23k_raw())
    if args.max_records:
        records = records[: args.max_records]

    train, val, test = split_records(
        records,
        SplitConfig(
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        ),
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "wrapper_v1.txt").write_text(WRAPPER_V1, encoding="utf-8")
    (out_dir / "wrapper_id.txt").write_text(f"{WRAPPER_ID}\n", encoding="utf-8")

    write_jsonl(out_dir / "m23k_train.jsonl", (build_wrapped_record(r, "train") for r in train))
    write_jsonl(out_dir / "m23k_val.jsonl", (build_wrapped_record(r, "val") for r in val))
    write_jsonl(out_dir / "m23k_test.jsonl", (build_wrapped_record(r, "test") for r in test))

    print(
        f"[m23k_v2] records={len(records)} train={len(train)} val={len(val)} test={len(test)}"
    )
    print(f"[m23k_v2] wrote: {out_dir}/(m23k_train.jsonl, m23k_val.jsonl, m23k_test.jsonl)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
