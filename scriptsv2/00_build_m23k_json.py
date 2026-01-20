# Sample usage:
# python scriptsv2/00_build_m23k_json.py \
#   --output_dir data/processed/m23k_v2 \
#   --seed 42 \
#   --train_ratio 0.9 \
#   --val_ratio 0.05 \
#   --test_ratio 0.05

from __future__ import annotations

import argparse
import hashlib
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import pandas as pd

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


def _require_parquet() -> None:
    try:
        import pyarrow  # noqa: F401
    except Exception as exc:  # pragma: no cover - runtime check
        raise SystemExit(
            "pyarrow is required for parquet outputs. Install it and retry."
        ) from exc


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


def _has_options(prompt: str) -> bool:
    text = prompt or ""
    for marker in ("A)", "B)", "C)", "D)", "E)", "A.", "B.", "C.", "D.", "E."):
        if marker in text:
            return True
    return False


def build_wrapped_record(raw_row: Dict[str, Any], split: str) -> Dict[str, Any]:
    norm = normalize_m23k_record(raw_row, cfg=NormalizeConfig(split=split))
    prompt = norm.get("input", {}).get("prompt", "")
    output = norm.get("output", {})
    answer = _pick_answer(output)
    created_at = datetime.utcnow().isoformat() + "Z"
    source_id = norm.get("id")
    record_id = f"m23k:{source_id}" if source_id else None
    return {
        "id": record_id or source_id,
        "x_raw": prompt,
        "x_wrapped": WRAPPER_V1.format(x=prompt),
        "y": answer,
        "meta": {
            "wrapper_id": WRAPPER_ID,
            "source_id": source_id,
            "source": norm.get("source"),
            "split": split,
            "answer": output,
            "upstream_metadata": norm.get("meta", {}).get("metadata"),
            "has_options": _has_options(prompt),
            "created_at": created_at,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/m23k",
        help="Output directory.",
    )
    ap.add_argument(
        "--configs_dir",
        type=str,
        default="configs",
        help="Directory for fixed configs (wrapper).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.9)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--test_ratio", type=float, default=0.05)
    ap.add_argument("--max_records", type=int, default=None)
    args = ap.parse_args()

    _require_parquet()

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

    cfg_dir = Path(args.configs_dir)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "wrapper_v1.txt").write_text(WRAPPER_V1, encoding="utf-8")
    (cfg_dir / "wrapper_id.txt").write_text(f"{WRAPPER_ID}\n", encoding="utf-8")

    train_rows = [build_wrapped_record(r, "train") for r in train]
    val_rows = [build_wrapped_record(r, "val") for r in val]
    test_rows = [build_wrapped_record(r, "test") for r in test]

    write_jsonl(out_dir / "m23k_wrapped_train.jsonl", train_rows)
    write_jsonl(out_dir / "m23k_wrapped_val.jsonl", val_rows)
    if test_rows:
        write_jsonl(out_dir / "m23k_wrapped_test.jsonl", test_rows)

    manifest_rows = []
    for row in train_rows + val_rows + test_rows:
        x_wrapped = row.get("x_wrapped") or ""
        manifest_rows.append(
            {
                "id": row.get("id"),
                "split": row.get("meta", {}).get("split"),
                "source": row.get("meta", {}).get("source"),
                "wrapper_id": row.get("meta", {}).get("wrapper_id"),
                "length_tokens": len(x_wrapped.split()),
                "has_options": row.get("meta", {}).get("has_options"),
            }
        )
    manifest_path = out_dir / "m23k_manifest.parquet"
    pd.DataFrame(manifest_rows).to_parquet(manifest_path, index=False)

    wrapper_hash = hashlib.sha1(WRAPPER_V1.encode("utf-8")).hexdigest()
    all_rows = train_rows + val_rows + test_rows
    avg_len = sum(len(r.get("x_wrapped", "").split()) for r in all_rows) / max(
        1, len(all_rows)
    )
    has_options_rate = sum(
        1 for r in all_rows if r.get("meta", {}).get("has_options")
    ) / max(1, len(all_rows))

    print(
        f"[m23k] records={len(records)} train={len(train)} val={len(val)} test={len(test)}"
    )
    print(
        f"[m23k] wrote: {out_dir}/(m23k_wrapped_train.jsonl, m23k_wrapped_val.jsonl, m23k_wrapped_test.jsonl)"
    )
    print(f"[m23k] wrote: {manifest_path}")
    print(f"[m23k] wrapper hash={wrapper_hash}")
    print(f"[m23k] avg_tokens={avg_len:.2f} has_options_rate={has_options_rate:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
