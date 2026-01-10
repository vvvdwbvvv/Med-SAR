# Sample usage:
# python scripts/11_real_target_eval.py \
#   --test_jsonl data/processed/mimic_test.jsonl \
#   --input_field question_shifted \
#   --label_field answer_string \
#   --ckpts sft=models/doctor_sft dapt=models/doctor_dapt medsar=models/doctor_round_5 \
#   --out_csv results/real_target_eval.csv

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def load_jsonl(p: Path) -> List[Dict]:
    rows = []
    with p.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def evaluate_model(
    rows: List[Dict], model_ckpt: str, input_field: str, label_field: str
) -> Dict[str, float]:
    # TODO: replace with your evaluation code
    # return {"acc":..., "f1":..., "ece":...}
    return {"acc": 0.0, "f1": 0.0, "ece": 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_jsonl", type=str, required=True)
    ap.add_argument("--input_field", type=str, default="question_shifted")
    ap.add_argument("--label_field", type=str, default="answer_string")
    ap.add_argument(
        "--ckpts",
        type=str,
        nargs="+",
        required=True,
        help="name=path pairs, e.g., sft=... dapt=... medsar=...",
    )
    ap.add_argument("--out_csv", type=str, required=True)
    args = ap.parse_args()

    rows = load_jsonl(Path(args.test_jsonl))
    ckpts = []
    for kv in args.ckpts:
        name, path = kv.split("=", 1)
        ckpts.append((name, path))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "acc", "f1", "ece"])
        w.writeheader()
        for name, path in ckpts:
            metrics = evaluate_model(
                rows,
                model_ckpt=path,
                input_field=args.input_field,
                label_field=args.label_field,
            )
            w.writerow({"model": name, **metrics})

    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
