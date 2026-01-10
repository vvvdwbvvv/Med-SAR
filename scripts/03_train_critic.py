# Sample usage:
# python scripts/03_train_critic.py \
#   --mimic_txt data/processed/corpus.txt \
#   --m23k_dev data/processed/m23k_val.jsonl \
#   --base distilbert-base-uncased \
#   --out models/critic \
#   --n_pos 20000 \
#   --n_neg 20000 \
#   --level 0.3 \
#   --seed 0

from __future__ import annotations
import argparse
import random
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from med_sar.corruptions import CorruptConfig, mixed


def load_lines(p: Path, n: int | None = None):
    lines = [l.strip() for l in p.open() if l.strip()]
    if n:
        random.shuffle(lines)
        lines = lines[:n]
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic_txt", type=str, required=True)
    ap.add_argument("--m23k_dev", type=str, required=True)
    ap.add_argument("--base", type=str, default="distilbert-base-uncased")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--n_pos", type=int, default=20000)
    ap.add_argument("--n_neg", type=int, default=20000)
    ap.add_argument("--level", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)

    mimic = load_lines(Path(args.mimic_txt), n=args.n_pos)

    # neg: generate from m23k questions
    import json

    m23k_rows = [json.loads(l) for l in Path(args.m23k_dev).open()]
    random.shuffle(m23k_rows)
    m23k_rows = m23k_rows[: args.n_neg]
    neg = []
    for i, r in enumerate(m23k_rows):
        neg.append(
            mixed(r["question"], CorruptConfig(level=args.level, seed=args.seed + i))
        )

    texts = mimic + neg
    labels = [1] * len(mimic) + [0] * len(neg)

    ds = Dataset.from_dict({"text": texts, "label": labels}).shuffle(seed=args.seed)
    ds = ds.train_test_split(test_size=0.05, seed=args.seed)

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)

    def tok_fn(ex):
        return tok(ex["text"], truncation=True, max_length=256)

    ds_tok = ds.map(tok_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=2)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        num_train_epochs=1,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        logging_steps=50,
        fp16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["test"],
    )
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"critic saved to {args.out}")


if __name__ == "__main__":
    main()
