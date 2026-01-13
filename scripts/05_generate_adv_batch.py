# Sample usage:
# python scripts/05_generate_adv_batvh.py \
#   --m23k data/processed/m23k_train.jsonl \
#   --critic models/critic \
#   --out outputs/adv_batch.jsonl \
#   --level 0.3 \
#   --seed 0

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from med_sar.corruptions import CorruptConfig, mixed
from med_sar.semantic_checks import semantic_report


def load_jsonl(p: Path):
    return [json.loads(row) for row in p.open()]


@torch.inference_mode()
def critic_scores(texts: List[str], ckpt: str) -> List[float]:
    tok = AutoTokenizer.from_pretrained(ckpt, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).cuda().eval()
    out = []
    for i in range(0, len(texts), 32):
        batch = texts[i : i + 32]
        x = tok(
            batch, return_tensors="pt", truncation=True, max_length=256, padding=True
        ).to("cuda")
        logits = model(**x).logits
        prob = torch.softmax(logits, dim=-1)[:, 1]  # label=1 is target-like
        out.extend(prob.detach().cpu().tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m23k", type=str, required=True)
    ap.add_argument("--critic", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--level", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = load_jsonl(Path(args.m23k))

    adv_texts = []
    for i, r in enumerate(rows):
        adv = mixed(r["question"], CorruptConfig(level=args.level, seed=args.seed + i))
        r["x_adv"] = adv
        adv_texts.append(adv)

    scores = critic_scores(adv_texts, args.critic)

    out_rows = []
    for r, s in zip(rows, scores):
        rep = semantic_report(r["question"], r["x_adv"])
        r["style_score"] = float(s)
        r["neg_flip"] = rep.negation_flip
        r["num_mismatch"] = rep.number_mismatch
        r["entity_jaccard"] = rep.entity_jaccard
        out_rows.append(r)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote adv batch to {args.out}")


if __name__ == "__main__":
    main()
