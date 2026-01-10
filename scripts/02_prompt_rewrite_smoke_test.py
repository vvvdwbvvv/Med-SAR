# Sample usage:
# python scripts/02_prompt_rewrite_smoke_test.py \
#   --m23k data/processed/m23k_val.jsonl \
#   --out outputs/smoke/rewrite_samples.jsonl \
#   --n 50 \
#   --level 0.3 \
#   --seed 0

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

from med_sar.corruptions import CorruptConfig, mixed


def load_jsonl(p: Path):
    rows = []
    with p.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m23k", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--level", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    rows = load_jsonl(Path(args.m23k))
    sample = random.sample(rows, k=min(args.n, len(rows)))

    out_rows = []
    for i, r in enumerate(sample):
        clean = r["question"]
        adv = mixed(clean, CorruptConfig(level=args.level, seed=args.seed + i))
        out_rows.append({"clean": clean, "adv": adv})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(out_rows)} to {args.out}")


if __name__ == "__main__":
    main()
