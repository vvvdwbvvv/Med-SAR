# Sample usage:
# python scripts/09_controlled_shift_benchmark.py \
#   --input_test data/processed/m23k_test.jsonl \
#   --out_dir outputs/controlled_shifts \
#   --levels 0.1 0.2 0.3 0.4 \
#   --shifts abbrev tele shuffle drop mixed \
#   --seed 0

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List

from med_sar.corruptions import CorruptConfig, abbrev_jargon, telegraphic, shuffle_sentences, ellipsis_drop, mixed

SHIFT_FUNCS = {
    "abbrev": abbrev_jargon,
    "tele": telegraphic,
    "shuffle": shuffle_sentences,
    "drop": ellipsis_drop,
    "mixed": mixed,
}

def load_jsonl(p: Path) -> List[Dict]:
    rows = []
    with p.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def save_jsonl(p: Path, rows: List[Dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def build_shifted(rows: List[Dict], shift: str, level: float, seed: int) -> List[Dict]:
    fn = SHIFT_FUNCS[shift]
    out = []
    for i, r in enumerate(rows):
        cfg = CorruptConfig(level=level, seed=seed + i)
        x = r["question"]  # adapt to your schema
        x2 = fn(x, cfg)
        rr = dict(r)
        rr["question_shifted"] = x2
        rr["shift"] = shift
        rr["level"] = level
        out.append(rr)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_test", type=str, required=True, help="source test jsonl with {question, answer,...}")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--levels", type=float, nargs="+", default=[0.1,0.2,0.3,0.4])
    ap.add_argument("--shifts", type=str, nargs="+", default=["abbrev","tele","shuffle","drop","mixed"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = load_jsonl(Path(args.input_test))
    out_dir = Path(args.out_dir)

    for shift in args.shifts:
        for level in args.levels:
            shifted = build_shifted(rows, shift=shift, level=level, seed=args.seed)
            save_jsonl(out_dir / f"{shift}_lvl{level:.1f}.jsonl", shifted)

    print(f"Saved shifted testsets to: {out_dir}")

if __name__ == "__main__":
    main()