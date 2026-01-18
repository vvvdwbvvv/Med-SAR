# Sample usage:
# python scriptsv2/08_visualize_tsne.py \
#   --clean_jsonl data/processed/m23k_v2/m23k_test.jsonl \
#   --adv_jsonl outputs/adv_train.jsonl \
#   --mimic_parquet data/processed/mimic_v2/mimic_notes.parquet \
#   --out_png outputs/tsne_v2.png

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer


def load_jsonl_text(p: Path, field: str, n: int, seed: int):
    rows = [json.loads(line) for line in p.open() if line.strip()]
    random.seed(seed)
    rows = random.sample(rows, k=min(n, len(rows)))
    return [r.get(field) or r.get("x_wrapped") or r.get("x_raw") or "" for r in rows]


def load_parquet_text(p: Path, n: int, seed: int, text_col: str):
    df = pd.read_parquet(p)
    texts = df[text_col].dropna().astype(str).tolist()
    random.seed(seed)
    return random.sample(texts, k=min(n, len(texts)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_jsonl", type=str, required=True)
    ap.add_argument("--adv_jsonl", type=str, required=True)
    ap.add_argument("--mimic_parquet", type=str, required=True)
    ap.add_argument("--text_col", type=str, default="text")
    ap.add_argument("--out_png", type=str, required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2"
    )
    args = ap.parse_args()

    clean = load_jsonl_text(Path(args.clean_jsonl), "x_wrapped", args.n, args.seed)
    adv = load_jsonl_text(Path(args.adv_jsonl), "x_adv", args.n, args.seed)
    mimic = load_parquet_text(Path(args.mimic_parquet), args.n, args.seed, args.text_col)

    enc = SentenceTransformer(args.model)
    X = enc.encode(clean + adv + mimic, batch_size=64, show_progress_bar=True)
    X2 = TSNE(n_components=2, random_state=args.seed, perplexity=30).fit_transform(X)

    n = len(clean)
    m = len(adv)
    plt.figure(figsize=(8, 6))
    plt.scatter(X2[:n, 0], X2[:n, 1], s=4, label="clean")
    plt.scatter(X2[n : n + m, 0], X2[n : n + m, 1], s=4, label="adv")
    plt.scatter(X2[n + m :, 0], X2[n + m :, 1], s=4, label="mimic")
    plt.legend()
    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_png, dpi=200)
    print(f"saved {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
