# Sample usage:
# python scripts/08_visualize_tsne.py \
#   --clean_jsonl data/processed/m23k_test.jsonl \
#   --adv_jsonl outputs/adv_batch.jsonl \
#   --mimic_txt data/processed/corpus.txt \
#   --out_png outputs/tsne_visualization.png \
#   --n 1000 \
#   --seed 0 \
#   --model sentence-transformers/all-MiniLM-L6-v2

from __future__ import annotations
import argparse
import random
from pathlib import Path
import json

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer


def load_jsonl_text(p: Path, field: str, n: int, seed: int):
    rows = [json.loads(line) for line in p.open()]
    random.seed(seed)
    rows = random.sample(rows, k=min(n, len(rows)))
    return [r[field] for r in rows]


def load_txt(p: Path, n: int, seed: int):
    lines = [line.strip() for line in p.open() if line.strip()]
    random.seed(seed)
    return random.sample(lines, k=min(n, len(lines)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_jsonl", type=str, required=True)
    ap.add_argument("--adv_jsonl", type=str, required=True)
    ap.add_argument("--mimic_txt", type=str, required=True)
    ap.add_argument("--out_png", type=str, required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2"
    )
    args = ap.parse_args()

    clean = load_jsonl_text(Path(args.clean_jsonl), "question", args.n, args.seed)
    adv = load_jsonl_text(Path(args.adv_jsonl), "x_adv", args.n, args.seed)
    mimic = load_txt(Path(args.mimic_txt), args.n, args.seed)

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


if __name__ == "__main__":
    main()
