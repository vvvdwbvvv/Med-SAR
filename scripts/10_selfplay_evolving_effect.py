# Sample usage:
# python scripts/10_selfplay_evolving_effect.py \
#   --source_train data/processed/m23k_train.jsonl \
#   --source_dev data/processed/m23k_val.jsonl \
#   --out_dir outputs/selfplay \
#   --rounds 5 \
#   --adv_level 0.3 \
#   --seed 0 \
#   --ablation full

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from med_sar.semantic_checks import semantic_report, aggregate_reports
from med_sar.corruptions import CorruptConfig, mixed

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

# ----- Placeholder interfaces: replace with your actual implementations -----

def critic_style_score(texts: List[str]) -> List[float]:
    # return target-like probability/logit per text
    # TODO: replace with DistilBERT/DeBERTa classifier inference
    return [0.5 for _ in texts]

def doctor_loss_on_batch(texts: List[str], labels: List[str]) -> List[float]:
    # return per-example loss for current doctor checkpoint
    # TODO: replace with your HF model forward loss (teacher forcing if generative)
    return [1.0 for _ in texts]

def train_doctor_on_adv(train_rows: List[Dict], out_dir: Path) -> None:
    # TODO: call your LoRA fine-tune script here; write checkpoint to out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "DONE.txt").write_text("placeholder\n", encoding="utf-8")

# -------------------------------------------------------------------------

def generate_adv_batch(rows: List[Dict], level: float, seed: int) -> List[Dict]:
    out = []
    for i, r in enumerate(rows):
        cfg = CorruptConfig(level=level, seed=seed + i)
        x = r["question"]
        x_adv = mixed(x, cfg)  # placeholder: swap to your G prompt rewrite later
        rr = dict(r)
        rr["x_adv"] = x_adv
        out.append(rr)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_train", type=str, required=True)
    ap.add_argument("--source_dev", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--adv_level", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ablation", type=str, default="full",
                   choices=["full","freeze_g","no_critic","no_semantic"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"round_metrics_{args.ablation}.csv"

    train_rows = load_jsonl(Path(args.source_train))
    dev_rows = load_jsonl(Path(args.source_dev))

    # if freeze_g: generate once and reuse
    frozen_adv = None
    if args.ablation == "freeze_g":
        frozen_adv = generate_adv_batch(train_rows, level=args.adv_level, seed=args.seed)
        save_jsonl(out_dir / "adv_round_1_frozen.jsonl", frozen_adv)

    with log_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "round","hardness_mean","style_mean",
            "neg_flip_rate","num_mismatch_rate","entity_jaccard_mean",
        ])
        w.writeheader()

        for r in range(1, args.rounds + 1):
            # --- Attack phase: generate adv ---
            if frozen_adv is not None:
                adv_rows = frozen_adv
            else:
                adv_rows = generate_adv_batch(train_rows, level=args.adv_level, seed=args.seed + 1000 * r)
                save_jsonl(out_dir / f"adv_round_{r}.jsonl", adv_rows)

            x_clean = [x["question"] for x in adv_rows]
            x_adv = [x["x_adv"] for x in adv_rows]
            y = [x.get("answer_string","") for x in adv_rows]  # adapt label field

            # --- Metrics: hardness ---
            loss_clean = doctor_loss_on_batch(x_clean, y)
            loss_adv = doctor_loss_on_batch(x_adv, y)
            hardness = sum((la - lc) for la, lc in zip(loss_adv, loss_clean)) / max(1, len(loss_adv))

            # --- Metrics: style-likeness (critic) ---
            if args.ablation == "no_critic":
                style_scores = [0.0 for _ in x_adv]
            else:
                style_scores = critic_style_score(x_adv)
            style_mean = sum(style_scores) / max(1, len(style_scores))

            # --- Metrics: semantic violations ---
            if args.ablation == "no_semantic":
                reports = []
            else:
                reports = [semantic_report(c, a) for c, a in zip(x_clean, x_adv)]
            sem = aggregate_reports(reports) if reports else {
                "neg_flip_rate": 1.0, "num_mismatch_rate": 1.0, "entity_jaccard_mean": 0.0
            }

            # --- Defense phase: train doctor on adv ---
            # (你可以把 clean+adv 混合；或只用 adv；這裡先示意)
            ckpt_dir = out_dir / f"doctor_round_{r}"
            train_doctor_on_adv(adv_rows, ckpt_dir)

            w.writerow({
                "round": r,
                "hardness_mean": hardness,
                "style_mean": style_mean,
                **sem,
            })
            f.flush()

    print(f"Saved round metrics to: {log_path}")

if __name__ == "__main__":
    main()