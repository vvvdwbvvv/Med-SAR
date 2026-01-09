# Sample usage:
# python scripts/06_loop_train.py \
#   --train data/processed/m23k_train.jsonl \
#   --dev data/processed/m23k_val.jsonl \
#   --critic models/critic \
#   --base_model meta-llama/Llama-3.1-8B-Instruct \
#   --out_dir outputs/loop_train \
#   --rounds 5 \
#   --level0 0.2 \
#   --level_step 0.05

from __future__ import annotations
import argparse, subprocess
from pathlib import Path
import csv, json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, required=True)
    ap.add_argument("--dev", type=str, required=True)
    ap.add_argument("--critic", type=str, required=True)
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--level0", type=float, default=0.2)
    ap.add_argument("--level_step", type=float, default=0.05)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    log = out / "round_metrics.csv"

    with log.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["round","adv_level","style_mean","neg_flip_rate","num_mismatch_rate","entity_jaccard_mean"])
        w.writeheader()

        doctor_ckpt = str(out / "doctor_round_0")
        # round 0: SFT on clean
        subprocess.check_call([
            "python","scripts/05_sft_doctor.py",
            "--train", args.train,
            "--dev", args.dev,
            "--base", args.base_model,
            "--out", doctor_ckpt
        ])

        for r in range(1, args.rounds+1):
            level = args.level0 + (r-1)*args.level_step
            adv_path = out / f"adv_round_{r}.jsonl"

            subprocess.check_call([
                "python","scripts/06_generate_adv_batch.py",
                "--m23k", args.train,
                "--critic", args.critic,
                "--out", str(adv_path),
                "--level", str(level),
            ])

            # aggregate metrics
            rows = [json.loads(l) for l in adv_path.open()]
            style_mean = sum(x["style_score"] for x in rows)/len(rows)
            neg_rate = sum(1 for x in rows if x["neg_flip"])/len(rows)
            num_rate = sum(1 for x in rows if x["num_mismatch"])/len(rows)
            jac_mean = sum(x["entity_jaccard"] for x in rows)/len(rows)

            # train doctor on adv (MVP: just treat adv as new training question)
            # You can modify 05_sft_doctor.py to accept input_field="x_adv"
            # For now, we overwrite question with x_adv.
            tmp_train = out / f"train_adv_round_{r}.jsonl"
            with tmp_train.open("w") as g:
                for x in rows:
                    x2 = dict(x)
                    x2["question"] = x["x_adv"]
                    g.write(json.dumps(x2, ensure_ascii=False)+"\n")

            doctor_ckpt = str(out / f"doctor_round_{r}")
            subprocess.check_call([
                "python","scripts/05_sft_doctor.py",
                "--train", str(tmp_train),
                "--dev", args.dev,
                "--base", args.base_model,
                "--out", doctor_ckpt
            ])

            w.writerow({
                "round": r,
                "adv_level": level,
                "style_mean": style_mean,
                "neg_flip_rate": neg_rate,
                "num_mismatch_rate": num_rate,
                "entity_jaccard_mean": jac_mean,
            })
            f.flush()

    print(f"done. metrics: {log}")

if __name__ == "__main__":
    main()