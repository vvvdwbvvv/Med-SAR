# Sample usage:
# python scripts/07_eval_baseline.py \
#   --shift_dir outputs/controlled_shifts \
#   --out_csv results/baseline_eval.csv \
#   --models sft=models/doctor_sft medsar=models/doctor_round_5

from __future__ import annotations
import argparse, csv, json
from pathlib import Path

def load_jsonl(p: Path):
    return [json.loads(l) for l in p.open()]

def eval_model(rows, ckpt, input_field="question_shifted", label_field="answer_string"):
    # TODO: replace with your real evaluation
    # return {"acc":..., "f1":..., "ece":...}
    return {"acc": 0.0, "f1": 0.0, "ece": 0.0}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift_dir", type=str, required=True)  # outputs from scripts/10_controlled_shift_benchmark.py
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--models", type=str, nargs="+", required=True, help="name=ckpt ...")
    args = ap.parse_args()

    shift_dir = Path(args.shift_dir)
    files = sorted(shift_dir.glob("*.jsonl"))

    models = []
    for kv in args.models:
        name, path = kv.split("=", 1)
        models.append((name, path))

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_csv).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model","shift","level","acc","f1","ece"])
        w.writeheader()

        for fp in files:
            rows = load_jsonl(fp)
            shift = rows[0].get("shift", fp.stem.split("_")[0])
            level = float(rows[0].get("level", fp.stem.split("lvl")[-1]))
            for name, ckpt in models:
                m = eval_model(rows, ckpt)
                w.writerow({"model": name, "shift": shift, "level": level, **m})

    print(f"wrote {args.out_csv}")

if __name__ == "__main__":
    main()