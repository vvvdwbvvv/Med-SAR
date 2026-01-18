# Sample usage:
# python scriptsv2/03_train_domain_probe.py \
#   --m23k data/processed/m23k_v2/m23k_train.jsonl \
#   --calibration outputs/calibration.json \
#   --out outputs/domain_probe.json

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
from typing import List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from med_sar.operators.chain import apply_chain, sample_chain
from med_sar.operators.calibration import load_calibration, level_for_t
from med_sar.operators.proxies import compute_proxies
from utils.io import read_jsonl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m23k", type=str, required=True)
    ap.add_argument("--calibration", type=str, default=None)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--t", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = list(read_jsonl(args.m23k))
    rng = random.Random(args.seed)
    sample = rng.sample(rows, k=min(args.n, len(rows)))

    calibration = load_calibration(args.calibration)

    X: List[List[float]] = []
    y: List[int] = []

    for i, r in enumerate(sample):
        clean = r.get("x_wrapped") or r.get("x_raw") or ""
        ops = sample_chain(rng)
        adv = apply_chain(
            clean,
            ops,
            t=args.t,
            seed=args.seed + i,
            level_fn=lambda op, t_val: level_for_t(op, t_val, calibration),
        )
        clean_feat = compute_proxies(clean)
        adv_feat = compute_proxies(adv)
        X.append(list(clean_feat.values()))
        y.append(0)
        X.append(list(adv_feat.values()))
        y.append(1)

    X_arr = np.array(X)
    y_arr = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=0.2, random_state=args.seed
    )

    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"accuracy": acc, "n": len(y_arr)}, indent=2),
        encoding="utf-8",
    )

    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
