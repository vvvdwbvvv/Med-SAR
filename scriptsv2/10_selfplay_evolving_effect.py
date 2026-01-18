# Sample usage:
# python scriptsv2/10_selfplay_evolving_effect.py \
#   --policy_logs outputs/loop_train_v2/policy_selection_logs.jsonl \
#   --out_csv outputs/selfplay_policy_summary.csv

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_logs", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_json", type=str, default=None)
    args = ap.parse_args()

    rows = []
    with Path(args.policy_logs).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))

    summary = []
    for r in rows:
        summary.append(
            {
                "round": r.get("round"),
                "best_t": r.get("best_t"),
                "best_ops": ",".join(r.get("best_ops", [])),
                "best_score": r.get("best_score"),
            }
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["round", "best_t", "best_ops", "best_score"])
        w.writeheader()
        w.writerows(summary)

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
