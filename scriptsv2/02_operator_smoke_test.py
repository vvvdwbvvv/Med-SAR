# Sample usage:
# python scriptsv2/02_operator_smoke_test.py \
#   --m23k data/processed/m23k_v2/m23k_val.jsonl \
#   --calibration outputs/calibration.json \
#   --guard_spec outputs/fact_guard_spec.yaml \
#   --out outputs/smoke/operator_samples.jsonl

from __future__ import annotations

import argparse
import difflib
import json
import random
from pathlib import Path
import sys
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from med_sar.guard.rules import GuardConfig, check_guard
from med_sar.operators.chain import apply_chain, parse_ops, sample_chain
from med_sar.operators.calibration import load_calibration, level_for_t
from utils.io import read_jsonl


def _load_guard_config(path: str | None) -> GuardConfig:
    if not path:
        return GuardConfig()
    import yaml

    data = yaml.safe_load(Path(path).read_text()) or {}
    return GuardConfig(
        min_entity_jaccard=float(data.get("min_entity_jaccard", 0.5)),
        min_length_ratio=float(data.get("min_length_ratio", 0.5)),
        max_length_ratio=float(data.get("max_length_ratio", 1.8)),
        allow_unit_changes=bool(data.get("allow_unit_changes", False)),
    )


def _print_diff(a: str, b: str) -> str:
    diff = difflib.unified_diff(
        a.splitlines(), b.splitlines(), lineterm="", fromfile="clean", tofile="adv"
    )
    return "\n".join(diff)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m23k", type=str, required=True)
    ap.add_argument("--calibration", type=str, default=None)
    ap.add_argument("--guard_spec", type=str, default=None)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t_values", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    ap.add_argument("--ops", type=str, default=None)
    ap.add_argument("--max_previews", type=int, default=2)
    args = ap.parse_args()

    rows = list(read_jsonl(args.m23k))
    rng = random.Random(args.seed)
    sample = rng.sample(rows, k=min(args.n, len(rows)))

    calibration = load_calibration(args.calibration)
    guard_cfg = _load_guard_config(args.guard_spec)

    out_rows: List[Dict] = []
    previewed = 0
    accept = 0
    reject = 0

    for i, r in enumerate(sample):
        clean = r.get("x_wrapped") or r.get("x_raw") or ""
        for t in args.t_values:
            if args.ops:
                ops = parse_ops(args.ops)
            else:
                ops = sample_chain(rng)
            adv = apply_chain(
                clean,
                ops,
                t=t,
                seed=args.seed + i,
                level_fn=lambda op, t_val: level_for_t(op, t_val, calibration),
            )
            guard = check_guard(clean, adv, guard_cfg)
            if guard.passed:
                accept += 1
            else:
                reject += 1
                if previewed < args.max_previews:
                    print("\n--- diff preview ---")
                    print(_print_diff(clean, adv))
                    print(f"reasons: {guard.reasons}")
                    previewed += 1
            out_rows.append(
                {
                    "id": r.get("id"),
                    "x_wrapped": clean,
                    "x_adv": adv,
                    "t": t,
                    "ops": ops,
                    "guard_pass": guard.passed,
                    "guard_reasons": guard.reasons,
                }
            )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = accept + reject
    print(f"accept={accept} reject={reject} total={total}")
    print(f"wrote {len(out_rows)} to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
