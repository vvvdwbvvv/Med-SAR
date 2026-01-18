# Sample usage:
# python scriptsv2/03_calibrate_operator_strength.py \
#   --mimic_manifest data/processed/mimic_v2/mimic_manifest.parquet \
#   --out outputs/calibration.json \
#   --operators_out outputs/operators.yaml

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from med_sar.operators.calibration import default_calibration
from med_sar.operators.library import OPERATOR_SPECS


def _require_parquet() -> None:
    try:
        import pyarrow  # noqa: F401
    except Exception as exc:  # pragma: no cover - runtime check
        raise SystemExit(
            "pyarrow is required for parquet inputs. Install it and retry."
        ) from exc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic_manifest", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--operators_out", type=str, required=True)
    ap.add_argument("--t_grid", type=float, nargs="+", default=[i / 10 for i in range(11)])
    ap.add_argument("--proxy_csv", type=str, default=None)
    args = ap.parse_args()

    _require_parquet()

    df = pd.read_parquet(args.mimic_manifest)
    proxy_cols = [
        "newline_ratio",
        "colon_ratio",
        "digit_ratio",
        "header_density",
        "abbrev_ratio",
    ]
    proxy_stats: Dict[str, Dict[str, float]] = {}
    for proxy in proxy_cols:
        if proxy not in df.columns:
            proxy_stats[proxy] = {"median": 0.0, "p90": 0.0}
            continue
        proxy_stats[proxy] = {
            "median": float(df[proxy].median()),
            "p90": float(df[proxy].quantile(0.9)),
        }

    calibration = default_calibration(args.t_grid)
    calibration["proxy_stats"] = proxy_stats

    proxy_targets: Dict[str, Dict[str, float]] = {}
    for proxy, stats in proxy_stats.items():
        targets: Dict[str, float] = {}
        for t in args.t_grid:
            target = stats["median"] + float(t) * (stats["p90"] - stats["median"])
            targets[f"{t:.2f}"] = float(target)
        proxy_targets[proxy] = targets
    calibration["proxy_targets"] = proxy_targets

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    operators_payload = {
        "operators": [
            {
                "name": spec.name,
                "description": spec.description,
                "min_level": spec.min_level,
                "max_level": spec.max_level,
                "default_level": spec.default_level,
                "proxy_focus": spec.proxy_focus,
                "level_map": calibration["operators"][spec.name]["level_map"],
            }
            for spec in OPERATOR_SPECS
        ]
    }
    Path(args.operators_out).write_text(
        yaml.safe_dump(operators_payload, sort_keys=False), encoding="utf-8"
    )

    proxy_csv = Path(args.proxy_csv) if args.proxy_csv else out_path.with_suffix(".csv")
    with proxy_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["proxy", "t", "target"])
        for proxy, targets in proxy_targets.items():
            for t, val in targets.items():
                w.writerow([proxy, t, val])

    print(f"wrote calibration to {out_path}")
    print(f"wrote operators to {args.operators_out}")
    print(f"wrote proxy targets to {proxy_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
