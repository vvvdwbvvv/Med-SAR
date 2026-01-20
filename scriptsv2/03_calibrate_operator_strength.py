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

PROXY_DEFS = [
    {
        "name": "proxy_newline_ratio",
        "description": "newline characters per total chars",
    },
    {
        "name": "proxy_colon_ratio",
        "description": "colon characters per total chars",
    },
    {
        "name": "proxy_digit_ratio",
        "description": "digit characters per total chars",
    },
    {
        "name": "proxy_header_density",
        "description": "header-like lines per non-empty line",
    },
    {
        "name": "proxy_abbrev_ratio",
        "description": "all-caps abbreviations per token",
    },
]


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
    ap.add_argument("--out", type=str, default="configs/calibration.json")
    ap.add_argument("--operators_out", type=str, default="configs/operators.yaml")
    ap.add_argument("--proxies_out", type=str, default="configs/proxies.yaml")
    ap.add_argument(
        "--t_grid", type=float, nargs="+", default=[i / 10 for i in range(11)]
    )
    ap.add_argument("--proxy_csv", type=str, default=None)
    args = ap.parse_args()

    _require_parquet()

    df = pd.read_parquet(args.mimic_manifest)
    proxy_cols = [
        ("proxy_newline_ratio", "newline_ratio"),
        ("proxy_colon_ratio", "colon_ratio"),
        ("proxy_digit_ratio", "digit_ratio"),
        ("proxy_header_density", "header_density"),
        ("proxy_abbrev_ratio", "abbrev_ratio"),
    ]
    proxy_stats: Dict[str, Dict[str, float]] = {}
    for proxy_name, legacy_name in proxy_cols:
        col = proxy_name if proxy_name in df.columns else legacy_name
        if col not in df.columns:
            proxy_stats[proxy_name] = {"median": 0.0, "p90": 0.0}
            continue
        proxy_stats[proxy_name] = {
            "median": float(df[col].median()),
            "p90": float(df[col].quantile(0.9)),
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
        "chain_rule": {"max_len": 2, "allow_repeat": False},
        "t_grid": [float(t) for t in args.t_grid],
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
        ],
    }
    Path(args.operators_out).write_text(
        yaml.safe_dump(operators_payload, sort_keys=False), encoding="utf-8"
    )

    proxies_payload = {
        "proxy_set_version": "proxy5_v1",
        "proxies": [
            {
                **proxy_def,
                **proxy_stats.get(proxy_def["name"], {"median": 0.0, "p90": 0.0}),
            }
            for proxy_def in PROXY_DEFS
        ],
    }
    Path(args.proxies_out).write_text(
        yaml.safe_dump(proxies_payload, sort_keys=False), encoding="utf-8"
    )

    proxy_csv = (
        Path(args.proxy_csv)
        if args.proxy_csv
        else Path("runs/calibration/proxy_targets.csv")
    )
    proxy_csv.parent.mkdir(parents=True, exist_ok=True)
    with proxy_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["proxy", "t", "target"])
        for proxy, targets in proxy_targets.items():
            for t, val in targets.items():
                w.writerow([proxy, t, val])

    print(f"wrote calibration to {out_path}")
    print(f"wrote operators to {args.operators_out}")
    print(f"wrote proxies to {args.proxies_out}")
    print(f"wrote proxy targets to {proxy_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
