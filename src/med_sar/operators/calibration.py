from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .library import OPERATOR_SPECS, get_spec


def load_calibration(path: str | Path | None) -> Dict[str, Any] | None:
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _closest_key(mapping: Dict[str, Any], t: float) -> str:
    # Keys stored as strings; pick closest numeric.
    best_key = None
    best_dist = None
    for k in mapping.keys():
        try:
            val = float(k)
        except ValueError:
            continue
        dist = abs(val - t)
        if best_dist is None or dist < best_dist:
            best_key = k
            best_dist = dist
    if best_key is None:
        raise KeyError("No numeric keys in calibration mapping")
    return best_key


def level_for_t(op_name: str, t: float, calibration: Dict[str, Any] | None) -> float:
    if calibration is None:
        return t
    ops = calibration.get("operators", {})
    if op_name not in ops:
        return t
    levels = ops[op_name].get("level_map", {})
    if not levels:
        return t
    key = _closest_key(levels, t)
    return float(levels[key])


def build_linear_levels(t_grid: Iterable[float]) -> Dict[str, Dict[str, float]]:
    levels: Dict[str, Dict[str, float]] = {}
    for spec in OPERATOR_SPECS:
        mapping: Dict[str, float] = {}
        for t in t_grid:
            level = spec.min_level + float(t) * (spec.max_level - spec.min_level)
            mapping[f"{t:.2f}"] = float(level)
        levels[spec.name] = mapping
    return levels


def default_calibration(t_grid: Iterable[float]) -> Dict[str, Any]:
    return {
        "t_grid": [float(t) for t in t_grid],
        "operators": {
            spec.name: {
                "min_level": spec.min_level,
                "max_level": spec.max_level,
                "default_level": spec.default_level,
                "proxy_focus": spec.proxy_focus,
                "level_map": build_linear_levels(t_grid)[spec.name],
            }
            for spec in OPERATOR_SPECS
        },
    }
