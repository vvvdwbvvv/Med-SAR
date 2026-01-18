from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

from med_sar.corruptions import (
    CorruptConfig,
    abbrev_jargon,
    telegraphic,
    shuffle_units,
    ellipsis_drop,
)


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    description: str
    min_level: float
    max_level: float
    default_level: float
    proxy_focus: str
    fn: Callable[[str, CorruptConfig], str]


def _apply(
    fn: Callable[[str, CorruptConfig], str], text: str, level: float, seed: int
) -> str:
    cfg = CorruptConfig(level=level, seed=seed)
    return fn(text, cfg)


OPERATOR_SPECS: List[OperatorSpec] = [
    OperatorSpec(
        name="abbrev_jargon",
        description="Replace common medical phrases with abbreviations.",
        min_level=0.0,
        max_level=0.9,
        default_level=0.3,
        proxy_focus="abbrev_ratio",
        fn=abbrev_jargon,
    ),
    OperatorSpec(
        name="telegraphic",
        description="Drop stopwords to mimic telegraphic clinical style.",
        min_level=0.0,
        max_level=0.8,
        default_level=0.3,
        proxy_focus="digit_ratio",
        fn=telegraphic,
    ),
    OperatorSpec(
        name="shuffle_units",
        description="Shuffle sentence/line units without repetition.",
        min_level=0.0,
        max_level=0.9,
        default_level=0.35,
        proxy_focus="newline_ratio",
        fn=shuffle_units,
    ),
    OperatorSpec(
        name="ellipsis_drop",
        description="Drop non-critical units with ellipsis-style gaps.",
        min_level=0.0,
        max_level=0.85,
        default_level=0.3,
        proxy_focus="header_density",
        fn=ellipsis_drop,
    ),
]


_OPERATOR_INDEX: Dict[str, OperatorSpec] = {spec.name: spec for spec in OPERATOR_SPECS}


def operator_names() -> List[str]:
    return [spec.name for spec in OPERATOR_SPECS]


def get_spec(name: str) -> OperatorSpec:
    if name not in _OPERATOR_INDEX:
        raise KeyError(f"Unknown operator: {name}")
    return _OPERATOR_INDEX[name]


def apply_operator(name: str, text: str, level: float, seed: int) -> str:
    spec = get_spec(name)
    return _apply(spec.fn, text, level, seed)
