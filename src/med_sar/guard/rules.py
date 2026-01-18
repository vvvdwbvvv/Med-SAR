from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .extract import (
    extract_entities,
    extract_numbers,
    extract_units,
    has_negation,
)


@dataclass(frozen=True)
class GuardConfig:
    min_entity_jaccard: float = 0.5
    min_length_ratio: float = 0.5
    max_length_ratio: float = 1.8
    allow_unit_changes: bool = False


@dataclass(frozen=True)
class GuardResult:
    passed: bool
    reasons: List[str]
    metrics: Dict[str, float]
    primary_reason: Optional[str]


_REASON_MAP = {
    "number_mismatch": "F1",
    "negation_flip": "F2",
    "unit_mismatch": "F3",
    "entity_drop": "F4",
    "length_ratio": "F5",
}


def _entity_jaccard(a: str, b: str) -> float:
    ent_a = extract_entities(a)
    ent_b = extract_entities(b)
    if not ent_a and not ent_b:
        return 1.0
    return len(ent_a & ent_b) / max(1, len(ent_a | ent_b))


def check_guard(clean: str, adv: str, cfg: GuardConfig) -> GuardResult:
    reasons: List[str] = []

    neg_flip = has_negation(clean) != has_negation(adv)
    if neg_flip:
        reasons.append("negation_flip")

    nums_clean = extract_numbers(clean)
    nums_adv = extract_numbers(adv)
    num_mismatch = nums_clean != nums_adv
    if num_mismatch:
        reasons.append("number_mismatch")

    unit_mismatch = False
    if not cfg.allow_unit_changes:
        units_clean = extract_units(clean)
        units_adv = extract_units(adv)
        unit_mismatch = units_clean != units_adv
        if unit_mismatch:
            reasons.append("unit_mismatch")

    entity_jaccard = _entity_jaccard(clean, adv)
    if entity_jaccard < cfg.min_entity_jaccard:
        reasons.append("entity_drop")

    len_ratio = (len(adv) / max(1, len(clean))) if clean else 0.0
    if len_ratio < cfg.min_length_ratio or len_ratio > cfg.max_length_ratio:
        reasons.append("length_ratio")

    passed = len(reasons) == 0
    primary_reason = None
    if reasons:
        primary_reason = _REASON_MAP.get(reasons[0], "F?")

    metrics = {
        "negation_flip": float(neg_flip),
        "number_mismatch": float(num_mismatch),
        "unit_mismatch": float(unit_mismatch),
        "entity_jaccard": float(entity_jaccard),
        "length_ratio": float(len_ratio),
    }

    return GuardResult(
        passed=passed, reasons=reasons, metrics=metrics, primary_reason=primary_reason
    )
