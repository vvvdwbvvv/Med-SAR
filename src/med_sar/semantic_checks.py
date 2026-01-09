# src/med_sar/semantic_checks.py
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

NEG_TRIGGERS = [
    "no ", "denies", "deny", "without", "not ", "negative for", "neg",
]
# very lightweight number capture
NUM_RE = re.compile(r"(?<!\w)(\d+(\.\d+)?)(?!\w)")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-]+")

@dataclass
class SemanticReport:
    negation_flip: bool
    number_mismatch: bool
    entity_jaccard: float

def _has_negation(text: str) -> bool:
    t = text.lower()
    return any(tr in t for tr in NEG_TRIGGERS)

def _numbers(text: str) -> Set[str]:
    return set(m.group(1) for m in NUM_RE.finditer(text))

def _entities(text: str, min_len: int = 4) -> Set[str]:
    # heuristic: keep longer alphabetic tokens as "entities"
    toks = [m.group(0).lower() for m in TOKEN_RE.finditer(text)]
    return set(t for t in toks if len(t) >= min_len)

def semantic_report(clean: str, adv: str) -> SemanticReport:
    neg_clean = _has_negation(clean)
    neg_adv = _has_negation(adv)
    neg_flip = (neg_clean != neg_adv)

    nums_clean = _numbers(clean)
    nums_adv = _numbers(adv)
    num_mismatch = (nums_clean != nums_adv)

    ent_clean = _entities(clean)
    ent_adv = _entities(adv)
    if not ent_clean and not ent_adv:
        jac = 1.0
    else:
        jac = len(ent_clean & ent_adv) / max(1, len(ent_clean | ent_adv))

    return SemanticReport(
        negation_flip=neg_flip,
        number_mismatch=num_mismatch,
        entity_jaccard=jac,
    )

def aggregate_reports(reports: List[SemanticReport]) -> Dict[str, float]:
    if not reports:
        return {"neg_flip_rate": 0.0, "num_mismatch_rate": 0.0, "entity_jaccard_mean": 1.0}
    neg = sum(r.negation_flip for r in reports) / len(reports)
    num = sum(r.number_mismatch for r in reports) / len(reports)
    jac = sum(r.entity_jaccard for r in reports) / len(reports)
    return {"neg_flip_rate": neg, "num_mismatch_rate": num, "entity_jaccard_mean": jac}
