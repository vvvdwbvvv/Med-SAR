from __future__ import annotations

import re
from typing import Set

from utils.text import normalize_text

NEG_TOKENS = {
    "no",
    "not",
    "without",
    "denies",
    "deny",
    "negative",
    "neg",
    "none",
}

NUM_RE = re.compile(r"(?<!\w)(\d+(?:\.\d+)?)(?!\w)")
UNIT_RE = re.compile(
    r"(?<!\w)(?:mg|g|kg|mcg|ml|l|bpm|mmhg|cm|mm|mmol|meq|u/l|iu|%)(?!\w)",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-]+")


def has_negation(text: str) -> bool:
    t = normalize_text(text).lower()
    return any(tok in t for tok in NEG_TOKENS)


def extract_numbers(text: str) -> Set[str]:
    return set(m.group(1) for m in NUM_RE.finditer(text))


def extract_units(text: str) -> Set[str]:
    return set(m.group(0).lower() for m in UNIT_RE.finditer(text))


def extract_entities(text: str, min_len: int = 4) -> Set[str]:
    toks = [m.group(0).lower() for m in TOKEN_RE.finditer(text)]
    return set(t for t in toks if len(t) >= min_len)
