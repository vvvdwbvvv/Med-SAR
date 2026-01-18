from __future__ import annotations

import re
from typing import Dict

from utils.text import normalize_text

_HEADER_RE = re.compile(r"^[A-Z][A-Z/\- ]{1,24}:\s*$")
_TOKEN_RE = re.compile(r"[A-Za-z]+")


def compute_proxies(text: str) -> Dict[str, float]:
    text = normalize_text(text)
    if not text:
        return {
            "newline_ratio": 0.0,
            "colon_ratio": 0.0,
            "digit_ratio": 0.0,
            "header_density": 0.0,
            "abbrev_ratio": 0.0,
        }

    length_chars = len(text)
    newline_ratio = text.count("\n") / max(1, length_chars)
    colon_ratio = text.count(":") / max(1, length_chars)
    digit_ratio = sum(1 for c in text if c.isdigit()) / max(1, length_chars)

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    header_count = sum(1 for line in lines if _HEADER_RE.match(line))
    header_density = header_count / max(1, len(lines))

    tokens = _TOKEN_RE.findall(text)
    if tokens:
        abbrev = sum(1 for tok in tokens if tok.isupper() and 2 <= len(tok) <= 5)
        abbrev_ratio = abbrev / max(1, len(tokens))
    else:
        abbrev_ratio = 0.0

    return {
        "newline_ratio": float(newline_ratio),
        "colon_ratio": float(colon_ratio),
        "digit_ratio": float(digit_ratio),
        "header_density": float(header_density),
        "abbrev_ratio": float(abbrev_ratio),
    }
