# src/med_sar/corruptions.py
from __future__ import annotations
import random
import re
from dataclasses import dataclass
from typing import Dict, List

_SENT_SPLIT = re.compile(r"(?<=[\.\?!])\s+")

DEFAULT_ABBREV = {
    "patient": "pt",
    "history": "hx",
    "shortness of breath": "sob",
    "chest pain": "cp",
    "blood pressure": "bp",
    "heart rate": "hr",
    "respiratory rate": "rr",
    "temperature": "temp",
    "denies": "neg",
}

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
    "because",
    "as",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "without",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "this",
    "that",
    "these",
    "those",
}


@dataclass
class CorruptConfig:
    level: float  # 0.0~0.5 typically
    seed: int = 0


def _split_sentences(text: str) -> List[str]:
    sents = _SENT_SPLIT.split(text.strip())
    return [s.strip() for s in sents if s.strip()]


def abbrev_jargon(
    text: str, cfg: CorruptConfig, abbrev: Dict[str, str] | None = None
) -> str:
    random.seed(cfg.seed)
    abbrev = abbrev or DEFAULT_ABBREV
    out = text
    # replace phrases first (longer keys first)
    for k in sorted(abbrev.keys(), key=len, reverse=True):
        if random.random() < cfg.level:
            out = re.sub(rf"\b{re.escape(k)}\b", abbrev[k], out, flags=re.IGNORECASE)
    return out


def telegraphic(text: str, cfg: CorruptConfig) -> str:
    random.seed(cfg.seed)
    tokens = re.split(r"(\W+)", text)  # keep separators
    kept = []
    for t in tokens:
        if (
            re.match(r"^\w+$", t)
            and t.lower() in STOPWORDS
            and random.random() < cfg.level
        ):
            continue
        kept.append(t)
    # optionally shorten whitespace
    out = "".join(kept)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def shuffle_sentences(text: str, cfg: CorruptConfig) -> str:
    random.seed(cfg.seed)
    sents = _split_sentences(text)
    if len(sents) < 2:
        return text
    # shuffle a portion proportional to level
    k = max(2, int(len(sents) * cfg.level))
    idx = list(range(len(sents)))
    random.shuffle(idx)
    pick = sorted(idx[:k])
    subset = [sents[i] for i in pick]
    random.shuffle(subset)
    for j, i in enumerate(pick):
        sents[i] = subset[j]
    return " ".join(sents)


def ellipsis_drop(text: str, cfg: CorruptConfig) -> str:
    random.seed(cfg.seed)
    sents = _split_sentences(text)
    if len(sents) < 2:
        return text
    # drop fraction of sentences, but keep at least 1
    drop_n = min(len(sents) - 1, int(len(sents) * cfg.level))
    drop_idx = set(random.sample(range(len(sents)), k=drop_n))
    kept = [s for i, s in enumerate(sents) if i not in drop_idx]
    return " ".join(kept)


def mixed(text: str, cfg: CorruptConfig) -> str:
    # apply multiple corruptions
    out = text
    out = abbrev_jargon(out, cfg)
    out = telegraphic(out, cfg)
    out = shuffle_sentences(out, cfg)
    out = ellipsis_drop(out, cfg)
    return out
