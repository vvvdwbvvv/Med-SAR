# src/med_sar/corruptions.py
import random
import re
from dataclasses import dataclass
from typing import List

_SENT_SPLIT = re.compile(r"(?<=[\.\?!])\s+")
_LINE_SPLIT = re.compile(r"\n+")  # clinical notes often newline-delimited
NUM_RE = re.compile(r"(?<!\w)\d+(\.\d+)?(?!\w)")

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

NEG_TOKENS = {"no", "not", "without", "denies", "deny", "negative", "neg", "none"}
# STOPWORDS: remove "without" from stopwords (critical)
STOPWORDS = {...}  # your set but DO NOT include without / no / not

SECTION_TEMPLATES = [
    "HPI: {q}\nA/P: ?",
    "CC: {q}\nAssessment: ",
    "ED Note: {q}\nPlan:",
]


@dataclass
class CorruptConfig:
    level: float
    seed: int = 0
    protect_negation: bool = True
    protect_numbers: bool = True
    protect_lines_with_caps_headers: bool = True  # e.g., "HPI:", "PMH:"


def wrap_note_style(text: str, cfg: CorruptConfig) -> str:
    rng = random.Random(cfg.seed)
    if rng.random() > cfg.level:
        return text
    tpl = rng.choice(SECTION_TEMPLATES)
    return tpl.format(q=text)


def _split_units(text: str) -> List[str]:
    """Prefer newline/section units; fall back to sentence split."""
    t = text.strip()
    parts = [p.strip() for p in _LINE_SPLIT.split(t) if p.strip()]
    if len(parts) >= 2:
        return parts
    sents = [s.strip() for s in _SENT_SPLIT.split(t) if s.strip()]
    return sents if sents else [t]


def _has_negation(text: str) -> bool:
    t = text.lower()
    return any(tok in t for tok in NEG_TOKENS)


def _has_number(text: str) -> bool:
    return NUM_RE.search(text) is not None


def telegraphic(text: str, cfg: CorruptConfig) -> str:
    rng = random.Random(cfg.seed)
    tokens = re.split(r"(\W+)", text)
    word_positions = [
        i
        for i, t in enumerate(tokens)
        if re.match(r"^\w+$", t) and t.lower() in STOPWORDS
    ]
    if not word_positions:
        return re.sub(r"\s+", " ", text).strip()

    K = max(1, int(len(word_positions) * cfg.level))
    drop = set(rng.sample(word_positions, k=min(K, len(word_positions))))

    kept = []
    for i, t in enumerate(tokens):
        if i in drop:
            continue
        kept.append(t)
    return re.sub(r"\s+", " ", "".join(kept)).strip()


def abbrev_jargon(text: str, cfg: CorruptConfig, abbrev=None) -> str:
    rng = random.Random(cfg.seed)
    abbrev = abbrev or DEFAULT_ABBREV
    out = text
    matches = []
    for k in sorted(abbrev.keys(), key=len, reverse=True):
        # count occurrences
        if re.search(rf"\b{re.escape(k)}\b", out, flags=re.IGNORECASE):
            matches.append(k)

    if not matches:
        return out

    # choose how many keys to replace (budget)
    K = max(1, int(len(matches) * cfg.level))
    chosen = rng.sample(matches, k=min(K, len(matches)))

    for k in chosen:
        out = re.sub(rf"\b{re.escape(k)}\b", abbrev[k], out, flags=re.IGNORECASE)
    return out


def shuffle_units(text: str, cfg: CorruptConfig) -> str:
    rng = random.Random(cfg.seed)
    units = _split_units(text)
    if len(units) < 2:
        return text

    # protect header-like lines
    protected = []
    movable = []
    for u in units:
        if cfg.protect_lines_with_caps_headers and re.match(
            r"^[A-Z][A-Z/ ]{1,10}:$", u.strip()
        ):
            protected.append(u)
        else:
            movable.append(u)

    k = max(2, int(len(movable) * cfg.level))
    k = min(k, len(movable))
    idx = list(range(len(movable)))
    rng.shuffle(idx)
    pick = sorted(idx[:k])
    subset = [movable[i] for i in pick]
    rng.shuffle(subset)
    for j, i in enumerate(pick):
        movable[i] = subset[j]

    # naive merge: keep protected at original positions is hard; simplest: append protected back
    out_units = movable + protected
    return "\n".join(out_units)


def ellipsis_drop(text: str, cfg: CorruptConfig) -> str:
    rng = random.Random(cfg.seed)
    units = _split_units(text)
    if len(units) < 2:
        return text

    # do not drop units with negation/numbers if protection is on
    candidates = []
    locked = []
    for u in units:
        if (cfg.protect_negation and _has_negation(u)) or (
            cfg.protect_numbers and _has_number(u)
        ):
            locked.append(u)
        else:
            candidates.append(u)

    # drop only from candidates
    if not candidates:
        return "\n".join(units)

    drop_n = min(len(candidates), int(len(units) * cfg.level))
    drop_n = min(drop_n, len(units) - 1)  # keep at least 1 total
    drop_idx = set(rng.sample(range(len(candidates)), k=drop_n))
    kept = [u for i, u in enumerate(candidates) if i not in drop_idx] + locked
    rng.shuffle(kept) if cfg.level > 0.35 else None  # optional mild reorder
    return "\n".join(kept)


def mixed(text: str, cfg: CorruptConfig) -> str:
    # apply multiple corruptions
    out = text
    out = wrap_note_style(out, cfg)
    out = abbrev_jargon(out, cfg)
    out = telegraphic(out, cfg)
    out = shuffle_units(out, cfg)
    out = ellipsis_drop(out, cfg)
    return out
