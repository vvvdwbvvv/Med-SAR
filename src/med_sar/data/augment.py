import re
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple


# ---------------------------
# A) Normalization (apply to ALL samples)
# ---------------------------

PHI_SPAN_RE = re.compile(r"\[\*\*.*?\*\*\]", flags=re.DOTALL)

SECTION_HEADER_RE = re.compile(
    r"^(?P<header>[A-Z][A-Z0-9 /,\-\(\)]{2,})(:)?\s*$"
)

MULTI_NL_RE = re.compile(r"\n{3,}")
MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def mask_phi_spans(text: str, token: str = "[PHI]") -> str:
    return PHI_SPAN_RE.sub(token, text)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip trailing spaces per line, collapse multiple internal spaces
    lines = [MULTI_SPACE_RE.sub(" ", ln).rstrip() for ln in text.split("\n")]
    text = "\n".join(lines).strip()
    text = MULTI_NL_RE.sub("\n\n", text)
    return text


def mask_section_headers(text: str, token: str = "[SECTION]", prob: float = 1.0, rng: random.Random = None) -> str:
    rng = rng or random.Random(0)
    out_lines = []
    for ln in text.split("\n"):
        m = SECTION_HEADER_RE.match(ln.strip())
        if m and rng.random() < prob:
            out_lines.append(f"{token}:")
        else:
            out_lines.append(ln)
    return "\n".join(out_lines)


def normalize_all(text: str, rng: random.Random) -> str:
    text = mask_phi_spans(text, token="[PHI]")
    text = normalize_whitespace(text)
    # Mask some headers to reduce shortcut learning; tune prob (0.3–1.0)
    text = mask_section_headers(text, token="[SECTION]", prob=0.5, rng=rng)
    text = normalize_whitespace(text)
    return text


# ---------------------------
# B) Corruptions (apply to create NEGATIVES)
# ---------------------------

FOOTER_START_RE = re.compile(r"^(Dictated By:|MEDQUIST|JOB#:|JOB#|D:\s|\s*T:\s)", flags=re.IGNORECASE)


def drop_footer_blocks(text: str) -> str:
    """
    Removes common footer/signature blocks by cutting from the first footer marker to end.
    Conservative: only triggers if marker appears in last ~35% of document.
    """
    lines = text.split("\n")
    n = len(lines)
    cutoff = int(n * 0.65)
    for i in range(cutoff, n):
        if FOOTER_START_RE.search(lines[i].strip()):
            return "\n".join(lines[:i]).rstrip()
    return text


def remove_section_headers(text: str) -> str:
    out_lines = []
    for ln in text.split("\n"):
        if SECTION_HEADER_RE.match(ln.strip()):
            continue
        out_lines.append(ln)
    return "\n".join(out_lines)


def paragraph_shuffle(text: str, rng: random.Random, keep_first: bool = True) -> str:
    # paragraphs split by blank lines
    paras = re.split(r"\n\s*\n", text.strip())
    if len(paras) <= 2:
        return text
    first = paras[0]
    rest = paras[1:]
    rng.shuffle(rest)
    shuffled = [first] + rest if keep_first else rest + [first]
    return "\n\n".join(shuffled)


_SENT_SPLIT_RE = re.compile(r"(?<=[\.\?\!])\s+(?=[A-Z\[])")

def sentence_shuffle_within_paragraphs(text: str, rng: random.Random, min_sents: int = 3) -> str:
    paras = re.split(r"\n\s*\n", text.strip())
    new_paras = []
    for p in paras:
        sents = _SENT_SPLIT_RE.split(p.strip())
        if len(sents) >= min_sents:
            rng.shuffle(sents)
            new_paras.append(" ".join(sents))
        else:
            new_paras.append(p)
    return "\n\n".join(new_paras)


def punctuation_overclean(text: str) -> str:
    # Simulate "too clean" generator output: remove repeated punctuation patterns, normalize colons.
    text = re.sub(r":\s*", " ", text)            # remove colon structure
    text = re.sub(r"\s+([,;])", r"\1", text)     # trim spaces before punctuation
    text = re.sub(r"[,;]{2,}", ",", text)        # collapse repeated punctuation
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return text


ABBREV_MAP = {
    "CT": "computed tomography",
    "IV": "intravenous",
    "CXR": "chest x-ray",
    "S/P": "status post",
}

def expand_abbreviations(text: str, rng: random.Random, prob: float = 0.5) -> str:
    # Replace a random subset of abbreviations (word-boundary aware)
    for abbr, full in ABBREV_MAP.items():
        if rng.random() < prob:
            text = re.sub(rf"\b{re.escape(abbr)}\b", full, text)
    return text


def suppress_all_caps(text: str) -> str:
    # Lowercase ALL CAPS tokens longer than 4 chars (e.g., RADIOLOGIC -> radiologic)
    def repl(m):
        w = m.group(0)
        return w.lower()
    return re.sub(r"\b[A-Z]{5,}\b", repl, text)


def delete_some_numbers(text: str, rng: random.Random, prob: float = 0.3) -> str:
    # Remove some numeric tokens (not PHI since already masked).
    def repl(m):
        return "" if rng.random() < prob else m.group(0)
    return re.sub(r"\b\d+(\.\d+)?\b", repl, text)


# ---------------------------
# Corruption pipeline builder
# ---------------------------

@dataclass
class CorruptionConfig:
    seed: int = 0
    num_transforms: Tuple[int, int] = (1, 3)  # apply between 1 and 3 corruptions
    # weights for choosing corruptions
    transform_weights: Dict[str, float] = None


def build_default_config(seed: int = 0) -> CorruptionConfig:
    return CorruptionConfig(
        seed=seed,
        num_transforms=(1, 3),
        transform_weights={
            "drop_footer": 1.2,
            "remove_headers": 1.0,
            "para_shuffle": 1.0,
            "sent_shuffle": 0.8,
            "punct_overclean": 0.8,
            "expand_abbrev": 0.6,
            "suppress_caps": 0.6,
            "delete_numbers": 0.5,
        }
    )


TRANSFORMS: Dict[str, Callable[[str, random.Random], str]] = {
    "drop_footer": lambda t, rng: drop_footer_blocks(t),
    "remove_headers": lambda t, rng: remove_section_headers(t),
    "para_shuffle": lambda t, rng: paragraph_shuffle(t, rng=rng, keep_first=True),
    "sent_shuffle": lambda t, rng: sentence_shuffle_within_paragraphs(t, rng=rng),
    "punct_overclean": lambda t, rng: punctuation_overclean(t),
    "expand_abbrev": lambda t, rng: expand_abbreviations(t, rng=rng, prob=0.7),
    "suppress_caps": lambda t, rng: suppress_all_caps(t),
    "delete_numbers": lambda t, rng: delete_some_numbers(t, rng=rng, prob=0.35),
}


def make_negative(text: str, cfg: CorruptionConfig) -> str:
    rng = random.Random(cfg.seed)
    # Normalize first so negatives aren't trivially detectable by PHI/whitespace
    t = normalize_all(text, rng=rng)

    weights = cfg.transform_weights or build_default_config(cfg.seed).transform_weights
    names = list(weights.keys())
    w = [weights[n] for n in names]

    k = rng.randint(cfg.num_transforms[0], cfg.num_transforms[1])
    chosen = rng.sample(names, k=k) if k <= len(names) else names

    for name in chosen:
        t = TRANSFORMS[name](t, rng)
        t = normalize_whitespace(t)

    return t


def make_pair(text: str, seed: int = 0) -> Dict[str, str]:
    rng = random.Random(seed)
    real = normalize_all(text, rng=rng)
    neg = make_negative(text, cfg=build_default_config(seed=seed))
    return {"real": real, "neg": neg}


# ---------------------------
# Example usage on your sample
# ---------------------------
if __name__ == "__main__":
    sample = """Admission Date: [**2151-7-16**] Discharge Date: [**2151-8-4**]

Service:
ADDENDUM:

RADIOLOGIC STUDIES: Radiologic studies also included a chest
CT, which confirmed cavitary lesions in the left lung apex
consistent with infectious process/tuberculosis. This also
moderate-sized left pleural effusion.

HEAD CT: Head CT showed no intracranial hemorrhage or mass
effect, but old infarction consistent with past medical
history.

ABDOMINAL CT: Abdominal CT showed lesions of
T10 and sacrum most likely secondary to osteoporosis. These can
be followed by repeat imaging as an outpatient.

[**First Name8 (NamePattern2) **] [**First Name4 (NamePattern1) 1775**]
[**Last Name (NamePattern1) **], M.D. [**MD Number(1) 1776**]

Dictated By:[**Hospital 1807**]
MEDQUIST36

D: [**2151-8-5**] 12:11
T: [**2151-8-5**] 12:21
JOB#: [**Job Number 1808**]"""

    pair = make_pair(sample, seed=42)
    print("----- REAL (normalized) -----")
    print(pair["real"][:800])
    print("\n----- NEG (corrupted) -----")
    print(pair["neg"][:800])
