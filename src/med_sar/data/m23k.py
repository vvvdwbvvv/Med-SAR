from dataclasses import dataclass
from typing import Any, Dict
from datasets import load_dataset

from utils.text import normalize_text


@dataclass(frozen=True)
class NormalizeConfig:
    split: str = "train"
    task: str = "mcq_rationale"
    # If True, also copy upstream `text` into input.context (when non-empty and not redundant).
    place_text_in_context: bool = True
    # If True, require at least one of answer_string/distilled_answer_string.
    require_answer_text: bool = True


def load_m23k_raw():
    ds = load_dataset("UCSC-VLAA/m23k-tokenized")
    split = "train"
    print(split, len(ds[split]))
    print(ds[split].column_names)

    ex = ds[split][0]
    print(ex)

    return ds[split]


_UPSTREAM_KEYS = {
    "answer_idx",
    "source",
    "metadata",
    "prompt",
    "answer_letter",
    "answer_string",
    "reasoning",
    "distilled_answer_string",
    "text",
}


def normalize_m23k_record(
    row: Dict[str, Any],
    *,
    cfg: NormalizeConfig = NormalizeConfig(),
) -> Dict[str, Any]:
    """
    Convert one upstream m23k record into canonical schema.
    """

    # Soft check: warn-free; do not fail on extra keys.
    prompt = normalize_text(row.get("prompt"))
    answer_idx = row.get("answer_idx", None)

    if answer_idx is not None and isinstance(answer_idx, str) and answer_idx.isdigit():
        answer_idx = int(answer_idx)

    answer_letter = normalize_text(row.get("answer_letter"))

    answer_string = normalize_text(row.get("answer_string"))
    distilled_answer_string = normalize_text(row.get("distilled_answer_string"))
    reasoning = normalize_text(row.get("reasoning"))
    text = normalize_text(row.get("text"))

    upstream_source = normalize_text(row.get("source"))
    metadata = row.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        # Preserve but keep canonical as dict; if upstream is string, wrap it.
        metadata = {"_metadata": metadata}

    # Decide whether to place upstream `text` into input.context.
    context = ""
    if cfg.place_text_in_context and text:
        # If text is identical to prompt, do not duplicate.
        context = "" if text == prompt else text

    # Stable ID payload: prefer any unique upstream id if present in metadata.
    # We do not assume a specific key name; try common candidates.
    upstream_id = None
    for k in ("id", "example_id", "uid", "qid", "question_id", "instance_id"):
        if k in metadata and metadata[k] is not None:
            upstream_id = str(metadata[k])
            break

    canonical: Dict[str, Any] = {
        "id": f"m23k-{cfg.split}-{upstream_id}",
        "source": "m23k",
        "split": cfg.split,
        "task": cfg.task,
        "input": {
            "prompt": prompt,
            "context": context,
            "choices": None,
        },
        "output": {
            "answer_idx": answer_idx,
            "answer_letter": answer_letter or None,
            "answer_string": answer_string or None,
            "distilled_answer_string": distilled_answer_string or None,
            "reasoning": reasoning or None,
        },
        "meta": {
            "upstream_source": upstream_source or None,
            "metadata": metadata,
            "raw_fields": {
                "text": text or None,
            },
        },
    }
    return canonical
