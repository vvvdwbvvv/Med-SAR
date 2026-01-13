from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

ANSWER_RE = re.compile(r"<answer>\s*([A-Za-z])\s*</answer>", re.IGNORECASE)
FALLBACK_RE = re.compile(r"answer\s*[:\-]\s*([A-Za-z])", re.IGNORECASE)


def _extract_answer(text: str) -> Optional[str]:
    if not text:
        return None
    match = ANSWER_RE.search(text)
    if match:
        return match.group(1).upper()
    match = FALLBACK_RE.search(text)
    if match:
        return match.group(1).upper()
    return None


def _index_to_label(idx: int, options: Any) -> Optional[str]:
    if options is None:
        return None
    if isinstance(options, dict):
        keys = list(options.keys())
        if 0 <= idx < len(keys):
            return str(keys[idx]).upper()
        return None
    if isinstance(options, list):
        if not options:
            return None
        if isinstance(options[0], dict):
            labels = []
            for opt in options:
                label = opt.get("label") or opt.get("key") or opt.get("option")
                labels.append(label)
            if 0 <= idx < len(labels):
                label = labels[idx]
                return str(label).upper() if label is not None else None
            return None
        letters = [chr(ord("A") + i) for i in range(len(options))]
        if 0 <= idx < len(letters):
            return letters[idx]
    return None


def _normalize_label(value: Any, options: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if len(stripped) == 1 and stripped.isalpha():
            return stripped.upper()
        if stripped.isdigit():
            return _index_to_label(int(stripped), options)
        if isinstance(options, dict):
            for key, opt in options.items():
                if isinstance(opt, str) and opt.strip().lower() == stripped.lower():
                    return str(key).upper()
        if isinstance(options, list):
            for idx, opt in enumerate(options):
                if isinstance(opt, str) and opt.strip().lower() == stripped.lower():
                    return _index_to_label(idx, options)
                if isinstance(opt, dict):
                    text = opt.get("text") or opt.get("value") or opt.get("answer")
                    if (
                        isinstance(text, str)
                        and text.strip().lower() == stripped.lower()
                    ):
                        label = opt.get("label") or opt.get("key") or opt.get("option")
                        if label:
                            return str(label).upper()
                        return _index_to_label(idx, options)
        return None
    if isinstance(value, int):
        return _index_to_label(value, options)
    return None


def _get_gold_label(item: Dict[str, Any]) -> Optional[str]:
    options = item.get("options")
    for key in ("answer_letter", "answer_idx", "answer", "label", "gold"):
        if key in item:
            label = _normalize_label(item.get(key), options)
            if label:
                return label
    return None


def get_results(result_file: str | Path) -> Dict[str, float]:
    result_path = Path(result_file)
    rows = json.loads(result_path.read_text())
    total = len(rows)
    evaluated = 0
    correct = 0
    missing_pred = 0
    missing_gold = 0

    for item in rows:
        pred = _extract_answer(item.get("output", ""))
        gold = _get_gold_label(item)
        if pred is None:
            missing_pred += 1
            continue
        if gold is None:
            missing_gold += 1
            continue
        evaluated += 1
        if pred == gold:
            correct += 1

    acc = correct / evaluated if evaluated else 0.0
    print(
        f"Accuracy: {acc:.4f} ({correct}/{evaluated}) | "
        f"missing_pred={missing_pred} missing_gold={missing_gold} total={total}"
    )
    return {
        "accuracy": acc,
        "correct": float(correct),
        "evaluated": float(evaluated),
        "missing_pred": float(missing_pred),
        "missing_gold": float(missing_gold),
        "total": float(total),
    }
