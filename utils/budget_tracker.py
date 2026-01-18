from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def count_tokens(text: str) -> int:
    return len(text.split())


@dataclass
class BudgetTracker:
    max_tokens: int | None = None
    max_examples: int | None = None
    total_tokens: int = 0
    total_examples: int = 0

    def add_text(self, text: str) -> None:
        self.total_tokens += count_tokens(text)
        self.total_examples += 1

    def add_texts(self, texts: Iterable[str]) -> None:
        for text in texts:
            self.add_text(text)

    def exceeds_budget(self) -> bool:
        if self.max_tokens is not None and self.total_tokens > self.max_tokens:
            return True
        if self.max_examples is not None and self.total_examples > self.max_examples:
            return True
        return False

    def to_dict(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "total_examples": self.total_examples,
            "max_tokens": self.max_tokens,
            "max_examples": self.max_examples,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
