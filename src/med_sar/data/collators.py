# src/med_sar/data/collators.py

from typing import Any, Dict, List

import torch


class CriticCollator:
    def __init__(self, tokenizer, max_length: int = 256):
        self.tok = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [b["text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

        enc = self.tok(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc["labels"] = labels
        # keep kind/meta for debugging if needed
        enc["kind"] = [b["kind"] for b in batch]
        enc["meta"] = [b["meta"] for b in batch]
        return enc
