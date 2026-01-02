# binary classifier stub
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class Critic:
    def __init__(self, name_or_path: str):
        self.tok = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            name_or_path, num_labels=2
        )
