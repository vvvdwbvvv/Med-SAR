
# LLM wrapper stu
from transformers import AutoTokenizer, AutoModelForCausalLM

class Doctor:
    def __init__(self, name_or_path: str):
        self.tok = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(name_or_path)

    def format_prompt(self, note_text: str) -> str:
        return (
            "You are a clinical reasoning agent.\n"
            "Given the following patient note, provide reasoning then a final diagnosis.\n\n"
            f"NOTE:\n{note_text}\n\n"
            "REASONING:\n"
        )
