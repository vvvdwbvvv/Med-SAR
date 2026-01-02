from transformers import AutoTokenizer, AutoModelForCausalLM

# prompt-based rewrite
class Generator:
    def __init__(self, name_or_path: str):
        self.tok = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(name_or_path)

    def build_rewrite_prompt(self, clean_text: str) -> str:
        return (
            "Rewrite the following medical query into a hasty ICU doctor's note style.\n"
            "Use abbreviations (pt, sob, hx), omit grammar, allow missing info, keep core meaning.\n\n"
            f"CLEAN:\n{clean_text}\n\n"
            "NOISY_NOTE:\n"
        )
