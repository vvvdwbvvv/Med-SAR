from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _pick_device(device: Optional[str]) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _pick_dtype(device: torch.device) -> torch.dtype:
    if device.type in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def _load_chat_template(model_path: str) -> Optional[str]:
    template_path = Path(model_path) / "chat_template.jinja"
    if template_path.exists():
        return template_path.read_text()
    return None


def _apply_chat_template(tokenizer, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


@dataclass
class GenerateConfig:
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    max_input_tokens: int = -1
    use_chat_template: bool = False


class BaseRunner:
    def generate(
        self, prompts: Sequence[str], *, cfg: GenerateConfig
    ) -> Tuple[List[str], List[str]]:
        raise NotImplementedError


class OpenAICompatRunner(BaseRunner):
    def __init__(self, *, port: int, model: Optional[str] = None):
        import openai

        base_url = f"http://127.0.0.1:{port}/v1"
        if hasattr(openai, "Client"):
            self._client = openai.Client(base_url=base_url, api_key="EMPTY")
        else:
            self._client = openai.OpenAI(base_url=base_url, api_key="EMPTY")

        if model is None:
            models = self._client.models.list().data
            if not models:
                raise RuntimeError(
                    "No models available from the local API; pass --model."
                )
            model = models[0].id
        self.model = model

    def generate(
        self, prompts: Sequence[str], *, cfg: GenerateConfig
    ) -> Tuple[List[str], List[str]]:
        response = self._client.completions.create(
            model=self.model,
            prompt=list(prompts),
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_new_tokens,
        )
        raw_preds = [choice.text for choice in response.choices]
        postprocessed_preds = [
            pred.replace("</s>", "").lstrip(" ") for pred in raw_preds
        ]
        return postprocessed_preds, raw_preds


class TransformersRunner(BaseRunner):
    def __init__(
        self,
        *,
        model_path: str,
        base_model: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.device = _pick_device(device)
        dtype = _pick_dtype(self.device)

        print(f"Loading tokenizer from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if (
            getattr(tokenizer, "pad_token_id", None) is None
            and getattr(tokenizer, "eos_token_id", None) is not None
        ):
            tokenizer.pad_token_id = tokenizer.eos_token_id
        chat_template = _load_chat_template(model_path)
        if chat_template and not getattr(tokenizer, "chat_template", None):
            tokenizer.chat_template = chat_template

        adapter_cfg_path = Path(model_path) / "adapter_config.json"
        if base_model is None and adapter_cfg_path.exists():
            base_model = json.loads(adapter_cfg_path.read_text()).get(
                "base_model_name_or_path"
            )

        print(f"Loading model from {base_model or model_path}...")
        print(f"Using dtype: {dtype}, device: {self.device}")

        model = AutoModelForCausalLM.from_pretrained(
            base_model or model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",  # Add automatic device mapping
            low_cpu_mem_usage=True,  # Reduce CPU memory usage
            max_memory={0: "15GB"}
            if torch.cuda.is_available()
            else None,  # Limit GPU memory for Colab
        )

        adapter_path = Path(model_path) / "adapter_model.safetensors"
        if adapter_path.exists():
            print(f"Loading adapter from {model_path}...")
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, model_path)

        model.eval()
        # Don't call model.to() when using device_map="auto"
        # model.to(self.device)

        self.tokenizer = tokenizer
        self.model = model

        print("✓ Model loaded successfully")
        if torch.cuda.is_available():
            print(
                f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f}GB"
            )

    def generate(
        self, prompts: Sequence[str], *, cfg: GenerateConfig
    ) -> Tuple[List[str], List[str]]:
        rendered: List[str] = []
        if cfg.use_chat_template:
            rendered = [_apply_chat_template(self.tokenizer, p) for p in prompts]
        else:
            rendered = list(prompts)

        if cfg.max_input_tokens > 0:
            trimmed: List[str] = []
            for p in rendered:
                ids = self.tokenizer.encode(p, add_special_tokens=False)
                if len(ids) > cfg.max_input_tokens:
                    ids = ids[: cfg.max_input_tokens]
                    trimmed.append(self.tokenizer.decode(ids))
                else:
                    trimmed.append(p)
            rendered = trimmed

        enc = self.tokenizer(
            rendered,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        do_sample = cfg.temperature is not None and cfg.temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=do_sample,
                temperature=cfg.temperature if do_sample else None,
                top_p=cfg.top_p if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated suffix (per-example).
        input_lens = enc["attention_mask"].sum(dim=1).tolist()
        decoded_raw: List[str] = []
        decoded_post: List[str] = []
        for seq, in_len in zip(out, input_lens):
            gen_ids = seq[int(in_len) :]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
            decoded_raw.append(text)
            decoded_post.append(text.replace("</s>", "").lstrip(" "))
        return decoded_post, decoded_raw
