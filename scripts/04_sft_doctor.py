# Sample usage:
# python scripts/04_sft_doctor.py \
#   --train data/processed/m23k/m23k_train.jsonl \
#   --dev data/processed/m23k/m23k_val.jsonl \
#   --base meta-llama/Llama-3.2-3B-Instruct \
#   --out models/doctor_sft \
#   --max_len 1024

from __future__ import annotations
import argparse
import torch
import json
from pathlib import Path
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)


def load_jsonl(p: Path):
    rows = []
    for line in p.open():
        rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, required=True)
    ap.add_argument("--dev", type=str, required=True)
    ap.add_argument(
        "--base", type=str, required=True
    )  # e.g. meta-llama/Llama-3.1-8B-Instruct (local)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--max_len", type=int, default=1024)
    args = ap.parse_args()

    tr = load_jsonl(Path(args.train))
    dv = load_jsonl(Path(args.dev))

    def format_ex(r):
        q = r.get("question") or r.get("prompt") or ""
        a = r.get("answer_string") or r.get("answer") or ""
        return f"Question:\n{q}\n\nAnswer:\n{a}"  # Doubly escaped newlines for correct parsing

    ds_train = Dataset.from_dict({"text": [format_ex(r) for r in tr]})
    ds_dev = Dataset.from_dict({"text": [format_ex(r) for r in dv]})

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tok_fn(ex):
        return tok(ex["text"], truncation=True, max_length=args.max_len)

    ds_train = ds_train.map(tok_fn, batched=True, remove_columns=["text"])
    ds_dev = ds_dev.map(tok_fn, batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        attn_implementation="sdpa",
        torch_dtype=torch.float16,
        use_cache=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "v_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Wrap the model with LoRA adapters
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        num_train_epochs=30,
        eval_strategy="steps",
        max_steps=3000,
        eval_steps=30,
        save_steps=30,
        logging_steps=30,
        use_liger_kernel=True,
        fp16=True,
        gradient_checkpointing=True,
        report_to="none",
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds_train,
        eval_dataset=ds_dev,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"doctor saved to {args.out}")


if __name__ == "__main__":
    main()
