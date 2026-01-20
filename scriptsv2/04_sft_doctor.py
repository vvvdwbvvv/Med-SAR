# Sample usage:
# python scriptsv2/04_sft_doctor.py \
#   --train data/processed/m23k_v2/m23k_train.jsonl \
#   --dev data/processed/m23k_v2/m23k_val.jsonl \
#   --base meta-llama/Llama-3.2-3B-Instruct \
#   --out models/doctor_sft_v2 \
#   --input_field x_wrapped

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from utils.budget_tracker import BudgetTracker


def load_jsonl(p: Path):
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _hash_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_tokenizer(tok) -> str:
    vocab = tok.get_vocab()
    items = sorted(vocab.items(), key=lambda kv: kv[0])
    h = hashlib.sha1()
    for token, idx in items:
        h.update(token.encode("utf-8", errors="ignore"))
        h.update(str(idx).encode("ascii", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()


def _extract_answer(row: dict, answer_field: str) -> str:
    if answer_field in row:
        val = row.get(answer_field)
    else:
        val = row.get("y")
    if isinstance(val, dict):
        for key in (
            "distilled_answer_string",
            "answer_string",
            "answer_letter",
            "answer_idx",
        ):
            if val.get(key) not in (None, ""):
                return str(val[key])
        return ""
    if val is None:
        return ""
    return str(val)


def format_ex(row: dict, input_field: str, answer_field: str) -> str:
    q = (
        row.get(input_field)
        or row.get("x_wrapped")
        or row.get("x_adv")
        or row.get("x_raw")
        or ""
    )
    a = _extract_answer(row, answer_field)
    return f"Question:\n{q}\n\nAnswer:\n{a}"


class JsonlLogger(TrainerCallback):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        payload = {"step": state.global_step, **logs}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, required=True)
    ap.add_argument("--dev", type=str, required=True)
    ap.add_argument("--adv_train", type=str, default=None)
    ap.add_argument("--input_field", type=str, default="x_wrapped")
    ap.add_argument("--answer_field", type=str, default="y")
    ap.add_argument("--invariance_aug", action="store_true")
    ap.add_argument("--base", type=str, required=True)
    ap.add_argument("--out", type=str, default="runs/sft_doctor/model")
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--budget_report", type=str, default=None)
    ap.add_argument("--max_tokens", type=int, default=None)
    ap.add_argument("--max_examples", type=int, default=None)
    args = ap.parse_args()

    train_path = Path(args.train)
    dev_path = Path(args.dev)
    train_rows = load_jsonl(train_path)
    if args.invariance_aug and args.adv_train:
        adv_path = Path(args.adv_train)
        adv_rows = load_jsonl(adv_path)
        train_rows = train_rows + adv_rows

    dev_rows = load_jsonl(dev_path)

    train_texts = [
        format_ex(r, args.input_field, args.answer_field) for r in train_rows
    ]
    tracker = BudgetTracker(
        max_tokens=args.max_tokens,
        max_examples=args.max_examples,
    )
    tracker.add_texts(train_texts)
    if tracker.exceeds_budget():
        raise SystemExit("Training budget exceeded; adjust max_tokens/max_examples.")
    if args.budget_report:
        tracker.save(args.budget_report)

    ds_train = Dataset.from_dict({"text": train_texts})
    ds_dev = Dataset.from_dict(
        {"text": [format_ex(r, args.input_field, args.answer_field) for r in dev_rows]}
    )

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
        use_cache=False,
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
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        num_train_epochs=1,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=500,
        use_liger_kernel=False,
        fp16=True,
        gradient_checkpointing=True,
        report_to="wandb" if args.wandb_run_name else "none",
        run_name=args.wandb_run_name,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        push_to_hub=False,
    )

    out_dir = Path(args.out)
    run_dir = out_dir.parent if out_dir.name == "model" else out_dir

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds_train,
        eval_dataset=ds_dev,
        data_collator=collator,
        callbacks=[JsonlLogger(run_dir / "train_log.jsonl")],
    )

    trainer.train()
    trainer.save_model(args.out)
    if trainer.tokenizer is not None:
        trainer.tokenizer.save_pretrained(args.out)

    dataset_hash = {
        "train": _hash_file(train_path),
        "dev": _hash_file(dev_path),
    }
    if args.invariance_aug and args.adv_train:
        dataset_hash["adv_train"] = _hash_file(Path(args.adv_train))

    config_payload = {
        "base_model": args.base,
        "tokenizer_hash": _hash_tokenizer(tok),
        "dataset_hash": dataset_hash,
        "steps": trainer.state.global_step,
        "batch": targs.per_device_train_batch_size,
        "gradient_accumulation_steps": targs.gradient_accumulation_steps,
        "learning_rate": targs.learning_rate,
        "compute_budget": tracker.to_dict(),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2), encoding="utf-8"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
