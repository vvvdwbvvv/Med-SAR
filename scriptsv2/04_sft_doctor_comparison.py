# scriptsv2/04_sft_doctor.py
# Sample usage:
# python scriptsv2/04_sft_doctor.py \
#   --train data/processed/m23k_v2/m23k_train.jsonl \
#   --dev data/processed/m23k_v2/m23k_val.jsonl \
#   --base meta-llama/Llama-3.2-3B-Instruct \
#   --out models/doctor_sft_v2 \
#   --input_field x_wrapped \
#   --loss_reduction erm

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import sys
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

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
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


def _norm_note_type(v) -> str:
    if v is None:
        return "unknown"
    v = str(v).strip().lower()
    return v.replace(" ", "_")


def _get_group_id(row: dict, group_mode: str, group_field: Optional[str]) -> str:
    if group_mode == "note_type":
        nt = row.get("note_type") or row.get("note_type_clean")
        return f"type_{_norm_note_type(nt)}"
    if group_mode == "time_only":
        tb = row.get("time_bucket")
        return f"t{tb if tb is not None else 'unknown'}"
    if group_mode == "time_type":
        tb = row.get("time_bucket")
        nt = row.get("note_type") or row.get("note_type_clean")
        return f"t{tb if tb is not None else 'unknown'}|type_{_norm_note_type(nt)}"
    if group_mode == "custom_field":
        if not group_field:
            return "all"
        v = row.get(group_field)
        return str(v) if v not in (None, "") else "unknown"
    return "all"


def _apply_min_group_count(group_ids: List[str], min_count: int) -> List[str]:
    if min_count <= 0:
        return group_ids
    c = Counter(group_ids)
    return [g if c[g] >= min_count else "other" for g in group_ids]


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


class CollatorWithGroup:
    """
    HF default collators tend to drop non-tensor fields.
    We preserve group_id as a Python list[str] so Trainer.compute_loss can use it.
    """
    def __init__(self, base_collator):
        self.base = base_collator

    def __call__(self, features):
        group_ids = [f.get("group_id", "all") for f in features]
        batch = self.base(features)
        batch["group_id"] = group_ids
        return batch


class RobustReductionTrainer(Trainer):
    """
    loss_reduction:
      - erm: mean over examples
      - cvar: mean of top rho fraction example losses
      - group_dro: Group DRO over group_id (time_bucket x note_type, etc.)
    """
    def __init__(
        self,
        *args,
        loss_reduction: str = "erm",
        cvar_rho: float = 0.2,
        gdro_eta: float = 0.1,
        gdro_min_prob: float = 1e-3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_reduction = loss_reduction
        self.cvar_rho = float(cvar_rho)
        self.gdro_eta = float(gdro_eta)
        self.gdro_min_prob = float(gdro_min_prob)

        # Group DRO state: unnormalized weights q_g
        self._q: Dict[str, float] = defaultdict(lambda: 1.0)

    @staticmethod
    def _per_example_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute per-example mean NLL for causal LM with ignore_index=-100.
        logits: (B, T, V)
        labels: (B, T)
        """
        # shift for causal LM
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        B, Tm1, V = shift_logits.shape
        loss_flat = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(B, Tm1)

        mask = (shift_labels != -100).float()
        denom = mask.sum(dim=1).clamp_min(1.0)
        per_ex = (loss_flat * mask).sum(dim=1) / denom
        return per_ex  # (B,)

    def _reduce_erm(self, per_ex: torch.Tensor) -> torch.Tensor:
        return per_ex.mean()

    def _reduce_cvar(self, per_ex: torch.Tensor) -> torch.Tensor:
        rho = max(1e-6, min(1.0, self.cvar_rho))
        k = max(1, int(torch.ceil(torch.tensor(rho * per_ex.numel(), device=per_ex.device)).item()))
        topk = torch.topk(per_ex, k=k, largest=True).values
        return topk.mean()

    def _reduce_group_dro(self, per_ex: torch.Tensor, group_ids: List[str]) -> torch.Tensor:
        # group mean losses
        group_to_losses: Dict[str, List[torch.Tensor]] = defaultdict(list)
        for li, gi in zip(per_ex, group_ids):
            group_to_losses[gi].append(li)

        group_mean = {}
        for g, losses in group_to_losses.items():
            group_mean[g] = torch.stack(losses).mean()

        # update q_g <- q_g * exp(eta * loss_g)
        with torch.no_grad():
            for g, lg in group_mean.items():
                self._q[g] = float(self._q[g]) * float(torch.exp(self.gdro_eta * lg.detach()).item())

            # normalize with floor
            keys = list(self._q.keys())
            q_vals = torch.tensor([self._q[k] for k in keys], dtype=torch.float32, device=per_ex.device)
            q_vals = q_vals / q_vals.sum().clamp_min(1e-12)

            # apply min prob floor, renormalize
            q_vals = torch.clamp(q_vals, min=self.gdro_min_prob)
            q_vals = q_vals / q_vals.sum().clamp_min(1e-12)

            for k, v in zip(keys, q_vals.tolist()):
                self._q[k] = float(v)

        # weighted sum of group losses
        loss = torch.zeros((), device=per_ex.device)
        for g, lg in group_mean.items():
            w = float(self._q.get(g, 0.0))
            loss = loss + (w * lg)
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        group_ids = inputs.pop("group_id", None)  # list[str] from collator
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get("labels")

        # per-example losses
        per_ex = self._per_example_nll(logits, labels)

        if self.loss_reduction == "erm":
            loss = self._reduce_erm(per_ex)
        elif self.loss_reduction == "cvar":
            loss = self._reduce_cvar(per_ex)
        elif self.loss_reduction == "group_dro":
            if group_ids is None:
                raise ValueError("group_dro requires group_id in batch. Ensure dataset has group_id + CollatorWithGroup.")
            loss = self._reduce_group_dro(per_ex, group_ids)
        else:
            raise ValueError(f"Unknown loss_reduction: {self.loss_reduction}")

        # helpful logs
        self.log({
            "loss_reduction": 0.0,  # placeholder scalar; avoids HF complaining about non-numeric
            "loss_per_ex_mean": float(per_ex.mean().detach().cpu().item()),
            "loss_per_ex_p90": float(torch.quantile(per_ex.detach(), 0.9).cpu().item()),
        })
        if self.loss_reduction == "group_dro":
            # log top few group weights (as separate scalar keys)
            top = sorted(self._q.items(), key=lambda kv: kv[1], reverse=True)[:5]
            for i, (g, w) in enumerate(top):
                # sanitize key
                key = f"gdro_q_top{i}"
                self.log({key: float(w)})
        return (loss, outputs) if return_outputs else loss


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

    # NEW: loss reduction
    ap.add_argument(
        "--loss_reduction",
        type=str,
        default="erm",
        choices=["erm", "cvar", "group_dro"],
    )
    ap.add_argument("--cvar_rho", type=float, default=0.2)

    # NEW: group config (N10)
    ap.add_argument(
        "--group_mode",
        type=str,
        default="time_type",
        choices=["time_type", "note_type", "time_only", "custom_field"],
    )
    ap.add_argument("--group_field", type=str, default=None)
    ap.add_argument("--min_group_count", type=int, default=0)

    # NEW: Group DRO params
    ap.add_argument("--gdro_eta", type=float, default=0.1)
    ap.add_argument("--gdro_min_prob", type=float, default=1e-3)

    args = ap.parse_args()

    train_path = Path(args.train)
    dev_path = Path(args.dev)
    train_rows = load_jsonl(train_path)
    if args.invariance_aug and args.adv_train:
        adv_path = Path(args.adv_train)
        adv_rows = load_jsonl(adv_path)
        train_rows = train_rows + adv_rows

    dev_rows = load_jsonl(dev_path)

    train_texts = [format_ex(r, args.input_field, args.answer_field) for r in train_rows]
    dev_texts = [format_ex(r, args.input_field, args.answer_field) for r in dev_rows]

    # Build group_id from row metadata (expects time_bucket + note_type in your wrapped dataset)
    train_groups = [_get_group_id(r, args.group_mode, args.group_field) for r in train_rows]
    dev_groups = [_get_group_id(r, args.group_mode, args.group_field) for r in dev_rows]

    train_groups = _apply_min_group_count(train_groups, args.min_group_count)
    dev_groups = _apply_min_group_count(dev_groups, args.min_group_count)

    # quick sanity print
    print("Top train groups:", Counter(train_groups).most_common(10))

    tracker = BudgetTracker(
        max_tokens=args.max_tokens,
        max_examples=args.max_examples,
    )
    tracker.add_texts(train_texts)
    if tracker.exceeds_budget():
        raise SystemExit("Training budget exceeded; adjust max_tokens/max_examples.")
    if args.budget_report:
        tracker.save(args.budget_report)

    ds_train = Dataset.from_dict({"text": train_texts, "group_id": train_groups})
    ds_dev = Dataset.from_dict({"text": dev_texts, "group_id": dev_groups})

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tok_fn(batch):
        out = tok(batch["text"], truncation=True, max_length=args.max_len)
        out["group_id"] = batch["group_id"]
        return out

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

    base_collator = DataCollatorForLanguageModeling(tok, mlm=False)
    collator = CollatorWithGroup(base_collator)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        num_train_epochs=1,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=500,
        use_liger_kernel=True,
        fp16=True,
        gradient_checkpointing=True,
        report_to="wandb" if args.wandb_run_name else "none",
        run_name=args.wandb_run_name,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        push_to_hub=False,
    )

    out_dir = Path(args.out)
    run_dir = out_dir.parent if out_dir.name == "model" else out_dir

    trainer = RobustReductionTrainer(
        model=model,
        args=targs,
        train_dataset=ds_train,
        eval_dataset=ds_dev,
        data_collator=collator,
        callbacks=[JsonlLogger(run_dir / "train_log.jsonl")],
        loss_reduction=args.loss_reduction,
        cvar_rho=args.cvar_rho,
        gdro_eta=args.gdro_eta,
        gdro_min_prob=args.gdro_min_prob,
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
        "loss_reduction": args.loss_reduction,
        "cvar_rho": args.cvar_rho,
        "group_mode": args.group_mode,
        "group_field": args.group_field,
        "min_group_count": args.min_group_count,
        "gdro_eta": args.gdro_eta,
        "gdro_min_prob": args.gdro_min_prob,
        "train_group_top10": Counter(train_groups).most_common(10),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2), encoding="utf-8"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())