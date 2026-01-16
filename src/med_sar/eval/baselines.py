# python scripts/07_eval_baseline.py \
#   --models sft=/content/doctor_sft_merged \
#   --datasets medmcqa pubmedqa mmlu_clinical \
#   --out_csv results/additional_datasets.csv \
#   --batch_size 8
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from datasets import load_dataset
from tqdm import tqdm

from med_sar.eval.runners import GenerateConfig, TransformersRunner
from med_sar.eval.scorer import get_results


# -----------------------------
# Dataset configs (true schemas)
# -----------------------------
DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "medmcqa": {
        "dataset": "openlifescienceai/medmcqa",
        "split": "validation",  # test doesn't have labels
        "question_path": ["question"],
        "context_path": None,
        "options": {"type": "fields", "fields": ["opa", "opb", "opc", "opd"]},
        "answer": {"type": "index", "path": ["cop"]},  # int 0-3
    },
    "pubmedqa_ols": {
        "dataset": "openlifescienceai/pubmedqa",
        "split": "validation",
        "question_path": ["data", "Question"],
        "context_path": ["data", "Context"],  # list[str]
        "options": {"type": "path", "path": ["data", "Options"]},  # dict {"A": "...", "B": "...", "C": "..."}
        "answer": {"type": "letter", "path": ["data", "Correct Option"]},  # "A"/"B"/"C"
    },
    "mmlu_clinical": {
        "dataset": "openlifescienceai/mmlu_clinical_knowledge",
        "split": "test",
        "question_path": ["data", "Question"],
        "context_path": None,
        "options": {"type": "path", "path": ["data", "Options"]},  # dict {"A": "...", "B": "...", "C": "...", "D": "..."}
        "answer": {"type": "letter", "path": ["data", "Correct Option"]},  # "A"/"B"/"C"/"D"
    },
    "pubmedqa": {
        "dataset": "qiaojin/PubMedQA",
        "name": "pqa_labeled",
        "split": "train",
        "question_path": ["question"],
        "context_path": ["context", "contexts"],  # list[str]
        "options": {"type": "static", "options": {"A": "yes", "B": "no", "C": "maybe"}},
        "answer": {
            "type": "map",
            "path": ["final_decision"],  # "yes"/"no"/"maybe"
            "mapping": {"yes": "A", "no": "B", "maybe": "C"},
        },
    },
}


# -----------------------------
# Helpers: nested access / normalize
# -----------------------------
def get_path(item: Mapping[str, Any], path: Optional[List[str]]) -> Any:
    if not path:
        return None

    cur: Any = item
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _nonempty_strs(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for v in values:
        s = str(v).strip()
        if s:
            out.append(s)
    return out


def normalize_context(ctx: Any) -> str:
    """
    Normalize context into a single string.
    - list[str] -> join
    - dict with 'contexts' or 'Context' -> join
    - str -> itself
    - None -> ""
    """
    if ctx is None:
        return ""

    if isinstance(ctx, str):
        return ctx.strip()

    if isinstance(ctx, list):
        return " ".join(_nonempty_strs(ctx))

    if isinstance(ctx, dict):
        if isinstance(ctx.get("contexts"), list):
            return " ".join(_nonempty_strs(ctx["contexts"]))
        if isinstance(ctx.get("Context"), list):
            return " ".join(_nonempty_strs(ctx["Context"]))

    return str(ctx).strip()


def _letter_ordered_keys(d: Mapping[str, Any]) -> Optional[List[str]]:
    # Returns ordered letter keys if the keys look like single letters.
    keys = [str(k).strip().upper() for k in d.keys()]
    if not keys or not all(len(k) == 1 and k.isalpha() for k in keys):
        return None
    return sorted(set(keys))


def format_options_from_dict(opt_dict: Mapping[str, Any]) -> str:
    ordered = _letter_ordered_keys(opt_dict)
    keys = ordered if ordered is not None else [str(k) for k in opt_dict.keys()]

    lines: List[str] = []
    for k in keys:
        v = opt_dict.get(k)
        if v is None:
            v = opt_dict.get(str(k).lower())
        if v is None:
            v = opt_dict.get(str(k).upper())
        if v is None:
            continue

        text = str(v).strip()
        if not text:
            continue
        lines.append(f"{str(k).strip().upper()}. {text}")

    return "\n".join(lines)


def _format_options_from_list(options: List[Any]) -> str:
    lines: List[str] = []
    for i, v in enumerate(options):
        text = str(v).strip()
        if not text:
            continue
        lines.append(f"{chr(65 + i)}. {text}")
    return "\n".join(lines)


def extract_options(item: Mapping[str, Any], cfg: Mapping[str, Any]) -> str:
    opt_cfg = cfg.get("options")
    if not opt_cfg:
        return ""

    opt_type = opt_cfg.get("type")

    if opt_type == "fields":
        fields: List[str] = opt_cfg["fields"]
        values = [item.get(f) for f in fields]
        return _format_options_from_list(values)

    if opt_type == "path":
        obj = get_path(item, opt_cfg["path"])
        if isinstance(obj, Mapping):
            return format_options_from_dict(obj)
        if isinstance(obj, list):
            return _format_options_from_list(obj)
        return ""

    if opt_type == "static":
        obj = opt_cfg["options"]
        if isinstance(obj, Mapping):
            return format_options_from_dict(obj)
        return ""

    return ""


def extract_gold_label(item: Mapping[str, Any], cfg: Mapping[str, Any]) -> str:
    ans_cfg: Mapping[str, Any] = cfg["answer"]
    ans_type = ans_cfg["type"]
    raw = get_path(item, ans_cfg["path"])

    if ans_type == "index":
        if isinstance(raw, int):
            return chr(65 + raw)
        if isinstance(raw, str) and raw.strip().isdigit():
            return chr(65 + int(raw.strip()))
        raise ValueError(f"Expected index int for answer, got: {raw!r}")

    if ans_type == "letter":
        if raw is None:
            raise ValueError("Missing letter answer")
        s = str(raw).strip().upper()
        if len(s) == 1 and s.isalpha():
            return s
        raise ValueError(f"Expected letter answer, got: {raw!r}")

    if ans_type == "map":
        if raw is None:
            raise ValueError("Missing mapped answer")
        key = str(raw).strip().lower()
        mapping: Mapping[str, str] = ans_cfg["mapping"]
        if key not in mapping:
            raise ValueError(f"Unknown mapping key {key!r} for answer")
        return mapping[key]

    raise ValueError(f"Unknown answer type: {ans_type}")


def format_prompt(item: Mapping[str, Any], cfg: Mapping[str, Any]) -> str:
    parts: List[str] = []

    ctx = get_path(item, cfg.get("context_path"))
    ctx_str = normalize_context(ctx)
    if ctx_str:
        parts.append(f"Context: {ctx_str}\n")

    q = get_path(item, cfg["question_path"])
    if q is None:
        raise ValueError(f"Missing question at path {cfg['question_path']}")
    parts.append(f"Question: {str(q).strip()}\n")

    opt_str = extract_options(item, cfg)
    if opt_str:
        parts.append(opt_str + "\n")

    parts.append("Please answer with the correct option letter in the format: <answer> A </answer>")
    return "\n".join(parts)


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_on_dataset(
    model_path: str,
    dataset_name: str,
    batch_size: int = 32,
    max_new_tokens: int = 32,
) -> Dict[str, float]:
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    cfg = DATASET_CONFIGS[dataset_name]

    print(f"Loading {cfg['dataset']}...")
    if "name" in cfg:
        ds = load_dataset(cfg["dataset"], cfg["name"], split=cfg["split"])
    else:
        ds = load_dataset(cfg["dataset"], split=cfg["split"])

    print(f"Dataset size: {len(ds)}")

    runner = TransformersRunner(model_path=model_path)

    results: List[Dict[str, Any]] = []
    total_batches = (len(ds) + batch_size - 1) // batch_size

    for b in tqdm(range(total_batches), desc=f"Evaluating {dataset_name}"):
        start = b * batch_size
        end = min((b + 1) * batch_size, len(ds))
        batch = ds.select(range(start, end))

        prompts = [format_prompt(row, cfg) for row in batch]

        preds, _ = runner.generate(
            prompts,
            cfg=GenerateConfig(
                max_new_tokens=max_new_tokens,
                temperature=0.1,
            ),
        )

        for i, pred in enumerate(preds):
            row = batch[i]
            gold = extract_gold_label(row, cfg)
            q = get_path(row, cfg["question_path"])

            results.append(
                {
                    "question": str(q).strip() if q is not None else "",
                    "answer": gold,  # gold letter
                    "output": pred,  # model raw output
                }
            )

    out_dir = Path(f"results/{Path(model_path).name}/{dataset_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    result_file = out_dir / "result.json"
    result_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved results to {result_file}")

    metrics = get_results(result_file)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    ap.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_CONFIGS.keys()),
        default=list(DATASET_CONFIGS.keys()),
        help="Datasets to evaluate on",
    )
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--output_csv", type=str, default="results/additional_datasets.csv")
    args = ap.parse_args()

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_name = Path(args.model).name

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "dataset", "accuracy", "evaluated", "total"])
        writer.writeheader()

        for dataset_name in args.datasets:
            print("\n" + "=" * 60)
            print(f"Evaluating on {dataset_name}")
            print("=" * 60 + "\n")

            metrics = evaluate_on_dataset(
                args.model,
                dataset_name,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
            )

            writer.writerow(
                {
                    "model": model_name,
                    "dataset": dataset_name,
                    "accuracy": f"{metrics['accuracy']:.4f}",
                    "evaluated": metrics["evaluated"],
                    "total": metrics["total"],
                }
            )
            f.flush()

    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
