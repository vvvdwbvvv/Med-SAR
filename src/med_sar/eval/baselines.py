from __future__ import annotations

# Sample usage:
# PYTHONPATH=src python -m med_sar.eval.baselines \
#   --backend openai \
#   --eval_dataset zou-lab/BioMed-R1-Eval \
#   --eval_benchmark medqa \
#   --port 8001 \
#   --batch_size 32 \
#   --max_new_tokens 4096 \
#   --temperature 0.2 \
#   --use_chat_template \
#   --strict_prompt \
#   --reasoning
#
# Local Transformers (no server):
# PYTHONPATH=src python -m med_sar.eval.baselines \
#   --backend transformers \
#   --model models/doctor_sft \
#   --eval_dataset zou-lab/BioMed-R1-Eval \
#   --eval_benchmark medqa

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List
from datasets import load_dataset
from tqdm import tqdm

from med_sar.eval.runners import GenerateConfig, OpenAICompatRunner, TransformersRunner
from med_sar.eval.scorer import get_results


def postprocess_output(pred: str) -> str:
    pred = pred.replace("</s>", "")
    return pred.lstrip(" ")


def load_file(dataset_name: str, eval_benchmark: str) -> List[Dict[str, Any]]:
    dataset = load_dataset(dataset_name, eval_benchmark)["test"]
    input_data: List[Dict[str, Any]] = []
    for item in dataset:
        if "options" in item and isinstance(item["options"], str):
            try:
                item["options"] = json.loads(item["options"])
            except json.JSONDecodeError:
                pass
        input_data.append(item)
    return input_data


def format_options(options: Any) -> str:
    if not options:
        return ""
    if isinstance(options, dict):
        return "\n".join([f"{op}. {ans}" for op, ans in options.items()])
    if isinstance(options, list):
        lines: List[str] = []
        for idx, opt in enumerate(options):
            if isinstance(opt, dict):
                label = (
                    opt.get("label") or opt.get("key") or opt.get("option") or str(idx)
                )
                value = opt.get("text") or opt.get("value") or opt.get("answer")
                if value:
                    lines.append(f"{label}. {value}")
                else:
                    lines.append(str(label))
            else:
                lines.append(str(opt))
        return "\n".join(lines)
    return str(options)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dataset", type=str, required=True)
    parser.add_argument("--eval_benchmark", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=6000)
    parser.add_argument("--max_tokens", type=int, default=-1)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--strict_prompt", action="store_true")
    parser.add_argument("--task", type=str, default="api")
    parser.add_argument(
        "--backend",
        type=str,
        default="openai",
        choices=["openai", "transformers"],
        help="Generation backend: openai=OpenAI-compatible server, transformers=local HF generate().",
    )
    parser.add_argument("--port", type=int, default=30000, help="Only used for --backend openai.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="OpenAI model id (openai backend) or path/model id (transformers backend).",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Only used for --backend transformers when --model is a LoRA adapter; defaults from adapter_config.json.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Only used for --backend transformers (e.g., cuda, mps, cpu).",
    )
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--reasoning", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    if args.backend == "openai":
        print(f"Using OpenAI-compatible API server at port {args.port}")
        runner = OpenAICompatRunner(port=args.port, model=args.model)
        model_name = runner.model.split("/")[-1]
        print(f"Using model {runner.model}")
    else:
        if args.model is None:
            raise SystemExit("--model is required for --backend transformers (path to checkpoint).")
        runner = TransformersRunner(
            model_path=args.model, base_model=args.base_model, device=args.device
        )
        model_name = Path(args.model).name if os.path.exists(args.model) else args.model.split("/")[-1]
        print(f"Using local Transformers model from {args.model} on device={runner.device}")

    input_data = load_file(args.eval_dataset, args.eval_benchmark)

    if args.reasoning:
        query_prompt = (
            "You are a biomedical reasoning model. You must think step-by-step and reason carefully about the following question before answering. "
            "You must select the correct option from the given options (A, B, C, D, etc.). Your response must conclude with the correct option in the format: <answer> A </answer>.\n"
            "{question}\n"
            "{option_str}\n"
        )
    else:
        if args.strict_prompt:
            query_prompt = (
                "You are a biomedical expert. Given a question and options, you must select the correct option from the given options (A, B, C, D, etc.). "
                "Your response must be direct and conclude with the correct option in the format: <answer> A </answer>. Please do not return any other text.\n"
                "{question}\n"
                "{option_str}\n"
            )
        else:
            query_prompt = (
                "Please answer the following question and return the answer in the format: <answer>...</answer>.\n"
                "{question} What is the most likely diagnosis?\n"
                "{option_str}\n"
            )

    final_results: List[Dict[str, Any]] = []
    total_batches = len(input_data) // args.batch_size + int(
        len(input_data) % args.batch_size != 0
    )
    for idx in tqdm(range(total_batches)):
        batch = input_data[idx * args.batch_size : (idx + 1) * args.batch_size]
        if not batch:
            break

        for item in batch:
            if "question" not in item:
                if "prompt" in item:
                    item["question"] = item["prompt"]
                elif "query" in item:
                    item["question"] = item["query"]
            item["option_str"] = format_options(item.get("options"))
            item["input_str"] = query_prompt.format_map(item)

        processed_batch = [item["input_str"] for item in batch]
        if idx == 0 and processed_batch:
            print("Example:")
            print(processed_batch[0])
        preds, _ = runner.generate(
            processed_batch,
            cfg=GenerateConfig(
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                max_input_tokens=args.max_tokens,
                use_chat_template=args.use_chat_template,
            ),
        )

        for j, item in enumerate(batch):
            pred = preds[j]
            if not pred:
                continue
            item["output"] = pred
            final_results.append(item)

    if args.reasoning:
        task_folder = f"./results/{model_name}/{args.eval_benchmark}/reasoning"
    else:
        task_folder = f"./results/{model_name}/{args.eval_benchmark}/non-reasoning"
    os.makedirs(task_folder, exist_ok=True)

    result_file = os.path.join(task_folder, "result.json")
    with open(result_file, "w") as fw:
        json.dump(final_results, fw, ensure_ascii=False, indent=2)

    get_results(result_file)


if __name__ == "__main__":
    main()
