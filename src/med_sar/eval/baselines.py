from __future__ import annotations

# Sample usage:
# PYTHONPATH=src python -m med_sar.eval.baselines \
#   --eval_dataset zou-lab/BioMed-R1-Eval \
#   --eval_benchmark medqa \
#   --port 8001 \
#   --batch_size 32 \
#   --max_new_tokens 4096 \
#   --temperature 0.2 \
#   --use_chat_template \
#   --strict_prompt \
#   --reasoning

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import openai
from datasets import load_dataset
from jinja2 import Template
from tqdm import tqdm
from transformers import AutoTokenizer

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


def trim_prompt(prompt: str, tokenizer: AutoTokenizer, max_tokens: int) -> str:
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(input_ids) <= max_tokens:
        return prompt
    input_ids = input_ids[:max_tokens]
    return tokenizer.decode(input_ids)


def call_model(
    client: Any,
    prompts: Sequence[str],
    *,
    model: str,
    max_new_tokens: int,
    temperature: float,
    tokenizer: Optional[AutoTokenizer],
    template: Optional[Template],
    max_input_tokens: int,
    print_example: bool = False,
) -> Tuple[List[str], List[str]]:
    if print_example and prompts:
        print("Example:")
        print(prompts[0])
    if template is not None:
        prompts = [
            template.render(
                messages=[{"role": "user", "content": prompt}],
                bos_token=tokenizer.bos_token if tokenizer else "",
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
    if max_input_tokens > 0 and tokenizer is not None:
        prompts = [
            trim_prompt(prompt, tokenizer, max_input_tokens) for prompt in prompts
        ]

    response = client.completions.create(
        model=model,
        prompt=prompts,
        temperature=temperature,
        top_p=0.9,
        max_tokens=max_new_tokens,
    )
    raw_preds = [choice.text for choice in response.choices]
    postprocessed_preds = [postprocess_output(pred) for pred in raw_preds]
    return postprocessed_preds, raw_preds


def get_client(port: int):
    base_url = f"http://127.0.0.1:{port}/v1"
    if hasattr(openai, "Client"):
        return openai.Client(base_url=base_url, api_key="EMPTY")
    return openai.OpenAI(base_url=base_url, api_key="EMPTY")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dataset", type=str, required=True)
    parser.add_argument("--eval_benchmark", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=6000)
    parser.add_argument("--max_tokens", type=int, default=-1)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--strict_prompt", action="store_true")
    parser.add_argument("--task", type=str, default="api")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--reasoning", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    print(f"Using local API server at port {args.port}")
    client = get_client(args.port)
    model = args.model
    if model is None:
        models = client.models.list().data
        if not models:
            raise RuntimeError("No models available from the local API; pass --model.")
        model = models[0].id
    print(f"Using model {model}")

    tokenizer = None
    template = None
    if args.use_chat_template or args.max_tokens > 0:
        tokenizer = AutoTokenizer.from_pretrained(
            model, trust_remote_code=True, padding_side="left"
        )
    if args.use_chat_template:
        if not tokenizer or not tokenizer.chat_template:
            raise RuntimeError(
                "Tokenizer chat_template is required with --use_chat_template."
            )
        template = Template(tokenizer.chat_template)

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
        preds, _ = call_model(
            client,
            processed_batch,
            model=model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            tokenizer=tokenizer,
            template=template,
            max_input_tokens=args.max_tokens,
            print_example=(idx == 0),
        )

        for j, item in enumerate(batch):
            pred = preds[j]
            if not pred:
                continue
            item["output"] = pred
            final_results.append(item)

    model_name = model.split("/")[-1]
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
