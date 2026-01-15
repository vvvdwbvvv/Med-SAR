from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any
from datasets import load_dataset
from tqdm import tqdm

from med_sar.eval.runners import GenerateConfig, TransformersRunner
from med_sar.eval.scorer import get_results

DATASET_CONFIGS = {
    "medmcqa": {
        "dataset": "openlifescienceai/medmcqa",
        "split": "validation",  # use validation since test doesn't have labels
        "question_field": "question",
        "options_field": "options",
        "answer_field": "cop",  # correct option index (0-3)
    },
    "pubmedqa": {
        "dataset": "qiaojin/PubMedQA",
        "name": "pqa_labeled",  # need to specify subset
        "split": "test",
        "question_field": "question",
        "context_field": "context",
        "answer_field": "final_decision",
    },
    "mmlu_clinical": {
        "dataset": "openlifescienceai/mmlu_clinical_knowledge",
        "split": "test",
        "question_field": "question",
        "options_field": "choices",
        "answer_field": "answer",  # integer index
    },
    "pubmedqa_ols": {
        "dataset": "openlifescienceai/pubmedqa",
        "split": "test",
        "question_field": "question",
        "context_field": "context",
        "answer_field": "final_decision",
    },
}


def format_options(options: Any) -> str:
    """Format options into lettered list."""
    if options is None:
        return ""
    if isinstance(options, dict):
        options = list(options.values())
    if not isinstance(options, list):
        return ""

    lines = []
    for i, opt in enumerate(options):
        lines.append(f"{chr(65 + i)}. {opt}")
    return "\n".join(lines)


def format_prompt(item: Dict[str, Any], config: Dict[str, str]) -> str:
    """Format question with options if available."""
    prompt_parts = []

    # Add context if available
    if "context_field" in config and config["context_field"] in item:
        context = item[config["context_field"]]
        if isinstance(context, dict):
            # Handle PubMedQA context format
            context = " ".join(context.get("contexts", []))
        if context:
            prompt_parts.append(f"Context: {context}\n")

    # Add question
    question = item[config["question_field"]]
    prompt_parts.append(f"Question: {question}\n")

    # Add options if available
    if "options_field" in config and config["options_field"] in item:
        options = item[config["options_field"]]
        options_str = format_options(options)
        if options_str:
            prompt_parts.append(f"{options_str}\n")

    # Add instruction
    prompt_parts.append(
        "Please answer with the correct option letter in the format: <answer> A </answer>"
    )

    return "\n".join(prompt_parts)


def get_gold_label(item: Dict[str, Any], config: Dict[str, str]) -> str:
    """Extract gold label from item."""
    answer = item[config["answer_field"]]

    # Handle integer index (convert to letter)
    if isinstance(answer, int):
        return chr(65 + answer)  # 0->A, 1->B, 2->C, 3->D

    # Handle string answers
    return str(answer)


def evaluate_on_dataset(
    model_path: str, dataset_name: str, batch_size: int = 32, max_new_tokens: int = 512
) -> Dict[str, float]:
    """Evaluate model on a single dataset."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    config = DATASET_CONFIGS[dataset_name]

    # Load dataset
    print(f"Loading {config['dataset']}...")
    if "name" in config:
        dataset = load_dataset(config["dataset"], config["name"], split=config["split"])
    else:
        dataset = load_dataset(config["dataset"], split=config["split"])

    print(f"Dataset size: {len(dataset)}")

    # Initialize runner
    runner = TransformersRunner(model_path=model_path)

    results = []
    total_batches = len(dataset) // batch_size + int(len(dataset) % batch_size != 0)

    for idx in tqdm(range(total_batches), desc=f"Evaluating {dataset_name}"):
        batch_start = idx * batch_size
        batch_end = min((idx + 1) * batch_size, len(dataset))

        # Get batch items (handle different dataset types)
        if hasattr(dataset, "select"):
            batch = dataset.select(range(batch_start, batch_end))
        else:
            batch = [dataset[i] for i in range(batch_start, batch_end)]

        # Format prompts
        prompts = [format_prompt(item, config) for item in batch]

        # Generate predictions
        preds, _ = runner.generate(
            prompts,
            cfg=GenerateConfig(
                max_new_tokens=max_new_tokens,
                temperature=0.1,  # lower temperature for deterministic answers
                do_sample=False,
            ),
        )

        # Store results
        for i, pred in enumerate(preds):
            item = batch[i]
            gold_label = get_gold_label(item, config)

            results.append(
                {
                    "question": item[config["question_field"]],
                    "answer": gold_label,
                    "output": pred,
                }
            )

    # Save results
    output_dir = Path(f"results/{Path(model_path).name}/{dataset_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    result_file = output_dir / "result.json"
    with result_file.open("w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved results to {result_file}")

    # Get metrics
    metrics = get_results(result_file)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_CONFIGS.keys()),
        default=list(DATASET_CONFIGS.keys()),
        help="Datasets to evaluate on",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--output_csv", type=str, default="results/additional_datasets.csv"
    )
    args = parser.parse_args()

    import csv

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "dataset", "accuracy", "evaluated", "total"]
        )
        writer.writeheader()

        model_name = Path(args.model).name

        for dataset_name in args.datasets:
            print(f"\n{'=' * 50}")
            print(f"Evaluating on {dataset_name}")
            print(f"{'=' * 50}\n")

            metrics = evaluate_on_dataset(
                args.model, dataset_name, args.batch_size, args.max_new_tokens
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
