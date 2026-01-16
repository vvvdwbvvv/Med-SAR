# Sample usage:
# python scripts/07_eval_baseline.py \
#   --models sft=models/doctor_sft medsar=models/doctor_round_5 \
#   --datasets medmcqa pubmedqa mmlu_clinical \
#   --out_csv results/additional_datasets.csv

from __future__ import annotations
import argparse
import sys
import csv
from pathlib import Path
from prettytable import PrettyTable

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from med_sar.eval.baselines import evaluate_on_dataset, DATASET_CONFIGS


def print_results_table(results: dict, model_name: str):
    """Print results in a pretty table format."""
    print(f"\n{'=' * 80}")
    print(f"Results for {model_name}")
    print(f"{'=' * 80}\n")

    tb = PrettyTable()
    tb.field_names = ["Dataset", "Accuracy", "Correct", "Total"]

    total_correct = 0
    total_samples = 0

    for dataset, metrics in results.items():
        tb.add_row(
            [
                dataset,
                f"{metrics['accuracy']:.2%}",
                metrics["evaluated"],
                metrics["total"],
            ]
        )
        total_correct += metrics["evaluated"]
        total_samples += metrics["total"]

    # Add average row
    avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
    tb.add_row(["Average", f"{avg_acc:.2%}", total_correct, total_samples])

    print(tb)


def print_comparison_table(all_results: dict):
    """Print comparison table across all models."""
    print(f"\n{'=' * 80}")
    print("Model Comparison")
    print(f"{'=' * 80}\n")

    # Get all unique datasets
    all_datasets = set()
    for results in all_results.values():
        all_datasets.update(results.keys())
    all_datasets = sorted(all_datasets)

    tb = PrettyTable()
    tb.field_names = ["Model"] + all_datasets + ["Average"]

    for model_name, results in all_results.items():
        row = [model_name]
        accuracies = []
        for dataset in all_datasets:
            if dataset in results:
                acc = results[dataset]["accuracy"]
                row.append(f"{acc:.2%}")
                accuracies.append(acc)
            else:
                row.append("N/A")

        # Add average
        avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0
        row.append(f"{avg_acc:.2%}")
        tb.add_row(row)

    print(tb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="name=path pairs, e.g., sft=models/doctor_sft",
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_CONFIGS.keys()),
        default=list(DATASET_CONFIGS.keys()),
        help="Datasets to evaluate on",
    )
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--out_csv", type=str, required=True)
    args = ap.parse_args()

    # Parse model name=path pairs
    models = []
    for kv in args.models:
        name, path = kv.split("=", 1)
        models.append((name, path))

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    all_results = {}

    with Path(args.out_csv).open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["model", "dataset", "accuracy", "evaluated", "total"]
        )
        w.writeheader()

        for name, model_path in models:
            print(f"\n{'=' * 80}")
            print(f"Evaluating model: {name}")
            print(f"{'=' * 80}\n")

            model_results = {}

            for dataset_name in args.datasets:
                print(f"\n>>> Evaluating on {dataset_name}...")

                try:
                    metrics = evaluate_on_dataset(
                        model_path,
                        dataset_name,
                        args.batch_size,
                        args.max_new_tokens,
                    )

                    w.writerow(
                        {
                            "model": name,
                            "dataset": dataset_name,
                            "accuracy": f"{metrics['accuracy']:.4f}",
                            "evaluated": metrics["evaluated"],
                            "total": metrics["total"],
                        }
                    )

                    model_results[dataset_name] = metrics

                    print(
                        f"✓ Completed: {metrics['evaluated']}/{metrics['total']} = {metrics['accuracy']:.2%}"
                    )

                except Exception as e:
                    print(f"✗ Error evaluating on {dataset_name}: {e}")
                    import traceback

                    traceback.print_exc()

                    w.writerow(
                        {
                            "model": name,
                            "dataset": dataset_name,
                            "accuracy": "ERROR",
                            "evaluated": 0,
                            "total": 0,
                        }
                    )

                    model_results[dataset_name] = {
                        "accuracy": 0.0,
                        "evaluated": 0,
                        "total": 0,
                    }

                f.flush()

            all_results[name] = model_results
            print_results_table(model_results, name)

    # Print final comparison table
    if len(all_results) > 1:
        print_comparison_table(all_results)

    print(f"\n{'=' * 80}")
    print(f"✓ CSV results written to: {args.out_csv}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()

# Single model
# python scripts/07_eval_baseline.py \
#   --models sft=models/doctor_sft \
#   --datasets medmcqa pubmedqa mmlu_clinical \
#   --out_csv results/additional_datasets.csv

# # Multiple models with comparison
# python scripts/07_eval_baseline.py \
#   --models sft=models/doctor_sft medsar=models/doctor_round_5 base=meta-llama/Llama-3.2-3B-Instruct \
#   --out_csv results/model_comparison.csv
