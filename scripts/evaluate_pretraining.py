"""Evaluation script for comparing pre-trained vs from-scratch models.

This script provides a comprehensive evaluation framework for assessing
the effectiveness of the pre-training approach across multiple datasets.
"""

import argparse
import json

from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch

from data.downloaders.air_quality import AirQualityDataset
from data.downloaders.etth1 import ETTh1Dataset
from data.downloaders.financial_market import FinancialMarketDataset
from data.downloaders.human_activity import HumanActivityDataset
from torch.utils.data import DataLoader

from treac.models.multi_dataset_model import MultiDatasetModel


def get_dataset(dataset_name: str, split: str, sequence_length: int):
    """Get dataset by name and split."""
    datasets = {
        "etth1": lambda: ETTh1Dataset(
            train=(split == "train"), sequence_length=sequence_length
        ),
        "human_activity": lambda: HumanActivityDataset(
            split=split, seq_len=sequence_length
        ),
        "air_quality": lambda: AirQualityDataset(split=split, seq_len=sequence_length),
        "financial_market": lambda: FinancialMarketDataset(
            split=split, seq_len=sequence_length
        ),
    }

    if dataset_name not in datasets:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available: {list(datasets.keys())}"
        )

    return datasets[dataset_name]()


def evaluate_model(model, test_loader, device="cpu"):
    """Evaluate model on test set."""
    model.eval()
    model = model.to(device)

    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            x_num = batch["x_num"].to(device)
            x_cat = batch.get("x_cat", None)
            if x_cat is not None:
                x_cat = x_cat.to(device)
            y = batch["y"].to(device)

            # Forward pass
            outputs = model(x_num, x_cat)

            # Compute loss
            if y.ndim == 2:
                y = torch.argmax(y, dim=1)
            y = y.long()

            loss = model.loss_fn(outputs, y)
            total_loss += loss.item()

            # Compute accuracy
            predictions = torch.argmax(outputs, dim=1)
            correct_predictions += (predictions == y).sum().item()
            total_predictions += y.size(0)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_predictions

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "predictions": all_predictions,
        "targets": all_targets,
    }


def load_model_from_checkpoint(
    checkpoint_path: str, dataset, model_type: str = "pretrained"
):
    """Load model from checkpoint."""
    if model_type == "pretrained":
        model = MultiDatasetModel.from_pretrained(
            pretrained_path=checkpoint_path,
            num_classes=dataset.num_classes,
            freeze_encoder=False,
            mode="pretrain",
        )
    else:
        # Load from scratch model
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        hparams = checkpoint.get("hyper_parameters", {})

        model = MultiDatasetModel(
            max_numeric_features=dataset.numeric_features,
            max_categorical_features=dataset.categorical_features,
            num_classes=dataset.num_classes,
            mode="variable_features",
            **{
                k: v
                for k, v in hparams.items()
                if k
                not in [
                    "max_numeric_features",
                    "max_categorical_features",
                    "num_classes",
                    "mode",
                ]
            },
        )

        model.load_state_dict(checkpoint["state_dict"], strict=False)

    # Set dataset schema
    if hasattr(model, "set_dataset_schema"):
        model.set_dataset_schema(
            numeric_features=dataset.numeric_features,
            categorical_features=dataset.categorical_features,
            column_names=dataset.get_column_names()
            if hasattr(dataset, "get_column_names")
            else None,
        )

    return model


def run_evaluation_suite(args):
    """Run comprehensive evaluation across datasets."""
    print("=" * 70)
    print("PRE-TRAINING EVALUATION SUITE")
    print("=" * 70)

    datasets = ["etth1", "human_activity", "air_quality", "financial_market"]
    results = []

    for dataset_name in datasets:
        print(f"\nEvaluating on {dataset_name}...")
        print("-" * 50)

        # Create test dataset
        try:
            test_dataset = get_dataset(dataset_name, "test", args.sequence_length)
        except Exception:
            # If no test split, use validation
            test_dataset = get_dataset(dataset_name, "val", args.sequence_length)

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        dataset_results = {"dataset": dataset_name}

        # Evaluate pre-trained model (if available)
        pretrained_path = (
            Path(args.checkpoint_dir)
            / f"finetune_{dataset_name}"
            / "pretrained_freeze_False"
            / "best*.ckpt"
        )
        pretrained_checkpoints = (
            list(pretrained_path.parent.glob("best*.ckpt"))
            if pretrained_path.parent.exists()
            else []
        )

        if pretrained_checkpoints:
            print("Evaluating pre-trained model...")
            try:
                model = load_model_from_checkpoint(
                    str(pretrained_checkpoints[0]), test_dataset, "pretrained"
                )
                pretrained_results = evaluate_model(model, test_loader, args.device)
                dataset_results["pretrained"] = pretrained_results
                print(
                    f"  Pre-trained - Loss: {pretrained_results['loss']:.4f}, "
                    f"Accuracy: {pretrained_results['accuracy']:.4f}"
                )
            except Exception as e:
                print(f"  Error evaluating pre-trained model: {e}")
                dataset_results["pretrained"] = None
        else:
            print("  No pre-trained model checkpoint found")
            dataset_results["pretrained"] = None

        # Evaluate from-scratch model
        scratch_path = (
            Path(args.checkpoint_dir)
            / f"finetune_{dataset_name}"
            / "from_scratch"
            / "best*.ckpt"
        )
        scratch_checkpoints = (
            list(scratch_path.parent.glob("best*.ckpt"))
            if scratch_path.parent.exists()
            else []
        )

        if scratch_checkpoints:
            print("Evaluating from-scratch model...")
            try:
                model = load_model_from_checkpoint(
                    str(scratch_checkpoints[0]), test_dataset, "scratch"
                )
                scratch_results = evaluate_model(model, test_loader, args.device)
                dataset_results["from_scratch"] = scratch_results
                print(
                    f"  From-scratch - Loss: {scratch_results['loss']:.4f}, "
                    f"Accuracy: {scratch_results['accuracy']:.4f}"
                )
            except Exception as e:
                print(f"  Error evaluating from-scratch model: {e}")
                dataset_results["from_scratch"] = None
        else:
            print("  No from-scratch model checkpoint found")
            dataset_results["from_scratch"] = None

        results.append(dataset_results)

    return results


def generate_report(results: list[dict], output_dir: str):
    """Generate evaluation report."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create summary table
    summary_data = []

    for result in results:
        dataset = result["dataset"]

        row = {"Dataset": dataset}

        if result["pretrained"]:
            row["Pretrained_Loss"] = result["pretrained"]["loss"]
            row["Pretrained_Accuracy"] = result["pretrained"]["accuracy"]
        else:
            row["Pretrained_Loss"] = None
            row["Pretrained_Accuracy"] = None

        if result["from_scratch"]:
            row["FromScratch_Loss"] = result["from_scratch"]["loss"]
            row["FromScratch_Accuracy"] = result["from_scratch"]["accuracy"]
        else:
            row["FromScratch_Loss"] = None
            row["FromScratch_Accuracy"] = None

        # Calculate improvements
        if result["pretrained"] and result["from_scratch"]:
            loss_improvement = (
                (result["from_scratch"]["loss"] - result["pretrained"]["loss"])
                / result["from_scratch"]["loss"]
                * 100
            )
            acc_improvement = (
                (result["pretrained"]["accuracy"] - result["from_scratch"]["accuracy"])
                / result["from_scratch"]["accuracy"]
                * 100
            )
            row["Loss_Improvement_%"] = loss_improvement
            row["Accuracy_Improvement_%"] = acc_improvement
        else:
            row["Loss_Improvement_%"] = None
            row["Accuracy_Improvement_%"] = None

        summary_data.append(row)

    # Save summary table
    summary_df = pd.DataFrame(summary_data)
    summary_path = output_path / "evaluation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Save detailed results
    results_path = output_path / "detailed_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Generate text report
    report_path = output_path / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("PRE-TRAINING EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("SUMMARY\n")
        f.write("-" * 30 + "\n")
        f.write(summary_df.to_string(index=False) + "\n\n")

        f.write("DETAILED RESULTS\n")
        f.write("-" * 30 + "\n")

        for result in results:
            f.write(f"\nDataset: {result['dataset']}\n")
            f.write("-" * 20 + "\n")

            if result["pretrained"]:
                f.write("Pre-trained:\n")
                f.write(f"  Loss: {result['pretrained']['loss']:.4f}\n")
                f.write(f"  Accuracy: {result['pretrained']['accuracy']:.4f}\n")

            if result["from_scratch"]:
                f.write("From Scratch:\n")
                f.write(f"  Loss: {result['from_scratch']['loss']:.4f}\n")
                f.write(f"  Accuracy: {result['from_scratch']['accuracy']:.4f}\n")

            if result["pretrained"] and result["from_scratch"]:
                loss_imp = (
                    (result["from_scratch"]["loss"] - result["pretrained"]["loss"])
                    / result["from_scratch"]["loss"]
                    * 100
                )
                acc_imp = (
                    (
                        result["pretrained"]["accuracy"]
                        - result["from_scratch"]["accuracy"]
                    )
                    / result["from_scratch"]["accuracy"]
                    * 100
                )
                f.write("Improvements:\n")
                f.write(f"  Loss: {loss_imp:.2f}%\n")
                f.write(f"  Accuracy: {acc_imp:.2f}%\n")

        # Overall statistics
        f.write("\nOVERALL STATISTICS\n")
        f.write("-" * 30 + "\n")

        valid_improvements = [
            row for row in summary_data if row["Loss_Improvement_%"] is not None
        ]
        if valid_improvements:
            avg_loss_imp = sum(
                row["Loss_Improvement_%"] for row in valid_improvements
            ) / len(valid_improvements)
            avg_acc_imp = sum(
                row["Accuracy_Improvement_%"] for row in valid_improvements
            ) / len(valid_improvements)

            f.write(f"Average Loss Improvement: {avg_loss_imp:.2f}%\n")
            f.write(f"Average Accuracy Improvement: {avg_acc_imp:.2f}%\n")

            num_better_loss = sum(
                1 for row in valid_improvements if row["Loss_Improvement_%"] > 0
            )
            num_better_acc = sum(
                1 for row in valid_improvements if row["Accuracy_Improvement_%"] > 0
            )

            f.write(
                f"Datasets with better loss: {num_better_loss}/"
                f"{len(valid_improvements)}\n"
            )
            f.write(
                f"Datasets with better accuracy: {num_better_acc}/"
                f"{len(valid_improvements)}\n"
            )

    return summary_path, results_path, report_path


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate pre-training effectiveness")

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints/finetune",
        help="Directory containing fine-tuned model checkpoints",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--sequence_length", type=int, default=96, help="Sequence length"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data workers"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device for evaluation"
    )

    args = parser.parse_args()

    # Set random seed
    pl.seed_everything(42)

    # Run evaluation suite
    results = run_evaluation_suite(args)

    # Generate report
    summary_path, results_path, report_path = generate_report(results, args.output_dir)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Summary table: {summary_path}")
    print(f"Detailed results: {results_path}")
    print(f"Full report: {report_path}")

    # Print summary to console
    try:
        summary_df = pd.read_csv(summary_path)
        print("\nSUMMARY:")
        print(summary_df.to_string(index=False))
    except Exception as e:
        print(f"Error displaying summary: {e}")


if __name__ == "__main__":
    main()
