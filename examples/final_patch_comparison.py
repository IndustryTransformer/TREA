#!/usr/bin/env python3
"""
Final comparison: PatchTREA-C Baseline vs Column-Aware versions.
Clean implementation with just the two configurations we want to keep.
"""

import os
import time

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch


# Set tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from data.downloaders.etth1 import ETTh1Dataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, f1_score

from treac.models.multi_dataset_model import MultiDatasetModel
from treac.utils import get_checkpoint_path, get_output_path
from utils.datamodule import TimeSeriesDataModule


def inject_nans(dataset, nan_rate=0.05):
    """Inject NaN values into the dataset at specified rate."""
    print(f"Injecting NaN values at rate: {nan_rate:.1%}")

    # Create a copy of the dataset
    dataset_copy = type(dataset)(
        data_dir=dataset.data_dir,
        train=dataset.train,
        sequence_length=dataset.sequence_length,
        task=dataset.task,
        num_classes=dataset.num_classes,
        download=False,
    )

    # Inject NaNs into x_num
    x_num = dataset_copy.x_num.clone()
    nan_mask = torch.rand_like(x_num) < nan_rate
    x_num[nan_mask] = float("nan")
    dataset_copy.x_num = x_num

    # Calculate actual NaN percentage
    total_values = x_num.numel()
    nan_count = torch.isnan(x_num).sum().item()
    actual_nan_rate = nan_count / total_values

    print(
        f"Actual NaN rate: {actual_nan_rate:.1%} "
        f"({nan_count:,} / {total_values:,} values)"
    )

    return dataset_copy


def train_model(model, dm, model_name, max_epochs=10):
    """Train a model and return results."""

    # Callbacks - use centralized path management
    checkpoint_dir = get_checkpoint_path(f"final_patch/{model_name}")
    checkpoint = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min",
        verbose=True,
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint, early_stop],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        deterministic=True,
        logger=False,
    )

    # Train
    start_time = time.time()
    trainer.fit(model, dm)
    training_time = time.time() - start_time

    # Load best model for evaluation
    best_model = type(model).load_from_checkpoint(
        checkpoint.best_model_path, **model.hparams
    )
    best_model.eval()

    # Get predictions for metrics
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dm.val_dataloader():
            x_num = batch["x_num"]
            x_cat = batch.get("x_cat", None)
            labels = batch["y"]

            # Move tensors to device
            device = next(best_model.parameters()).device
            x_num = x_num.to(device)
            if x_cat is not None:
                x_cat = x_cat.to(device)
            labels = labels.to(device)

            if labels.ndim == 2:
                labels = torch.argmax(labels, dim=1)
            labels = labels.long()

            logits = best_model(x_num, x_cat)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    # Get final validation loss
    val_loss = trainer.callback_metrics.get("val_loss", float("nan"))
    if torch.is_tensor(val_loss):
        val_loss = val_loss.item()

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Val Loss": val_loss,
        "Training Time (s)": training_time,
        "Parameters": sum(p.numel() for p in model.parameters()),
        "Epochs": trainer.current_epoch + 1,
        "Use Columns": "Yes"
        if hasattr(model, "use_column_embeddings") and model.use_column_embeddings
        else "No",
    }


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    MAX_EPOCHS = 10
    NAN_RATE = 0.05

    print("=" * 80)
    print("Final PatchTREA-C Comparison: Baseline vs Column-Aware")
    print("=" * 80)

    # Load clean dataset
    print("\nLoading ETTh1 dataset...")
    train_dataset = ETTh1Dataset(train=True)
    val_dataset = ETTh1Dataset(train=False)

    # Inject NaNs
    print(f"\nInjecting {NAN_RATE:.1%} NaN values...")
    train_dataset_nan = inject_nans(train_dataset, nan_rate=NAN_RATE)
    val_dataset_nan = inject_nans(val_dataset, nan_rate=NAN_RATE)

    # Create data module
    dm = TimeSeriesDataModule(
        train_dataset=train_dataset_nan,
        val_dataset=val_dataset_nan,
        batch_size=64,
        num_workers=2,
    )

    # Get dataset info
    c_in = train_dataset_nan.x_num.shape[1]
    seq_len = train_dataset_nan.sequence_length
    num_classes = train_dataset_nan.num_classes
    column_names = dm.get_column_names()

    print(f"Train samples: {len(train_dataset_nan)}")
    print(f"Val samples: {len(val_dataset_nan)}")
    print("\nDataset info:")
    print(f"- Input channels: {c_in}")
    print(f"- Sequence length: {seq_len}")
    print(f"- Number of classes: {num_classes}")
    print(f"- Column names: {column_names}")

    results = []

    # 1. PatchTREA-C Baseline (no column embeddings)
    print("\n" + "=" * 60)
    print("Training PatchTREA-C Baseline (patches + dual-patch NaN)...")
    print("=" * 60)

    baseline_model = MultiDatasetModel.create_baseline(
        c_in=c_in,
        seq_len=seq_len,
        num_classes=num_classes,
        task="classification",
        patch_len=16,
        stride=8,
        d_model=128,
        n_head=8,
        num_layers=3,
        lr=1e-3,
        mode="standard",
    )

    baseline_results = train_model(
        baseline_model, dm, "PatchTREA-C-Baseline", max_epochs=MAX_EPOCHS
    )
    results.append(baseline_results)

    # 2. PatchTREA-C Column-Aware (with simple column embeddings)
    print("\n" + "=" * 60)
    print("Training PatchTREA-C Column-Aware (patches + dual-patch NaN + columns)...")
    print("=" * 60)

    column_model = MultiDatasetModel.create_column_aware(
        c_in=c_in,
        seq_len=seq_len,
        num_classes=num_classes,
        column_names=column_names,
        task="classification",
        patch_len=16,
        stride=8,
        d_model=128,
        n_head=8,
        num_layers=3,
        lr=1e-3,
        column_embedding_dim=16,
        mode="standard",
    )

    column_results = train_model(
        column_model, dm, "PatchTREA-C-Columns", max_epochs=MAX_EPOCHS
    )
    results.append(column_results)

    # Create comparison DataFrame
    df = pd.DataFrame(results)

    # Display results
    print("\n" + "=" * 80)
    print("FINAL PATCHTREA-C COMPARISON")
    print("=" * 80)
    print(df.to_string(index=False, float_format="%.6f"))

    # Save results to proper output directory
    output_path = get_output_path("final_patchTREA-C_comparison.csv", "comparisons")
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Analysis
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    baseline_acc = df[df["Model"] == "PatchTREA-C-Baseline"]["Accuracy"].iloc[0]
    column_acc = df[df["Model"] == "PatchTREA-C-Columns"]["Accuracy"].iloc[0]

    print(
        f"PatchTREA-C Baseline:          {baseline_acc:.4f} (patches + dual-patch NaN)"
    )
    print(
        f"PatchTREA-C Column-Aware:      {column_acc:.4f} "
        f"(+ lightweight column embeddings)"
    )
    print(
        f"Performance trade-off:       "
        f"{((column_acc - baseline_acc) / baseline_acc * 100):+.2f}%"
    )

    baseline_params = df[df["Model"] == "PatchTREA-C-Baseline"]["Parameters"].iloc[0]
    column_params = df[df["Model"] == "PatchTREA-C-Columns"]["Parameters"].iloc[0]
    param_overhead = (column_params - baseline_params) / baseline_params * 100

    print(
        f"\nParameter overhead:          +{param_overhead:.2f}% ("
        f"{column_params - baseline_params:,} params)"
    )

    print("\n🎯 CONCLUSION:")
    print(
        f"   ✅ Baseline PatchTREA-C: {baseline_acc:.1%} accuracy, "
        f"{baseline_params:,} params"
    )
    print(
        f"   ✅ Column-Aware PatchTREA-C: {column_acc:.1%} accuracy, "
        f"{column_params:,} params"
    )
    if abs(column_acc - baseline_acc) <= 0.05:  # Within 5%
        print("   🚀 READY FOR MULTI-DATASET TRAINING!")
        print("      Small performance trade-off acceptable for transferability gains.")
    else:
        print("   ⚠️  Consider tuning column embedding integration.")


if __name__ == "__main__":
    main()
