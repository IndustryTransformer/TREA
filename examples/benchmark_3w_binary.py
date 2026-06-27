"""Binary classification benchmark on 3W: Normal vs Anomaly.

Remaps the 10-class 3W labels to binary (class 0 → Normal, classes 1-9 → Anomaly)
and benchmarks TREA-C against LOF and RF baselines for direct comparison with
published results from Fernandes et al. (2024):
    - LOF on real data:      F1 = 0.87
    - LOF on simulated data: F1 = 0.92

Usage:
    uv run python examples/benchmark_3w_binary.py
    uv run python examples/benchmark_3w_binary.py --models trea_triple_stat_tokens,lof,rf --seeds 42,43,44
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import time

from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from pytorch_lightning.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from torch.utils.data import DataLoader, WeightedRandomSampler


sys.path.insert(0, ".")

from trea.models import TriplePatchTransformer
from utils.three_w import ThreeWDataset


DEEP_MODELS = {
    "trea_triple",
    "trea_triple_stat_tokens",
}
CLASSICAL_MODELS = {"rf", "lof"}
ALL_MODELS = DEEP_MODELS | CLASSICAL_MODELS


def remap_to_binary(dataset: ThreeWDataset) -> None:
    """Remap 10-class labels to binary: Normal (0) vs Anomaly (1)."""
    dataset.labels = np.where(dataset.labels == 0, 0, 1).astype(np.int64)

    unique, counts = np.unique(dataset.labels, return_counts=True)
    dataset.class_counts = np.zeros(2, dtype=np.int64)
    for cls, count in zip(unique, counts, strict=False):
        dataset.class_counts[cls] = count

    total = len(dataset.labels)
    dataset.class_weights = np.zeros(2, dtype=np.float32)
    for cls, count in zip(unique, counts, strict=False):
        dataset.class_weights[cls] = total / (count * 2)

    print(
        f"  Binary remap: Normal={dataset.class_counts[0]}, Anomaly={dataset.class_counts[1]}"
    )


class BinaryThreeWDataModule(pl.LightningDataModule):
    """DataModule for binary 3W with tempered weighted sampling."""

    def __init__(
        self,
        train_dataset: ThreeWDataset,
        val_dataset: ThreeWDataset,
        batch_size: int,
        num_workers: int,
        sampling_power: float,
        sampler_seed: int,
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sampling_power = sampling_power
        self.sampler_seed = sampler_seed

    def set_sampler_seed(self, seed: int) -> None:
        self.sampler_seed = seed

    def train_dataloader(self) -> DataLoader:
        class_counts = torch.from_numpy(self.train_dataset.class_counts).double()
        class_weights = torch.zeros_like(class_counts)
        present = class_counts > 0
        class_weights[present] = class_counts[present].pow(-self.sampling_power)
        sample_weights = class_weights[
            torch.from_numpy(self.train_dataset.labels).long()
        ]

        generator = torch.Generator()
        generator.manual_seed(self.sampler_seed)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Binary 3W benchmark: Normal vs Anomaly"
    )

    # Data
    parser.add_argument("--toolkit-path", type=str, default="/home/kailukowiak/Work/3W")
    parser.add_argument(
        "--data-path", type=str, default="/home/kailukowiak/Work/3W/dataset"
    )
    parser.add_argument("--window-size", type=int, default=96)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-split-seed", type=int, default=42)
    parser.add_argument("--max-files", type=int, default=None)

    # Protocol
    parser.add_argument(
        "--models",
        type=str,
        default="trea_triple_stat_tokens,rf,lof",
        help=f"Comma-separated model IDs from: {sorted(ALL_MODELS)}",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
        help="Comma-separated training/evaluation seeds",
    )

    # Deep model training
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sampling-power", type=float, default=0.5)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # Architecture
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--feature-attention-dim", type=int, default=32)
    parser.add_argument("--feature-attention-heads", type=int, default=4)

    # Classical baseline options
    parser.add_argument("--rf-n-estimators", type=int, default=500)
    parser.add_argument("--rf-n-jobs", type=int, default=-1)
    parser.add_argument("--rf-train-max-samples", type=int, default=200000)
    parser.add_argument("--lof-n-neighbors", type=int, default=20)
    parser.add_argument("--lof-train-max-samples", type=int, default=50000)

    # Outputs
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/benchmark_3w_binary",
    )
    parser.add_argument("--run-name", type=str, default="default")

    return parser.parse_args()


def extract_engineered_features(windows: np.ndarray) -> np.ndarray:
    """Create fixed-size feature vectors from [N, C, T] windows."""
    x = windows
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(x, axis=2)
        std = np.nanstd(x, axis=2)
        min_v = np.nanmin(x, axis=2)
        max_v = np.nanmax(x, axis=2)
        nan_rate = np.isnan(x).mean(axis=2)
        first = np.nan_to_num(x[:, :, 0], nan=0.0)
        last = np.nan_to_num(x[:, :, -1], nan=0.0)
        delta = last - first
        half = x.shape[2] // 2
        if half > 0:
            trend = np.nanmean(x[:, :, half:], axis=2) - np.nanmean(
                x[:, :, :half], axis=2
            )
        else:
            trend = np.zeros_like(mean)
        energy = np.nanmean(np.square(np.nan_to_num(x, nan=0.0)), axis=2)

    blocks = [mean, std, min_v, max_v, nan_rate, delta, trend, energy]
    out = np.concatenate(blocks, axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def binary_metrics(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Compute binary classification metrics (anomaly = positive class)."""
    tp = int(((preds == 1) & (targets == 1)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())
    tn = int(((preds == 0) & (targets == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

    # Also compute normal-class F1 for macro average
    prec_normal = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_normal = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_normal = (
        2 * prec_normal * rec_normal / (prec_normal + rec_normal)
        if (prec_normal + rec_normal) > 0
        else 0.0
    )
    macro_f1 = (f1_normal + f1) / 2.0

    return {
        "accuracy": accuracy,
        "anomaly_precision": precision,
        "anomaly_recall": recall,
        "anomaly_f1": f1,
        "normal_f1": f1_normal,
        "macro_f1": macro_f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


@torch.no_grad()
def evaluate_torch_binary(
    model: pl.LightningModule, dataloader: DataLoader
) -> dict[str, float]:
    """Evaluate a PyTorch model on binary classification."""
    device = model.device
    model.eval()

    all_preds = []
    all_targets = []

    for batch in dataloader:
        x_num = batch["x_num"].to(device)
        x_cat = batch["x_cat"].to(device)
        y = batch["y"]

        logits = model(x_num, x_cat)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        all_preds.append(preds)
        all_targets.append(y.numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    return binary_metrics(preds, targets)


def build_trea(
    info: dict[str, Any],
    args: argparse.Namespace,
    use_stat_tokens: bool,
) -> TriplePatchTransformer:
    """Build a TREA-C model for binary classification."""
    model = TriplePatchTransformer(
        C_num=info["n_numeric"],
        C_cat=info["n_categorical"],
        cat_cardinalities=info["cat_cardinalities"],
        T=info["sequence_length"],
        d_model=args.d_model,
        task="classification",
        num_classes=2,
        n_head=args.n_head,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pooling=args.pooling,
        lr=args.lr,
        patch_len=args.patch_len,
        stride=args.patch_stride,
        column_names=info["column_names"],
        use_column_embeddings=True,
        use_pre_patch_feature_attention=True,
        feature_attention_dim=args.feature_attention_dim,
        feature_attention_heads=args.feature_attention_heads,
        use_stat_tokens=use_stat_tokens,
    )
    model.loss_fn = nn.CrossEntropyLoss()

    def _configure_optimizers(self: pl.LightningModule):
        return torch.optim.AdamW(
            self.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    model.configure_optimizers = MethodType(_configure_optimizers, model)
    return model


def run_deep_model(
    model_name: str,
    seed: int,
    info: dict[str, Any],
    dm: BinaryThreeWDataModule,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Train and evaluate a deep model for binary classification."""
    pl.seed_everything(seed, workers=True)
    dm.set_sampler_seed(seed)

    use_stat_tokens = "stat_tokens" in model_name
    model = build_trea(info=info, args=args, use_stat_tokens=use_stat_tokens)

    # Override validation_step to log val_loss for early stopping
    def validation_step_binary(batch, batch_idx):
        out = model(batch["x_num"], batch["x_cat"])
        loss = model.loss_fn(out, batch["y"])
        model.log("val_loss", loss, prog_bar=True)
        return loss

    model.validation_step = validation_step_binary

    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=True)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        precision=args.precision,
        logger=False,
        enable_progress_bar=args.show_progress,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=[early_stop],
    )

    t0 = time.perf_counter()
    trainer.fit(model, datamodule=dm)
    train_seconds = time.perf_counter() - t0

    metrics = evaluate_torch_binary(model=model, dataloader=dm.val_dataloader())
    metrics["train_seconds"] = train_seconds
    metrics["num_params"] = int(sum(p.numel() for p in model.parameters()))
    return metrics


def run_rf_baseline(
    train_ds: ThreeWDataset,
    val_ds: ThreeWDataset,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Random Forest with engineered features for binary classification."""
    rng = np.random.default_rng(seed)

    x_train = extract_engineered_features(train_ds.windows)
    y_train = train_ds.labels.copy()
    x_val = extract_engineered_features(val_ds.windows)
    y_val = val_ds.labels.copy()

    if args.rf_train_max_samples and len(y_train) > args.rf_train_max_samples:
        idx = rng.choice(len(y_train), size=args.rf_train_max_samples, replace=False)
        x_train = x_train[idx]
        y_train = y_train[idx]

    clf = RandomForestClassifier(
        n_estimators=args.rf_n_estimators,
        random_state=seed,
        n_jobs=args.rf_n_jobs,
        class_weight="balanced_subsample",
    )

    t0 = time.perf_counter()
    clf.fit(x_train, y_train)
    train_seconds = time.perf_counter() - t0

    pred = clf.predict(x_val)
    metrics = binary_metrics(pred, y_val)
    metrics["train_seconds"] = train_seconds
    metrics["num_params"] = 0
    return metrics


def run_lof_baseline(
    train_ds: ThreeWDataset,
    val_ds: ThreeWDataset,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """LOF (Local Outlier Factor) baseline — trained on normal data only.

    This mirrors Fernandes et al. (2024): fit LOF on normal-class windows,
    then predict inlier (Normal) vs outlier (Anomaly) on the full val set.
    """
    rng = np.random.default_rng(seed)

    x_train_all = extract_engineered_features(train_ds.windows)
    y_train_all = train_ds.labels

    # LOF trains only on normal (class 0) data
    normal_mask = y_train_all == 0
    x_train_normal = x_train_all[normal_mask]

    # Subsample if too large
    if args.lof_train_max_samples and len(x_train_normal) > args.lof_train_max_samples:
        idx = rng.choice(
            len(x_train_normal), size=args.lof_train_max_samples, replace=False
        )
        x_train_normal = x_train_normal[idx]

    x_val = extract_engineered_features(val_ds.windows)
    y_val = val_ds.labels.copy()

    lof = LocalOutlierFactor(
        n_neighbors=args.lof_n_neighbors,
        novelty=True,
        n_jobs=-1,
    )

    t0 = time.perf_counter()
    lof.fit(x_train_normal)
    train_seconds = time.perf_counter() - t0

    # LOF predict: 1 = inlier (normal), -1 = outlier (anomaly)
    raw_pred = lof.predict(x_val)
    pred = np.where(raw_pred == -1, 1, 0)  # remap: -1→Anomaly(1), 1→Normal(0)

    metrics = binary_metrics(pred, y_val)
    metrics["train_seconds"] = train_seconds
    metrics["num_params"] = 0
    return metrics


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_report(
    runs: list[dict[str, Any]],
    models: list[str],
) -> str:
    """Build a markdown report comparing models against published baselines."""
    lines: list[str] = []
    lines.append("# 3W Binary Classification Benchmark: Normal vs Anomaly")
    lines.append("")
    lines.append("## Published Baselines (Fernandes et al. 2024)")
    lines.append("")
    lines.append("| Method | Data | F1 (Anomaly) |")
    lines.append("|--------|------|-------------:|")
    lines.append("| LOF (no feature eng.) | Simulated | 0.920 |")
    lines.append("| LOF (with feature eng.) | Simulated | 0.915 |")
    lines.append("| LOF (with feature eng.) | Real | 0.870 |")
    lines.append("| LOF (no feature eng.) | Real | 0.859 |")
    lines.append("| Isolation Forest | Mixed | 0.727 |")
    lines.append("| OCSVM | Mixed | 0.470 |")
    lines.append("")
    lines.append("## Our Results (Mixed real + simulated data)")
    lines.append("")
    lines.append(
        "| Model | Seeds | Anomaly F1 | Normal F1 | Macro F1 | "
        "Accuracy | Precision | Recall | Sec/run |"
    )
    lines.append(
        "|-------|------:|----------:|---------:|--------:|--------:|----------:|-------:|--------:|"
    )

    for model_name in models:
        model_runs = [r for r in runs if r["model"] == model_name]
        if not model_runs:
            continue

        n = len(model_runs)
        af1 = np.mean([r["anomaly_f1"] for r in model_runs])
        nf1 = np.mean([r["normal_f1"] for r in model_runs])
        mf1 = np.mean([r["macro_f1"] for r in model_runs])
        acc = np.mean([r["accuracy"] for r in model_runs])
        prec = np.mean([r["anomaly_precision"] for r in model_runs])
        rec = np.mean([r["anomaly_recall"] for r in model_runs])
        secs = np.mean([r["train_seconds"] for r in model_runs])

        lines.append(
            f"| {model_name} | {n} | {af1:.4f} | {nf1:.4f} | {mf1:.4f} | "
            f"{acc:.4f} | {prec:.4f} | {rec:.4f} | {secs:.1f} |"
        )

    lines.append("")
    lines.append("## Per-Run Detail")
    lines.append("")
    lines.append(
        "| Model | Seed | Anomaly F1 | Normal F1 | Macro F1 | Accuracy | TP | FP | FN | TN |"
    )
    lines.append(
        "|-------|-----:|----------:|---------:|--------:|--------:|---:|---:|---:|---:|"
    )

    for r in runs:
        lines.append(
            f"| {r['model']} | {r['seed']} | {r['anomaly_f1']:.4f} | "
            f"{r['normal_f1']:.4f} | {r['macro_f1']:.4f} | {r['accuracy']:.4f} | "
            f"{r['tp']} | {r['fp']} | {r['fn']} | {r['tn']} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Published LOF results use **one-class** learning (train on normal only). "
        "Our LOF baseline replicates this protocol."
    )
    lines.append(
        "- TREA-C and RF use **supervised binary** classification "
        "(trained on both normal and anomaly windows)."
    )
    lines.append(
        "- Fernandes et al. evaluate on real-only and simulated-only subsets separately. "
        "Our evaluation uses the full mixed dataset (real + simulated), "
        "which may differ in difficulty."
    )
    lines.append(
        "- TREA-C processes raw time series with NaN handling. "
        "RF and LOF use handcrafted statistical features (mean, std, min, max, etc.)."
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    invalid = [m for m in models if m not in ALL_MODELS]
    if invalid:
        raise ValueError(f"Unknown model ids: {invalid}. Valid: {sorted(ALL_MODELS)}")

    torch.set_float32_matmul_precision("medium")

    print("=" * 70)
    print("3W Binary Benchmark: Normal vs Anomaly")
    print("=" * 70)
    print(f"Models: {models}")
    print(f"Seeds: {seeds}")

    # Load data
    print("\nLoading training data...")
    train_ds = ThreeWDataset(
        toolkit_path=args.toolkit_path,
        data_path=args.data_path,
        window_size=args.window_size,
        stride=args.stride,
        split="train",
        val_fraction=args.val_fraction,
        max_files=args.max_files,
        seed=args.data_split_seed,
        augment=False,
    )

    print("\nLoading validation data...")
    val_ds = ThreeWDataset(
        toolkit_path=args.toolkit_path,
        data_path=args.data_path,
        window_size=args.window_size,
        stride=args.stride,
        split="val",
        val_fraction=args.val_fraction,
        normalization_stats=train_ds.normalization_stats,
        max_files=args.max_files,
        seed=args.data_split_seed,
        augment=False,
    )

    # Remap to binary
    print("\nRemapping labels to binary...")
    remap_to_binary(train_ds)
    remap_to_binary(val_ds)

    info = train_ds.get_feature_info()
    # Override num_classes since we remapped
    info["num_classes"] = 2

    print(f"\nFeatures: {info['n_numeric']} numeric, T={info['sequence_length']}")
    print(f"Train: {len(train_ds)} windows, Val: {len(val_ds)} windows")

    dm = BinaryThreeWDataModule(
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampling_power=args.sampling_power,
        sampler_seed=seeds[0],
    )

    # Run benchmarks
    runs: list[dict[str, Any]] = []
    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = out_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
                "args": vars(args),
                "models": models,
                "seeds": seeds,
                "num_train_samples": len(train_ds),
                "num_val_samples": len(val_ds),
                "train_class_counts": train_ds.class_counts.tolist(),
                "val_class_counts": val_ds.class_counts.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for model_name in models:
        print(f"\n{'=' * 70}")
        print(f"Model: {model_name}")
        print("=" * 70)

        for seed in seeds:
            print(f"\n[{model_name}] seed={seed} ...")

            if model_name in DEEP_MODELS:
                metrics = run_deep_model(
                    model_name=model_name,
                    seed=seed,
                    info=info,
                    dm=dm,
                    args=args,
                )
            elif model_name == "rf":
                metrics = run_rf_baseline(
                    train_ds=train_ds,
                    val_ds=val_ds,
                    seed=seed,
                    args=args,
                )
            elif model_name == "lof":
                metrics = run_lof_baseline(
                    train_ds=train_ds,
                    val_ds=val_ds,
                    seed=seed,
                    args=args,
                )
            else:
                raise ValueError(f"Unsupported model: {model_name}")

            row: dict[str, Any] = {"model": model_name, "seed": seed, **metrics}
            runs.append(row)

            print(
                f"  anomaly_f1={metrics['anomaly_f1']:.4f} "
                f"normal_f1={metrics['normal_f1']:.4f} "
                f"macro_f1={metrics['macro_f1']:.4f} "
                f"accuracy={metrics['accuracy']:.4f} "
                f"(TP={metrics['tp']} FP={metrics['fp']} "
                f"FN={metrics['fn']} TN={metrics['tn']})"
            )

    # Save results
    fieldnames = [
        "model",
        "seed",
        "anomaly_f1",
        "normal_f1",
        "macro_f1",
        "accuracy",
        "anomaly_precision",
        "anomaly_recall",
        "tp",
        "fp",
        "fn",
        "tn",
        "train_seconds",
        "num_params",
    ]
    write_csv(out_dir / "runs.csv", runs, fieldnames)

    report = build_report(runs, models)
    (out_dir / "report.md").write_text(report, encoding="utf-8")

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<30} {'Anomaly F1':>10} {'Macro F1':>10} {'Accuracy':>10}")
    print("-" * 65)

    for model_name in models:
        model_runs = [r for r in runs if r["model"] == model_name]
        if not model_runs:
            continue
        af1 = np.mean([r["anomaly_f1"] for r in model_runs])
        mf1 = np.mean([r["macro_f1"] for r in model_runs])
        acc = np.mean([r["accuracy"] for r in model_runs])
        print(f"{model_name:<30} {af1:>10.4f} {mf1:>10.4f} {acc:>10.4f}")

    print(f"\n{'Published LOF (real)':<30} {'0.8700':>10} {'—':>10} {'—':>10}")
    print(f"{'Published LOF (simulated)':<30} {'0.9200':>10} {'—':>10} {'—':>10}")

    print(f"\nResults saved to: {out_dir}")
    print(f"  - {out_dir / 'runs.csv'}")
    print(f"  - {out_dir / 'report.md'}")
    print(f"  - {out_dir / 'config.json'}")


if __name__ == "__main__":
    main()
