"""Benchmark 3W models with multi-seed non-inferiority analysis.

This script is designed to support claims like:
"Our candidate architecture is statistically non-inferior to strong baselines."

It runs selected models on the same 3W split across multiple random seeds,
computes validation macro-F1/accuracy, and exports:
- run-level results (`runs.csv`)
- aggregated summary (`summary.csv`)
- non-inferiority comparisons (`noninferiority.csv`)
- human-readable report (`report.md`)

Usage:
    uv run python examples/benchmark_3w_noninferiority.py \
      --models trea_triple,patchtstnan,multidataset_none,rf_stat_features \
      --candidate-model trea_triple \
      --seeds 42,43,44,45,46 \
      --max-epochs 30 \
      --margin 0.01
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

from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, WeightedRandomSampler


sys.path.insert(0, ".")

from trea.models import MultiDatasetModel, PatchTSTNan, TriplePatchTransformer
from utils.three_w import EVENT_NAMES, ThreeWDataset


DEEP_MODELS = {
    "trea_triple",
    "trea_triple_no_feature_attn",
    "trea_triple_stat_tokens",
    "trea_triple_stat_tokens_no_feature_attn",
    "patchtstnan",
    "multidataset_none",
    "multidataset_auto",
}
CLASSICAL_MODELS = {"rf_stat_features"}
ALL_MODELS = DEEP_MODELS | CLASSICAL_MODELS


class ThreeWDataModule(pl.LightningDataModule):
    """DataModule for 3W with tempered weighted sampling."""

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
        description="Multi-seed 3W benchmark with non-inferiority analysis"
    )

    # Data
    parser.add_argument("--toolkit-path", type=str, default="/home/kailukowiak/Work/3W")
    parser.add_argument(
        "--data-path", type=str, default="/home/kailukowiak/Work/3W/dataset"
    )
    parser.add_argument("--window-size", type=int, default=96)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--data-split-seed",
        type=int,
        default=42,
        help="Controls train/val file split for 3W loader",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Cap files loaded for both train and val (debug/smoke runs)",
    )

    # Protocol
    parser.add_argument(
        "--models",
        type=str,
        default=(
            "trea_triple,patchtstnan,multidataset_none,multidataset_auto,"
            "rf_stat_features"
        ),
        help=f"Comma-separated model IDs from: {sorted(ALL_MODELS)}",
    )
    parser.add_argument(
        "--candidate-model",
        type=str,
        default="trea_triple",
        help="Model to test for non-inferiority vs all other selected models",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44,45,46",
        help="Comma-separated training/evaluation seeds",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.01,
        help="Non-inferiority margin on macro-F1 (candidate - comparator)",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=4000)

    # Deep model training
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sampling-power", type=float, default=0.5)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        choices=["16-mixed", "32-true"],
    )
    parser.add_argument("--log-every-n-steps", type=int, default=100)
    parser.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Show Lightning progress bars (default: true). "
            "Use --no-show-progress to disable."
        ),
    )

    # Shared architecture
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=8)

    # Triple model options
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument(
        "--feature-attention-dim",
        type=int,
        default=32,
        help="Used for trea_triple when feature attention is enabled",
    )
    parser.add_argument("--feature-attention-heads", type=int, default=4)

    # Classical baseline options
    parser.add_argument("--rf-n-estimators", type=int, default=500)
    parser.add_argument("--rf-max-depth", type=int, default=None)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=1)
    parser.add_argument("--rf-n-jobs", type=int, default=-1)
    parser.add_argument(
        "--rf-train-max-samples",
        type=int,
        default=200000,
        help="Subsample cap for RF training rows",
    )
    parser.add_argument(
        "--rf-use-cross-features",
        action="store_true",
        help="Add pairwise mean-product features for RF baseline",
    )
    parser.add_argument(
        "--rf-max-cross-pairs",
        type=int,
        default=64,
        help="Max pairwise cross features when --rf-use-cross-features is enabled",
    )

    # Outputs
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/benchmark_3w_noninferiority",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="default",
    )

    return parser.parse_args()


def parse_csv_ints(raw: str) -> list[int]:
    values = [s.strip() for s in raw.split(",")]
    return [int(s) for s in values if s]


def parse_csv_models(raw: str) -> list[str]:
    models = [m.strip() for m in raw.split(",") if m.strip()]
    invalid = [m for m in models if m not in ALL_MODELS]
    if invalid:
        raise ValueError(
            f"Unknown model ids: {invalid}. Valid model ids: {sorted(ALL_MODELS)}"
        )
    if not models:
        raise ValueError("At least one model must be provided in --models")
    return models


def attach_adamw_optimizer(
    model: pl.LightningModule, lr: float, weight_decay: float
) -> None:
    """Replace configure_optimizers for consistent benchmark budget."""

    def _configure_optimizers(self: pl.LightningModule):
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

    model.configure_optimizers = MethodType(_configure_optimizers, model)


def build_trea_triple(
    info: dict[str, Any],
    args: argparse.Namespace,
    use_pre_patch_feature_attention: bool,
    use_stat_tokens: bool,
) -> TriplePatchTransformer:
    model = TriplePatchTransformer(
        C_num=info["n_numeric"],
        C_cat=info["n_categorical"],
        cat_cardinalities=info["cat_cardinalities"],
        T=info["sequence_length"],
        d_model=args.d_model,
        task="classification",
        num_classes=info["num_classes"],
        n_head=args.n_head,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pooling=args.pooling,
        lr=args.lr,
        patch_len=args.patch_len,
        stride=args.patch_stride,
        column_names=info["column_names"],
        use_column_embeddings=True,
        use_pre_patch_feature_attention=use_pre_patch_feature_attention,
        feature_attention_dim=args.feature_attention_dim,
        feature_attention_heads=args.feature_attention_heads,
        use_stat_tokens=use_stat_tokens,
    )
    model.loss_fn = nn.CrossEntropyLoss()
    attach_adamw_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    return model


def build_patchtstnan(info: dict[str, Any], args: argparse.Namespace) -> PatchTSTNan:
    model = PatchTSTNan(
        C_num=info["n_numeric"],
        C_cat=0,
        T=info["sequence_length"],
        d_model=args.d_model,
        patch_len=args.patch_len,
        stride=args.patch_stride,
        num_classes=info["num_classes"],
        n_head=args.n_head,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        task="classification",
    )
    attach_adamw_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    return model


def build_multidataset(
    info: dict[str, Any], args: argparse.Namespace, column_embedding_strategy: str
) -> MultiDatasetModel:
    model = MultiDatasetModel(
        max_numeric_features=info["n_numeric"],
        max_categorical_features=0,
        num_classes=info["num_classes"],
        mode="variable_features",
        task="classification",
        patch_len=args.patch_len,
        stride=args.patch_stride,
        d_model=args.d_model,
        n_head=args.n_head,
        num_layers=args.num_layers,
        max_sequence_length=info["sequence_length"],
        lr=args.lr,
        column_embedding_strategy=column_embedding_strategy,
        use_feature_masks=True,
        categorical_cardinalities=[],
    )
    model.set_dataset_schema(
        numeric_features=info["n_numeric"],
        categorical_features=0,
        column_names=info["column_names"],
    )
    attach_adamw_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    return model


def macro_f1_from_predictions(
    preds: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> tuple[float, list[float]]:
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(targets.tolist(), preds.tolist(), strict=False):
        cm[t, p] += 1

    per_class_f1: list[float] = []
    for i in range(num_classes):
        tp = cm[i, i].item()
        fp = (cm[:, i].sum() - cm[i, i]).item()
        fn = (cm[i, :].sum() - cm[i, i]).item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_class_f1.append(float(f1))

    macro_f1 = float(sum(per_class_f1) / len(per_class_f1))
    return macro_f1, per_class_f1


@torch.no_grad()
def evaluate_torch_model(
    model: pl.LightningModule, dataloader: DataLoader, num_classes: int
) -> dict[str, Any]:
    device = model.device
    model.eval()

    all_preds = []
    all_targets = []

    for batch in dataloader:
        x_num = batch["x_num"].to(device)
        x_cat = batch.get("x_cat", None)
        if x_cat is not None:
            x_cat = x_cat.to(device)
        y = batch["y"].to(device)

        logits = model(x_num, x_cat)
        preds = torch.argmax(logits, dim=1)

        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    acc = float((preds == targets).float().mean().item())
    macro_f1, per_class_f1 = macro_f1_from_predictions(preds, targets, num_classes)

    return {
        "val_acc": acc,
        "val_f1_macro": macro_f1,
        "val_f1_per_class": per_class_f1,
    }


def extract_engineered_features(
    windows: np.ndarray,
    use_cross_features: bool,
    max_cross_pairs: int,
) -> np.ndarray:
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
    base = np.concatenate(blocks, axis=1)
    base = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)

    if not use_cross_features:
        return base.astype(np.float32)

    n_channels = mean.shape[1]
    pair_features: list[np.ndarray] = []
    pair_count = 0
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            pair_features.append(mean[:, i] * mean[:, j])
            pair_count += 1
            if pair_count >= max_cross_pairs:
                break
        if pair_count >= max_cross_pairs:
            break

    if pair_features:
        cross = np.stack(pair_features, axis=1)
        cross = np.nan_to_num(cross, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.concatenate([base, cross], axis=1)
    else:
        out = base

    return out.astype(np.float32)


def run_rf_baseline(
    train_ds: ThreeWDataset,
    val_ds: ThreeWDataset,
    num_classes: int,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)

    x_train = extract_engineered_features(
        train_ds.windows,
        use_cross_features=args.rf_use_cross_features,
        max_cross_pairs=args.rf_max_cross_pairs,
    )
    y_train = train_ds.labels.copy()
    x_val = extract_engineered_features(
        val_ds.windows,
        use_cross_features=args.rf_use_cross_features,
        max_cross_pairs=args.rf_max_cross_pairs,
    )
    y_val = val_ds.labels.copy()

    if (
        args.rf_train_max_samples is not None
        and len(y_train) > args.rf_train_max_samples
    ):
        idx = rng.choice(len(y_train), size=args.rf_train_max_samples, replace=False)
        x_train = x_train[idx]
        y_train = y_train[idx]

    clf = RandomForestClassifier(
        n_estimators=args.rf_n_estimators,
        max_depth=args.rf_max_depth,
        min_samples_leaf=args.rf_min_samples_leaf,
        random_state=seed,
        n_jobs=args.rf_n_jobs,
        class_weight="balanced_subsample",
    )

    t0 = time.perf_counter()
    clf.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - t0

    pred = clf.predict(x_val)
    pred_t = torch.from_numpy(pred.astype(np.int64))
    target_t = torch.from_numpy(y_val.astype(np.int64))

    val_acc = float((pred_t == target_t).float().mean().item())
    val_f1_macro, val_f1_per_class = macro_f1_from_predictions(
        pred_t, target_t, num_classes=num_classes
    )

    return {
        "val_acc": val_acc,
        "val_f1_macro": val_f1_macro,
        "val_f1_per_class": val_f1_per_class,
        "train_seconds": fit_seconds,
        "num_params": 0,
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        val = float(values[0])
        return val, val

    idx = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    samples = values[idx]
    sample_means = samples.mean(axis=1)
    lo = float(np.quantile(sample_means, alpha / 2))
    hi = float(np.quantile(sample_means, 1 - alpha / 2))
    return lo, hi


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_runs(
    runs: list[dict[str, Any]],
    models: list[str],
    bootstrap_samples: int,
    ci_alpha: float = 0.05,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(0)
    summary_rows: list[dict[str, Any]] = []

    for model_name in models:
        model_runs = [r for r in runs if r["model"] == model_name]
        f1 = np.asarray([r["val_f1_macro"] for r in model_runs], dtype=np.float64)
        acc = np.asarray([r["val_acc"] for r in model_runs], dtype=np.float64)
        train_s = np.asarray([r["train_seconds"] for r in model_runs], dtype=np.float64)

        f1_lo, f1_hi = bootstrap_mean_ci(
            f1, n_bootstrap=bootstrap_samples, alpha=ci_alpha, rng=rng
        )
        acc_lo, acc_hi = bootstrap_mean_ci(
            acc, n_bootstrap=bootstrap_samples, alpha=ci_alpha, rng=rng
        )

        summary_rows.append(
            {
                "model": model_name,
                "n_runs": int(len(model_runs)),
                "val_f1_macro_mean": float(np.mean(f1)),
                "val_f1_macro_std": float(np.std(f1, ddof=1)) if len(f1) > 1 else 0.0,
                "val_f1_macro_ci95_low": f1_lo,
                "val_f1_macro_ci95_high": f1_hi,
                "val_acc_mean": float(np.mean(acc)),
                "val_acc_std": float(np.std(acc, ddof=1)) if len(acc) > 1 else 0.0,
                "val_acc_ci95_low": acc_lo,
                "val_acc_ci95_high": acc_hi,
                "train_seconds_mean": float(np.mean(train_s)),
            }
        )

    summary_rows.sort(key=lambda x: x["val_f1_macro_mean"], reverse=True)
    return summary_rows


def noninferiority_analysis(
    runs: list[dict[str, Any]],
    candidate_model: str,
    models: list[str],
    margin: float,
    bootstrap_samples: int,
    ci_alpha: float = 0.05,
) -> tuple[list[dict[str, Any]], bool]:
    rng = np.random.default_rng(1)
    rows: list[dict[str, Any]] = []

    candidate_runs = [r for r in runs if r["model"] == candidate_model]
    by_candidate_seed = {
        int(r["seed"]): float(r["val_f1_macro"])
        for r in candidate_runs
    }

    overall_pass = True
    for comparator in models:
        if comparator == candidate_model:
            continue

        comp_runs = [r for r in runs if r["model"] == comparator]
        by_comp_seed = {int(r["seed"]): float(r["val_f1_macro"]) for r in comp_runs}
        common_seeds = sorted(set(by_candidate_seed) & set(by_comp_seed))

        if not common_seeds:
            row = {
                "candidate_model": candidate_model,
                "comparator_model": comparator,
                "n_paired_seeds": 0,
                "margin": margin,
                "delta_mean": float("nan"),
                "delta_ci95_low": float("nan"),
                "delta_ci95_high": float("nan"),
                "p_bootstrap_delta_le_neg_margin": float("nan"),
                "noninferior": False,
                "superior": False,
            }
            rows.append(row)
            overall_pass = False
            continue

        deltas = np.asarray(
            [by_candidate_seed[s] - by_comp_seed[s] for s in common_seeds],
            dtype=np.float64,
        )

        delta_mean = float(np.mean(deltas))
        delta_lo, delta_hi = bootstrap_mean_ci(
            deltas, n_bootstrap=bootstrap_samples, alpha=ci_alpha, rng=rng
        )

        if len(deltas) == 1:
            p_boot = 1.0 if deltas[0] <= -margin else 0.0
        else:
            idx = rng.integers(0, len(deltas), size=(bootstrap_samples, len(deltas)))
            sample_means = deltas[idx].mean(axis=1)
            p_boot = float(np.mean(sample_means <= -margin))

        noninferior = bool(delta_lo > -margin)
        superior = bool(delta_lo > 0.0)
        overall_pass = overall_pass and noninferior

        rows.append(
            {
                "candidate_model": candidate_model,
                "comparator_model": comparator,
                "n_paired_seeds": int(len(common_seeds)),
                "margin": margin,
                "delta_mean": delta_mean,
                "delta_ci95_low": delta_lo,
                "delta_ci95_high": delta_hi,
                "p_bootstrap_delta_le_neg_margin": p_boot,
                "noninferior": noninferior,
                "superior": superior,
            }
        )

    return rows, overall_pass


def build_model(model_name: str, info: dict[str, Any], args: argparse.Namespace):
    if model_name == "trea_triple":
        return build_trea_triple(
            info=info,
            args=args,
            use_pre_patch_feature_attention=True,
            use_stat_tokens=False,
        )
    if model_name == "trea_triple_no_feature_attn":
        return build_trea_triple(
            info=info,
            args=args,
            use_pre_patch_feature_attention=False,
            use_stat_tokens=False,
        )
    if model_name == "trea_triple_stat_tokens":
        return build_trea_triple(
            info=info,
            args=args,
            use_pre_patch_feature_attention=True,
            use_stat_tokens=True,
        )
    if model_name == "trea_triple_stat_tokens_no_feature_attn":
        return build_trea_triple(
            info=info,
            args=args,
            use_pre_patch_feature_attention=False,
            use_stat_tokens=True,
        )
    if model_name == "patchtstnan":
        return build_patchtstnan(info=info, args=args)
    if model_name == "multidataset_none":
        return build_multidataset(
            info=info,
            args=args,
            column_embedding_strategy="none",
        )
    if model_name == "multidataset_auto":
        return build_multidataset(
            info=info,
            args=args,
            column_embedding_strategy="auto_expanding",
        )
    raise ValueError(f"Unsupported deep model name: {model_name}")


def run_deep_model_once(
    model_name: str,
    seed: int,
    info: dict[str, Any],
    dm: ThreeWDataModule,
    args: argparse.Namespace,
) -> dict[str, Any]:
    pl.seed_everything(seed, workers=True)
    dm.set_sampler_seed(seed)

    model = build_model(model_name=model_name, info=info, args=args)
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        precision=args.precision,
        logger=False,
        enable_progress_bar=args.show_progress,
        gradient_clip_val=args.gradient_clip_val,
        log_every_n_steps=args.log_every_n_steps,
        num_sanity_val_steps=0,
        limit_val_batches=0,
    )

    t0 = time.perf_counter()
    trainer.fit(model, datamodule=dm)
    train_seconds = time.perf_counter() - t0

    metrics = evaluate_torch_model(
        model=model,
        dataloader=dm.val_dataloader(),
        num_classes=info["num_classes"],
    )

    metrics["train_seconds"] = train_seconds
    metrics["num_params"] = int(sum(p.numel() for p in model.parameters()))
    return metrics


def build_report_markdown(
    summary_rows: list[dict[str, Any]],
    noninf_rows: list[dict[str, Any]],
    candidate_model: str,
    margin: float,
    overall_noninferior: bool,
) -> str:
    lines: list[str] = []
    lines.append("# 3W Non-Inferiority Benchmark Report")
    lines.append("")
    lines.append("## Summary (Validation)")
    lines.append("")
    lines.append(
        "| Model | Runs | Macro-F1 mean | Macro-F1 CI95 | "
        "Acc mean | Acc CI95 | Train sec/run |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            "| "
            f"{row['model']} | {row['n_runs']} | "
            f"{row['val_f1_macro_mean']:.4f} | "
            f"[{row['val_f1_macro_ci95_low']:.4f}, "
            f"{row['val_f1_macro_ci95_high']:.4f}] | "
            f"{row['val_acc_mean']:.4f} | "
            f"[{row['val_acc_ci95_low']:.4f}, {row['val_acc_ci95_high']:.4f}] | "
            f"{row['train_seconds_mean']:.1f} |"
        )

    lines.append("")
    lines.append(
        "## Non-Inferiority: candidate "
        f"`{candidate_model}` vs comparators (margin={margin:.4f})"
    )
    lines.append("")
    lines.append(
        "| Comparator | Paired seeds | Δ mean (candidate - comparator) | "
        "Δ CI95 | Non-inferior | Superior |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in noninf_rows:
        lines.append(
            "| "
            f"{row['comparator_model']} | {row['n_paired_seeds']} | "
            f"{row['delta_mean']:.4f} | "
            f"[{row['delta_ci95_low']:.4f}, {row['delta_ci95_high']:.4f}] | "
            f"{row['noninferior']} | {row['superior']} |"
        )

    lines.append("")
    lines.append(
        "Overall non-inferiority verdict for "
        f"`{candidate_model}` against all selected comparators: "
        f"`{overall_noninferior}`"
    )
    lines.append("")
    lines.append("Interpretation rule used:")
    lines.append(
        "- Candidate is non-inferior if the lower 95% CI bound "
        f"of Δ is greater than `-{margin:.4f}`."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    seeds = parse_csv_ints(args.seeds)
    models = parse_csv_models(args.models)
    if not seeds:
        raise ValueError("At least one seed must be provided in --seeds")

    if args.candidate_model not in models:
        raise ValueError(
            f"--candidate-model {args.candidate_model!r} must be in --models {models}"
        )

    torch.set_float32_matmul_precision("medium")

    print("=" * 80)
    print("3W Benchmark: Multi-Seed Non-Inferiority")
    print("=" * 80)
    print(f"Models: {models}")
    print(f"Candidate model: {args.candidate_model}")
    print(f"Seeds: {seeds}")
    print(f"Margin (macro-F1): {args.margin}")
    print(f"Data split seed: {args.data_split_seed}")

    # Load data once for a fixed split seed, then vary model init/sampler seeds.
    print("\nLoading 3W train split...")
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

    print("\nLoading 3W val split...")
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

    info = train_ds.get_feature_info()
    print(
        f"\nSchema: {info['n_numeric']} numeric, "
        f"{info['n_categorical']} categorical, classes={info['num_classes']}"
    )

    dm = ThreeWDataModule(
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampling_power=args.sampling_power,
        sampler_seed=seeds[0],
    )

    runs: list[dict[str, Any]] = []
    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "config.json"
    config_payload = {
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "args": vars(args),
        "models": models,
        "seeds": seeds,
        "dataset_info": {
            "n_numeric": info["n_numeric"],
            "n_categorical": info["n_categorical"],
            "num_classes": info["num_classes"],
            "sequence_length": info["sequence_length"],
            "column_names": info["column_names"],
            "num_train_samples": len(train_ds),
            "num_val_samples": len(val_ds),
        },
    }
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    for model_name in models:
        print("\n" + "-" * 80)
        print(f"Model: {model_name}")
        print("-" * 80)

        for seed in seeds:
            print(f"[{model_name}] seed={seed} ...")
            if model_name in DEEP_MODELS:
                metrics = run_deep_model_once(
                    model_name=model_name,
                    seed=seed,
                    info=info,
                    dm=dm,
                    args=args,
                )
            elif model_name == "rf_stat_features":
                metrics = run_rf_baseline(
                    train_ds=train_ds,
                    val_ds=val_ds,
                    num_classes=info["num_classes"],
                    seed=seed,
                    args=args,
                )
            else:
                raise ValueError(f"Unsupported model name: {model_name}")

            row: dict[str, Any] = {
                "model": model_name,
                "seed": seed,
                "val_acc": metrics["val_acc"],
                "val_f1_macro": metrics["val_f1_macro"],
                "train_seconds": metrics["train_seconds"],
                "num_params": metrics["num_params"],
                "num_train_samples": len(train_ds),
                "num_val_samples": len(val_ds),
                "val_f1_per_class_json": json.dumps(metrics["val_f1_per_class"]),
            }
            for class_idx, f1_value in enumerate(metrics["val_f1_per_class"]):
                row[f"f1_class_{class_idx}"] = f1_value

            runs.append(row)
            print(
                f"  val_f1_macro={row['val_f1_macro']:.4f} "
                f"val_acc={row['val_acc']:.4f} "
                f"train_seconds={row['train_seconds']:.1f}"
            )

    summary_rows = summarize_runs(
        runs=runs,
        models=models,
        bootstrap_samples=args.bootstrap_samples,
    )
    noninf_rows, overall_noninferior = noninferiority_analysis(
        runs=runs,
        candidate_model=args.candidate_model,
        models=models,
        margin=args.margin,
        bootstrap_samples=args.bootstrap_samples,
    )

    run_fields = [
        "model",
        "seed",
        "val_acc",
        "val_f1_macro",
        "train_seconds",
        "num_params",
        "num_train_samples",
        "num_val_samples",
        "val_f1_per_class_json",
    ] + [f"f1_class_{i}" for i in range(info["num_classes"])]
    summary_fields = [
        "model",
        "n_runs",
        "val_f1_macro_mean",
        "val_f1_macro_std",
        "val_f1_macro_ci95_low",
        "val_f1_macro_ci95_high",
        "val_acc_mean",
        "val_acc_std",
        "val_acc_ci95_low",
        "val_acc_ci95_high",
        "train_seconds_mean",
    ]
    noninf_fields = [
        "candidate_model",
        "comparator_model",
        "n_paired_seeds",
        "margin",
        "delta_mean",
        "delta_ci95_low",
        "delta_ci95_high",
        "p_bootstrap_delta_le_neg_margin",
        "noninferior",
        "superior",
    ]

    runs_path = out_dir / "runs.csv"
    summary_path = out_dir / "summary.csv"
    noninf_path = out_dir / "noninferiority.csv"
    report_path = out_dir / "report.md"

    write_csv(runs_path, runs, run_fields)
    write_csv(summary_path, summary_rows, summary_fields)
    write_csv(noninf_path, noninf_rows, noninf_fields)
    report_text = build_report_markdown(
        summary_rows=summary_rows,
        noninf_rows=noninf_rows,
        candidate_model=args.candidate_model,
        margin=args.margin,
        overall_noninferior=overall_noninferior,
    )
    report_path.write_text(report_text, encoding="utf-8")

    print("\n" + "=" * 80)
    print("Summary (sorted by macro-F1)")
    print("=" * 80)
    for row in summary_rows:
        print(
            f"{row['model']:<30} "
            f"F1={row['val_f1_macro_mean']:.4f} "
            f"CI95=[{row['val_f1_macro_ci95_low']:.4f},"
            f"{row['val_f1_macro_ci95_high']:.4f}] "
            f"Acc={row['val_acc_mean']:.4f}"
        )

    print("\n" + "=" * 80)
    print(
        f"Non-Inferiority vs candidate={args.candidate_model} "
        f"(margin={args.margin:.4f})"
    )
    print("=" * 80)
    for row in noninf_rows:
        print(
            f"{row['comparator_model']:<30} "
            f"Delta={row['delta_mean']:+.4f} "
            f"CI95=[{row['delta_ci95_low']:+.4f},{row['delta_ci95_high']:+.4f}] "
            f"noninferior={row['noninferior']} superior={row['superior']}"
        )

    print(
        "\nOverall non-inferiority verdict for "
        f"{args.candidate_model}: {overall_noninferior}"
    )
    print(f"\nSaved: {runs_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {noninf_path}")
    print(f"Saved: {report_path}")
    print(f"Saved: {config_path}")

    print("\nPer-class labels:")
    for i in range(info["num_classes"]):
        print(f"  {i}: {EVENT_NAMES.get(i, f'Class {i}')}")


if __name__ == "__main__":
    main()
