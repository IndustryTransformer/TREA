"""Compare 3W fine-tuning from pretrained weights vs from scratch.

This script trains `MultiDatasetModel` on 3W twice:
1. Encoder initialized from a causal pretraining checkpoint.
2. Same architecture initialized from scratch.

It reports validation accuracy and macro-F1 for both runs.

Usage:
    uv run python examples/finetune_3w_pretrain_compare.py \
      --pretrained-ckpt checkpoints/pretrain_multi_dataset_causal/last.ckpt
"""

from __future__ import annotations

import argparse
import sys

from pathlib import Path
from typing import Any


sys.path.insert(0, ".")

import pytorch_lightning as pl
import torch

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, WeightedRandomSampler

from trea.models import MultiDatasetModel
from utils.three_w import EVENT_NAMES, ThreeWDataset


class ThreeWDataModule(pl.LightningDataModule):
    """DataModule for 3W with tempered weighted sampling."""

    def __init__(
        self,
        train_dataset: ThreeWDataset,
        val_dataset: ThreeWDataset,
        batch_size: int,
        num_workers: int,
        sampling_power: float,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sampling_power = sampling_power

    def train_dataloader(self) -> DataLoader:
        class_counts = torch.from_numpy(self.train_dataset.class_counts).double()
        class_weights = torch.zeros_like(class_counts)
        present = class_counts > 0
        class_weights[present] = class_counts[present].pow(-self.sampling_power)
        sample_weights = class_weights[
            torch.from_numpy(self.train_dataset.labels).long()
        ]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
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
        description="Fine-tune 3W: pretrained vs scratch comparison"
    )
    parser.add_argument(
        "--pretrained-ckpt",
        type=str,
        default="checkpoints/pretrain_multi_dataset_causal/last.ckpt",
    )
    parser.add_argument("--toolkit-path", type=str, default="/home/kailukowiak/Work/3W")
    parser.add_argument(
        "--data-path", type=str, default="/home/kailukowiak/Work/3W/dataset"
    )
    parser.add_argument("--window-size", type=int, default=96)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sampling-power", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/finetune_3w_pretrain_compare",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="checkpoints/finetune_3w_pretrain_compare",
    )
    return parser.parse_args()


def _macro_f1_from_predictions(
    preds: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> tuple[float, list[float]]:
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(targets.tolist(), preds.tolist(), strict=False):
        cm[t, p] += 1

    per_class_f1 = []
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
        per_class_f1.append(f1)

    macro_f1 = float(sum(per_class_f1) / len(per_class_f1))
    return macro_f1, per_class_f1


@torch.no_grad()
def evaluate_model(
    model: MultiDatasetModel, dataloader: DataLoader, num_classes: int
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
    macro_f1, per_class_f1 = _macro_f1_from_predictions(preds, targets, num_classes)
    return {
        "val_acc": acc,
        "val_f1_macro": macro_f1,
        "val_f1_per_class": per_class_f1,
    }


def _load_pretrain_hparams(pretrained_ckpt: str) -> dict[str, Any]:
    ckpt = torch.load(pretrained_ckpt, map_location="cpu")
    if "hyper_parameters" not in ckpt:
        raise RuntimeError(f"No hyper_parameters in checkpoint: {pretrained_ckpt}")
    return ckpt["hyper_parameters"]


def _set_finetune_lr(model: MultiDatasetModel, lr: float) -> None:
    # configure_optimizers reads self.hparams.lr
    model.hparams.lr = lr


def _build_scratch_model(
    info: dict[str, Any], pretrain_hparams: dict[str, Any], lr: float
) -> MultiDatasetModel:
    model = MultiDatasetModel(
        max_numeric_features=info["n_numeric"],
        max_categorical_features=0,
        num_classes=info["num_classes"],
        mode="variable_features",
        task="classification",
        patch_len=int(pretrain_hparams.get("patch_len", 16)),
        stride=int(pretrain_hparams.get("stride", 8)),
        d_model=int(pretrain_hparams.get("d_model", 192)),
        n_head=int(pretrain_hparams.get("n_head", 8)),
        num_layers=int(pretrain_hparams.get("num_layers", 4)),
        max_sequence_length=int(info["sequence_length"]),
        lr=lr,
        column_embedding_strategy="none",
        use_feature_masks=bool(pretrain_hparams.get("use_feature_masks", True)),
        categorical_cardinalities=[],
    )
    model.set_dataset_schema(
        numeric_features=info["n_numeric"],
        categorical_features=0,
        column_names=info["column_names"],
    )
    return model


def _build_pretrained_model(
    pretrained_ckpt: str, info: dict[str, Any], lr: float, freeze_encoder: bool
) -> MultiDatasetModel:
    model = MultiDatasetModel.from_pretrained(
        pretrained_path=pretrained_ckpt,
        num_classes=info["num_classes"],
        freeze_encoder=freeze_encoder,
    )
    _set_finetune_lr(model, lr)
    model.set_dataset_schema(
        numeric_features=info["n_numeric"],
        categorical_features=0,
        column_names=info["column_names"],
    )
    return model


def train_and_eval(
    run_name: str,
    model: MultiDatasetModel,
    dm: ThreeWDataModule,
    args: argparse.Namespace,
    num_classes: int,
) -> dict[str, Any]:
    logger = TensorBoardLogger(
        save_dir=args.log_dir, name="3w_compare", version=run_name
    )
    ckpt_cb = ModelCheckpoint(
        dirpath=Path(args.ckpt_dir) / run_name,
        filename="epoch-{epoch:02d}",
        save_last=True,
        save_top_k=-1,
        every_n_epochs=1,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        logger=logger,
        callbacks=[ckpt_cb],
        log_every_n_steps=50,
        gradient_clip_val=1.0,
        num_sanity_val_steps=0,
        limit_val_batches=0,
    )

    trainer.fit(model, dm)

    eval_ckpt = ckpt_cb.last_model_path
    if eval_ckpt:
        state = torch.load(eval_ckpt, map_location="cpu")
        model.load_state_dict(state["state_dict"], strict=True)
    else:
        eval_ckpt = "<no-checkpoint>"

    metrics = evaluate_model(model, dm.val_dataloader(), num_classes=num_classes)
    metrics["checkpoint"] = eval_ckpt
    return metrics


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")

    pretrained_path = Path(args.pretrained_ckpt)
    if not pretrained_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")

    print("=" * 72)
    print("3W Fine-Tuning Comparison: Pretrained vs Scratch (MultiDatasetModel)")
    print("=" * 72)

    print("\nLoading 3W train split...")
    train_ds = ThreeWDataset(
        toolkit_path=args.toolkit_path,
        data_path=args.data_path,
        window_size=args.window_size,
        stride=args.stride,
        split="train",
        val_fraction=args.val_fraction,
        max_files=args.max_files,
        seed=args.seed,
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
        seed=args.seed,
        augment=False,
    )

    dm = ThreeWDataModule(
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampling_power=args.sampling_power,
    )

    info = train_ds.get_feature_info()
    print(
        f"\nSchema: {info['n_numeric']} numeric, "
        f"{info['n_categorical']} categorical, classes={info['num_classes']}"
    )
    print(f"Using pretrained checkpoint: {pretrained_path}")

    pretrain_hparams = _load_pretrain_hparams(str(pretrained_path))

    print("\n[Run 1/2] Fine-tune from pretrained encoder")
    pretrained_model = _build_pretrained_model(
        pretrained_ckpt=str(pretrained_path),
        info=info,
        lr=args.lr,
        freeze_encoder=args.freeze_encoder,
    )
    pretrained_metrics = train_and_eval(
        run_name="from_pretrained",
        model=pretrained_model,
        dm=dm,
        args=args,
        num_classes=info["num_classes"],
    )

    print("\n[Run 2/2] Train from scratch (same architecture)")
    scratch_model = _build_scratch_model(
        info=info,
        pretrain_hparams=pretrain_hparams,
        lr=args.lr,
    )
    scratch_metrics = train_and_eval(
        run_name="from_scratch",
        model=scratch_model,
        dm=dm,
        args=args,
        num_classes=info["num_classes"],
    )

    print("\n" + "=" * 72)
    print("Validation Comparison")
    print("=" * 72)
    print(
        f"Pretrained: val_acc={pretrained_metrics['val_acc']:.4f} "
        f"val_f1_macro={pretrained_metrics['val_f1_macro']:.4f}"
    )
    print(
        f"Scratch:    val_acc={scratch_metrics['val_acc']:.4f} "
        f"val_f1_macro={scratch_metrics['val_f1_macro']:.4f}"
    )

    delta = pretrained_metrics["val_f1_macro"] - scratch_metrics["val_f1_macro"]
    print(f"Delta macro-F1 (pretrained - scratch): {delta:+.4f}")
    print(f"Pretrained checkpoint: {pretrained_metrics['checkpoint']}")
    print(f"Scratch checkpoint:    {scratch_metrics['checkpoint']}")

    print("\nPer-class F1 (pretrained):")
    for i, f1 in enumerate(pretrained_metrics["val_f1_per_class"]):
        name = EVENT_NAMES.get(i, f"Class {i}")
        print(f"  {i:>2} {name:<32} {f1:.4f}")

    print("\nPer-class F1 (scratch):")
    for i, f1 in enumerate(scratch_metrics["val_f1_per_class"]):
        name = EVENT_NAMES.get(i, f"Class {i}")
        print(f"  {i:>2} {name:<32} {f1:.4f}")


if __name__ == "__main__":
    main()
