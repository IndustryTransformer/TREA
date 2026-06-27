"""Train TriplePatchTransformer on the 3W petroleum well anomaly detection dataset.

Uses the ThreeWToolkit to load raw parquet files with NaN values preserved,
letting TREA-C's triple-encoded attention handle missing values natively.

Usage:
    uv run python examples/train_3w.py

Prerequisites:
    Clone the 3W repo: git clone https://github.com/petrobras/3W.git
    Update TOOLKIT_PATH below to point to your clone.
"""

import sys


sys.path.insert(0, ".")

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchmetrics import Accuracy, F1Score

from treac.models import TriplePatchTransformer
from utils.three_w import ThreeWDataset


# ============================================================
# CONFIGURATION — Update these for your environment
# ============================================================
TOOLKIT_PATH = "/home/kailukowiak/Work/3W"
DATA_PATH = "/home/kailukowiak/Work/3W/dataset"

WINDOW_SIZE = 96  # Back to proven window size
STRIDE = 96  # Non-overlapping
RARE_CLASS_STRIDE = None  # Disabled — WeightedRandomSampler already handles balance
SAMPLING_POWER = 0.5  # 0.5=sqrt inverse-frequency (less aggressive than full balancing)
BATCH_SIZE = 256
MAX_EPOCHS = 150
D_MODEL = 256
N_HEAD = 8
NUM_LAYERS = 4
DROPOUT = 0.3  # Strong regularization to combat overfitting
WEIGHT_DECAY = 1e-2
PATCH_LEN = 16  # Default patch config (proven with T=96)
PATCH_STRIDE = 8
POOLING = "mean"
USE_PRE_PATCH_FEATURE_ATTENTION = True
FEATURE_ATTENTION_DIM = 32
FEATURE_ATTENTION_HEADS = 4
USE_STAT_TOKENS = True
# Wide semantic column identity: frozen text encoding of each sensor's
# description + a learned d_model projection, added to the per-feature tokens.
# Supersedes the old 1-D `use_column_embeddings` scalar channel.
USE_SEMANTIC_COLUMNS = True
LR = 3e-4  # Higher peak LR for OneCycleLR
VAL_FRACTION = 0.2
NUM_WORKERS = 4
MAX_FILES = None  # Set to e.g. 50 for quick debugging
SEED = 42
# ============================================================


class ConfusionMatrixCallback(Callback):
    """Log confusion matrix to TensorBoard at the end of each validation epoch."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.preds = []
        self.targets = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        with torch.no_grad():
            logits = pl_module(batch["x_num"], batch["x_cat"])
            preds = torch.argmax(logits, dim=1)
            self.preds.append(preds.cpu())
            self.targets.append(batch["y"].cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        from torchmetrics import ConfusionMatrix

        from utils.three_w import EVENT_NAMES

        all_preds = torch.cat(self.preds)
        all_targets = torch.cat(self.targets)
        self.preds.clear()
        self.targets.clear()

        cm_metric = ConfusionMatrix(task="multiclass", num_classes=self.num_classes)
        cm = cm_metric(all_preds, all_targets).numpy()

        # Per-class F1 from confusion matrix
        per_class_f1 = []
        for i in range(self.num_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            per_class_f1.append(f1)
            pl_module.log(f"val_f1_class_{i}", f1, on_epoch=True)

        # Print per-class summary every 5 epochs
        if trainer.current_epoch % 5 == 0:
            print(f"\n  Per-class metrics (epoch {trainer.current_epoch}):")
            print(
                f"    {'':>2}  {'Prec':>5}  {'Rec':>5}  {'F1':>5}  {'Support':>7}  Name"
            )
            for i, f1 in enumerate(per_class_f1):
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                name = EVENT_NAMES.get(i, f"Class {i}")
                support = cm[i, :].sum()
                print(
                    f"    {i:>2}  {prec:.3f}  {rec:.3f}  {f1:.3f}  {support:>7d}  {name}"
                )
            print(f"    Macro F1: {np.mean(per_class_f1):.3f}")

        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — Epoch {trainer.current_epoch}")

        if trainer.logger and hasattr(trainer.logger, "experiment"):
            trainer.logger.experiment.add_figure(
                "confusion_matrix", fig, global_step=trainer.global_step
            )
        plt.close(fig)


class GradientNormCallback(Callback):
    """Log gradient norms for monitoring training stability."""

    def on_after_backward(self, trainer, pl_module):
        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                total_norm += p.grad.detach().data.norm(2).item() ** 2
        total_norm = total_norm**0.5
        pl_module.log("grad_norm", total_norm, on_step=True, on_epoch=False)


class ThreeWDataModule(pl.LightningDataModule):
    """DataModule for the 3W dataset with tempered weighted sampling."""

    def __init__(
        self,
        train_dataset: ThreeWDataset,
        val_dataset: ThreeWDataset,
        batch_size: int = 256,
        num_workers: int = 4,
        sampling_power: float = 0.5,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sampling_power = sampling_power

    def train_dataloader(self):
        # Tempered inverse-frequency sampling to reduce train/val prior mismatch.
        # power=1.0 is full inverse-frequency; 0.5 is a milder correction.
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

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )


def main():
    pl.seed_everything(SEED)
    torch.set_float32_matmul_precision("medium")

    print("=" * 70)
    print("3W Well Anomaly Classification — TREA-C Triple-Encoded Attention")
    print("=" * 70)

    # --- Data ---
    print("\nLoading training data...")
    train_ds = ThreeWDataset(
        toolkit_path=TOOLKIT_PATH,
        data_path=DATA_PATH,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        rare_class_stride=RARE_CLASS_STRIDE,
        split="train",
        val_fraction=VAL_FRACTION,
        max_files=MAX_FILES,
        seed=SEED,
        augment=False,
    )

    print("\nLoading validation data...")
    val_ds = ThreeWDataset(
        toolkit_path=TOOLKIT_PATH,
        data_path=DATA_PATH,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        split="val",
        val_fraction=VAL_FRACTION,
        normalization_stats=train_ds.normalization_stats,
        max_files=MAX_FILES,
        seed=SEED,
    )

    dm = ThreeWDataModule(
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        sampling_power=SAMPLING_POWER,
    )

    info = train_ds.get_feature_info()
    print(
        f"\nFeatures: {info['n_numeric']} numeric, {info['n_categorical']} categorical"
    )
    print(f"Sequence length: {info['sequence_length']}")
    print(f"Classes: {info['num_classes']}")
    print(
        "Pre-patch feature attention: "
        f"{USE_PRE_PATCH_FEATURE_ATTENTION} "
        f"(dim={FEATURE_ATTENTION_DIM}, heads={FEATURE_ATTENTION_HEADS})"
    )
    print(f"Stat tokens: {USE_STAT_TOKENS}")
    print(f"Semantic columns: {USE_SEMANTIC_COLUMNS}")

    # --- Model ---
    model = TriplePatchTransformer(
        C_num=info["n_numeric"],
        C_cat=info["n_categorical"],
        cat_cardinalities=info["cat_cardinalities"],
        T=info["sequence_length"],
        d_model=D_MODEL,
        task="classification",
        num_classes=info["num_classes"],
        n_head=N_HEAD,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        pooling=POOLING,
        lr=LR,
        patch_len=PATCH_LEN,
        stride=PATCH_STRIDE,
        column_names=info["column_names"],
        use_column_embeddings=False,
        use_pre_patch_feature_attention=USE_PRE_PATCH_FEATURE_ATTENTION,
        feature_attention_dim=FEATURE_ATTENTION_DIM,
        feature_attention_heads=FEATURE_ATTENTION_HEADS,
        use_stat_tokens=USE_STAT_TOKENS,
        use_semantic_columns=USE_SEMANTIC_COLUMNS,
        column_descriptions=info["column_descriptions"],
    )

    # Plain CE loss — WeightedRandomSampler already handles class balance at
    # the batch level, so adding class weights to the loss over-corrects.
    model.loss_fn = nn.CrossEntropyLoss()
    print("\nUsing plain CrossEntropyLoss with tempered weighted sampling")

    # Override optimizer: stronger weight_decay + OneCycleLR for better generalization
    steps_per_epoch = len(train_ds) // BATCH_SIZE + 1

    def configure_optimizers(self=model):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=LR,
            steps_per_epoch=steps_per_epoch,
            epochs=MAX_EPOCHS,
            pct_start=0.1,  # 10% warmup
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    model.configure_optimizers = configure_optimizers

    # --- F1 metrics: attach to model and override validation_step ---
    nc = info["num_classes"]
    model.val_f1_macro = F1Score(task="multiclass", num_classes=nc, average="macro").to(
        model.device
    )
    model.val_f1_per_class = F1Score(
        task="multiclass", num_classes=nc, average="none"
    ).to(model.device)
    model.val_acc_metric = Accuracy(
        task="multiclass", num_classes=nc, average="macro"
    ).to(model.device)

    _orig_val_step = model.validation_step

    def validation_step_with_f1(batch, batch_idx):
        out = model(batch["x_num"], batch["x_cat"])
        loss = model.loss_fn(out, batch["y"])
        preds = torch.argmax(out, dim=1)

        model.val_f1_macro(preds, batch["y"])
        model.val_f1_per_class(preds, batch["y"])
        model.val_acc_metric(preds, batch["y"])

        model.log("val_loss", loss, prog_bar=True)
        model.log("val_acc", model.val_acc_metric, prog_bar=True, on_epoch=True)
        model.log("val_f1_macro", model.val_f1_macro, prog_bar=True, on_epoch=True)
        return loss

    model.validation_step = validation_step_with_f1

    # --- Callbacks ---
    callbacks = [
        ModelCheckpoint(
            dirpath="./checkpoints/3w",
            filename="3w-{epoch:02d}-{val_f1_macro:.4f}",
            monitor="val_f1_macro",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_f1_macro",
            patience=25,
            mode="max",
            verbose=True,
        ),
        ConfusionMatrixCallback(num_classes=nc),
        GradientNormCallback(),
    ]

    logger = TensorBoardLogger("logs", name="3w_classification")

    # --- Train ---
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=100,
        precision="16-mixed",
        enable_progress_bar=True,
    )

    print("\nStarting training...")
    trainer.fit(model, dm)

    print("\nTraining complete!")
    print(f"Best checkpoint: {callbacks[0].best_model_path}")


if __name__ == "__main__":
    main()
