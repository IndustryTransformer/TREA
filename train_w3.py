"""
Training script for W3 dataset with real wells only.

Usage:
    uv run python train_w3.py
"""

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn

from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchmetrics import Accuracy, ConfusionMatrix, F1Score

from treac.models import TriplePatchTransformer


class W3Dataset(Dataset):
    """Dataset for W3 well data with well-level splitting support."""

    def __init__(
        self,
        parquet_path: str,
        seq_len: int = 96,
        filter_simulated: bool = True,
        max_samples: int | None = None,
        train_wells: set | None = None,
        normalization_stats: dict | None = None,
    ):
        """
        Args:
            parquet_path: Path to parquet file
            seq_len: Sequence length for time series windows
            filter_simulated: If True, exclude SIMULATED wells
            max_samples: Maximum number of samples to load (for memory management)
            train_wells: Set of well names to include (for well-level splitting)
            normalization_stats: Pre-computed normalization stats (mean, std) from
            training set
        """
        self.seq_len = seq_len

        # Load data
        print(f"Loading data from {parquet_path}...")
        if filter_simulated:
            # Load in chunks to filter efficiently
            df = pd.read_parquet(parquet_path)
            print(f"Original size: {len(df):,} rows")
            df = df[df["well_name"] != "SIMULATED"]
            print(f"After filtering SIMULATED: {len(df):,} rows")
        else:
            df = pd.read_parquet(parquet_path)

        # Filter by wells if specified (for well-level train/val split)
        if train_wells is not None:
            print(f"Filtering to {len(train_wells)} specified wells...")
            df = df[df["well_name"].isin(train_wells)].reset_index(drop=True)
            print(f"After well filtering: {len(df):,} rows")

        # Limit samples if specified
        if max_samples is not None and len(df) > max_samples:
            print(f"Limiting to {max_samples:,} samples")
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

        # Store well names for later use
        self.well_names = df["well_name"].values

        # Feature columns (27 numeric features)
        self.feature_cols = [
            "ABER-CKGL",
            "ABER-CKP",
            "ESTADO-DHSV",
            "ESTADO-M1",
            "ESTADO-M2",
            "ESTADO-PXO",
            "ESTADO-SDV-GL",
            "ESTADO-SDV-P",
            "ESTADO-W1",
            "ESTADO-W2",
            "ESTADO-XO",
            "P-ANULAR",
            "P-JUS-BS",
            "P-JUS-CKGL",
            "P-JUS-CKP",
            "P-MON-CKGL",
            "P-MON-CKP",
            "P-MON-SDV-P",
            "P-PDG",
            "PT-P",
            "P-TPT",
            "QBS",
            "QGL",
            "T-JUS-CKP",
            "T-MON-CKP",
            "T-PDG",
            "T-TPT",
        ]

        # Extract features and labels
        # First, handle extreme values that cause overflow
        features_raw = df[self.feature_cols].values

        # Clip extreme values before converting to float32
        features_raw = np.clip(features_raw, -1e6, 1e6)
        self.features = features_raw.astype(np.float32)

        self.labels = df["class"].values

        # Handle missing labels
        self.valid_indices = ~pd.isna(self.labels)
        self.features = self.features[self.valid_indices]
        self.labels = self.labels[self.valid_indices].astype(np.int64)

        print(f"Final dataset size: {len(self.labels):,} samples")
        print(f"Features shape: {self.features.shape}")

        # Better class distribution printing for sparse labels
        unique_classes, counts = np.unique(self.labels, return_counts=True)
        print("\nOriginal class distribution:")
        for cls, count in zip(unique_classes, counts, strict=False):
            print(f"  Class {cls}: {count:,} samples")

        # Create label mapping: classes 0-9 stay the same, 101-109 map to 10-18
        # This ensures labels are contiguous from 0 to num_classes-1
        self.label_mapping = {}
        remapped_idx = 0
        for cls in sorted(unique_classes):
            self.label_mapping[cls] = remapped_idx
            remapped_idx += 1

        # Apply label mapping
        self.labels = np.array(
            [self.label_mapping[label] for label in self.labels], dtype=np.int64
        )

        print("\nRemapped class distribution:")
        unique_remapped, counts_remapped = np.unique(self.labels, return_counts=True)
        for cls, orig_cls, count in zip(
            unique_remapped, sorted(unique_classes), counts_remapped, strict=False
        ):
            print(f"  Class {cls} (originally {orig_cls}): {count:,} samples")

        self.num_classes = len(unique_classes)
        print(f"\nTotal unique classes: {self.num_classes}")

        # Compute class weights for imbalanced dataset
        # Use inverse frequency: weight = 1 / (class_count / total_count)
        total_samples = len(self.labels)
        self.class_weights = np.zeros(self.num_classes, dtype=np.float32)
        for remapped_cls, count in zip(unique_remapped, counts_remapped, strict=False):
            self.class_weights[remapped_cls] = total_samples / (
                count * self.num_classes
            )

        print("\nClass weights (for loss function):")
        for cls, orig_cls, weight in zip(
            unique_remapped, sorted(unique_classes), self.class_weights, strict=False
        ):
            print(f"  Class {cls} (originally {orig_cls}): weight = {weight:.4f}")

        # Compute or use provided normalization statistics
        if normalization_stats is not None:
            # Use pre-computed stats from training set
            print("\nUsing pre-computed normalization statistics from training set")
            self.mean = normalization_stats["mean"]
            self.std = normalization_stats["std"]
        else:
            # Compute normalization statistics on this dataset (training set only)
            print("\nComputing normalization statistics...")

            # Calculate NaN rates per column
            nan_rates = np.isnan(self.features).mean(axis=0)
            print("\nNaN rates per feature:")
            for i, (col, rate) in enumerate(
                zip(self.feature_cols, nan_rates, strict=False)
            ):
                if rate > 0:
                    print(f"  {col}: {rate * 100:.2f}% NaN")

            # Identify columns that are all or mostly NaN
            high_nan_cols = [
                col
                for col, rate in zip(self.feature_cols, nan_rates, strict=False)
                if rate > 0.95
            ]
            if high_nan_cols:
                print(f"\nWARNING: {len(high_nan_cols)} columns have >95% NaN values:")
                for col in high_nan_cols:
                    print(f"  - {col}")

            # Compute mean and std
            self.mean = np.nanmean(self.features, axis=0)
            self.std = np.nanstd(self.features, axis=0)

            # Replace NaN mean/std with 0 and 1 respectively
            self.mean = np.nan_to_num(self.mean, nan=0.0)
            self.std = np.nan_to_num(self.std, nan=1.0)
            self.std[self.std == 0] = 1.0  # Avoid division by zero

    def __len__(self):
        # Number of possible windows
        return max(0, len(self.features) - self.seq_len + 1)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        # Extract sequence window
        seq = self.features[idx : idx + self.seq_len]

        # Normalize (keep NaNs so TriplePatchTransformer can create mask channels)
        seq_norm = (seq - self.mean) / self.std

        # IMPORTANT: Do NOT call np.nan_to_num here!
        # TriplePatchTransformer needs real NaNs to form the mask/value dual patches
        # The model will handle NaNs internally by creating separate mask channels

        # Label for the window (use label at the end of sequence)
        label = self.labels[idx + self.seq_len - 1]

        return torch.from_numpy(seq_norm), torch.tensor(label, dtype=torch.long)


class W3DataModule(pl.LightningDataModule):
    """DataModule for W3 dataset."""

    def __init__(
        self,
        train_path: str = "./data/W3/train.parquet",
        test_path: str = "./data/W3/test.parquet",
        seq_len: int = 96,
        batch_size: int = 256,
        num_workers: int = 4,
        max_train_samples: int | None = None,
        val_split: float = 0.1,
    ):
        super().__init__()
        self.train_path = train_path
        self.test_path = test_path
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_train_samples = max_train_samples
        self.val_split = val_split

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            # Load data to get well names for splitting
            print("Loading data for well-level train/val split...")
            df = pd.read_parquet(self.train_path)
            df = df[df["well_name"] != "SIMULATED"]

            # Limit samples if specified (sample wells, not individual rows)
            if self.max_train_samples is not None and len(df) > self.max_train_samples:
                # Sample wells proportionally to keep class distribution
                unique_wells = df["well_name"].unique()
                n_wells_to_keep = max(
                    10, int(len(unique_wells) * (self.max_train_samples / len(df)))
                )
                selected_wells = np.random.choice(
                    unique_wells, size=n_wells_to_keep, replace=False
                )
                df = df[df["well_name"].isin(selected_wells)]
                print(
                    f"Sampled {len(selected_wells)} wells, resulting in {len(df):,} samples"
                )

            # Get unique wells and split at well level
            unique_wells = df["well_name"].unique()
            print(f"\nTotal unique wells: {len(unique_wells)}")

            # Shuffle and split wells
            np.random.seed(42)
            np.random.shuffle(unique_wells)
            n_val_wells = max(2, int(len(unique_wells) * self.val_split))
            val_wells = set(unique_wells[:n_val_wells])
            train_wells = set(unique_wells[n_val_wells:])

            print(f"Train wells: {len(train_wells)}")
            print(f"Val wells: {len(val_wells)}")

            # Create training dataset first (to compute normalization stats)
            self.train_dataset = W3Dataset(
                self.train_path,
                seq_len=self.seq_len,
                filter_simulated=True,
                max_samples=None,  # Already limited by well selection
                train_wells=train_wells,
                normalization_stats=None,  # Will compute
            )

            # Get normalization stats from training set
            normalization_stats = {
                "mean": self.train_dataset.mean,
                "std": self.train_dataset.std,
            }

            # Create validation dataset using training stats
            self.val_dataset = W3Dataset(
                self.train_path,
                seq_len=self.seq_len,
                filter_simulated=True,
                max_samples=None,
                train_wells=val_wells,
                normalization_stats=normalization_stats,
            )

            print("\nFinal dataset sizes:")
            print(f"Train windows: {len(self.train_dataset):,}")
            print(f"Val windows: {len(self.val_dataset):,}")

        if stage == "test" or stage is None:
            self.test_dataset = W3Dataset(
                self.test_path,
                seq_len=self.seq_len,
                filter_simulated=False,  # Test set doesn't have SIMULATED
            )
            print(f"Test size: {len(self.test_dataset):,}")

    def train_dataloader(self):
        # Create weighted sampler for better class balance
        # Each sample gets weight based on its class (inverse frequency)
        print("\nCreating weighted sampler for balanced training...")

        # Get labels for all windows in the dataset
        # Access the underlying dataset if it's wrapped
        if hasattr(self.train_dataset, "dataset"):
            base_dataset = self.train_dataset.dataset
        else:
            base_dataset = self.train_dataset

        # Get class for each window (label at end of each sequence)
        sample_weights = []
        for idx in range(len(self.train_dataset)):
            # Get the window's label
            window_start = idx
            label_idx = window_start + self.seq_len - 1
            if label_idx < len(base_dataset.labels):
                label = base_dataset.labels[label_idx]
                weight = base_dataset.class_weights[label]
                sample_weights.append(weight)
            else:
                # Shouldn't happen, but handle gracefully
                sample_weights.append(1.0)

        sample_weights = torch.DoubleTensor(sample_weights)

        # Create sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,  # Allow sampling with replacement for rare classes
        )

        print(f"Weighted sampler created with {len(sample_weights):,} samples")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,  # Use sampler instead of shuffle
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True,
        )


class GradientNormCallback(Callback):
    """Callback to log gradient norms for debugging training dynamics."""

    def on_after_backward(self, trainer, pl_module):
        """Log gradient norms after backward pass."""
        # Calculate total gradient norm
        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5

        # Log to tensorboard
        pl_module.log("grad_norm", total_norm, on_step=True, on_epoch=False)


class ConfusionMatrixCallback(Callback):
    """Callback to log confusion matrix at the end of each validation epoch."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.confusion_matrix = ConfusionMatrix(
            task="multiclass", num_classes=num_classes
        )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Update confusion matrix with validation batch predictions."""
        x, y = batch
        logits = pl_module(x)
        preds = torch.argmax(logits, dim=1)
        # Move to CPU to avoid device mismatch
        self.confusion_matrix.update(preds.cpu(), y.cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log confusion matrix as an image to TensorBoard."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Compute confusion matrix
        cm = self.confusion_matrix.compute().cpu().numpy()
        self.confusion_matrix.reset()

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=True)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix - Epoch {trainer.current_epoch}")

        # Log to tensorboard
        trainer.logger.experiment.add_figure(
            "confusion_matrix", fig, global_step=trainer.global_step
        )
        plt.close(fig)


class W3Model(pl.LightningModule):
    """Lightning wrapper for W3 classification."""

    def __init__(
        self,
        c_in: int = 27,
        seq_len: int = 96,
        num_classes: int = 17,
        d_model: int = 128,
        n_head: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        pooling: str = "mean",
        lr: float = 1e-4,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        # Create model
        self.model = TriplePatchTransformer(
            C_num=c_in,
            C_cat=0,  # No categorical features
            cat_cardinalities=[],
            T=seq_len,
            d_model=d_model,
            task="classification",
            num_classes=num_classes,
            n_head=n_head,
            num_layers=num_layers,
            dropout=dropout,
            pooling=pooling,
            lr=lr,
        )

        # Loss function with class weights for imbalanced data
        if class_weights is not None:
            print("\nUsing weighted loss with class weights")
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            print("\nUsing unweighted loss")
            self.criterion = nn.CrossEntropyLoss()

        # Metrics for evaluation
        self.val_acc = Accuracy(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_per_class_acc = Accuracy(
            task="multiclass", num_classes=num_classes, average="none"
        )
        self.val_per_class_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="none"
        )

        self.test_acc = Accuracy(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_per_class_acc = Accuracy(
            task="multiclass", num_classes=num_classes, average="none"
        )
        self.test_per_class_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="none"
        )

    def forward(self, x):
        # TriplePatchTransformer expects (x_num, x_cat)
        # x has shape [B, T, C], need to transpose to [B, C, T]
        x_num = x.transpose(1, 2)
        x_cat = torch.zeros(
            x.shape[0], 0, x.shape[1], dtype=torch.long, device=x.device
        )
        return self.model(x_num, x_cat)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)

        # Update metrics
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.val_per_class_acc(preds, y)
        self.val_per_class_f1(preds, y)

        # Log macro metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc_macro", self.val_acc, prog_bar=True, on_epoch=True)
        self.log("val_f1_macro", self.val_f1, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        """Log per-class metrics at the end of validation epoch."""
        per_class_acc = self.val_per_class_acc.compute()
        per_class_f1 = self.val_per_class_f1.compute()

        # Log per-class metrics
        for i in range(len(per_class_acc)):
            self.log(f"val_acc_class_{i}", per_class_acc[i], on_epoch=True)
            self.log(f"val_f1_class_{i}", per_class_f1[i], on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)

        # Update metrics
        self.test_acc(preds, y)
        self.test_f1(preds, y)
        self.test_per_class_acc(preds, y)
        self.test_per_class_f1(preds, y)

        # Log macro metrics
        self.log("test_loss", loss)
        self.log("test_acc_macro", self.test_acc, on_epoch=True)
        self.log("test_f1_macro", self.test_f1, on_epoch=True)

        return loss

    def on_test_epoch_end(self):
        """Log per-class metrics at the end of test epoch."""
        per_class_acc = self.test_per_class_acc.compute()
        per_class_f1 = self.test_per_class_f1.compute()

        # Log per-class metrics
        for i in range(len(per_class_acc)):
            self.log(f"test_acc_class_{i}", per_class_acc[i], on_epoch=True)
            self.log(f"test_f1_class_{i}", per_class_f1[i], on_epoch=True)

        # Print summary to console
        print("\nPer-class Test Results:")
        print("=" * 60)
        for i in range(len(per_class_acc)):
            print(f"  Class {i}: Acc={per_class_acc[i]:.4f}, F1={per_class_f1[i]:.4f}")
        print("=" * 60)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


def main():
    # Set random seeds
    pl.seed_everything(42)

    # Set matmul precision for better performance on CUDA devices with Tensor Cores
    torch.set_float32_matmul_precision("medium")

    # Configuration
    CONFIG = {
        "seq_len": 96,
        "batch_size": 256,
        "num_workers": 8,
        "max_train_samples": 5_000_000,  # Limit to 5M samples (adjust as needed)
        "val_split": 0.1,
        "max_epochs": 50,
        "d_model": 128,
        "n_head": 8,
        "num_layers": 3,
        "dropout": 0.1,
        "pooling": "mean",
        "lr": 1e-4,
    }

    print("=" * 80)
    print("W3 Well Classification Training")
    print("=" * 80)
    print("\nConfiguration:")
    for key, val in CONFIG.items():
        print(f"  {key}: {val}")
    print()

    # Create data module
    datamodule = W3DataModule(
        seq_len=CONFIG["seq_len"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
        max_train_samples=CONFIG["max_train_samples"],
        val_split=CONFIG["val_split"],
    )

    # Setup to get the number of classes from the dataset
    datamodule.setup(stage="fit")

    # Get num_classes and class_weights from the dataset
    if hasattr(datamodule.train_dataset, "dataset"):
        # Handle RandomSplit wrapper
        train_dataset = datamodule.train_dataset.dataset
    else:
        train_dataset = datamodule.train_dataset

    num_classes = train_dataset.num_classes
    class_weights = torch.from_numpy(train_dataset.class_weights).float()

    print(f"\nDetected {num_classes} classes in dataset")

    # Create model
    model = W3Model(
        c_in=27,
        seq_len=CONFIG["seq_len"],
        num_classes=num_classes,
        d_model=CONFIG["d_model"],
        n_head=CONFIG["n_head"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
        pooling=CONFIG["pooling"],
        lr=CONFIG["lr"],
        class_weights=class_weights,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints/w3",
        filename="w3-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        verbose=True,
    )

    confusion_matrix_callback = ConfusionMatrixCallback(num_classes=num_classes)
    gradient_norm_callback = GradientNormCallback()

    # Logger
    logger = TensorBoardLogger("logs", name="w3_classification")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=CONFIG["max_epochs"],
        accelerator="auto",
        devices=1,
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            confusion_matrix_callback,
            gradient_norm_callback,
        ],
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=100,
        precision="16-mixed",  # Use mixed precision for faster training
    )

    # Train
    print("\nStarting training...")
    trainer.fit(model, datamodule)

    # Test
    print("\nRunning test evaluation...")
    trainer.test(model, datamodule, ckpt_path="best")

    print("\nTraining complete!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
