"""
Test for overfitting on ETTh1 dataset by applying the "ultra-fast"
training configuration, including per-sequence normalization.

This script is designed to diagnose if the per-sequence normalization is the
primary cause of the overfitting observed in the W3 dataset training.
"""

import numpy as np
import pytorch_lightning as pl
import torch

from data.downloaders.etth1 import ETTh1Dataset
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from treac.models import PatchTSTNan


def per_sequence_collate_fn(batch):
    """
    Custom collate function to apply per-sequence normalization.
    This mimics the behavior of the W3StreamingDataset.
    """
    # Default collate to stack samples into a batch
    collated_batch = torch.utils.data.default_collate(batch)
    x_num = collated_batch["x_num"]  # Shape: [B, C, T]

    # Transpose to [B, T, C] for easier iteration
    x_num_permuted = x_num.permute(0, 2, 1)

    # Apply per-sequence normalization
    normalized_sequences = []
    for seq in x_num_permuted:
        # seq shape: [T, C]
        # Convert to numpy for normalization logic from W3 script
        numeric_values = seq.numpy()

        # Handle NaN/inf values
        numeric_values = np.nan_to_num(numeric_values, nan=0.0, posinf=0.0, neginf=0.0)

        # Per-feature normalization within the sequence
        numeric_values_norm = np.zeros_like(numeric_values)
        for col_idx in range(numeric_values.shape[1]):
            col_data = numeric_values[:, col_idx]
            mean_val = np.mean(col_data)
            std_val = np.std(col_data)
            if std_val > 1e-8:
                normalized = (col_data - mean_val) / std_val
                normalized = np.clip(normalized, -3, 3)
                numeric_values_norm[:, col_idx] = normalized
            else:
                numeric_values_norm[:, col_idx] = 0.0

        normalized_sequences.append(
            torch.tensor(numeric_values_norm, dtype=torch.float32)
        )

    # Stack and transpose back to [B, C, T]
    x_num_normalized = torch.stack(normalized_sequences).permute(0, 2, 1)

    collated_batch["x_num"] = x_num_normalized
    return collated_batch


class CustomDataModule(pl.LightningDataModule):
    """DataModule that uses the custom collate function."""

    def __init__(self, train_dataset, val_dataset, batch_size, num_workers):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=per_sequence_collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=per_sequence_collate_fn,
            pin_memory=True,
        )


def test_overfitting_on_etth1():
    """Test PatchTSTNan on ETTh1 with ultra-fast settings."""
    print("=" * 60)
    print("Testing for Overfitting on ETTh1 using Ultra-Fast Config")
    print("=" * 60)

    # Configuration from ultra_fast script
    SEQUENCE_LENGTH = 96  # Standard for ETTh1
    BATCH_SIZE = 256  # Larger batch size
    MAX_EPOCHS = 20  # More epochs to observe overfitting
    NUM_CLASSES = 3  # Dummy classes for ETTh1 classification task

    # Load dataset
    print("\nLoading ETTh1 dataset...")
    train_dataset = ETTh1Dataset(
        data_dir="./data/etth1",
        train=True,
        sequence_length=SEQUENCE_LENGTH,
        task="classification",
        num_classes=NUM_CLASSES,
        download=True,
    )
    val_dataset = ETTh1Dataset(
        data_dir="./data/etth1",
        train=False,
        sequence_length=SEQUENCE_LENGTH,
        task="classification",
        num_classes=NUM_CLASSES,
        download=False,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create data module with custom collate function
    dm = CustomDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
    )

    # Get data dimensions
    feature_info = train_dataset.get_feature_info()
    c_in = feature_info["n_numeric"]
    seq_len = feature_info["sequence_length"]
    num_classes = feature_info["num_classes"]

    print("\nDataset info:")
    print(f"- Input channels: {c_in}")
    print(f"- Sequence length: {seq_len}")
    print(f"- Number of classes: {num_classes}")

    # Create model with "ultra-fast" configuration
    print("\nCreating large PatchTSTNan model...")
    model = PatchTSTNan(
        c_in=c_in,
        seq_len=seq_len,
        num_classes=num_classes,
        patch_len=16,
        stride=8,
        d_model=128,  # Larger model dimension
        n_head=8,  # More attention heads
        num_layers=4,  # Deeper model
        dropout=0.15,  # Low dropout
        lr=5e-4,  # High learning rate
        task="classification",
    )

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {model_params:,}")

    # Create trainer with "ultra-fast" settings
    print(f"\nStarting training for {MAX_EPOCHS} epochs...")
    logger = TensorBoardLogger("tb_logs", name="test_overfitting_etth1")
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices=1,
        logger=logger,
        precision="16-mixed",
        enable_progress_bar=True,
        log_every_n_steps=10,
        val_check_interval=1.0,  # Validate once per epoch
    )

    # Train model
    try:
        trainer.fit(model, dm)
        print("\nTraining completed successfully!")
        if trainer.callback_metrics:
            print("Final metrics:")
            for key, value in trainer.callback_metrics.items():
                if isinstance(value, torch.Tensor):
                    print(f"- {key}: {value.item():.4f}")
                else:
                    print(f"- {key}: {value}")

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_overfitting_on_etth1()
