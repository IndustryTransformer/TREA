"""Quick test of fixed TREA-C on ETTh1."""

import sys

from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch

from torch.utils.data import Dataset


sys.path.insert(0, str(Path(__file__).parent.parent))

from treac.models.triple_attention import TriplePatchTransformer
from utils.datamodule import TimeSeriesDataModule


class ETTh1Dataset(Dataset):
    """Simple ETTh1 dataset loader from CSV."""

    def __init__(
        self,
        data_dir: str,
        train: bool = True,
        sequence_length: int = 96,
        task: str = "classification",
        num_classes: int = 3,
    ):
        self.sequence_length = sequence_length
        self.task = task
        self.num_classes = num_classes

        # Load CSV
        import os

        csv_path = os.path.join(data_dir, "ETTh1.csv")
        df = pd.read_csv(csv_path)

        # Extract numeric features (exclude date column)
        feature_cols = [col for col in df.columns if col != "date"]
        data = df[feature_cols].values  # Shape: [T_total, C]

        # Use 80/20 train/val split
        split_idx = int(0.8 * len(data))
        data_split = data[:split_idx] if train else data[split_idx:]

        # Create sequences with sliding window
        self.x_num = []
        self.y = []

        for i in range(len(data_split) - sequence_length):
            seq = data_split[i : i + sequence_length]  # [T, C]
            self.x_num.append(torch.FloatTensor(seq).T)  # [C, T]

            # Create classification target based on next value trend
            next_val = data_split[i + sequence_length, 0]
            current_val = data_split[i + sequence_length - 1, 0]
            change = (next_val - current_val) / (abs(current_val) + 1e-8)

            if change < -0.01:
                label = 0  # Decrease
            elif change > 0.01:
                label = 2  # Increase
            else:
                label = 1  # Stable

            self.y.append(label)

        self.x_num = torch.stack(self.x_num)
        self.y = torch.LongTensor(self.y)

        # Store feature info
        self.C_num = len(feature_cols)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "x_num": self.x_num[idx],
            "x_cat": torch.zeros(0, self.sequence_length, dtype=torch.long),
            "y": self.y[idx],
        }


print("Loading ETTh1 dataset...")
train_dataset = ETTh1Dataset(data_dir="../data/etth1", train=True, sequence_length=96)
val_dataset = ETTh1Dataset(data_dir="../data/etth1", train=False, sequence_length=96)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")

# Create data module
dm = TimeSeriesDataModule(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    batch_size=64,
    num_workers=4,
)

# Test a single batch
sample_batch = next(iter(dm.train_dataloader()))
print("\nBatch shapes:")
print(f"  x_num: {sample_batch['x_num'].shape}")
print(f"  y: {sample_batch['y'].shape}")

# Create TREA-C model with patch-based processing
print("\nCreating TREA-C model...")
model = TriplePatchTransformer(
    C_num=train_dataset.C_num,
    C_cat=0,
    cat_cardinalities=[],
    T=96,
    d_model=128,
    n_head=8,
    num_layers=3,
    patch_len=16,
    stride=8,
    task="classification",
    num_classes=3,
    lr=1e-3,
)

print(f"Number of patches: {model.num_patches}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test forward pass
print("\nTesting forward pass...")
with torch.no_grad():
    output = model(sample_batch["x_num"], sample_batch["x_cat"])
    print(f"Output shape: {output.shape}")
    print(f"Output sample: {output[0]}")

# Train for 3 epochs
print("\nTraining for 3 epochs...")
trainer = pl.Trainer(
    max_epochs=3,
    accelerator="auto",
    devices=1,
    enable_progress_bar=True,
    log_every_n_steps=50,
)

trainer.fit(model, dm)

print("\n✓ TREA-C fixed and working!")
