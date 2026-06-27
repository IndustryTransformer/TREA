"""Train and compare models on NASA Turbofan dataset."""

import sys
import time

from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


sys.path.insert(0, str(Path(__file__).parent.parent))

from treac.models.triple_attention import TriplePatchTransformer
from utils.datamodule import TimeSeriesDataModule


# Try to import turbofan dataset, but don't fail if it doesn't exist
try:
    from data.downloaders.nasa_turbofan import NASATurbofanDataset

    HAS_TURBOFAN_LOADER = True
except (ImportError, ModuleNotFoundError):
    HAS_TURBOFAN_LOADER = False


class PatchTSTClassifier(pl.LightningModule):
    """PatchTST adapted for classification."""

    def __init__(
        self,
        c_in: int,
        seq_len: int,
        num_classes: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_head: int = 8,
        num_layers: int = 3,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1

        # Patch embedding
        self.patch_embedding = nn.Linear(patch_len * c_in, d_model)

        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.head = nn.Linear(d_model, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def create_patches(self, x):
        """Create patches from input tensor."""
        B, C, T = x.shape
        patches = []

        for i in range(0, T - self.patch_len + 1, self.stride):
            patch = x[:, :, i : i + self.patch_len]  # [B, C, patch_len]
            patch = patch.reshape(B, -1)  # [B, C * patch_len]
            patches.append(patch)

        patches = torch.stack(patches, dim=1)  # [B, num_patches, C * patch_len]
        return patches

    def forward(self, x_num, x_cat=None):
        patches = self.create_patches(x_num)

        # Embed patches
        x = self.patch_embedding(patches)
        x = x + self.pos_embedding

        # Transformer
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)  # [B, d_model]

        # Classification
        return self.head(x)

    def training_step(self, batch, batch_idx):
        out = self(batch["x_num"], batch.get("x_cat"))
        loss = self.loss_fn(out, batch["y"])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch["x_num"], batch.get("x_cat"))
        loss = self.loss_fn(out, batch["y"])
        self.log("val_loss", loss)

        preds = torch.argmax(out, dim=1)
        acc = (preds == batch["y"]).float().mean()
        self.log("val_acc", acc)

        return {"val_loss": loss, "preds": preds, "labels": batch["y"]}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class CNNClassifier(pl.LightningModule):
    """1D CNN baseline for time series classification."""

    def __init__(
        self,
        c_in: int,
        seq_len: int,
        num_classes: int,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(c_in, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )

        # Calculate output size
        with torch.no_grad():
            dummy = torch.zeros(1, c_in, seq_len)
            out = self.feature_extractor(dummy)
            flat_size = out.shape[1] * out.shape[2]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x_num, x_cat=None):
        x = self.feature_extractor(x_num)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        out = self(batch["x_num"], batch.get("x_cat"))
        loss = self.loss_fn(out, batch["y"])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch["x_num"], batch.get("x_cat"))
        loss = self.loss_fn(out, batch["y"])
        self.log("val_loss", loss)

        preds = torch.argmax(out, dim=1)
        acc = (preds == batch["y"]).float().mean()
        self.log("val_acc", acc)

        return {"val_loss": loss, "preds": preds, "labels": batch["y"]}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def train_model(model, dm, model_name, max_epochs=30):
    """Train a model and return results."""
    start_time = time.time()

    # Callbacks
    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"checkpoints/turbofan_comparison/{model_name}",
        filename="{epoch:02d}-{val_loss:.3f}",
        save_top_k=1,
        mode="min",
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=5, verbose=True, mode="min")

    # Logger
    logger = TensorBoardLogger("logs", name=f"turbofan_{model_name}")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint, early_stop],
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=50,
    )

    print(f"Training {model_name}...")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    trainer.fit(model, dm)

    # Validate on best model
    trainer.validate(ckpt_path="best", datamodule=dm)

    # Calculate metrics
    val_results = trainer.validate(model, dm, verbose=False)
    val_loss = val_results[0]["val_loss"]
    val_acc = val_results[0]["val_acc"]

    training_time = time.time() - start_time

    return {
        "Model": model_name,
        "Val Accuracy": f"{val_acc:.4f}",
        "Val Loss": f"{val_loss:.4f}",
        "Parameters": f"{sum(p.numel() for p in model.parameters()):,}",
        "Training Time (s)": f"{training_time:.1f}",
        "Epochs": trainer.current_epoch + 1,
    }


def main():
    # Try to load NASA dataset if loader is available
    use_turbofan = False
    if HAS_TURBOFAN_LOADER:
        try:
            # First check if data exists
            train_dataset = NASATurbofanDataset(
                data_dir="./data/nasa_turbofan",
                subset="FD001",
                train=True,
                sequence_length=50,
                task="classification",
                num_classes=3,
                download=True,  # Auto-download via kagglehub
            )

            val_dataset = NASATurbofanDataset(
                data_dir="./data/nasa_turbofan",
                subset="FD001",
                train=False,
                sequence_length=50,
                task="classification",
                num_classes=3,
                download=False,
            )

            print("Loaded NASA Turbofan dataset:")
            print(f"Train samples: {len(train_dataset)}")
            print(f"Val samples: {len(val_dataset)}")
            use_turbofan = True

        except Exception as e:
            print(f"Failed to load turbofan: {e}")

    if not use_turbofan:
        print("\n" + "=" * 60)
        print("NASA Turbofan dataset not found!")
        print("=" * 60)
        print("\nTo use this dataset, please:")
        print(
            "1. Download from: https://data.nasa.gov/download/nk8v-ckry/application%2Fzip"
        )
        print("2. Extract the zip file")
        print("3. Copy the .txt files to: ./data/nasa_turbofan/")
        print("\nAlternatively, let's use a publicly available dataset...")

        # Fallback to a different dataset or provide instructions
        print("\n" + "=" * 60)
        print("Alternative: Using Synthetic Data with Realistic Parameters")
        print("=" * 60)

        from utils.dataset_base import SyntheticTimeSeriesDataset

        # Create synthetic data mimicking industrial sensors
        train_dataset = SyntheticTimeSeriesDataset(
            num_samples=10000,
            T=100,  # Longer sequences
            C_num=21,  # Similar to turbofan sensors
            C_cat=3,  # Operational modes
            cat_cardinalities=[5, 10, 3],  # Different categorical ranges
            num_classes=3,
            task="classification",
            missing_ratio=0.05,  # Some missing data
        )

        val_dataset = SyntheticTimeSeriesDataset(
            num_samples=2000,
            T=100,
            C_num=21,
            C_cat=3,
            cat_cardinalities=[5, 10, 3],
            num_classes=3,
            task="classification",
            missing_ratio=0.05,
        )

        print("\nUsing synthetic data mimicking industrial sensors:")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")

    # Create data module
    dm = TimeSeriesDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=64,
        num_workers=4,
    )

    # Get feature info
    if hasattr(train_dataset, "get_feature_info"):
        feature_info = train_dataset.get_feature_info()
    else:
        feature_info = {
            "n_numeric": train_dataset.C_num,
            "n_categorical": train_dataset.C_cat,
            "cat_cardinalities": train_dataset.cat_cardinalities,
        }

    # Get dimensions for models
    c_in = feature_info["n_numeric"]
    seq_len = train_dataset[0]["x_num"].shape[1]
    num_classes = 3

    print("\nDataset info:")
    print(f"- Numeric features: {c_in}")
    print(f"- Sequence length: {seq_len}")
    print(f"- Number of classes: {num_classes}")

    results = []
    max_epochs = 15

    # 1. Train TREA-C (our model)
    print("\n" + "=" * 60)
    print("Training TREA-C (Triple-Patch Transformer)...")
    print("=" * 60)

    treac_model = TriplePatchTransformer(
        C_num=c_in,
        C_cat=feature_info["n_categorical"],
        cat_cardinalities=feature_info["cat_cardinalities"],
        T=seq_len,
        d_model=128,
        n_head=8,
        num_layers=4,
        task="classification",
        num_classes=num_classes,
        lr=1e-3,
    )

    treac_results = train_model(treac_model, dm, "TREA-C", max_epochs=max_epochs)
    results.append(treac_results)

    # 2. Train PatchTST
    print("\n" + "=" * 60)
    print("Training PatchTST...")
    print("=" * 60)

    patchtst_model = PatchTSTClassifier(
        c_in=c_in,
        seq_len=seq_len,
        num_classes=num_classes,
        patch_len=16,
        stride=8,
        d_model=128,
        n_head=8,
        num_layers=3,
        lr=1e-3,
    )

    patchtst_results = train_model(
        patchtst_model, dm, "PatchTST", max_epochs=max_epochs
    )
    results.append(patchtst_results)

    # 3. Train CNN baseline
    print("\n" + "=" * 60)
    print("Training CNN Baseline...")
    print("=" * 60)

    cnn_model = CNNClassifier(
        c_in=c_in,
        seq_len=seq_len,
        num_classes=num_classes,
        lr=1e-3,
    )

    cnn_results = train_model(cnn_model, dm, "CNN", max_epochs=max_epochs)
    results.append(cnn_results)

    # Create comparison DataFrame
    df = pd.DataFrame(results)

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    print(df.to_string(index=False))

    # Save to CSV
    csv_filename = "outputs/turbofan_model_comparison.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nResults saved to: {csv_filename}")

    # Create summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    best_acc_idx = df["Val Accuracy"].str.replace("%", "").astype(float).idxmax()
    best_model = df.iloc[best_acc_idx]

    print(f"🏆 Best Model: {best_model['Model']}")
    print(f"   Accuracy: {best_model['Val Accuracy']}")
    print(f"   Parameters: {best_model['Parameters']}")
    print(f"   Training Time: {best_model['Training Time (s)']}s")

    # Parameter efficiency
    print("\n📊 Parameter Efficiency:")
    for _, row in df.iterrows():
        acc = float(row["Val Accuracy"].replace("%", ""))
        params = int(row["Parameters"].replace(",", ""))
        efficiency = acc / (params / 1000)  # Accuracy per 1K parameters
        print(f"   {row['Model']}: {efficiency:.2f} acc/1K params")


if __name__ == "__main__":
    main()
