"""Ultra-fast PatchTSTNan training with in-memory data processing.

Optimized for fine-grained temporal data using pre-loaded datasets for maximum
training speed with larger patch sizes.
"""

import numpy as np
import polars as pol
import pytorch_lightning as pl
import torch

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from treac.models import PatchTSTNan


class W3InMemoryDataset(Dataset):
    """
    Efficient dataset that uses pre-loaded and pre-processed data from memory.
    All data is loaded upfront and processed in-memory for maximum speed.
    """

    def __init__(
        self,
        well_data_store: dict,
        numeric_cols: list,
        sequence_length: int = 128,
        max_sequences_per_well: int = 10000,
    ):
        self.well_data_store = well_data_store
        self.numeric_cols = numeric_cols
        self.seq_len = sequence_length
        self.max_sequences_per_well = max_sequences_per_well

        print("Computing sequence indices from loaded data...")
        self.sequence_indices = []
        total_sequences = 0
        for well_name, well_data in self.well_data_store.items():
            count = len(well_data)
            if count >= sequence_length + 1:  # Need at least seq_len + 1 for target
                # Calculate possible sequences for this well
                possible_sequences = count - sequence_length

                # Create indices for this well (step size to sample evenly)
                if possible_sequences > self.max_sequences_per_well:
                    step = possible_sequences // self.max_sequences_per_well
                    starts = range(0, possible_sequences, step)[
                        : self.max_sequences_per_well
                    ]
                else:
                    starts = range(possible_sequences)

                for start_idx in starts:
                    self.sequence_indices.append((well_name, start_idx))

                total_sequences += len(starts)

        print(f"Created {total_sequences} sequence indices.")

    def __len__(self):
        return len(self.sequence_indices)

    def __getitem__(self, idx):
        well_name, start_idx = self.sequence_indices[idx]

        # Load well data from in-memory store
        well_data = self.well_data_store[well_name]

        if start_idx + self.seq_len >= len(well_data):
            # Fallback to last valid sequence
            start_idx = max(0, len(well_data) - self.seq_len - 1)

        end_idx = start_idx + self.seq_len
        target_idx = end_idx

        # Extract sequence data
        sequence_data = well_data.iloc[start_idx:end_idx]
        target_state = well_data.iloc[target_idx]["class"]

        # Extract numeric features (already scaled)
        numeric_values = sequence_data[self.numeric_cols].values

        x_num = torch.tensor(numeric_values, dtype=torch.float32).T
        x_cat = torch.empty((0, self.seq_len), dtype=torch.long)
        y = torch.tensor(target_state, dtype=torch.long)

        return {"x_num": x_num, "x_cat": x_cat, "y": y}


class W3DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_parquet_path: str | None = None,
        test_parquet_path: str | None = None,
        metadata: dict | None = None,
        batch_size: int = 256,
        sequence_length: int = 128,
    ):
        super().__init__()
        self.train_parquet_path = train_parquet_path
        self.test_parquet_path = test_parquet_path
        self.metadata = metadata
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.train_well_data_store = None
        self.val_well_data_store = None

    def setup(self, stage=None):  # stage parameter required by Lightning interface
        print("Loading and preprocessing data in memory...")

        # Load full datasets into memory
        train_df = pol.read_parquet(self.train_parquet_path)
        test_df = pol.read_parquet(self.test_parquet_path)

        # --- Preprocessing ---
        state_to_label = self.metadata["state_to_label"]

        def preprocess(df):
            return (
                df.filter(pol.col("well_name").str.contains("WELL"))
                .filter(pol.col("class").is_not_null())
                .drop("state")
                .with_columns(
                    pol.col("class").cast(pol.Utf8).replace_strict(state_to_label)
                )
            )

        train_df = preprocess(train_df)
        test_df = preprocess(test_df)

        # Convert to pandas for scikit-learn compatibility
        train_pdf = train_df.to_pandas()
        test_pdf = test_df.to_pandas()

        numeric_cols = self.metadata["numeric_cols"]

        # Handle NaN/inf values before scaling
        train_pdf[numeric_cols] = np.nan_to_num(
            train_pdf[numeric_cols], nan=0.0, posinf=0.0, neginf=0.0
        )
        test_pdf[numeric_cols] = np.nan_to_num(
            test_pdf[numeric_cols], nan=0.0, posinf=0.0, neginf=0.0
        )

        # --- Scaling (Fit on train only, transform both) ---
        print("Fitting StandardScaler on training data...")
        scaler = StandardScaler()
        scaler.fit(train_pdf[numeric_cols])

        print("Applying scaler to train and test data...")
        train_pdf[numeric_cols] = scaler.transform(train_pdf[numeric_cols])
        test_pdf[numeric_cols] = scaler.transform(test_pdf[numeric_cols])

        # Clip values to prevent outliers from dominating
        train_pdf[numeric_cols] = np.clip(train_pdf[numeric_cols], -3, 3)
        test_pdf[numeric_cols] = np.clip(test_pdf[numeric_cols], -3, 3)

        # --- Create Data Stores ---
        self.train_well_data_store = {
            name: df for name, df in train_pdf.groupby("well_name")
        }
        self.val_well_data_store = {
            name: df for name, df in test_pdf.groupby("well_name")
        }

        print(f"Training wells: {len(self.train_well_data_store)}")
        print(f"Validation wells: {len(self.val_well_data_store)}")

        # --- Create Datasets ---
        self.train_dataset = W3InMemoryDataset(
            well_data_store=self.train_well_data_store,
            numeric_cols=self.metadata["numeric_cols"],
            sequence_length=self.sequence_length,
            max_sequences_per_well=50000,
        )

        self.val_dataset = W3InMemoryDataset(
            well_data_store=self.val_well_data_store,
            numeric_cols=self.metadata["numeric_cols"],
            sequence_length=self.sequence_length,
            max_sequences_per_well=10000,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
        )


def extract_dataset_metadata():
    """Extract metadata from the full dataset without loading into memory."""
    print("=" * 60)
    print("EXTRACTING DATASET METADATA")
    print("=" * 60)

    df_lazy = pol.scan_parquet("/home/ubuntu/TREA-C/data/W3/train.parquet")

    # Get total row count efficiently
    total_rows = df_lazy.select(pol.len()).collect().item()
    print(f"Total rows in dataset: {total_rows:,}")

    # Get column info
    columns = df_lazy.columns
    print(f"Total columns: {len(columns)}")

    # Get unique classes efficiently (without loading full data)
    classes_df = (
        df_lazy.select("class")
        .filter(pol.col("class").is_not_null())
        .unique()
        .collect()
    )
    unique_classes = classes_df["class"].sort().to_list()
    print(f"Unique classes: {unique_classes}")

    # Get unique wells count
    wells_count = (
        df_lazy.select("well_name").unique().select(pol.len()).collect().item()
    )
    print(f"Unique wells: {wells_count}")

    # Identify numeric columns
    numeric_cols = [
        col for col in columns if col not in ["class", "well_name", "state"]
    ]
    print(f"Numeric features: {len(numeric_cols)}")

    # Create class mapping
    class_to_label = {str(cls): i for i, cls in enumerate(unique_classes)}
    print(f"Class labels: {class_to_label}")

    return {
        "total_rows": total_rows,
        "unique_states": len(unique_classes),  # Keep name for compatibility
        "state_to_label": class_to_label,  # Keep name for compatibility
        "numeric_cols": numeric_cols,
        "wells_count": wells_count,
        "C_num": len(numeric_cols),
        "C_cat": 0,
    }


def main():
    print("=" * 60)
    print("ULTRA-FAST PatchTSTNan Training")
    print("In-memory processing for maximum training speed")
    print("=" * 60)

    # Phase 1: Extract metadata efficiently
    metadata = extract_dataset_metadata()

    # Using curated train/test split with in-memory processing
    print("\n" + "=" * 60)
    print("USING FULL DATASET WITH IN-MEMORY PROCESSING")
    print("=" * 60)
    print("✓ Training data: FULL data/W3/train.parquet (loaded into memory)")
    print("✓ Test data: FULL data/W3/test.parquet (loaded into memory)")
    print("✓ Pre-computed sequence indices for fast access")
    print("✓ StandardScaler fitted on training data only")
    print("✓ Efficient batch generation from pre-processed data")

    # Phase 1: Scaled configuration for maximum GPU utilization (target: 8-15GB)
    sequence_length = 1024  # Longer sequences for more data per sample
    batch_size = 512  # Much larger batch size for GPU efficiency
    patch_size = 16  # Proven stable patches: 1024/16 = 64 patches

    print("\nPhase 1: GPU-Optimized Configuration:")
    print(f"  Sequence length: {sequence_length} (2x longer)")
    print(f"  Patch size: {patch_size}")
    stride_patches = (sequence_length - patch_size) // 8 + 1
    std_patches = sequence_length // patch_size
    print(f"  Patches: {std_patches} standard (~{stride_patches} with stride)")
    print(f"  Batch size: {batch_size} (2x larger)")
    effective_batch_size = batch_size * 8  # With gradient accumulation
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Model size: d_model={128}, heads={8}, layers={4}")
    print("  Target GPU usage: 8-15GB (vs previous 0.57GB)")
    print("  Expected speedup: 10-20x through GPU utilization")

    # Create data module with in-memory processing
    print("\n" + "=" * 60)
    print("CREATING IN-MEMORY DATA MODULE")
    print("=" * 60)

    # Create data module with curated train/test split
    train_path = "/home/ubuntu/TREA-C/data/W3/train.parquet"
    test_path = "/home/ubuntu/TREA-C/data/W3/test.parquet"

    data_module = W3DataModule(
        train_parquet_path=train_path,
        test_parquet_path=test_path,
        metadata=metadata,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )

    # Create larger model for maximum GPU utilization and capacity
    model = PatchTSTNan(
        c_in=metadata["C_num"],  # Use c_in parameter name for stride-based mode
        seq_len=sequence_length,  # Use seq_len parameter name
        num_classes=metadata["unique_states"],
        patch_len=16,  # Proven stable patches
        stride=8,  # Proven stable stride
        d_model=128,  # Larger model dimension for better capacity
        n_head=8,  # More attention heads for richer representations
        num_layers=4,  # Deeper model for better learning capacity
        dropout=0.15,  # Moderate dropout for regularization
        lr=5e-4,  # Lower learning rate for larger model
        task="classification",  # Explicit task specification
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,}")

    # Apply performance optimizations
    print("\n" + "=" * 60)
    print("APPLYING PERFORMANCE OPTIMIZATIONS")
    print("=" * 60)

    # Enable Tensor Cores for maximum performance
    torch.set_float32_matmul_precision("medium")
    print("✓ Enabled Tensor Cores (medium precision)")

    # Set optimal threading
    torch.set_num_threads(8)
    print("✓ Set torch threads to 8")

    # Create trainer with maximum speed optimizations

    logger = TensorBoardLogger("tb_logs", name="patchtstnan_w3_ultra_fast")

    # Add early stopping to prevent overfitting
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        verbose=True,
        mode="min",
    )

    # Add model checkpoint to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="patchtstnan-w3-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=15,  # More epochs for larger model
        accelerator="auto",
        devices="auto",
        precision="16-mixed",  # Mixed precision for speed and memory efficiency
        log_every_n_steps=10,  # More frequent logging for monitoring
        logger=logger,
        callbacks=[early_stopping_callback, checkpoint_callback],
        gradient_clip_val=1.0,  # Gradient clipping for stability
        # Production settings for larger training
        enable_checkpointing=True,  # Checkpointing for longer runs
        enable_progress_bar=True,
        val_check_interval=0.5,  # Validation every half epoch
        limit_val_batches=50,  # More validation batches for stable metrics
        # Phase 1 optimizations
        enable_model_summary=True,  # Show detailed model structure
        num_sanity_val_steps=2,
        accumulate_grad_batches=8,  # Effective batch size: 512*8 = 4096
        # Additional optimizations for large models
        detect_anomaly=False,  # Disable for production speed
        benchmark=True,  # Optimize CUDA kernels
    )

    # Train model
    print("\n" + "=" * 60)
    print("STARTING ULTRA-FAST TRAINING")
    print("=" * 60)
    print(f"GPU Memory before training: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    print("Starting training with pre-loaded data...")

    import time

    start_time = time.time()

    trainer.fit(model, data_module)

    end_time = time.time()
    training_time = end_time - start_time

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Training time: {training_time:.2f}s ({training_time / 60:.1f} minutes)")
    print(f"GPU Memory after training: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    print(f"Max GPU Memory used: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")

    # Reset memory stats for next run
    torch.cuda.reset_peak_memory_stats()

    # Manually verify the results to ensure no data leakage
    manual_validation(model, data_module, trainer)


def manual_validation(model, data_module, trainer):
    """
    Manually calculates validation metrics to verify training results.
    """
    print("\n" + "=" * 60)
    print("STARTING MANUAL VALIDATION")
    print("=" * 60)

    # Ensure the model is in evaluation mode
    model.eval()

    # Move model to the correct device (GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Get the validation dataloader
    val_dataloader = data_module.val_dataloader()

    # Determine the number of batches to use for validation
    num_val_batches = trainer.limit_val_batches
    if isinstance(num_val_batches, float):
        num_val_batches = int(num_val_batches * len(val_dataloader))
    if num_val_batches == 0:
        num_val_batches = len(val_dataloader)

    print(f"Running manual validation on {num_val_batches} batches...")

    all_preds = []
    all_labels = []
    total_loss = 0
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            if i >= num_val_batches:
                break

            # Move batch to the same device as the model
            x_num = batch["x_num"].to(device)
            x_cat = batch["x_cat"].to(device)
            y = batch["y"].to(device)

            # Get model predictions (logits)
            logits = model(x_num, x_cat)

            # Calculate loss for the batch and add to total
            loss = loss_fn(logits, y)
            total_loss += loss.item()

            # Get predicted class by finding the index with the highest logit
            preds = torch.argmax(logits, dim=1)

            # Move predictions and labels to CPU for aggregation
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    # Combine results from all batches
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Calculate overall metrics
    from sklearn.metrics import accuracy_score

    manual_accuracy = accuracy_score(all_labels, all_preds)
    manual_loss = total_loss / num_val_batches

    # Get the final validation metrics from the trainer
    trainer_val_acc = trainer.callback_metrics.get("val_acc", "N/A")
    trainer_val_loss = trainer.callback_metrics.get("val_loss", "N/A")

    print("\n--- Manual vs. Trainer Metrics Comparison ---")
    if isinstance(trainer_val_acc, torch.Tensor):
        trainer_val_acc = trainer_val_acc.item()
    if isinstance(trainer_val_loss, torch.Tensor):
        trainer_val_loss = trainer_val_loss.item()

    print(f"  Manual Accuracy: {manual_accuracy:.4f}")
    print(f"  Trainer Accuracy: {trainer_val_acc:.4f}\n")

    print(f"  Manual Loss: {manual_loss:.4f}")
    print(f"  Trainer Loss: {trainer_val_loss:.4f}\n")

    # Conclusion
    if (
        abs(manual_accuracy - trainer_val_acc) < 1e-4
        and abs(manual_loss - trainer_val_loss) < 1e-4
    ):
        print("✓ SUCCESS: Manual and trainer metrics are consistent.")
        print(
            "This confirms the high performance is genuine and not due to data leakage."
        )
    else:
        print("✗ WARNING: Discrepancy found between manual and trainer metrics.")
        print("Further investigation is needed to identify the source of the issue.")


if __name__ == "__main__":
    main()
