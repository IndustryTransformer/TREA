"""Test PatchTSTNan on ETTh1 dataset to validate the consolidated implementation."""

import pytorch_lightning as pl
import torch

from data.downloaders.etth1 import ETTh1Dataset
from pytorch_lightning.loggers import TensorBoardLogger

from treac.models import PatchTSTNan
from utils.datamodule import TimeSeriesDataModule


def test_patchtstnan():
    """Test PatchTSTNan on ETTh1 dataset."""
    print("=" * 60)
    print("Testing PatchTSTNan on ETTh1 Dataset")
    print("=" * 60)

    # Configuration
    SEQUENCE_LENGTH = 96
    NUM_CLASSES = 3
    BATCH_SIZE = 32
    MAX_EPOCHS = 3  # Just a few epochs to test

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

    # Create data module
    dm = TimeSeriesDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,  # Avoid multiprocessing issues
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

    # Test data loading
    print("\nTesting data loading...")
    dm.setup("fit")
    train_loader = dm.train_dataloader()

    # Get a sample batch to inspect
    sample_batch = next(iter(train_loader))
    print("Sample batch shapes:")
    print(f"- x_num: {sample_batch['x_num'].shape}")
    print(
        f"- x_cat: {sample_batch['x_cat'].shape if 'x_cat' in sample_batch else 'N/A'}"
    )
    print(f"- y: {sample_batch['y'].shape}")
    print(f"- y unique values: {torch.unique(sample_batch['y'])}")
    print(f"- x_num contains NaN: {torch.isnan(sample_batch['x_num']).any()}")
    x_min, x_max = sample_batch["x_num"].min(), sample_batch["x_num"].max()
    print(f"- x_num range: [{x_min:.3f}, {x_max:.3f}]")

    # Create model with stride-based patching (like in compare_models_etth1.py)
    print("\nCreating PatchTSTNan model...")

    model = PatchTSTNan(
        c_in=c_in,  # Use c_in parameter name for compatibility
        seq_len=seq_len,
        num_classes=num_classes,
        patch_len=16,
        stride=8,
        d_model=64,  # Smaller model for testing
        n_head=4,
        num_layers=2,
        lr=1e-3,
        task="classification",
    )

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {params:,}")

    # Test forward pass
    print("\nTesting forward pass...")
    model.eval()
    with torch.no_grad():
        try:
            output = model(sample_batch["x_num"], sample_batch.get("x_cat"))
            print(f"Output shape: {output.shape}")
            print(f"Output contains NaN: {torch.isnan(output).any()}")
            print(f"Output contains Inf: {torch.isinf(output).any()}")
            print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

            # Test loss calculation
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(output, sample_batch["y"])
            print(f"Sample loss: {loss.item():.4f}")
            print(f"Loss is NaN: {torch.isnan(loss)}")

        except Exception as e:
            print(f"Error in forward pass: {e}")
            return

    # Create trainer
    print(f"\nStarting training for {MAX_EPOCHS} epochs...")

    logger = TensorBoardLogger("tb_logs", name="test_patchtstnan_etth1")

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices=1,
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=10,
        val_check_interval=0.5,  # Check validation more frequently
        limit_train_batches=50,  # Limit to 50 batches per epoch for testing
        limit_val_batches=20,  # Limit validation batches
    )

    # Train model
    try:
        trainer.fit(model, dm)
        print("\nTraining completed successfully!")

        # Check final metrics
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
    test_patchtstnan()
