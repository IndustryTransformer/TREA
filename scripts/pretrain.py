"""Pre-training script for variable feature time series model using self-supervised
learning.

This script demonstrates the pre-training approach for time series models,
training on multiple datasets simultaneously using SSL objectives.
"""

import argparse

from pathlib import Path
from typing import Any

import pytorch_lightning as pl

from data.downloaders.air_quality import AirQualityDataset
from data.downloaders.etth1 import ETTh1Dataset
from data.downloaders.financial_market import FinancialMarketDataset
from data.downloaders.human_activity import HumanActivityDataset
from data.downloaders.multi_dataset_loader import MultiDatasetDataModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from treac.models.multi_dataset_model import MultiDatasetModel


def create_datasets(target_sequence_length: int = 96) -> dict[str, dict[str, Any]]:
    """Create train and validation datasets for pre-training.

    Args:
        target_sequence_length: Target sequence length for standardization

    Returns:
        Dictionary with train and val datasets
    """
    print("Creating datasets for pre-training...")

    # Create training datasets
    train_datasets = {
        "etth1": ETTh1Dataset(train=True, sequence_length=target_sequence_length),
        "human_activity": HumanActivityDataset(
            split="train", seq_len=target_sequence_length
        ),
        "air_quality": AirQualityDataset(split="train", seq_len=target_sequence_length),
        "financial_market": FinancialMarketDataset(
            split="train", seq_len=target_sequence_length
        ),
    }

    # Create validation datasets
    val_datasets = {
        "etth1": ETTh1Dataset(train=False, sequence_length=target_sequence_length),
        "human_activity": HumanActivityDataset(
            split="val", seq_len=target_sequence_length
        ),
        "air_quality": AirQualityDataset(split="val", seq_len=target_sequence_length),
        "financial_market": FinancialMarketDataset(
            split="val", seq_len=target_sequence_length
        ),
    }

    # Print dataset information
    print("\nDataset Summary:")
    print("-" * 50)
    for name, dataset in train_datasets.items():
        print(f"{name}:")
        print(f"  Train samples: {len(dataset)}")
        print(f"  Val samples: {len(val_datasets[name])}")

        # Get feature info using different methods depending on dataset
        if hasattr(dataset, "numeric_features"):
            numeric_features = dataset.numeric_features
            categorical_features = dataset.categorical_features
            num_classes = dataset.num_classes
            sequence_length = dataset.sequence_length
        elif hasattr(dataset, "get_feature_info"):
            info = dataset.get_feature_info()
            numeric_features = info["n_numeric"]
            categorical_features = info["n_categorical"]
            num_classes = info["num_classes"]
            sequence_length = info["sequence_length"]
        else:
            # Fallback to examining the data directly
            sample = dataset[0]
            numeric_features = sample["x_num"].shape[0] if "x_num" in sample else 0
            categorical_features = (
                sample["x_cat"].shape[0]
                if "x_cat" in sample and sample["x_cat"] is not None
                else 0
            )
            num_classes = getattr(dataset, "num_classes", 3)
            sequence_length = getattr(
                dataset,
                "sequence_length",
                sample["x_num"].shape[1] if "x_num" in sample else 96,
            )

        print(
            f"  Features: {numeric_features} numeric, "
            f"{categorical_features} categorical"
        )
        print(f"  Classes: {num_classes}")
        print(f"  Sequence length: {sequence_length}")
        print()

    return {"train": train_datasets, "val": val_datasets}


def create_model(datasets: dict[str, dict[str, Any]], args) -> MultiDatasetModel:
    """Create pre-training model with proper schema configuration.

    Args:
        datasets: Dataset dictionary
        args: Command line arguments

    Returns:
        Configured MultiDatasetModel model
    """
    print("Creating pre-training model...")

    # Calculate maximum features across all datasets
    numeric_features_list = []
    categorical_features_list = []
    classes_list = []

    for dataset in datasets["train"].values():
        if hasattr(dataset, "numeric_features"):
            numeric_features_list.append(dataset.numeric_features)
            categorical_features_list.append(dataset.categorical_features)
            classes_list.append(dataset.num_classes)
        elif hasattr(dataset, "get_feature_info"):
            info = dataset.get_feature_info()
            numeric_features_list.append(info["n_numeric"])
            categorical_features_list.append(info["n_categorical"])
            classes_list.append(info["num_classes"])
        else:
            # Fallback to examining the data directly
            sample = dataset[0]
            numeric_features = sample["x_num"].shape[0] if "x_num" in sample else 0
            categorical_features = (
                sample["x_cat"].shape[0]
                if "x_cat" in sample and sample["x_cat"] is not None
                else 0
            )
            num_classes = getattr(dataset, "num_classes", 3)
            numeric_features_list.append(numeric_features)
            categorical_features_list.append(categorical_features)
            classes_list.append(num_classes)

    max_numeric_features = max(numeric_features_list)
    max_categorical_features = max(categorical_features_list)
    max_classes = max(classes_list)

    print("Model configuration:")
    print(f"  Max numeric features: {max_numeric_features}")
    print(f"  Max categorical features: {max_categorical_features}")
    print(f"  Max classes: {max_classes}")
    print(f"  Target sequence length: {args.sequence_length}")

    # Determine SSL objectives (default to True unless explicitly disabled)
    use_masked_patch = args.use_masked_patch or not args.no_masked_patch
    use_temporal_order = args.use_temporal_order or not args.no_temporal_order
    use_contrastive = args.use_contrastive or not args.no_contrastive

    # If no explicit flags, use all objectives by default
    if not any(
        [
            args.use_masked_patch,
            args.use_temporal_order,
            args.use_contrastive,
            args.no_masked_patch,
            args.no_temporal_order,
            args.no_contrastive,
        ]
    ):
        use_masked_patch = use_temporal_order = use_contrastive = True

    # Create model
    model = MultiDatasetModel(
        max_numeric_features=max_numeric_features,
        max_categorical_features=max_categorical_features,
        num_classes=max_classes,
        mode="pretrain",
        # SSL configuration
        ssl_objectives={
            "masked_patch": use_masked_patch,
            "temporal_order": use_temporal_order,
            "contrastive": use_contrastive,
        },
        mask_ratio=args.mask_ratio,
        temporal_shuffle_ratio=args.temporal_shuffle_ratio,
        contrastive_temperature=args.contrastive_temperature,
        augmentation_strength=args.augmentation_strength,
        lambda_masked=args.lambda_masked,
        lambda_temporal=args.lambda_temporal,
        lambda_contrastive=args.lambda_contrastive,
        lambda_supervised=args.lambda_supervised,
        # Model architecture
        d_model=args.d_model,
        n_head=args.n_head,
        num_layers=args.num_layers,
        patch_len=args.patch_len,
        stride=args.stride,
        lr=args.lr,
        # Feature handling
        use_feature_masks=True,
        column_embedding_strategy="auto_expanding",
        max_sequence_length=args.sequence_length,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


def main():
    """Main pre-training function."""
    parser = argparse.ArgumentParser(description="Pre-train time series model with SSL")

    # Data parameters
    parser.add_argument(
        "--sequence_length", type=int, default=96, help="Target sequence length"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data workers"
    )
    parser.add_argument(
        "--oversample_smaller",
        action="store_true",
        default=True,
        help="Oversample smaller datasets",
    )

    # Model parameters
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument(
        "--n_head", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--num_layers", type=int, default=3, help="Number of transformer layers"
    )
    parser.add_argument("--patch_len", type=int, default=16, help="Patch length")
    parser.add_argument("--stride", type=int, default=8, help="Patch stride")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    # SSL objective parameters
    parser.add_argument(
        "--use_masked_patch", action="store_true", help="Use masked patch prediction"
    )
    parser.add_argument(
        "--use_temporal_order",
        action="store_true",
        help="Use temporal order prediction",
    )
    parser.add_argument(
        "--use_contrastive", action="store_true", help="Use contrastive learning"
    )
    parser.add_argument(
        "--no_masked_patch", action="store_true", help="Disable masked patch prediction"
    )
    parser.add_argument(
        "--no_temporal_order",
        action="store_true",
        help="Disable temporal order prediction",
    )
    parser.add_argument(
        "--no_contrastive", action="store_true", help="Disable contrastive learning"
    )

    parser.add_argument(
        "--mask_ratio", type=float, default=0.15, help="Masking ratio for MPP"
    )
    parser.add_argument(
        "--temporal_shuffle_ratio",
        type=float,
        default=0.3,
        help="Shuffle ratio for temporal order",
    )
    parser.add_argument(
        "--contrastive_temperature",
        type=float,
        default=0.1,
        help="Contrastive temperature",
    )
    parser.add_argument(
        "--augmentation_strength",
        type=float,
        default=0.1,
        help="Data augmentation strength",
    )

    # Loss weights
    parser.add_argument(
        "--lambda_masked", type=float, default=1.0, help="Weight for masked patch loss"
    )
    parser.add_argument(
        "--lambda_temporal",
        type=float,
        default=0.5,
        help="Weight for temporal order loss",
    )
    parser.add_argument(
        "--lambda_contrastive",
        type=float,
        default=0.3,
        help="Weight for contrastive loss",
    )
    parser.add_argument(
        "--lambda_supervised",
        type=float,
        default=0.1,
        help="Weight for supervised loss",
    )

    # Training parameters
    parser.add_argument(
        "--max_epochs", type=int, default=50, help="Maximum training epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints/pretrain",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--log_dir", type=str, default="./logs/pretrain", help="Logging directory"
    )

    # Device parameters
    parser.add_argument(
        "--accelerator", type=str, default="auto", help="Accelerator type"
    )
    parser.add_argument("--devices", type=str, default="auto", help="Number of devices")
    parser.add_argument(
        "--precision", type=str, default="32", help="Training precision"
    )

    args = parser.parse_args()

    # Set random seed
    pl.seed_everything(42)

    print("=" * 70)
    print("TIME SERIES PRE-TRAINING WITH SELF-SUPERVISED LEARNING")
    print("=" * 70)
    print()

    # Create datasets
    datasets = create_datasets(args.sequence_length)

    # Create data module
    print("Setting up data module...")
    dm = MultiDatasetDataModule(
        train_datasets=datasets["train"],
        val_datasets=datasets["val"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_sequence_length=args.sequence_length,
        oversample_smaller=args.oversample_smaller,
    )

    # Create model
    model = create_model(datasets, args)

    # Set up logging and checkpointing
    logger = TensorBoardLogger(
        save_dir=args.log_dir, name="pretrain_experiment", version=None
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="pretrain-{epoch:02d}-{train/total_loss:.2f}",
        monitor="train/total_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="train/total_loss", min_delta=0.001, patience=args.patience, mode="min"
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=1,
        log_every_n_steps=50,
        val_check_interval=0.5,  # Check validation twice per epoch
        enable_progress_bar=True,
    )

    print("\nStarting pre-training...")
    print(
        f"SSL objectives: "
        f"{[obj for obj, enabled in model.ssl_objectives_config.items() if enabled]}"
    )
    print(f"Training for up to {args.max_epochs} epochs")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print(f"Logs saved to: {args.log_dir}")
    print()

    # Train model
    trainer.fit(model, dm)

    print("\n" + "=" * 70)
    print("PRE-TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Final loss: {trainer.callback_metrics.get('train/total_loss', 'N/A')}")

    # Save final model state
    final_checkpoint_path = Path(args.checkpoint_dir) / "final_pretrained_model.ckpt"
    trainer.save_checkpoint(final_checkpoint_path)
    print(f"Final model saved to: {final_checkpoint_path}")

    print("\nTo use this pre-trained model for fine-tuning:")
    print(
        "model = MultiDatasetModel.from_pretrained(\n"
        f"    '{final_checkpoint_path}',\n"
        "    num_classes=<your_classes>,\n"
        "    mode='pretrain'\n"
        ")"
    )


if __name__ == "__main__":
    main()
