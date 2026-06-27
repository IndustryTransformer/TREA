"""Fine-tuning script for pre-trained time series model.

This script demonstrates how to fine-tune a pre-trained model on a specific
downstream task, comparing performance against training from scratch.
"""

import argparse

from pathlib import Path

import pytorch_lightning as pl

from data.downloaders.air_quality import AirQualityDataset
from data.downloaders.etth1 import ETTh1Dataset
from data.downloaders.financial_market import FinancialMarketDataset
from data.downloaders.human_activity import HumanActivityDataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from treac.models.multi_dataset_model import MultiDatasetModel


def get_dataset(dataset_name: str, split: str, sequence_length: int):
    """Get dataset by name."""
    datasets = {
        "etth1": lambda: ETTh1Dataset(
            train=(split == "train"), sequence_length=sequence_length
        ),
        "human_activity": lambda: HumanActivityDataset(
            split=split, seq_len=sequence_length
        ),
        "air_quality": lambda: AirQualityDataset(split=split, seq_len=sequence_length),
        "financial_market": lambda: FinancialMarketDataset(
            split=split, seq_len=sequence_length
        ),
    }

    if dataset_name not in datasets:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available: {list(datasets.keys())}"
        )

    return datasets[dataset_name]()


def create_pretrained_model(
    pretrained_path: str, num_classes: int, freeze_encoder: bool = True
) -> MultiDatasetModel:
    """Load pre-trained model for fine-tuning."""
    print(f"Loading pre-trained model from: {pretrained_path}")
    print(f"Freeze encoder: {freeze_encoder}")

    model = MultiDatasetModel.from_pretrained(
        pretrained_path=pretrained_path,
        num_classes=num_classes,
        freeze_encoder=freeze_encoder,
        mode="pretrain",
    )

    return model


def create_from_scratch_model(dataset, args) -> MultiDatasetModel:
    """Create model to train from scratch for comparison."""
    print("Creating model to train from scratch...")

    model = MultiDatasetModel(
        max_numeric_features=dataset.numeric_features,
        max_categorical_features=dataset.categorical_features,
        num_classes=dataset.num_classes,
        d_model=args.d_model,
        n_head=args.n_head,
        num_layers=args.num_layers,
        patch_len=args.patch_len,
        stride=args.stride,
        lr=args.lr,
        use_feature_masks=True,
        column_embedding_strategy="auto_expanding",
        max_sequence_length=args.sequence_length,
        mode="variable_features",
    )

    # Set dataset schema
    model.set_dataset_schema(
        numeric_features=dataset.numeric_features,
        categorical_features=dataset.categorical_features,
        column_names=dataset.get_column_names()
        if hasattr(dataset, "get_column_names")
        else None,
    )

    return model


def create_dataloaders(
    dataset_name: str, sequence_length: int, batch_size: int, num_workers: int
):
    """Create train and validation dataloaders."""
    print(f"Creating dataloaders for {dataset_name}...")

    # Create datasets
    train_dataset = get_dataset(dataset_name, "train", sequence_length)

    # For validation, use 'val' if available, otherwise 'test'
    try:
        val_dataset = get_dataset(dataset_name, "val", sequence_length)
    except Exception:
        try:
            val_dataset = get_dataset(dataset_name, "test", sequence_length)
        except Exception:
            # If no val/test split, use the train=False version
            val_dataset = get_dataset(
                dataset_name, "train", sequence_length
            )  # This will create val split
            val_dataset.split = "val"  # Override split

    print("Dataset info:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(
        f"  Features: {train_dataset.numeric_features} numeric, "
        f"{train_dataset.categorical_features} categorical"
    )
    print(f"  Classes: {train_dataset.num_classes}")
    print(f"  Sequence length: {train_dataset.sequence_length}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_dataset


def run_experiment(
    model, train_loader, val_loader, dataset, args, experiment_name: str
):
    """Run training experiment."""
    print(f"\nRunning experiment: {experiment_name}")
    print("-" * 50)

    # Set dataset schema for the model
    if hasattr(model, "set_dataset_schema"):
        model.set_dataset_schema(
            numeric_features=dataset.numeric_features,
            categorical_features=dataset.categorical_features,
            column_names=dataset.get_column_names()
            if hasattr(dataset, "get_column_names")
            else None,
        )

    # Set up logging and checkpointing
    logger = TensorBoardLogger(
        save_dir=args.log_dir, name=f"finetune_{args.dataset}", version=experiment_name
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(args.checkpoint_dir) / experiment_name,
        filename="best-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.001, patience=args.patience, mode="min"
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        gradient_clip_val=1.0,
        log_every_n_steps=20,
        enable_progress_bar=True,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Trainable parameters: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)

    # Test on validation set
    results = trainer.test(model, val_loader)

    return {
        "experiment_name": experiment_name,
        "best_val_loss": checkpoint_callback.best_model_score.item()
        if checkpoint_callback.best_model_score
        else None,
        "final_results": results[0] if results else {},
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "best_checkpoint": checkpoint_callback.best_model_path,
    }


def main():
    """Main fine-tuning function."""
    parser = argparse.ArgumentParser(
        description="Fine-tune pre-trained time series model"
    )

    # Data parameters
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["etth1", "human_activity", "air_quality", "financial_market"],
        help="Dataset to fine-tune on",
    )
    parser.add_argument(
        "--sequence_length", type=int, default=96, help="Sequence length"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data workers"
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

    # Pre-training parameters
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        help="Path to pre-trained model checkpoint",
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        default=False,
        help="Freeze encoder during fine-tuning",
    )
    parser.add_argument(
        "--compare_from_scratch",
        action="store_true",
        default=True,
        help="Also train model from scratch for comparison",
    )

    # Training parameters
    parser.add_argument(
        "--max_epochs", type=int, default=50, help="Maximum training epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=15, help="Early stopping patience"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints/finetune",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--log_dir", type=str, default="./logs/finetune", help="Logging directory"
    )

    # Device parameters
    parser.add_argument(
        "--accelerator", type=str, default="auto", help="Accelerator type"
    )
    parser.add_argument("--devices", type=str, default="auto", help="Number of devices")
    parser.add_argument(
        "--precision", type=str, default="16-mixed", help="Training precision"
    )

    args = parser.parse_args()

    # Set random seed
    pl.seed_everything(42)

    print("=" * 70)
    print("TIME SERIES MODEL FINE-TUNING")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Pre-trained model: {args.pretrained_path}")
    print(f"Freeze encoder: {args.freeze_encoder}")
    print(f"Compare with from-scratch: {args.compare_from_scratch}")
    print()

    # Create dataloaders
    train_loader, val_loader, dataset = create_dataloaders(
        args.dataset, args.sequence_length, args.batch_size, args.num_workers
    )

    results = []

    # Experiment 1: Fine-tune pre-trained model (if available)
    if args.pretrained_path and Path(args.pretrained_path).exists():
        print("\n" + "=" * 50)
        print("EXPERIMENT 1: FINE-TUNING PRE-TRAINED MODEL")
        print("=" * 50)

        model = create_pretrained_model(
            args.pretrained_path, dataset.num_classes, args.freeze_encoder
        )

        experiment_name = f"pretrained_freeze_{args.freeze_encoder}"
        result = run_experiment(
            model, train_loader, val_loader, dataset, args, experiment_name
        )
        results.append(result)

        # Also try with unfrozen encoder
        if args.freeze_encoder:
            print("\n" + "=" * 50)
            print("EXPERIMENT 2: FINE-TUNING WITH UNFROZEN ENCODER")
            print("=" * 50)

            model_unfrozen = create_pretrained_model(
                args.pretrained_path, dataset.num_classes, freeze_encoder=False
            )

            experiment_name = "pretrained_freeze_False"
            result = run_experiment(
                model_unfrozen, train_loader, val_loader, dataset, args, experiment_name
            )
            results.append(result)

    # Experiment 3: Train from scratch (if requested)
    if args.compare_from_scratch:
        print("\n" + "=" * 50)
        print("EXPERIMENT 3: TRAINING FROM SCRATCH")
        print("=" * 50)

        model_scratch = create_from_scratch_model(dataset, args)

        experiment_name = "from_scratch"
        result = run_experiment(
            model_scratch, train_loader, val_loader, dataset, args, experiment_name
        )
        results.append(result)

    # Print comparison results
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPARISON")
    print("=" * 70)

    for result in results:
        print(f"\n{result['experiment_name'].upper()}:")
        print(
            f"  Best validation loss: {result['best_val_loss']:.4f}"
            if result["best_val_loss"]
            else "  N/A"
        )
        print(f"  Total parameters: {result['total_params']:,}")
        print(f"  Trainable parameters: {result['trainable_params']:,}")
        print(f"  Best checkpoint: {result['best_checkpoint']}")

        if result["final_results"]:
            for metric, value in result["final_results"].items():
                if isinstance(value, int | float):
                    print(f"  {metric}: {value:.4f}")

    # Find best result
    if results:
        best_result = min(
            results,
            key=lambda x: x["best_val_loss"] if x["best_val_loss"] else float("inf"),
        )
        print(f"\nBEST PERFORMING MODEL: {best_result['experiment_name']}")
        print(f"Best validation loss: {best_result['best_val_loss']:.4f}")

    print(f"\nAll results saved to: {args.log_dir}")


if __name__ == "__main__":
    main()
