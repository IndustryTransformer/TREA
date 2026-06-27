"""Train TriplePatchTransformer on Human Activity Recognition dataset."""

import sys


sys.path.append(".")

import pytorch_lightning as pl

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from examples.download_har_dataset import HARDataset
from treac.models.triple_attention import TriplePatchTransformer
from utils.datamodule import TimeSeriesDataModule


def main():
    print("Loading HAR dataset...")

    # Load datasets
    train_dataset = HARDataset(train=True, download=True)
    val_dataset = HARDataset(train=False, download=True)

    print("\nDataset loaded successfully!")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Number of channels: {train_dataset[0]['x_num'].shape[0]}")
    print(f"Sequence length: {train_dataset[0]['x_num'].shape[1]}")
    print(f"Activities: {train_dataset.activity_names}")

    # Create data module
    dm = TimeSeriesDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=64,
        num_workers=4,
    )

    # Create model
    model = TriplePatchTransformer(
        C_num=9,  # 9 sensor channels (3-axis acc + 3-axis gyro + 3-axis total acc)
        C_cat=2,  # Dummy categorical features
        cat_cardinalities=[1, 1],  # Dummy
        T=128,  # HAR uses 128 timesteps (2.56 seconds at 50Hz)
        d_model=64,
        n_head=4,
        num_layers=3,
        task="classification",
        num_classes=6,  # 6 activities
    )

    print(
        f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters"
    )

    # Callbacks
    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/har",
        filename="TREA-C-har-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=15, mode="min")

    # Logger
    logger = TensorBoardLogger("logs", name="har_activity_recognition")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint, early_stop],
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        val_check_interval=0.5,  # Check validation twice per epoch
    )

    # Train
    print("\nStarting training on HAR dataset...")
    print("This is real sensor data from smartphones!")
    trainer.fit(model, dm)

    # Evaluate final performance on validation set
    print("\nEvaluating final performance...")
    results = trainer.validate(model, dm)

    print("\nTraining complete!")
    print(f"Best model saved to: {checkpoint.best_model_path}")
    print(f"Final validation results: {results}")


if __name__ == "__main__":
    main()
