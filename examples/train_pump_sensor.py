"""Train TriplePatchTransformer on Industrial Pump Sensor dataset."""

import sys


sys.path.append(".")

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl

from data.downloaders.pump_sensor import PumpSensorDataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from treac.models.triple_attention import TriplePatchTransformer
from utils.datamodule import TimeSeriesDataModule


def visualize_pump_data(dataset, num_samples=3):
    """Visualize some pump sensor data."""
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4 * num_samples))

    feature_info = dataset.get_feature_info()

    for i in range(num_samples):
        sample = dataset[i]
        x_num = sample["x_num"].numpy()
        label = sample["y"].item()
        fault_type = feature_info["fault_types"][label]

        # Plot vibration signals
        ax1 = axes[i, 0] if num_samples > 1 else axes[0]
        time = np.arange(x_num.shape[1])
        ax1.plot(time, x_num[0], label="Vib X", alpha=0.7)
        ax1.plot(time, x_num[1], label="Vib Y", alpha=0.7)
        ax1.plot(time, x_num[2], label="Vib Z", alpha=0.7)
        ax1.set_title(f"Sample {i}: Vibration Signals - {fault_type}")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Vibration")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot other sensors
        ax2 = axes[i, 1] if num_samples > 1 else axes[1]
        ax2_twin = ax2.twinx()

        # Pressure and flow on left axis
        ax2.plot(time, x_num[4], "b-", label="Pressure", alpha=0.7)
        ax2.plot(time, x_num[5], "g-", label="Flow Rate", alpha=0.7)
        ax2.set_ylabel("Pressure / Flow", color="b")
        ax2.tick_params(axis="y", labelcolor="b")

        # Temperature on right axis
        ax2_twin.plot(time, x_num[3], "r-", label="Temperature", alpha=0.7)
        ax2_twin.set_ylabel("Temperature", color="r")
        ax2_twin.tick_params(axis="y", labelcolor="r")

        ax2.set_title(f"Sample {i}: Process Variables - {fault_type}")
        ax2.set_xlabel("Time")
        ax2.grid(True, alpha=0.3)

        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    plt.savefig("pump_sensor_visualization.png", dpi=150)
    print("Saved visualization to pump_sensor_visualization.png")
    plt.close()


def main():
    print("=" * 60)
    print("Industrial Pump Sensor Fault Detection")
    print("=" * 60)

    # Create datasets
    print("\nCreating pump sensor datasets...")
    train_dataset = PumpSensorDataset(
        train=True,
        sequence_length=100,
        num_samples=5000,
        task="classification",
        num_classes=4,
    )

    val_dataset = PumpSensorDataset(
        train=False,
        sequence_length=100,
        num_samples=1000,
        task="classification",
        num_classes=4,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Get feature info
    feature_info = train_dataset.get_feature_info()
    print("\nDataset characteristics:")
    print(f"- Numeric features: {feature_info['numeric_names']}")
    print(f"- Categorical features: {feature_info['categorical_names']}")
    print(f"- Fault types: {feature_info['fault_types']}")

    # Visualize some samples
    print("\nVisualizing sample data...")
    visualize_pump_data(train_dataset, num_samples=3)

    # Create data module
    dm = TimeSeriesDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=64,
        num_workers=4,
    )

    # Create model
    model = TriplePatchTransformer(
        C_num=feature_info["n_numeric"],
        C_cat=feature_info["n_categorical"],
        cat_cardinalities=feature_info["cat_cardinalities"],
        T=100,
        d_model=96,
        n_head=6,
        num_layers=4,
        task="classification",
        num_classes=4,
    )

    print(
        f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters"
    )

    # Callbacks
    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/pump_sensor",
        filename="pump-fault-{epoch:02d}-{val_loss:.3f}",
        save_top_k=3,
        mode="min",
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=20, mode="min")

    # Logger
    logger = TensorBoardLogger(save_dir="logs", name="pump_sensor_fault_detection")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint, early_stop],
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        val_check_interval=0.5,
    )

    # Train
    print("\nStarting training for pump fault detection...")
    print("This simulates a real industrial predictive maintenance scenario!")
    trainer.fit(model, dm)

    # Test final performance
    print("\nEvaluating on validation set...")
    results = trainer.test(model, dm.val_dataloader())

    print("\nTraining complete!")
    print(f"Best model saved to: {checkpoint.best_model_path}")
    print(f"Test results: {results}")

    # Show per-class accuracy if available
    print("\nFault detection performance:")
    for i, fault in enumerate(feature_info["fault_types"]):
        print(f"  - {fault}: Check TensorBoard for detailed metrics")


if __name__ == "__main__":
    main()
