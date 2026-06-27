"""Download and prepare Human Activity Recognition (HAR) dataset.

This is a real sensor dataset that's publicly available and commonly used.
It contains accelerometer and gyroscope data from smartphones.
"""

import urllib.request
import zipfile

from pathlib import Path

import numpy as np
import torch


def download_har_dataset(data_dir="./data/har"):
    """Download UCI HAR Dataset."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
    zip_path = data_path / "har.zip"

    if not zip_path.exists():
        print("Downloading HAR dataset...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete!")

    # Extract
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_path)

    print("Dataset ready!")
    return data_path / "UCI HAR Dataset"


def load_har_data(base_path, train=True):
    """Load HAR dataset files."""
    subset = "train" if train else "test"

    # Load sensor data
    y = np.loadtxt(base_path / subset / f"y_{subset}.txt", dtype=int) - 1  # 0-indexed

    # Reshape to [samples, channels, time]
    # Original shape: [samples, 561 features]
    # 561 = 9 sensors * 17 statistics * 3 axes + some additional features
    # We'll use the raw sensor readings instead

    # Load raw inertial signals
    signals_path = base_path / subset / "Inertial Signals"

    # 9 signal types (3 axes each for acc and gyro on body and total)
    signal_names = [
        "body_acc_x",
        "body_acc_y",
        "body_acc_z",
        "body_gyro_x",
        "body_gyro_y",
        "body_gyro_z",
        "total_acc_x",
        "total_acc_y",
        "total_acc_z",
    ]

    signals = []
    for signal in signal_names:
        filename = signals_path / f"{signal}_{subset}.txt"
        signal_data = np.loadtxt(filename)
        signals.append(signal_data)

    # Stack signals: [samples, channels, time]
    X_raw = np.stack(signals, axis=1)

    return torch.FloatTensor(X_raw), torch.LongTensor(y)


class HARDataset(torch.utils.data.Dataset):
    """Human Activity Recognition dataset wrapper."""

    def __init__(self, data_dir="./data/har", train=True, download=True):
        if download:
            base_path = download_har_dataset(data_dir)
        else:
            base_path = Path(data_dir) / "UCI HAR Dataset"

        self.X, self.y = load_har_data(base_path, train)

        # HAR has 6 activities
        self.num_classes = 6
        self.activity_names = [
            "WALKING",
            "WALKING_UPSTAIRS",
            "WALKING_DOWNSTAIRS",
            "SITTING",
            "STANDING",
            "LAYING",
        ]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # HAR has no categorical features, create dummy ones
        return {
            "x_num": self.X[idx],  # [9 channels, 128 timesteps]
            "x_cat": torch.zeros(2, 128, dtype=torch.long),  # dummy
            "y": self.y[idx],
        }


if __name__ == "__main__":
    # Test loading
    print("Testing HAR dataset loading...")
    dataset = HARDataset(train=True)
    print(f"Train samples: {len(dataset)}")
    print(f"Sample shape: {dataset[0]['x_num'].shape}")
    print(f"Activities: {dataset.activity_names}")

    # Show class distribution
    unique, counts = torch.unique(dataset.y, return_counts=True)
    print("\nClass distribution:")
    for i, (cls, count) in enumerate(zip(unique, counts, strict=False)):
        print(f"  {dataset.activity_names[cls]}: {count} samples")
