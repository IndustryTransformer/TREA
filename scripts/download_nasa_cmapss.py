"""Download NASA C-MAPSS dataset from Kaggle."""

import shutil

from pathlib import Path

import kagglehub


def download_nasa_cmapss():
    """Download NASA C-MAPSS turbofan dataset."""

    print("Downloading NASA C-MAPSS dataset from Kaggle...")

    # Download dataset
    path = kagglehub.dataset_download("behrad3d/nasa-cmaps")

    print(f"Downloaded to: {path}")

    # Copy to our data folder
    target_dir = Path("./data/nasa_cmapss")
    target_dir.mkdir(parents=True, exist_ok=True)

    # List downloaded files
    source_path = Path(path)
    # Files are in CMaps subfolder
    cmaps_path = source_path / "CMaps"
    files = list(cmaps_path.glob("*.txt"))

    print(f"\nFound {len(files)} files")
    print("Copying to data/nasa_cmapss/...")

    for file in files:
        shutil.copy2(file, target_dir / file.name)
        print(f"  - {file.name}")

    print(f"\nDataset ready in: {target_dir}")
    return target_dir


if __name__ == "__main__":
    download_nasa_cmapss()
