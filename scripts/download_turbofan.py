"""Download NASA Turbofan dataset directly."""

import urllib.request
import zipfile

from pathlib import Path


def download_turbofan_dataset(data_dir="./data/nasa_turbofan"):
    """Download NASA C-MAPSS Turbofan dataset."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # The dataset is hosted on NASA's data portal
    # Direct link to the CMAPSSData.zip file
    url = "https://data.nasa.gov/api/views/nk8v-ckry/files/4f8ab55a-92eb-4834-a737-976e3c9a7d78?download=true&filename=CMAPSSData.zip"

    zip_path = data_path / "CMAPSSData.zip"

    if not zip_path.exists():
        print("Downloading NASA C-MAPSS Turbofan dataset...")
        print("This may take a few minutes (file size: ~200MB)...")

        # Download with progress
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            print(f"Progress: {percent:.1f}%", end="\r")

        urllib.request.urlretrieve(url, zip_path, download_progress)
        print("\nDownload complete!")

    # Extract the dataset
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        # Extract only the text files we need
        for file in zip_ref.namelist():
            if file.endswith(".txt") and "CMAPSSData/" in file:
                # Extract to our data directory, removing the CMAPSSData prefix
                target_path = data_path / Path(file).name
                with zip_ref.open(file) as source, open(target_path, "wb") as target:
                    target.write(source.read())

    print("Dataset ready!")

    # List extracted files
    txt_files = list(data_path.glob("*.txt"))
    print(f"\nExtracted {len(txt_files)} files:")
    for f in sorted(txt_files)[:10]:  # Show first 10
        print(f"  - {f.name}")
    if len(txt_files) > 10:
        print(f"  ... and {len(txt_files) - 10} more files")

    return data_path


if __name__ == "__main__":
    download_turbofan_dataset()
