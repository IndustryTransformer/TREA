"""Path utilities for consistent output directory management."""

from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory.

    Returns:
        Path to the project root (where pyproject.toml is located)
    """
    # Start from this file and go up to find pyproject.toml
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent

    # Fallback: assume we're in treac/utils/ and go up 2 levels
    return Path(__file__).resolve().parent.parent.parent


def get_outputs_dir() -> Path:
    """Get the outputs directory, creating it if it doesn't exist.

    Returns:
        Path to data/datasets/outputs/ directory
    """
    outputs_dir = get_project_root() / "datasets" / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return outputs_dir


def get_checkpoints_dir() -> Path:
    """Get the checkpoints directory, creating it if it doesn't exist.

    Returns:
        Path to checkpoints/ directory
    """
    checkpoints_dir = get_project_root() / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    return checkpoints_dir


def get_output_path(filename: str, subdir: str | None = None) -> Path:
    """Get a path for output file in the data/datasets/outputs directory.

    Args:
        filename: Name of the output file
        subdir: Optional subdirectory within outputs

    Returns:
        Full path to the output file
    """
    outputs_dir = get_outputs_dir()

    if subdir:
        outputs_dir = outputs_dir / subdir
        outputs_dir.mkdir(parents=True, exist_ok=True)

    return outputs_dir / filename


def get_checkpoint_path(model_name: str, filename: str = "best") -> Path:
    """Get a path for model checkpoint.

    Args:
        model_name: Name of the model/experiment
        filename: Checkpoint filename (default: "best")

    Returns:
        Full path to the checkpoint directory
    """
    checkpoint_dir = get_checkpoints_dir() / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


if __name__ == "__main__":
    # Test the path utilities
    print(f"Project root: {get_project_root()}")
    print(f"Outputs dir: {get_outputs_dir()}")
    print(f"Checkpoints dir: {get_checkpoints_dir()}")
    print(f"Sample output path: {get_output_path('test.csv')}")
    print(f"Sample checkpoint path: {get_checkpoint_path('test_model')}")
