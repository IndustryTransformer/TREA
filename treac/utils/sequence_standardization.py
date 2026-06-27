"""Sequence length standardization utilities for multi-dataset training."""

import numpy as np
import torch


def pad_or_truncate_sequence(
    x: torch.Tensor,
    target_length: int,
    pad_value: float = 0.0,
    truncate_strategy: str = "end",
) -> torch.Tensor:
    """Pad or truncate sequences to target length.

    Args:
        x: Input tensor [batch_size, channels, sequence_length]
        target_length: Target sequence length
        pad_value: Value to use for padding (recommend 0.0 or nan)
        truncate_strategy: 'start', 'end', or 'random'

    Returns:
        Standardized tensor [batch_size, channels, target_length]
    """
    B, C, T = x.shape

    if target_length == T:
        return x

    elif target_length < T:
        # Truncate
        if truncate_strategy == "end":
            return x[:, :, :target_length]
        elif truncate_strategy == "start":
            return x[:, :, -target_length:]
        elif truncate_strategy == "random":
            start_idx = torch.randint(0, T - target_length + 1, (1,)).item()
            return x[:, :, start_idx : start_idx + target_length]
        else:
            raise ValueError(f"Unknown truncate_strategy: {truncate_strategy}")

    else:
        # Pad
        pad_amount = target_length - T
        if pad_value != 0.0:
            padding = torch.full(
                (B, C, pad_amount), pad_value, dtype=x.dtype, device=x.device
            )
        else:
            padding = torch.zeros((B, C, pad_amount), dtype=x.dtype, device=x.device)

        return torch.cat([x, padding], dim=2)


def create_attention_mask(
    original_lengths: torch.Tensor, target_length: int
) -> torch.Tensor:
    """Create attention mask for padded sequences.

    Args:
        original_lengths: Original sequence lengths [batch_size]
        target_length: Target sequence length

    Returns:
        Attention mask [batch_size, target_length] (1 = real data, 0 = padding)
    """
    batch_size = original_lengths.shape[0]
    mask = torch.zeros(batch_size, target_length, dtype=torch.bool)

    for i, length in enumerate(original_lengths):
        mask[i, : min(length, target_length)] = True

    return mask


def adaptive_target_length(
    sequence_lengths: list[int], strategy: str = "percentile_95"
) -> int:
    """Adaptively choose target length based on dataset statistics.

    Args:
        sequence_lengths: List of sequence lengths in the dataset
        strategy: 'max', 'mean', 'median', 'percentile_95', 'percentile_99'

    Returns:
        Recommended target sequence length
    """
    lengths = np.array(sequence_lengths)

    if strategy == "max":
        return int(lengths.max())
    elif strategy == "mean":
        return int(lengths.mean())
    elif strategy == "median":
        return int(np.median(lengths))
    elif strategy == "percentile_95":
        return int(np.percentile(lengths, 95))
    elif strategy == "percentile_99":
        return int(np.percentile(lengths, 99))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


class SequenceStandardizer:
    """Utility class for standardizing sequences across datasets."""

    def __init__(
        self,
        target_length: int,
        pad_value: float = 0.0,
        truncate_strategy: str = "end",
        track_original_lengths: bool = True,
    ):
        """Initialize sequence standardizer.

        Args:
            target_length: Target sequence length
            pad_value: Value for padding
            truncate_strategy: How to truncate long sequences
            track_original_lengths: Whether to track original lengths for masking
        """
        self.target_length = target_length
        self.pad_value = pad_value
        self.truncate_strategy = truncate_strategy
        self.track_original_lengths = track_original_lengths

    def __call__(self, batch: dict) -> dict:
        """Standardize a batch of sequences.

        Args:
            batch: Dictionary containing 'x_num' and optionally other keys

        Returns:
            Standardized batch with optional attention masks
        """
        x_num = batch["x_num"]
        B, C, T = x_num.shape

        # Store original lengths if tracking
        if self.track_original_lengths:
            original_lengths = torch.full((B,), T, dtype=torch.long)
            batch["attention_mask"] = create_attention_mask(
                original_lengths, self.target_length
            )
            batch["original_lengths"] = original_lengths

        # Standardize sequence length
        batch["x_num"] = pad_or_truncate_sequence(
            x_num,
            self.target_length,
            pad_value=self.pad_value,
            truncate_strategy=self.truncate_strategy,
        )

        return batch


# Example usage for common scenarios
def get_standardizer_for_datasets(
    datasets: dict, strategy: str = "percentile_95"
) -> SequenceStandardizer:
    """Create a standardizer optimized for multiple datasets.

    Args:
        datasets: Dict mapping dataset names to sequence lengths
        strategy: Strategy for choosing target length

    Returns:
        Configured SequenceStandardizer
    """
    all_lengths = []
    for dataset_name, lengths in datasets.items():
        all_lengths.extend(lengths)
        print(
            f"{dataset_name}: {len(lengths)} samples, "
            f"lengths {min(lengths)}-{max(lengths)} (avg: {np.mean(lengths):.1f})"
        )

    target_length = adaptive_target_length(all_lengths, strategy)

    print(f"\nRecommended target length ({strategy}): {target_length}")
    truncated_count = sum(1 for length in all_lengths if length > target_length)
    total_samples = len(all_lengths)
    percentage = (100 * truncated_count / total_samples) if total_samples > 0 else 0
    print(f"Will truncate {truncated_count} samples ({percentage:.1f}%)")

    return SequenceStandardizer(
        target_length=target_length,
        pad_value=float("nan"),  # Use NaN for padding - gets handled by dual-patch
        truncate_strategy="random",  # Random truncation for better data diversity
    )


if __name__ == "__main__":
    # Test sequence standardization
    print("Testing sequence standardization...")

    # Simulate different datasets with different sequence lengths
    datasets = {
        "Short Series": [32, 45, 38, 41, 35, 42],
        "Medium Series": [96, 102, 89, 95, 108, 91],
        "Long Series": [256, 312, 278, 295, 301, 289],
        "Mixed Series": [45, 96, 127, 89, 234, 67, 156],
    }

    # Create standardizer
    standardizer = get_standardizer_for_datasets(datasets)

    # Test with sample data
    x = torch.randn(4, 3, 150)  # [batch, channels, time]
    x[0, :, 100:] = float("nan")  # Some NaNs

    batch = {"x_num": x, "y": torch.randint(0, 3, (4,))}

    print(f"\nOriginal shape: {batch['x_num'].shape}")

    standardized_batch = standardizer(batch)

    print(f"Standardized shape: {standardized_batch['x_num'].shape}")
    print(f"Attention mask shape: {standardized_batch['attention_mask'].shape}")
    print(f"Original lengths: {standardized_batch['original_lengths']}")

    # Check NaN handling
    nan_count = torch.isnan(standardized_batch["x_num"]).sum()
    print(f"NaN count after standardization: {nan_count}")
