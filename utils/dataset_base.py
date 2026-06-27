import pandas as pd
import torch

from torch.utils.data import Dataset

from utils.data_config import DatasetConfig


class SyntheticTimeSeriesDataset(Dataset):
    """Synthetic dataset for testing and development."""

    def __init__(
        self,
        num_samples: int = 1000,
        T: int = 64,
        C_num: int = 4,
        C_cat: int = 2,
        cat_cardinalities: list | None = None,
        num_classes: int = 3,
        task: str = "classification",
        missing_ratio: float = 0.1,
    ):
        """Initialize synthetic dataset.

        Args:
            num_samples: Number of samples
            T: Sequence length
            C_num: Number of numeric features
            C_cat: Number of categorical features
            cat_cardinalities: List of cardinalities for each categorical feature
            num_classes: Number of classes for classification
            task: 'classification' or 'regression'
            missing_ratio: Ratio of missing values to introduce
        """
        self.task = task
        self.num_classes = num_classes
        self.T = T
        self.C_num = C_num
        self.C_cat = C_cat

        # Default cardinalities if not provided
        if cat_cardinalities is None:
            cat_cardinalities = [5] * C_cat
        self.cat_cardinalities = cat_cardinalities

        # Generate numeric data with some missing values
        self.x_num = torch.randn(num_samples, C_num, T)
        if missing_ratio > 0:
            mask = torch.rand(num_samples, C_num, T) < missing_ratio
            self.x_num[mask] = float("nan")

        # Generate categorical data
        self.x_cat = torch.stack(
            [
                torch.randint(0, cardinality, (num_samples, T))
                for cardinality in cat_cardinalities
            ],
            dim=1,
        )

        # Generate targets
        if task == "classification":
            # Create a complex pattern that requires learning temporal and
            # cross-channel patterns
            # Use a fixed seed for reproducible complex patterns
            gen = torch.Generator().manual_seed(42)

            # Generate random weights for combining features
            feature_weights = torch.randn(C_num, generator=gen)

            # Create signal by combining:
            # 1. Weighted average of channels (ignoring NaN)
            channel_features = []
            for i in range(C_num):
                # Replace NaN with 0 for this computation
                channel_clean = torch.nan_to_num(self.x_num[:, i, :], nan=0.0)
                channel_features.append(channel_clean)

            # Stack and weight channels
            channels_stacked = torch.stack(channel_features, dim=1)  # [B, C_num, T]
            weighted_channels = (channels_stacked * feature_weights.view(1, -1, 1)).sum(
                dim=1
            )  # [B, T]

            # 2. Temporal aggregation with overlapping windows
            signal_components = []
            window_size = T // 4
            for i in range(0, T - window_size + 1, window_size // 2):
                window = weighted_channels[:, i : i + window_size]
                # Compute both mean and std for this window
                signal_components.append(window.mean(dim=1))
                signal_components.append(window.std(dim=1))

            # 3. Include categorical dynamics
            if C_cat > 0:
                # Count unique values in first categorical channel
                cat_unique = torch.zeros(num_samples)
                for i in range(num_samples):
                    cat_unique[i] = len(torch.unique(self.x_cat[i, 0, :]))
                signal_components.append(cat_unique / 10.0)  # Normalize

            # Combine all components
            all_features = torch.stack(signal_components, dim=1)  # [B, n_features]

            # Use random projection to combine features
            projection = torch.randn(all_features.shape[1], generator=gen)
            combined_signal = all_features @ projection

            # Add moderate noise
            noise = torch.randn(num_samples, generator=gen) * 0.5
            final_signal = combined_signal + noise

            # Create balanced classes using quantiles
            sorted_signal, _ = torch.sort(final_signal)
            n_per_class = num_samples // num_classes

            self.y = torch.zeros(num_samples, dtype=torch.long)
            for i in range(num_classes):
                if i < num_classes - 1:
                    threshold = sorted_signal[(i + 1) * n_per_class]
                    mask = (final_signal >= sorted_signal[i * n_per_class]) & (
                        final_signal < threshold
                    )
                else:
                    mask = final_signal >= sorted_signal[i * n_per_class]
                self.y[mask] = i

        else:  # regression
            # Create a more complex regression target
            # Combine multiple channels with non-linear transformations
            signal_components = []

            # Use means and stds from different channels
            for i in range(min(2, C_num)):
                signal_components.append(torch.nanmean(self.x_num[:, i, :], dim=1))
                # torch.nanstd not available in all versions, use manual calculation
                x_i = self.x_num[:, i, :]
                mask = ~torch.isnan(x_i)
                mean = torch.nanmean(x_i, dim=1, keepdim=True)
                std = torch.sqrt(torch.nanmean((x_i - mean) ** 2, dim=1))
                signal_components.append(std)

            # Non-linear combination
            weights = torch.randn(len(signal_components))
            # Stack and sum to get proper type inference
            weighted_comps = [
                w * comp for w, comp in zip(weights, signal_components, strict=False)
            ]
            combined = torch.stack(weighted_comps).sum(dim=0)

            # Add non-linearity and noise
            self.y = (
                torch.tanh(combined).unsqueeze(1) + torch.randn(num_samples, 1) * 0.2
            )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"x_num": self.x_num[idx], "x_cat": self.x_cat[idx], "y": self.y[idx]}


class TimeSeriesDataset(Dataset):
    """Dataset for real time series data from pandas DataFrame."""

    def __init__(self, df: pd.DataFrame, config: DatasetConfig, transform=None):
        """Initialize dataset from DataFrame and config.

        Args:
            df: DataFrame with time series data
            config: DatasetConfig instance with metadata
            transform: Optional data transformations
        """
        self.config = config
        self.transform = transform

        # Process data based on config
        self._process_dataframe(df)

    def _process_dataframe(self, df: pd.DataFrame):
        """Process DataFrame into tensors."""
        # This assumes data is in a specific format
        # Actual implementation would depend on data structure

        n_samples = self.config.n_samples
        T = self.config.sequence_length

        # Initialize tensors
        self.x_num = torch.zeros(n_samples, self.config.n_numeric, T)
        self.x_cat = torch.zeros(
            n_samples, self.config.n_categorical, T, dtype=torch.long
        )

        # Process target
        if self.config.task == "classification":
            self.y = torch.zeros(n_samples, dtype=torch.long)
        else:
            self.y = torch.zeros(n_samples, 1)

        # TODO: Actual data processing logic would go here
        # This is a placeholder - real implementation would reshape
        # the DataFrame into the required tensor format

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = {"x_num": self.x_num[idx], "x_cat": self.x_cat[idx], "y": self.y[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample
