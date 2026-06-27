"""Configuration dataclass for dataset metadata."""

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class DatasetConfig:
    """Configuration for time series dataset.

    This dataclass stores metadata about the dataset including feature names,
    categorical cardinalities, and other properties derived from the DataFrame.
    """

    # Numeric features
    numeric_features: list[str]
    n_numeric: int

    # Categorical features
    categorical_features: list[str]
    n_categorical: int
    categorical_cardinalities: list[int]

    # Time series properties
    sequence_length: int
    n_samples: int

    # Task properties
    task: str  # 'classification' or 'regression'
    target_column: str

    # Optional fields with defaults
    categorical_mappings: dict[str, dict[str, int]] = field(default_factory=dict)
    n_classes: int | None = None  # For classification
    has_missing_values: bool = False
    missing_value_ratio: float = 0.0

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        numeric_features: list[str],
        categorical_features: list[str],
        target_column: str,
        sequence_length: int,
        task: str = "classification",
    ) -> "DatasetConfig":
        """Create config from a pandas DataFrame.

        Args:
            df: Input DataFrame with time series data
            numeric_features: List of numeric feature column names
            categorical_features: List of categorical feature column names
            target_column: Name of target column
            sequence_length: Length of each time series sequence
            task: 'classification' or 'regression'

        Returns:
            DatasetConfig instance
        """
        # Validate inputs
        all_features = numeric_features + categorical_features + [target_column]
        missing_cols = set(all_features) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

        # Calculate categorical cardinalities and create mappings
        categorical_cardinalities = []
        categorical_mappings = {}

        for cat_col in categorical_features:
            unique_values = df[cat_col].dropna().unique()
            n_unique = len(unique_values)
            categorical_cardinalities.append(n_unique)

            # Create mapping from value to index
            categorical_mappings[cat_col] = {
                val: idx for idx, val in enumerate(sorted(unique_values))
            }

        # Calculate missing value statistics
        has_missing = df[numeric_features].isna().any().any()
        missing_ratio = (
            df[numeric_features].isna().sum().sum() / (len(df) * len(numeric_features))
            if numeric_features
            else 0.0
        )

        # Determine number of classes for classification
        n_classes = None
        if task == "classification":
            n_classes = df[target_column].nunique()

        # Calculate number of samples (assuming df is in long format)
        # This might need adjustment based on actual data format
        n_samples = len(df) // sequence_length

        return cls(
            numeric_features=numeric_features,
            n_numeric=len(numeric_features),
            categorical_features=categorical_features,
            n_categorical=len(categorical_features),
            categorical_cardinalities=categorical_cardinalities,
            categorical_mappings=categorical_mappings,
            sequence_length=sequence_length,
            n_samples=n_samples,
            task=task,
            target_column=target_column,
            n_classes=n_classes,
            has_missing_values=has_missing,
            missing_value_ratio=missing_ratio,
        )

    def get_model_params(self) -> dict:
        """Get parameters needed for model initialization."""
        return {
            "C_num": self.n_numeric,
            "C_cat": self.n_categorical,
            "cat_cardinalities": self.categorical_cardinalities,
            "T": self.sequence_length,
            "task": self.task,
            "num_classes": self.n_classes if self.task == "classification" else None,
        }
