"""PyTorch Lightning DataModule for TREA-C."""

import pytorch_lightning as pl

from torch.utils.data import DataLoader, random_split

from utils.data_config import DatasetConfig
from utils.dataset_base import SyntheticTimeSeriesDataset, TimeSeriesDataset


class TimeSeriesDataModule(pl.LightningDataModule):
    """DataModule for time series data."""

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        train_val_split: float = 0.8,
        # For pre-made datasets
        train_dataset=None,
        val_dataset=None,
        test_dataset=None,
        # For synthetic data
        synthetic: bool = True,
        num_samples: int = 1000,
        T: int = 64,
        C_num: int = 4,
        C_cat: int = 2,
        cat_cardinalities: list[int] | None = None,
        task: str = "classification",
        num_classes: int = 3,
        missing_ratio: float = 0.1,
        # For real data
        config: DatasetConfig | None = None,
        train_df=None,
        val_df=None,
        test_df=None,
    ):
        """Initialize DataModule.

        Args:
            batch_size: Batch size
            num_workers: Number of workers for data loading
            train_val_split: Train/validation split ratio
            synthetic: Whether to use synthetic data
            num_samples: Number of synthetic samples
            T: Sequence length
            C_num: Number of numeric features
            C_cat: Number of categorical features
            cat_cardinalities: Cardinalities for categorical features
            task: 'classification' or 'regression'
            num_classes: Number of classes
            missing_ratio: Ratio of missing values
            config: DatasetConfig for real data
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
        """
        super().__init__()
        # Only save hyperparameters if using synthetic data
        # When using pre-made datasets, the params aren't relevant and cause conflicts
        if train_dataset is None and val_dataset is None:
            self.save_hyperparameters()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.synthetic = synthetic

        # Pre-made datasets
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset
        self._test_dataset = test_dataset

        # Synthetic data params
        self.num_samples = num_samples
        self.T = T
        self.C_num = C_num
        self.C_cat = C_cat
        self.cat_cardinalities = cat_cardinalities
        self.task = task
        self.num_classes = num_classes
        self.missing_ratio = missing_ratio

        # Real data params
        self.config = config
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

    def setup(self, stage: str | None = None):
        """Set up datasets."""
        # If pre-made datasets are provided, use them directly
        if self._train_dataset is not None:
            self.train_dataset = self._train_dataset
            self.val_dataset = self._val_dataset
            self.test_dataset = self._test_dataset
            return

        if self.synthetic:
            # Create synthetic datasets
            if stage == "fit" or stage is None:
                full_dataset = SyntheticTimeSeriesDataset(
                    num_samples=self.num_samples,
                    T=self.T,
                    C_num=self.C_num,
                    C_cat=self.C_cat,
                    cat_cardinalities=self.cat_cardinalities,
                    task=self.task,
                    num_classes=self.num_classes,
                    missing_ratio=self.missing_ratio,
                )

                # Split into train/val
                train_size = int(self.train_val_split * len(full_dataset))
                val_size = len(full_dataset) - train_size
                self.train_dataset, self.val_dataset = random_split(
                    full_dataset, [train_size, val_size]
                )

            if stage == "test":
                self.test_dataset = SyntheticTimeSeriesDataset(
                    num_samples=200,  # Smaller test set
                    T=self.T,
                    C_num=self.C_num,
                    C_cat=self.C_cat,
                    cat_cardinalities=self.cat_cardinalities,
                    task=self.task,
                    num_classes=self.num_classes,
                    missing_ratio=self.missing_ratio,
                )
        else:
            # Use real data
            if not self.config:
                raise ValueError("DatasetConfig required for real data")

            if stage == "fit" or stage is None:
                if self.train_df is not None:
                    self.train_dataset = TimeSeriesDataset(self.train_df, self.config)
                if self.val_df is not None:
                    self.val_dataset = TimeSeriesDataset(self.val_df, self.config)

            if stage == "test":
                if self.test_df is not None:
                    self.test_dataset = TimeSeriesDataset(self.test_df, self.config)

    def train_dataloader(self):
        """Get train dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Get validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Get test dataloader."""
        if hasattr(self, "test_dataset"):
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        return None
