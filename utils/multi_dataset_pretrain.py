"""Utilities for multi-dataset pretraining with variable feature schemas.

This module provides:
- Dataset source descriptors with schema metadata
- A collator that pads numeric/categorical features to unified feature space
- A DataModule for mixed-source pretraining
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytorch_lightning as pl
import torch

from torch.utils.data import ConcatDataset, DataLoader, Dataset


@dataclass
class DatasetSource:
    """Metadata wrapper for a dataset used in multi-dataset pretraining."""

    name: str
    dataset: Dataset
    n_numeric: int
    n_categorical: int = 0
    cat_cardinalities: list[int] | None = None
    num_classes: int = 2
    column_names: list[str] | None = None

    @classmethod
    def from_dataset(
        cls,
        name: str,
        dataset: Dataset,
        num_classes: int | None = None,
        n_numeric: int | None = None,
        n_categorical: int | None = None,
        cat_cardinalities: list[int] | None = None,
        column_names: list[str] | None = None,
    ) -> DatasetSource:
        """Build source metadata from dataset introspection + optional overrides."""
        info = {}
        if hasattr(dataset, "get_feature_info"):
            info = dataset.get_feature_info()  # type: ignore[attr-defined]

        # Infer from dataset sample if not provided by metadata.
        sample = dataset[0]
        sample_x_num = sample["x_num"]
        sample_x_cat = sample.get("x_cat", None)

        inferred_n_numeric = int(sample_x_num.shape[0])
        inferred_n_categorical = (
            int(sample_x_cat.shape[0]) if sample_x_cat is not None else 0
        )
        inferred_num_classes = int(
            info.get("num_classes", getattr(dataset, "num_classes", 2))
        )
        inferred_cat_cardinalities = info.get(
            "cat_cardinalities", [100] * inferred_n_categorical
        )
        inferred_column_names = info.get("column_names", None)

        return cls(
            name=name,
            dataset=dataset,
            n_numeric=int(
                n_numeric
                if n_numeric is not None
                else info.get("n_numeric", inferred_n_numeric)
            ),
            n_categorical=int(
                n_categorical
                if n_categorical is not None
                else info.get("n_categorical", inferred_n_categorical)
            ),
            cat_cardinalities=cat_cardinalities
            if cat_cardinalities is not None
            else inferred_cat_cardinalities,
            num_classes=int(
                num_classes if num_classes is not None else inferred_num_classes
            ),
            column_names=column_names
            if column_names is not None
            else inferred_column_names,
        )


class _DatasetAdapter(Dataset):
    """Attach dataset-level metadata to every sample item."""

    def __init__(self, source: DatasetSource):
        self.source = source

    def __len__(self) -> int:
        return len(self.source.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.source.dataset[idx]
        x_num = item["x_num"].float()
        x_cat = item.get("x_cat", None)
        y = item.get("y", None)
        if y is None:
            y = torch.tensor(-100, dtype=torch.long)

        return {
            "x_num": x_num,
            "x_cat": x_cat.long() if x_cat is not None else None,
            "y": y,
            "dataset_name": self.source.name,
            "num_classes": self.source.num_classes,
            "n_numeric": self.source.n_numeric,
            "n_categorical": self.source.n_categorical,
            "column_names": self.source.column_names,
        }


def _resize_time_axis(
    x: torch.Tensor, target_len: int, pad_value: float | int
) -> torch.Tensor:
    """Pad or truncate tensor on time axis to target length.

    Expects x shaped [C, T].
    """
    c, t = x.shape
    if t == target_len:
        return x
    if t > target_len:
        return x[:, :target_len]

    pad = torch.full((c, target_len - t), pad_value, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=1)


class MultiDatasetPretrainCollator:
    """Collate variable-schema samples into a unified padded batch."""

    def __init__(
        self,
        max_numeric_features: int,
        max_categorical_features: int,
        sequence_length: int | None = None,
    ):
        self.max_numeric_features = max_numeric_features
        self.max_categorical_features = max_categorical_features
        self.sequence_length = sequence_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        if not batch:
            raise ValueError("Empty batch in MultiDatasetPretrainCollator")

        # Use fixed sequence length if provided, otherwise longest in batch.
        if self.sequence_length is None:
            target_t = max(int(sample["x_num"].shape[1]) for sample in batch)
        else:
            target_t = self.sequence_length

        bsz = len(batch)
        x_num = torch.full(
            (bsz, self.max_numeric_features, target_t),
            float("nan"),
            dtype=torch.float32,
        )
        x_cat = torch.zeros(
            (bsz, self.max_categorical_features, target_t), dtype=torch.long
        )
        feature_mask = torch.zeros(
            (bsz, self.max_numeric_features + self.max_categorical_features, target_t),
            dtype=torch.float32,
        )
        y = torch.full((bsz,), -100, dtype=torch.long)
        dataset_name = []
        num_classes = torch.zeros((bsz,), dtype=torch.long)

        for i, sample in enumerate(batch):
            sample_x_num = _resize_time_axis(
                sample["x_num"].float(), target_t, float("nan")
            )
            sample_n_num = min(int(sample["n_numeric"]), self.max_numeric_features)
            x_num[i, :sample_n_num, :] = sample_x_num[:sample_n_num, :]
            feature_mask[i, :sample_n_num, :] = 1.0

            sample_x_cat = sample.get("x_cat", None)
            sample_n_cat = min(
                int(sample["n_categorical"]), self.max_categorical_features
            )
            if sample_x_cat is not None and sample_n_cat > 0:
                sample_x_cat = _resize_time_axis(sample_x_cat.long(), target_t, 0)
                x_cat[i, :sample_n_cat, :] = sample_x_cat[:sample_n_cat, :]
                feature_mask[
                    i,
                    self.max_numeric_features : self.max_numeric_features
                    + sample_n_cat,
                    :,
                ] = 1.0

            sample_y = sample.get("y", None)
            if sample_y is not None:
                if torch.is_tensor(sample_y):
                    y[i] = int(sample_y.item())
                else:
                    y[i] = int(sample_y)

            dataset_name.append(sample["dataset_name"])
            num_classes[i] = int(sample["num_classes"])

        return {
            "x_num": x_num,
            "x_cat": x_cat,
            "feature_mask": feature_mask,
            "y": y,
            "dataset_name": dataset_name,
            "num_classes": num_classes,
        }


class MultiDatasetPretrainDataModule(pl.LightningDataModule):
    """Lightning DataModule for multi-dataset pretraining."""

    def __init__(
        self,
        train_sources: list[DatasetSource],
        val_sources: list[DatasetSource] | None = None,
        batch_size: int = 64,
        num_workers: int = 4,
        sequence_length: int | None = None,
        shuffle_train: bool = True,
    ):
        super().__init__()
        self.train_sources = train_sources
        self.val_sources = val_sources or []
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sequence_length = sequence_length
        self.shuffle_train = shuffle_train

        self.max_numeric_features = max(source.n_numeric for source in train_sources)
        self.max_categorical_features = max(
            source.n_categorical for source in train_sources
        )
        self.max_categorical_cardinalities = (
            self._compute_max_categorical_cardinalities(train_sources)
        )

        self.collate_fn = MultiDatasetPretrainCollator(
            max_numeric_features=self.max_numeric_features,
            max_categorical_features=self.max_categorical_features,
            sequence_length=sequence_length,
        )

    def _compute_max_categorical_cardinalities(
        self, sources: list[DatasetSource]
    ) -> list[int]:
        if self.max_categorical_features == 0:
            return []

        max_cardinalities = [2] * self.max_categorical_features
        for source in sources:
            cards = source.cat_cardinalities or []
            for i in range(min(len(cards), self.max_categorical_features)):
                max_cardinalities[i] = max(max_cardinalities[i], int(cards[i]))
        return max_cardinalities

    def setup(self, stage: str | None = None):
        train_adapters = [_DatasetAdapter(source) for source in self.train_sources]
        self.train_dataset = ConcatDataset(train_adapters)

        if self.val_sources:
            val_adapters = [_DatasetAdapter(source) for source in self.val_sources]
            self.val_dataset = ConcatDataset(val_adapters)
        else:
            self.val_dataset = None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader | list:
        if self.val_dataset is None:
            # Lightning expects an iterable; use empty iterable to disable val loop.
            return []
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
            persistent_workers=self.num_workers > 0,
        )

    def model_schema_kwargs(self) -> dict[str, Any]:
        """Arguments to initialize MultiDatasetModel for this datamodule."""
        return {
            "max_numeric_features": self.max_numeric_features,
            "max_categorical_features": self.max_categorical_features,
            "categorical_cardinalities": self.max_categorical_cardinalities,
        }
