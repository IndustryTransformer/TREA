"""3W petroleum well dataset loader for TREA-C.

Wraps the ThreeWToolkit's ParquetDataset to produce windowed time series
in TREA-C's standard dict format (x_num, x_cat, y) with NaN values preserved
for triple-encoded attention.

Requires the 3W toolkit repo to be cloned locally:
    git clone https://github.com/petrobras/3W.git

Usage:
    from utils.three_w import ThreeWDataset

    train_ds = ThreeWDataset(
        toolkit_path="/path/to/3W",
        data_path="/path/to/3W/dataset",
        split="train",
    )
    val_ds = ThreeWDataset(
        toolkit_path="/path/to/3W",
        data_path="/path/to/3W/dataset",
        split="val",
        normalization_stats=train_ds.normalization_stats,
    )
"""

from __future__ import annotations

import hashlib
import re
import sys

from pathlib import Path

import numpy as np
import torch

from torch.utils.data import Dataset


SENSOR_COLUMNS = [
    "ABER-CKGL",
    "ABER-CKP",
    "ESTADO-DHSV",
    "ESTADO-M1",
    "ESTADO-M2",
    "ESTADO-PXO",
    "ESTADO-SDV-GL",
    "ESTADO-SDV-P",
    "ESTADO-W1",
    "ESTADO-W2",
    "ESTADO-XO",
    "P-ANULAR",
    "P-JUS-BS",
    "P-JUS-CKGL",
    "P-JUS-CKP",
    "P-MON-CKGL",
    "P-MON-CKP",
    "P-MON-SDV-P",
    "P-PDG",
    "PT-P",
    "P-TPT",
    "QBS",
    "QGL",
    "T-JUS-CKP",
    "T-MON-CKP",
    "T-PDG",
    "T-TPT",
]

EVENT_NAMES = {
    0: "Normal Operation",
    1: "Abrupt Increase of BSW",
    2: "Spurious Closure of DHSV",
    3: "Severe Slugging",
    4: "Flow Instability",
    5: "Rapid Productivity Loss",
    6: "Quick Restriction in PCK",
    7: "Scaling in PCK",
    8: "Hydrate in Production Line",
    9: "Hydrate in Service Line",
}

NUM_CLASSES = 10


def sanitize_numeric_values(values, value_policy="legacy_clip"):
    """Prepare float32 sensor values without silently flattening valid pressures.

    The opt-in policy preserves finite values up to 1e12 in magnitude, masking
    larger values as invalid before downcasting. The cutoff was chosen from a
    training-only audit: pressures reach 1e7-1e10, while extreme negative artifacts
    reach 1e15-1e42. It is not a learned bound or a test-derived clipping threshold.
    """
    x = np.array(values, dtype=np.float64, copy=True)
    if value_policy == "legacy_clip":
        np.clip(x, -1e6, 1e6, out=x)
    elif value_policy == "preserve_range":
        x[~np.isfinite(x) | (np.abs(x) > 1e12)] = np.nan
    else:
        raise ValueError(f"Unknown value_policy: {value_policy}")
    return x.astype(np.float32, copy=False)


def grouped_file_split(files, holdout_fraction=0.2, internal_val_fraction=0.0, seed=42):
    """Partition wells; optional validation is drawn only from development wells.

    Files without a WELL id are separate simulation instances, not physical wells.
    Fractions and group membership are fixed before reading signals or labels.
    """
    if not 0 < holdout_fraction < 1 or not 0 <= internal_val_fraction < 1:
        raise ValueError("Invalid split fractions")
    groups = np.array(
        [
            m.group(0) if (m := re.search(r"WELL-\d+", str(p))) else f"sim-{i}"
            for i, p in enumerate(files)
        ]
    )
    wells = np.array(sorted(set(groups)))
    order = np.random.RandomState(seed).permutation(len(wells))
    n_test = max(1, int(round(len(wells) * holdout_fraction)))
    test = set(wells[order[:n_test]])
    remaining = wells[order[n_test:]]
    n_val = (
        max(1, int(round(len(remaining) * internal_val_fraction)))
        if internal_val_fraction
        else 0
    )
    val = set(np.random.RandomState(seed + 1).permutation(remaining)[:n_val])
    train = set(remaining) - val
    if not train or (internal_val_fraction and not val):
        raise ValueError("Too few groups for the requested split")
    partitions = {
        "train": train,
        "val": val if internal_val_fraction else test,
        "test": test,
    }
    return groups, {
        name: np.flatnonzero(np.isin(groups, list(ids)))
        for name, ids in partitions.items()
    }


def frozen_group_split(files, manifest):
    """Validate complete frozen membership against the exact file inventory."""
    if list(map(str, files)) != sorted(map(str, files)):
        raise ValueError("Frozen simulation group IDs require canonical file order")
    digest = hashlib.sha256("\n".join(sorted(map(str, files))).encode()).hexdigest()
    if digest != manifest["file_inventory_sha256"]:
        raise ValueError("Frozen protocol file inventory mismatch")
    groups, _ = grouped_file_split(files)
    partitions = manifest["partitions"]
    if set(partitions) != {"train", "val", "test", "excluded"}:
        raise ValueError("Frozen protocol requires train/val/test/excluded")
    flattened = [g for gs in partitions.values() for g in gs]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Duplicate or overlapping frozen groups")
    if set(flattened) != set(groups):
        raise ValueError("Frozen protocol must cover every group exactly once")
    if any(not partitions[s] for s in ["train", "val", "test"]):
        raise ValueError("Empty frozen development or test partition")
    return groups, {
        name: np.flatnonzero(np.isin(groups, ids)) for name, ids in partitions.items()
    }


def _load_toolkit(toolkit_path: str):
    """Import ThreeWToolkit, adding toolkit_path to sys.path if needed."""
    toolkit_dir = str(Path(toolkit_path) / "toolkit")
    if toolkit_dir not in sys.path:
        sys.path.insert(0, toolkit_dir)

    from ThreeWToolkit.core.base_dataset import ParquetDatasetConfig
    from ThreeWToolkit.dataset import ParquetDataset

    return ParquetDataset, ParquetDatasetConfig


class ThreeWDataset(Dataset):
    """3W petroleum well dataset for TREA-C triple-encoded attention.

    Loads well event files via the ThreeWToolkit, applies sliding windows,
    normalizes with nanmean/nanstd (preserving NaNs), and returns data in
    TREA-C's standard dict format.

    Args:
        toolkit_path: Path to the cloned petrobras/3W repository root.
        data_path: Path to the 3W dataset directory (contains 0/-9/ subdirs).
            Defaults to ``{toolkit_path}/dataset``.
        window_size: Sliding window length (T).
        stride: Sliding window stride.
        split: One of "train", "val", or "test". Legacy two-way splitting
            aliases test to val. Set internal_val_fraction > 0 with well mode
            to reserve a distinct test partition before selecting validation.
        internal_val_fraction: Fraction of non-test groups used for internal
            validation. Zero preserves the legacy two-way protocol.
        val_fraction: Fraction of files held out for validation.
        normalization_stats: Pre-computed {"mean": array, "std": array} from
            the training set. If None, stats are computed from this dataset.
        max_files: Limit the number of files loaded (for debugging).
        seed: Random seed for reproducible file-level splitting.
        max_feature_nan_rate: Drop features with NaN rate above this threshold
            when building training stats.
    """

    # Classes with fewer than this many files get denser windowing
    RARE_CLASS_FILE_THRESHOLD = 100
    # Drop features that are effectively empty in train (all-NaN normalization hazard)
    MAX_FEATURE_NAN_RATE = 0.99

    def __init__(
        self,
        toolkit_path: str,
        data_path: str | None = None,
        window_size: int = 192,
        stride: int = 192,
        rare_class_stride: int | None = None,
        split: str = "train",
        split_mode: str = "random",
        val_fraction: float = 0.2,
        normalization_stats: dict[str, np.ndarray] | None = None,
        max_files: int | None = None,
        seed: int = 42,
        augment: bool = False,
        max_feature_nan_rate: float = MAX_FEATURE_NAN_RATE,
        internal_val_fraction: float = 0.0,
        value_policy: str = "legacy_clip",
        group_manifest: dict | None = None,
    ):
        self.window_size = window_size
        self.stride = stride
        self.rare_class_stride = rare_class_stride
        self.split = split
        self.augment = augment
        self.max_feature_nan_rate = max_feature_nan_rate
        if value_policy not in {"legacy_clip", "preserve_range"}:
            raise ValueError(f"Unknown value_policy: {value_policy}")
        self.value_policy = value_policy
        self.feature_names = list(SENSOR_COLUMNS)
        self.feature_indices = np.arange(len(SENSOR_COLUMNS), dtype=np.int64)
        self.n_features = len(self.feature_names)

        if data_path is None:
            data_path = str(Path(toolkit_path) / "dataset")

        ParquetDataset, ParquetDatasetConfig = _load_toolkit(toolkit_path)

        # Load all files with clean_data=False to preserve NaN values
        config = ParquetDatasetConfig(
            path=data_path,
            split=None,  # load all files (train/val/test split not implemented)
            clean_data=False,
            target_column="class",
            seed=seed,
        )
        parquet_ds = ParquetDataset(config)

        # Split for train/val. split_mode="random" splits at the file level (a
        # well's files may span train/val -> leaks well identity, since 3W
        # missingness is structural per well). split_mode="well" groups by well
        # so no well appears in both -> honest generalization to unseen wells.
        n_total = len(parquet_ds)
        rng = np.random.RandomState(seed)

        if split_mode == "well":
            well_of, partitions = grouped_file_split(
                parquet_ds.files_events, val_fraction, internal_val_fraction, seed
            )
            if group_manifest is not None:
                if not internal_val_fraction:
                    raise ValueError("Frozen protocol requires three-way splitting")
                well_of, partitions = frozen_group_split(
                    parquet_ds.files_events, group_manifest
                )
            train_indices, val_indices = partitions["train"], partitions["val"]
            print(
                f"[ThreeWDataset] grouped split: "
                f"{ {k: len(set(well_of[v])) for k, v in partitions.items()} }"
            )
        elif split_mode == "random":
            if internal_val_fraction or group_manifest is not None:
                raise ValueError(
                    "Internal grouped validation requires split_mode='well'"
                )
            well_of = np.array([f"file-{i}" for i in range(n_total)])
            file_indices = rng.permutation(n_total)
            n_val = int(n_total * val_fraction)
            train_indices = file_indices[: n_total - n_val]
            val_indices = file_indices[n_total - n_val :]
        else:
            raise ValueError(
                f"Unknown split_mode: {split_mode!r}. Use 'random' or 'well'."
            )

        if split == "train":
            selected_indices = train_indices
        elif split in ("val", "test"):
            # Preserve the legacy alias unless the three-way protocol is enabled.
            selected_indices = val_indices
            if split == "test" and internal_val_fraction:
                selected_indices = partitions["test"]
        else:
            raise ValueError(
                f"Unknown split: {split!r}. Use 'train', 'val', or 'test'."
            )

        if max_files is not None:
            selected_indices = selected_indices[:max_files]
        self.selected_files = [
            str(parquet_ds.files_events[i]) for i in selected_indices
        ]
        self.selected_groups = sorted(set(well_of[selected_indices]))

        # Count files per class directory to identify rare classes
        rare_classes = set()
        if rare_class_stride is not None:
            data_dir = Path(data_path)
            for cls_id in range(NUM_CLASSES):
                cls_dir = data_dir / str(cls_id)
                if cls_dir.exists():
                    n_cls_files = len(list(cls_dir.glob("*.parquet")))
                    if n_cls_files < self.RARE_CLASS_FILE_THRESHOLD:
                        rare_classes.add(cls_id)
            if rare_classes:
                print(
                    f"  Rare classes (< {self.RARE_CLASS_FILE_THRESHOLD} files): "
                    f"{sorted(rare_classes)}"
                )
                print(
                    f"  Using stride={rare_class_stride} for rare classes, "
                    f"stride={stride} otherwise"
                )

        # Extract windows from each file
        all_windows = []
        all_labels = []
        all_file_ids = []
        all_group_ids = []
        n_files = len(selected_indices)

        print(f"[ThreeWDataset] Loading {n_files} files for split={split!r}...")

        for i, file_idx in enumerate(selected_indices):
            try:
                item = parquet_ds[int(file_idx)]
            except (ValueError, KeyError) as e:
                print(f"  Skipping file {file_idx}: {e}")
                continue

            signal = item["signal"]
            label_df = item["label"]

            # Select only sensor columns (some may be missing in a file)
            available = [c for c in SENSOR_COLUMNS if c in signal.columns]
            missing = [c for c in SENSOR_COLUMNS if c not in signal.columns]

            features = sanitize_numeric_values(
                signal[available].to_numpy(dtype=np.float64, copy=False), value_policy
            )

            # Add NaN columns for any missing sensors
            if missing:
                nan_cols = np.full(
                    (features.shape[0], len(missing)), np.nan, dtype=np.float32
                )
                features = np.column_stack([features, nan_cols])
                # Reorder to match SENSOR_COLUMNS
                col_order = [
                    available.index(c)
                    if c in available
                    else len(available) + missing.index(c)
                    for c in SENSOR_COLUMNS
                ]
                features = features[:, col_order]

            # Extract labels and map transients (101-109 → 1-9)
            labels = label_df["class"].to_numpy(dtype=np.float64, copy=False)
            labels = np.where(np.isnan(labels), -1, labels).astype(np.int64)
            labels = np.where(labels >= 100, labels % 100, labels)

            # Determine stride for this file: use smaller stride for rare classes
            valid_labels = labels[labels >= 0]
            if len(valid_labels) > 0 and rare_classes:
                dominant_class = int(np.median(valid_labels))
                file_stride = (
                    rare_class_stride if dominant_class in rare_classes else stride
                )
            else:
                file_stride = stride

            # Slide windows
            T_file = len(features)
            if T_file < window_size:
                continue

            for start in range(0, T_file - window_size + 1, file_stride):
                end = start + window_size
                window_labels = labels[start:end]

                # Skip windows with invalid labels
                valid_mask = window_labels >= 0
                if valid_mask.sum() == 0:
                    continue

                # Majority label in window
                window_label = int(np.median(window_labels[valid_mask]))
                if window_label < 0 or window_label >= NUM_CLASSES:
                    continue

                # Store as (C, T) — channels first for TREA-C
                all_windows.append(features[start:end].T)  # [C, T]
                all_labels.append(window_label)
                all_file_ids.append(int(file_idx))
                all_group_ids.append(well_of[file_idx])

            if (i + 1) % 200 == 0 or i == n_files - 1:
                print(
                    f"  Processed {i + 1}/{n_files} files, "
                    f"{len(all_windows)} windows so far"
                )

        self.windows = np.array(all_windows, dtype=np.float32)  # [N, C, T]
        self.labels = np.array(all_labels, dtype=np.int64)  # [N]
        self.file_ids = np.asarray(all_file_ids, dtype=np.int64)
        self.group_ids = np.asarray(all_group_ids)

        print(
            f"[ThreeWDataset] {split}: {len(self.labels)} windows from {n_files} files"
        )

        if len(self.labels) == 0:
            raise RuntimeError(
                "No valid windows were extracted from the selected files."
            )

        # Keep train-selected feature subset aligned across splits via
        # normalization_stats.
        if normalization_stats is not None:
            if "feature_indices" in normalization_stats:
                self.feature_indices = np.asarray(
                    normalization_stats["feature_indices"], dtype=np.int64
                )
            elif "feature_names" in normalization_stats:
                index_by_name = {name: idx for idx, name in enumerate(SENSOR_COLUMNS)}
                self.feature_indices = np.asarray(
                    [
                        index_by_name[name]
                        for name in normalization_stats["feature_names"]
                    ],
                    dtype=np.int64,
                )
            else:
                self.feature_indices = np.arange(len(SENSOR_COLUMNS), dtype=np.int64)
        else:
            feature_nan_rates = np.isnan(self.windows).mean(axis=(0, 2))
            keep_mask = feature_nan_rates <= self.max_feature_nan_rate
            if not np.any(keep_mask):
                keep_mask[:] = True
            self.feature_indices = np.where(keep_mask)[0]
            dropped_indices = np.where(~keep_mask)[0]
            if len(dropped_indices) > 0:
                dropped = [
                    (
                        SENSOR_COLUMNS[idx],
                        float(feature_nan_rates[idx]),
                    )
                    for idx in dropped_indices.tolist()
                ]
                print(
                    f"  Dropping {len(dropped)} near-empty features "
                    f"(NaN rate > {self.max_feature_nan_rate:.2f}):"
                )
                for name, rate in dropped:
                    print(f"    - {name}: {rate:.3f}")

        self.feature_names = [
            SENSOR_COLUMNS[idx] for idx in self.feature_indices.tolist()
        ]
        self.windows = self.windows[:, self.feature_indices, :]
        self.n_features = len(self.feature_names)
        print(f"  Active features: {self.n_features}/{len(SENSOR_COLUMNS)}")

        # Class distribution
        unique, counts = np.unique(self.labels, return_counts=True)
        print(f"  Classes: {dict(zip(unique.tolist(), counts.tolist(), strict=False))}")

        # Compute class weights (inverse frequency)
        total = len(self.labels)
        self.class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        for cls, count in zip(unique, counts, strict=False):
            self.class_counts[cls] = count
        self.class_weights = np.zeros(NUM_CLASSES, dtype=np.float32)
        for cls, count in zip(unique, counts, strict=False):
            self.class_weights[cls] = total / (count * NUM_CLASSES)
        # Set weight=0 for absent classes to avoid NaN loss
        self.class_weights[self.class_weights == 0] = 0.0

        # Normalization
        if normalization_stats is not None:
            self.mean = np.asarray(normalization_stats["mean"], dtype=np.float32)
            self.std = np.asarray(normalization_stats["std"], dtype=np.float32)
            if (
                self.mean.shape[0] != self.n_features
                or self.std.shape[0] != self.n_features
            ):
                raise ValueError(
                    "Normalization stats feature dimension does not match active "
                    "features. Pass train_ds.normalization_stats from the same "
                    "feature selection."
                )
            self.std[self.std == 0] = 1.0
            print("  Using pre-computed normalization stats")
        else:
            self.mean = np.nanmean(self.windows, axis=(0, 2))  # [C]
            self.std = np.nanstd(self.windows, axis=(0, 2))  # [C]
            self.mean = np.nan_to_num(self.mean, nan=0.0)
            self.std = np.nan_to_num(self.std, nan=1.0)
            self.std[self.std == 0] = 1.0
            print("  Computed normalization stats from data")

        # Apply normalization: broadcast [C] over [N, C, T]
        self.windows = (self.windows - self.mean[np.newaxis, :, np.newaxis]) / self.std[
            np.newaxis, :, np.newaxis
        ]
        # NaN values propagate correctly: (NaN - mean) / std = NaN

    @property
    def normalization_stats(self) -> dict[str, np.ndarray]:
        """Return normalization stats for passing to val/test datasets."""
        return {
            "mean": self.mean,
            "std": self.std,
            "feature_indices": self.feature_indices,
            "feature_names": np.array(self.feature_names),
        }

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x = torch.from_numpy(self.windows[idx].copy())  # [C_num, T]

        if self.augment:
            # Light augmentation: jitter + scaling on non-NaN values only
            nan_mask = torch.isnan(x)
            # Work on non-NaN values: replace NaN with 0 temporarily
            x_clean = torch.nan_to_num(x, nan=0.0)
            # Gaussian jitter
            jitter = torch.randn_like(x_clean) * 0.01
            x_clean = x_clean + jitter
            # Per-channel random scaling
            scale = 1.0 + torch.randn(x_clean.shape[0], 1) * 0.05  # [C, 1]
            x_clean = x_clean * scale
            # Restore NaN positions
            x = x_clean.masked_fill(nan_mask, float("nan"))

        return {
            "x_num": x,  # [C_num, T]
            "x_cat": torch.zeros(0, self.window_size, dtype=torch.long),
            "y": torch.tensor(self.labels[idx], dtype=torch.long),
        }

    def get_feature_info(self) -> dict:
        """Return feature metadata for model construction."""
        from utils.three_w_columns import describe_columns

        return {
            "n_numeric": self.n_features,
            "n_categorical": 0,
            "cat_cardinalities": [],
            "sequence_length": self.window_size,
            "num_classes": NUM_CLASSES,
            "column_names": list(self.feature_names),
            "column_descriptions": describe_columns(self.feature_names),
        }
