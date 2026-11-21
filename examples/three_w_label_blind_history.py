"""Reconstruct causal inputs from sensors before applying annotation eligibility."""

import warnings

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from examples.benchmark_3w_binary import extract_engineered_features
from examples.three_w_causal_features import causal_history_features
from utils.three_w import _load_toolkit, sanitize_numeric_values


def sensor_features(values, mean, std, physical):
    """No labels/masks accepted: fixed source warm-up, then all sensor windows."""
    values = sanitize_numeric_values(values, "preserve_range")
    n, c = len(values) // 96, values.shape[1]
    windows = values[: n * 96].reshape(n, 96, c).transpose(0, 2, 1).copy()
    windows = (
        windows - np.asarray(mean, dtype=np.float32)[None, :, None]
    ) / np.asarray(std, dtype=np.float32)[None, :, None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        base = extract_engineered_features(windows)
    history = np.zeros((n, 4 * c), dtype=np.float32)
    burnin = 37 if physical else 0
    if n > burnin:
        history[burnin:] = causal_history_features(
            base[burnin:], np.zeros(n - burnin, dtype=np.int64), alpha=0.1
        )
    return windows, base, history


def eligible_window_indices(labels):
    """Select scored examples only AFTER all sensor/history features exist."""
    labels = np.where(np.isfinite(labels), labels, -1).astype(np.int64)
    labels = np.where(labels >= 100, labels % 100, labels)
    n = len(labels) // 96
    windows = labels[: n * 96].reshape(n, 96).astype(float)
    windows[windows < 0] = np.nan
    present = np.isfinite(windows).any(axis=1)
    rows = np.flatnonzero(present)
    median = np.nanmedian(windows[present], axis=1).astype(np.int64)
    valid = (median >= 0) & (median < 10)
    return rows[valid], (median[valid] != 0).astype(np.int64)


def file_inventory(args):
    dataset, config = _load_toolkit(args.toolkit_path)
    metadata = dataset(
        config(
            path=args.data_path,
            split=None,
            clean_data=False,
            target_column="class",
            seed=args.split_seed,
        )
    )
    return {str(path): i for i, path in enumerate(metadata.files_events)}


def reconstruct_split(data, manifest, inventory, compare_cached_history=True):
    """Raw reconstruction with annotation-blind history; verify cached alignment.

    Annotation values only select scored rows and check target alignment. Changing
    annotation availability at earlier windows cannot change a later input.
    """
    normalization = manifest["normalization"]
    names = normalization["feature_names"]
    root = Path(manifest["data_path"])
    ids = np.asarray(data["file_ids"])
    result = np.empty((len(ids), 12 * len(names)), dtype=np.float32)
    visited = np.zeros(len(ids), dtype=bool)
    report = {
        "split": manifest["split"],
        "files": 0,
        "scored_windows": 0,
        "sensor_windows": 0,
        "annotation_blind_history": True,
        "physical_burnin_windows": 37,
        "simulation_burnin_windows": 0,
        "cache_history_equivalence_required": compare_cached_history,
        "max_base_difference": 0.0,
        "max_history_difference": 0.0,
    }
    for file in manifest["files"]:
        path = root / file
        columns = [name for name in names if name in pq.read_schema(path).names]
        sensors = pd.read_parquet(path, columns=columns).reindex(columns=names)
        if (
            not isinstance(sensors.index, pd.DatetimeIndex)
            or sensors.index.hasnans
            or not sensors.index.is_monotonic_increasing
        ):
            raise ValueError(f"Nonchronological sensor input: {file}")
        windows, base, history = sensor_features(
            sensors.to_numpy(dtype=float),
            normalization["mean"],
            normalization["std"],
            "WELL-" in Path(file).name,
        )
        # Sensor features have already been constructed without annotations.
        labels = pd.read_parquet(path, columns=["class"])["class"].to_numpy()
        eligible, targets = eligible_window_indices(labels)
        rows = np.flatnonzero(ids == inventory[file])
        if len(rows) != len(eligible):
            raise ValueError(f"Cached/raw eligibility count mismatch: {file}")
        if len(rows):
            np.testing.assert_array_equal(targets, data["labels"][rows])
            np.testing.assert_allclose(
                windows[eligible], data["windows"][rows], atol=0, rtol=0, equal_nan=True
            )
            cached_base = np.asarray(data["features"][rows])
            np.testing.assert_allclose(
                base[eligible], cached_base, atol=1e-6, rtol=1e-6
            )
            report["max_base_difference"] = max(
                report["max_base_difference"],
                float(np.max(np.abs(base[eligible] - cached_base))),
            )
            if compare_cached_history:
                cached_history = causal_history_features(cached_base, ids[rows])
                np.testing.assert_allclose(
                    history[eligible],
                    cached_history,
                    atol=1e-6,
                    rtol=1e-6,
                    err_msg=f"Annotation-dependent history: {file}",
                )
                report["max_history_difference"] = max(
                    report["max_history_difference"],
                    float(np.max(np.abs(history[eligible] - cached_history))),
                )
            result[rows] = np.concatenate([base[eligible], history[eligible]], axis=1)
            visited[rows] = True
        report["files"] += 1
        report["scored_windows"] += len(rows)
        report["sensor_windows"] += len(base)
    if not visited.all():
        raise ValueError("Unmatched cached file IDs")
    return result, report
