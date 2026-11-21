"""Strictly past, annotation-blind initial sensor references for 3W."""

import warnings

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from examples.three_w_label_blind_history import (
    eligible_window_indices,
    sensor_features,
)


def initial_reference_features(base, reference_windows=37):
    """Four blocks relative to the first min(37, j) strictly earlier windows.

    An initial period is a reference, not an assumed healthy period. Use median
    means and median within-window std plus scaled MAD of means as the scale.
    Current-window offset, std, and range are signed-log compressed; unavailable
    references are explicitly masked. No labels, source identity, or future rows.
    """
    base = np.asarray(base, dtype=np.float64)
    if base.ndim != 2 or base.shape[1] % 8 or reference_windows < 1:
        raise ValueError("Expected eight summary blocks and a positive prefix length")
    n, width = base.shape
    c = width // 8
    if not c:
        raise ValueError("No sensors")
    out = np.zeros((n, 4 * c), dtype=np.float32)
    mean, std, low, high, missing, _, _, _ = np.split(base, 8, axis=1)
    observed = missing < 1

    def apply(start, stop):
        prefix = min(start, reference_windows)
        initial_mean = np.where(observed[:prefix], mean[:prefix], np.nan)
        initial_std = np.where(observed[:prefix], std[:prefix], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            location = np.nanmedian(initial_mean, axis=0)
            mad = np.nanmedian(np.abs(initial_mean - location), axis=0)
            scale = np.nanmedian(initial_std, axis=0) + 1.4826 * mad
        available = np.isfinite(location) & np.isfinite(scale)
        location = np.nan_to_num(location)
        scale = np.maximum(np.nan_to_num(scale), 1e-6)
        valid = available[None, :] & observed[start:stop]
        values = [
            (mean[start:stop] - location) / scale,
            std[start:stop] / scale,
            (high[start:stop] - low[start:stop]) / scale,
        ]
        for block, value in enumerate(values):
            compressed = np.sign(value) * np.log1p(np.abs(value))
            out[start:stop, block * c : (block + 1) * c] = np.where(
                valid, compressed, 0
            )
        out[start:stop, 3 * c :] = available.astype(np.float32)

    for j in range(1, min(n, reference_windows)):
        apply(j, j + 1)
    if n > reference_windows:
        apply(reference_windows, n)
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite initial-reference inputs")
    return out


def reconstruct_development(data, manifest, inventory):
    """Sensors first, annotations only select scored rows. Reject outer splits."""
    if manifest["split"] not in {"train", "val"}:
        raise ValueError("Initial-reference screening is development-only")
    stats = manifest["normalization"]
    names = stats["feature_names"]
    ids = np.asarray(data["file_ids"])
    result = np.empty((len(ids), 12 * len(names)), dtype=np.float32)
    visited = np.zeros(len(ids), dtype=bool)
    report = {
        "split": manifest["split"],
        "files": 0,
        "sensor_windows": 0,
        "scored_windows": 0,
        "annotation_blind": True,
        "reference_uses_unlabeled_prefix": True,
    }
    for file in manifest["files"]:
        path = Path(manifest["data_path"]) / file
        columns = [name for name in names if name in pq.read_schema(path).names]
        sensors = pd.read_parquet(path, columns=columns).reindex(columns=names)
        if (
            not isinstance(sensors.index, pd.DatetimeIndex)
            or sensors.index.hasnans
            or not sensors.index.is_monotonic_increasing
        ):
            raise ValueError(f"Nonchronological input: {file}")
        windows, base, _ = sensor_features(
            sensors.to_numpy(dtype=float), stats["mean"], stats["std"], physical=False
        )
        reference = initial_reference_features(base)
        # All sensor inputs exist before annotations are read.
        labels = pd.read_parquet(path, columns=["class"])["class"].to_numpy()
        eligible, targets = eligible_window_indices(labels)
        rows = np.flatnonzero(ids == inventory[file])
        if len(rows) != len(eligible):
            raise ValueError(f"Eligibility mismatch: {file}")
        if len(rows):
            np.testing.assert_array_equal(targets, data["labels"][rows])
            np.testing.assert_allclose(
                windows[eligible], data["windows"][rows], atol=0, rtol=0, equal_nan=True
            )
            np.testing.assert_allclose(
                base[eligible], data["features"][rows], atol=0, rtol=0
            )
            result[rows] = np.concatenate([base[eligible], reference[eligible]], axis=1)
            visited[rows] = True
        report["files"] += 1
        report["sensor_windows"] += len(base)
        report["scored_windows"] += len(rows)
    if not visited.all():
        raise ValueError("Unmatched cached rows")
    current = np.asarray(data["features"])[:, 4 * len(names) : 5 * len(names)] < 1
    available = result[:, -len(names) :] > 0
    for source, mask in [
        ("physical", np.char.startswith(data["group_ids"], "WELL-")),
        ("simulation", ~np.char.startswith(data["group_ids"], "WELL-")),
    ]:
        denominator = int(current[mask].sum())
        report[source] = {
            "observed_sensor_windows": denominator,
            "with_initial_reference": int((current[mask] & available[mask]).sum()),
            "fraction_with_reference": float(
                (current[mask] & available[mask]).sum() / denominator
            )
            if denominator
            else None,
        }
    return result, report
