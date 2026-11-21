from __future__ import annotations

import numpy as np
import pytest

from examples.benchmark_3w_binary import (
    ensemble_binary_outputs,
    extract_engineered_features,
    extract_naive_features,
)


def test_naive_features_preserve_temporal_order_and_nan() -> None:
    windows = np.arange(2 * 3 * 8, dtype=np.float32).reshape(2, 3, 8)
    windows[0, 1, 3] = np.nan

    features = extract_naive_features(windows, time_points=8)

    assert features.shape == (2, 24)
    np.testing.assert_allclose(features[1], windows[1].reshape(-1))
    assert np.isnan(features[0]).sum() == 1


def test_naive_features_subsample_evenly() -> None:
    windows = np.arange(10, dtype=np.float32).reshape(1, 1, 10)
    features = extract_naive_features(windows, time_points=4)
    np.testing.assert_array_equal(features, [[0.0, 3.0, 6.0, 9.0]])


def test_naive_features_validate_inputs() -> None:
    with pytest.raises(ValueError, match=r"\[N, C, T\]"):
        extract_naive_features(np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="positive"):
        extract_naive_features(np.zeros((2, 3, 4), dtype=np.float32), time_points=0)


def test_engineered_features_are_finite() -> None:
    windows = np.array(
        [[[[0.0, 1.0, np.nan, 3.0]], [[np.nan, np.nan, np.nan, np.nan]]]],
        dtype=np.float32,
    ).reshape(1, 2, 4)
    features = extract_engineered_features(windows)
    assert features.shape == (1, 16)
    assert np.isfinite(features).all()


def test_explicit_deep_ensemble_averages_probabilities() -> None:
    targets = np.array([0, 1, 1])
    outputs = [
        {
            "scores": np.array([0.2, 0.4, 0.9]),
            "targets": targets,
            "train_seconds": 2.0,
            "num_params": 10,
        },
        {
            "scores": np.array([0.4, 0.8, 0.7]),
            "targets": targets.copy(),
            "train_seconds": 3.0,
            "num_params": 10,
        },
    ]

    metrics = ensemble_binary_outputs(outputs)

    assert metrics["accuracy"] == 1.0
    assert metrics["train_seconds"] == 5.0
    assert metrics["num_params"] == 20


def test_explicit_deep_ensemble_rejects_misaligned_targets() -> None:
    output = {
        "scores": np.array([0.2, 0.8]),
        "targets": np.array([0, 1]),
        "train_seconds": 1.0,
        "num_params": 10,
    }
    mismatched = {**output, "targets": np.array([1, 0])}
    with pytest.raises(ValueError, match="Target order"):
        ensemble_binary_outputs([output, mismatched])
