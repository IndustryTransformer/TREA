from types import SimpleNamespace

import numpy as np
import pytest

from examples.benchmark_3w_binary import extract_engineered_features
from examples.benchmark_3w_initial_reference import validate_args
from examples.three_w_initial_reference import (
    initial_reference_features,
    reconstruct_development,
)


def features(values):
    return extract_engineered_features(np.asarray(values, dtype=np.float32))


def test_initial_reference_is_strictly_past_and_prefix_invariant():
    rng = np.random.default_rng(4)
    base = features(rng.normal(size=(70, 2, 96)))
    expected = initial_reference_features(base)
    np.testing.assert_array_equal(expected[0], 0)
    for length in [1, 2, 36, 37, 38, 53]:
        np.testing.assert_array_equal(
            initial_reference_features(base[:length]), expected[:length]
        )
    changed = base.copy()
    changed[45:, :2] += 1000
    np.testing.assert_array_equal(
        initial_reference_features(changed)[:45], expected[:45]
    )


def test_robust_reference_not_determined_by_a_single_initial_outlier():
    x = np.tile(np.array([[-1, 1]], dtype=np.float32), (40, 1, 48))
    base = features(x)
    changed = base.copy()
    changed[0, 0] = 1e8
    np.testing.assert_array_equal(
        initial_reference_features(base)[37:], initial_reference_features(changed)[37:]
    )


def test_reference_masks_missing_prefix_and_current_missingness():
    base = features(np.ones((40, 2, 96)))
    base[:37, 8] = 1  # First sensor absent throughout reference prefix.
    base[38, 9] = 1  # Second sensor absent in the current window.
    result = initial_reference_features(base)
    assert np.isfinite(result).all()
    assert result[37, 6] == 0 and result[37, 7] == 1
    np.testing.assert_array_equal(result[38, [1, 3, 5]], 0)


def test_reference_uses_earlier_unlabeled_sensor_values():
    base = features(np.zeros((45, 1, 96)))
    changed = base.copy()
    changed[:37, 0] = 10
    assert not np.array_equal(
        initial_reference_features(base)[37:], initial_reference_features(changed)[37:]
    )
    # No annotation argument exists; selecting different earlier scored rows
    # cannot enter the reference calculation.
    out = initial_reference_features(changed)
    np.testing.assert_array_equal(out[np.arange(37, 45)], out[37:])


def test_reference_invariant_to_positive_scale_and_shift_when_scale_nonzero():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(50, 2, 96))
    np.testing.assert_allclose(
        initial_reference_features(features(x)),
        initial_reference_features(features(x * 5 + 13)),
        atol=1e-5,
        rtol=1e-5,
    )


def test_no_outer_reconstruction_allowed():
    with pytest.raises(ValueError, match="development-only"):
        reconstruct_development({}, {"split": "test"}, {})


def test_invalid_reference_shape_and_empty_sequence():
    with pytest.raises(ValueError):
        initial_reference_features(np.zeros((4, 9)))
    assert initial_reference_features(np.zeros((0, 8))).shape == (0, 4)


def test_initial_reference_runner_rejects_outer_access_before_any_loading():
    with pytest.raises(ValueError, match="must not be loaded"):
        validate_args(SimpleNamespace(evaluate_test=True))
