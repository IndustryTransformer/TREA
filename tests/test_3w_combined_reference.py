import numpy as np
import pytest

from examples.benchmark_3w_combined_reference import combine_features


def test_combined_reference_retains_base_and_both_contexts():
    initial = np.arange(72, dtype=np.float32).reshape(3, 24)
    causal = initial.copy()
    causal[:, 16:] += 100
    result = combine_features(initial, causal)
    assert result.shape == (3, 32)
    np.testing.assert_array_equal(result[:, :24], initial)
    np.testing.assert_array_equal(result[:, 24:], causal[:, 16:])


def test_combined_reference_rejects_misaligned_base_rows():
    initial = np.zeros((2, 12), dtype=np.float32)
    causal = initial.copy()
    causal[1, 0] = 1
    with pytest.raises(AssertionError):
        combine_features(initial, causal)
    with pytest.raises(ValueError):
        combine_features(initial, np.zeros((2, 16)))
