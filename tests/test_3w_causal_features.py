import numpy as np
import pytest

from examples.three_w_causal_features import causal_history_features


def fixture():
    features = np.zeros((6, 16), dtype=np.float32)
    features[:, :2] = [[1, 5], [3, 7], [5, 9], [100, 200], [110, 220], [120, 240]]
    features[:, 2:4] = 2
    return features, np.array([0, 0, 0, 1, 1, 1])


def test_history_is_past_only_and_file_local():
    features, files = fixture()
    reference = causal_history_features(features, files, alpha=1)
    np.testing.assert_array_equal(reference[[0, 3]], 0)
    np.testing.assert_array_equal(reference[1, :4], [2, 2, 2, 2])
    changed = features.copy()
    changed[2, :2] = 1e5
    changed[3:, :2] = -1e5
    altered = causal_history_features(changed, files, alpha=1)
    np.testing.assert_array_equal(reference[:2], altered[:2])
    # Appending arbitrary future rows cannot alter an existing prefix.
    np.testing.assert_array_equal(
        reference[:2], causal_history_features(features[:2], files[:2], alpha=1)
    )


def test_missing_history_never_exposes_a_future_reference():
    features, files = fixture()
    features[:2, 8:10] = 1
    history = causal_history_features(features, files)
    np.testing.assert_array_equal(history[:3], 0)
    features[2, :2] = 1e8
    np.testing.assert_array_equal(causal_history_features(features, files)[:2], 0)
    assert np.isfinite(history).all()


def test_causal_features_validate_shape_and_decay():
    features, files = fixture()
    with pytest.raises(ValueError, match="alpha"):
        causal_history_features(features, files, alpha=0)
    with pytest.raises(ValueError, match="eight summary"):
        causal_history_features(features[:, :-1], files)
