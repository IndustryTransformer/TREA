import numpy as np

from examples.three_w_label_blind_history import (
    eligible_window_indices,
    sensor_features,
)


def test_prior_annotation_changes_do_not_change_shared_target_inputs():
    raw = np.repeat(np.arange(42, dtype=float), 96)[:, None]
    raw = np.column_stack([raw, raw * 2 + 1])
    _, base, history = sensor_features(raw, [0, 0], [1, 1], physical=True)
    labels = np.ones(len(raw))
    labels[: 37 * 96] = np.nan
    first, _ = eligible_window_indices(labels)
    labels[38 * 96 : 39 * 96] = np.nan
    second, _ = eligible_window_indices(labels)
    shared = np.intersect1d(first, second)
    _, second_base, second_history = sensor_features(raw, [0, 0], [1, 1], physical=True)
    np.testing.assert_array_equal(base[shared], second_base[shared])
    np.testing.assert_array_equal(history[shared], second_history[shared])
    np.testing.assert_array_equal(history[:38], 0)
    assert history[39, 0] == 2


def test_eligibility_matches_binary_median_rule_and_transients():
    labels = np.r_[
        np.full(96, np.nan), np.zeros(96), np.full(96, 101), np.full(96, 999)
    ]
    indices, target = eligible_window_indices(labels)
    np.testing.assert_array_equal(indices, [1, 2])
    np.testing.assert_array_equal(target, [0, 1])


def test_raw_history_remains_causal_when_future_sensor_values_change():
    raw = np.repeat(np.arange(6, dtype=float), 96)[:, None]
    _, base, history = sensor_features(raw, [2], [3], physical=False)
    changed = raw.copy()
    changed[4 * 96 :] = 1e6
    _, future_base, future_history = sensor_features(changed, [2], [3], physical=False)
    np.testing.assert_array_equal(base[:4], future_base[:4])
    np.testing.assert_array_equal(history[:4], future_history[:4])
