import numpy as np
import pytest

from examples.benchmark_3w_grouped import choose_threshold
from examples.diagnose_3w_calibration import (
    POLICIES,
    apply_policy,
    calibration_folds,
    threshold_for,
)


def test_macro_threshold_matches_existing_exact_tie_behavior():
    y = np.array([0, 1, 1, 0, 1, 0, 0])
    score = np.array([0.1, 0.1, 0.3, 0.7, 0.8, 0.9, 0.9])
    assert threshold_for(y, score) == choose_threshold(y, score)
    assert threshold_for(y, score, weights=np.ones(len(y))) == choose_threshold(
        y, score
    )


def test_balanced_and_prior_robust_threshold_find_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1])
    score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    for objective in ["macro", "balanced", "robust"]:
        threshold = threshold_for(y, score, objective)
        np.testing.assert_array_equal(score >= threshold, y)
    with pytest.raises(ValueError, match="Both classes"):
        threshold_for(np.zeros(6), score)


def test_calibration_folds_keep_groups_together_and_both_sources_in_each_fold():
    groups = np.repeat(
        [*[f"WELL-{i}" for i in range(8)], *[f"sim-{i}" for i in range(12)]], 3
    )
    assignment = calibration_folds(groups)
    np.testing.assert_array_equal(assignment, calibration_folds(groups))
    for group in np.unique(groups):
        assert len(np.unique(assignment[groups == group])) == 1
    for fold in range(4):
        held = groups[assignment == fold]
        assert np.char.startswith(held, "WELL-").any()
        assert (~np.char.startswith(held, "WELL-")).any()
        assert not set(held) & set(groups[assignment != fold])


@pytest.mark.parametrize("policy", POLICIES)
def test_policies_fit_without_target_labels_and_return_finite_scores(policy):
    y = np.tile([0, 0, 1, 1], 8)
    score = np.tile([0.1, 0.2, 0.8, 0.9], 8)
    groups = np.repeat(
        [*[f"WELL-{i}" for i in range(4)], *[f"sim-{i}" for i in range(4)]], 4
    )
    target_score = np.array([0.15, 0.85, 0.15, 0.85])
    target_groups = np.array(["WELL-new"] * 2 + ["sim-new"] * 2)
    prediction, probability, parameters = apply_policy(
        policy, y, score, groups, target_score, target_groups
    )
    assert prediction.dtype == bool
    assert np.isfinite(probability).all()
    assert parameters
    np.testing.assert_array_equal(prediction, [False, True, False, True])
