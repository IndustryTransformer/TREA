import numpy as np
import pytest

from examples.analyze_3w_outer_robustness import mean_fold_interval, pair_decomposition


def test_auc_pair_decomposition_preserves_ties_and_cross_fold_offsets():
    records = [
        {
            "labels": np.array([0, 0, 1, 1]),
            "neural_ensemble_score": np.array([0, 0, 1, 1]),
        },
        {
            "labels": np.array([0, 0, 1, 1]),
            "neural_ensemble_score": np.array([1, 1, 2, 2]),
        },
    ]
    result = pair_decomposition(records, "neural")
    assert result["pooled_auc"] == pytest.approx(0.875)
    assert result["within_fold_contribution"] == pytest.approx(0.5)
    assert result["cross_fold_contribution"] == pytest.approx(0.375)
    assert sum(c["pair_weight"] for c in result["cells"]) == 1


def test_equal_fold_interval_identical_models():
    r = {
        "labels": np.array([0, 1, 0, 1]),
        "groups": np.array(["WELL-a", "WELL-a", "SIM-a", "SIM-a"]),
        "neural_ensemble_score": np.array([0.1, 0.9, 0.2, 0.8]),
        "tree_ensemble_score": np.array([0.1, 0.9, 0.2, 0.8]),
    }
    result = mean_fold_interval([r, r], repeats=10)
    assert result["mean_fold_auc_delta_95_interval"] == [0, 0]
    assert result["valid_replicates"] == 10
