from types import SimpleNamespace

import numpy as np
import pytest

from examples import benchmark_3w_specialists as specialists


def test_specialists_refuse_confirmation_access(tmp_path):
    with pytest.raises(ValueError, match="consumed confirmation"):
        specialists.run(SimpleNamespace(evaluate_test=True, output_dir=str(tmp_path)))


def test_specialist_sampling_partitions_the_same_budget_without_overlap(
    tmp_path, monkeypatch
):
    calls = []
    groups = np.array(["WELL-a"] * 4 + ["sim-a"] * 4)
    labels = np.tile([0, 1], 4)
    data = {
        s: {
            "windows": np.zeros((8, 2, 4), dtype=np.float32),
            "features": np.arange(128, dtype=np.float32).reshape(8, 16),
            "labels": labels,
            "group_ids": groups,
            "file_ids": np.arange(8),
        }
        for s in ["train", "val"]
    }
    protocol = {"train": {"normalization": {"mean": [0, 0], "std": [1, 1]}}, "val": {}}
    monkeypatch.setattr(specialists, "load_data", lambda args: (data, protocol))
    monkeypatch.setattr(specialists.joblib, "dump", lambda *args: None)

    class Estimator:
        def __init__(self, **kwargs):
            pass

        def fit(self, x, y, sample_weight=None):
            calls.append((x.copy(), y.copy()))

        def predict_proba(self, x):
            p = np.linspace(0.1, 0.9, len(x))
            return np.column_stack([1 - p, p])

    monkeypatch.setattr(specialists, "XGBClassifier", Estimator)
    args = SimpleNamespace(
        evaluate_test=False,
        models="xgb",
        output_dir=str(tmp_path),
        threads=1,
        feature_family="all",
        relative_stats=False,
        seeds="42",
        train_cap=0,
        group_loss_power=0.5,
        preprocess="raw",
    )
    specialists.run(args)
    physical = np.load(tmp_path / "physical/train_indices_42.npy")
    simulation = np.load(tmp_path / "simulation/train_indices_42.npy")
    assert not set(physical) & set(simulation)
    np.testing.assert_array_equal(np.sort(np.r_[physical, simulation]), np.arange(8))
    assert sum(len(x) for x, _ in calls) == len(labels)
    assert np.char.startswith(groups[physical], "WELL-").all()
    assert not np.char.startswith(groups[simulation], "WELL-").any()
