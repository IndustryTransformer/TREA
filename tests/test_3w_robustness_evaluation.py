import json

import numpy as np
import pytest

from examples import evaluate_3w_robustness as evaluation


def test_pooled_metrics_uses_fixed_fold_decisions():
    y = np.array([0, 0, 1, 1])
    score = np.array([0.8, 0.9, 0.92, 0.95])
    decisions = np.array([0, 0, 1, 1])
    result = evaluation.pooled_metrics(y, score, decisions)
    assert result["auc"] == result["macro_f1"] == 1


def test_no_outer_access_when_any_validation_reproduction_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "docs/results"
    path.mkdir(parents=True)
    (path / "3w_robustness_selection.json").write_text(
        json.dumps({"folds": [{"fold": i} for i in range(4)]})
    )
    monkeypatch.setattr(evaluation, "verify_selection", lambda _: None)
    checked = []

    def prepare(fold, device):
        checked.append(fold["fold"])
        if fold["fold"] == 3:
            raise ValueError("Validation reproduction failed")
        return {}

    def forbid_outer(args):
        pytest.fail("Outer data must stay sealed")

    monkeypatch.setattr(evaluation, "prepare_fold", prepare)
    monkeypatch.setattr(evaluation, "load_data", forbid_outer)
    with pytest.raises(ValueError, match="reproduction failed"):
        evaluation.main()
    assert checked == list(range(4))
    assert not (tmp_path / "logs/benchmark_3w_binary/robustness_outer_v1").exists()


def test_selection_requires_frozen_evaluator():
    with pytest.raises(ValueError, match="Evaluator must be frozen"):
        evaluation.verify_selection(
            {
                "outer_groups_unread_at_freeze": True,
                "folds": [{"fold": i} for i in range(4)],
                "artifact_sha256": {},
            }
        )


def test_aggregate_rejects_duplicate_outer_groups():
    records = [
        {"labels": np.array([0, 1]), "groups": np.array(["a", "a"]), "fold": i}
        for i in range(2)
    ]
    with pytest.raises(ValueError, match="multiple outer folds"):
        evaluation.aggregate(records)


def test_paired_interval_identical_models_is_zero():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    score = np.array([0.1, 0.7, 0.3, 0.9, 0.2, 0.8, 0.4, 0.6])
    result = evaluation.paired_intervals(
        y,
        score,
        score,
        y,
        y,
        np.array(
            ["WELL-a", "WELL-a", "sim-a", "sim-a", "WELL-b", "WELL-b", "sim-b", "sim-b"]
        ),
        np.array([0, 0, 0, 0, 1, 1, 1, 1]),
        repeats=10,
    )
    assert result["auc_delta_95_interval"] == [0, 0]
    assert result["macro_f1_delta_95_interval"] == [0, 0]
    assert result["valid_replicates"] == 10


def test_aggregate_separates_seed_spread_and_fold_spread(monkeypatch):
    monkeypatch.setattr(evaluation, "paired_intervals", lambda *args: {})
    records = []
    for fold in range(4):
        y = np.array([0, 1, 0, 1])
        record = {
            "fold": fold,
            "labels": y,
            "groups": np.array([f"WELL-{fold}"] * 2 + [f"sim-{fold}"] * 2),
        }
        for family in ["neural", "tree"]:
            record[family] = {}
            for seed in ["42", "43", "44", "ensemble"]:
                # Identical ranking but deliberately different fixed decisions.
                prediction = y if seed != "44" else np.ones(4, dtype=int)
                record[family][seed] = {
                    "score": np.array([0.6, 0.9, 0.7, 0.8]),
                    "prediction": prediction,
                }
        records.append(record)
    result = evaluation.aggregate(records)
    neural = result["neural"]
    assert neural["ensemble"]["macro_f1"] == 1
    assert neural["ensemble"]["fold_macro_f1"] == [1] * 4
    assert neural["44"]["std_fold_macro_f1"] == 0
    assert neural["single_fit_summary"]["macro_f1"]["mean"] == pytest.approx(7 / 9)
    assert neural["single_fit_summary"]["macro_f1"]["std"] > 0
