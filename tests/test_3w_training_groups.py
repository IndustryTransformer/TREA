import numpy as np
import pytest
import torch
import torch.nn.functional as F

from examples.audit_3w_training_groups import row_outputs, summarize_groups


def test_member_losses_are_not_loss_of_average_prediction():
    logits = torch.tensor([[[10.0, 0.0], [0.0, 10.0]]])
    score, loss = row_outputs(logits, torch.tensor([0]))
    assert score.item() == pytest.approx(0.5)
    assert loss.item() > 4.9
    single_score, single_loss = row_outputs(logits[:, 0], torch.tensor([0]))
    assert single_loss.item() == pytest.approx(
        F.cross_entropy(logits[:, 0], torch.tensor([0])).item()
    )
    assert single_score.item() < 0.001


def test_group_mass_and_loss_decompose():
    report = summarize_groups(
        np.array([0, 1, 0, 1]),
        np.array([0.1, 0.9, 0.8, 0.2]),
        np.array([0.1, 0.1, 2.0, 2.0]),
        np.array(["WELL-A", "WELL-A", "SIM-A", "SIM-A"]),
        np.array([2.0, 2.0, 1.0, 1.0]),
        0.5,
    )
    assert sum(r["weight_mass_fraction"] for r in report["groups"]) == pytest.approx(1)
    assert sum(r["weighted_loss_fraction"] for r in report["groups"]) == pytest.approx(
        1
    )
    assert report["sources"]["physical"]["weight_mass_fraction"] == pytest.approx(2 / 3)
    assert report["sources"]["simulation"]["weighted_loss_fraction"] == pytest.approx(
        4 / 4.4
    )
    assert report["sources"]["physical"]["auc"] == 1
    assert report["groups"][0]["group"] == "SIM-A"


def test_single_class_auc_null_and_zero_loss_safe():
    report = summarize_groups(
        np.zeros(2),
        np.zeros(2),
        np.zeros(2),
        np.array(["WELL-A", "WELL-A"]),
        np.ones(2),
        0.5,
    )
    assert report["overall"]["auc"] is None
    assert report["overall"]["weighted_loss_fraction"] == 0


def test_rejects_misaligned_arrays():
    with pytest.raises(ValueError):
        summarize_groups(
            np.zeros(2),
            np.zeros(1),
            np.zeros(2),
            np.array(["WELL-A", "WELL-A"]),
            np.ones(2),
            0.5,
        )
