"""Guard the reference comparison's ensemble objective and selection boundary."""

import pytest
import torch

from scripts.nox_reference_models import regression_loss, select


def test_tabm_loss_does_not_cancel_opposing_member_errors():
    predictions = torch.tensor([[[-1.0], [1.0]]], requires_grad=True)
    loss = regression_loss(predictions, torch.tensor([0.0]))
    assert loss.item() == pytest.approx(1.0)
    loss.backward()
    assert predictions.grad.flatten().tolist() == [-1.0, 1.0]


def test_selection_ignores_test_scores():
    records = [
        {"candidate": 0, "rmse": {"val": 5.0, "iid": 1.0}},
        {"candidate": 1, "rmse": {"val": 4.0, "iid": 100.0}},
    ]
    assert select(records) == 1


def test_single_prediction_loss_matches_mse():
    predictions = torch.tensor([[2.0], [4.0]])
    target = torch.tensor([1.0, 2.0])
    assert regression_loss(predictions, target).item() == pytest.approx(2.5)
