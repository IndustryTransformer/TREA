"""Protect chronology, label budgets, and masked-only SSL gradients."""

import numpy as np
import torch

from scripts.nox_temporal_pretraining import FRACTIONS, draw_indices, masked_mse


def test_label_draws_are_nested_chronological_and_budgeted():
    historical = np.arange(22191)
    previous_fit, previous_val = set(), set()
    for fraction in FRACTIONS:
        fit, val = draw_indices(historical, fraction)
        assert fit.max() < val.min()
        assert set(fit).isdisjoint(val)
        assert len(fit) + len(val) == round(len(historical) * fraction)
        assert previous_fit <= set(fit) and previous_val <= set(val)
        previous_fit, previous_val = set(fit), set(val)
    assert previous_fit | previous_val == set(historical)


def test_ssl_loss_does_not_train_on_visible_cells():
    prediction = torch.tensor([[2.0, 1000.0]], requires_grad=True)
    loss = masked_mse(
        prediction, torch.zeros_like(prediction), torch.tensor([[True, False]])
    )
    assert loss.item() == 4.0
    loss.backward()
    assert prediction.grad.tolist() == [[4.0, 0.0]]
