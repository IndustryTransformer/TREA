from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from examples import three_w_cross_well_mixup as module
from examples.three_w_cross_well_mixup import CrossGroupPartners, mixed_loss


def test_partners_are_training_only_different_group_same_source():
    groups = np.array(["WELL-A"] * 5 + ["WELL-B"] * 2 + ["SIM-A", "SIM-B", "SIM-B"])
    sampler = CrossGroupPartners(groups)
    anchors = np.tile(np.arange(len(groups)), 1000)
    paired = sampler.sample(anchors, np.random.default_rng(42))
    np.testing.assert_array_equal(
        paired, sampler.sample(anchors, np.random.default_rng(42))
    )
    assert np.all(groups[anchors] != groups[paired])
    np.testing.assert_array_equal(
        np.char.startswith(groups[anchors], "WELL-"),
        np.char.startswith(groups[paired], "WELL-"),
    )
    assert paired.min() >= 0 and paired.max() < len(groups)


def test_cannot_pair_single_group_source():
    with pytest.raises(ValueError):
        CrossGroupPartners(np.array(["WELL-A", "WELL-A", "SIM-A", "SIM-B"]))


@pytest.mark.parametrize("members", [1, 3])
def test_weighted_loss_matches_manual_member_target_average(members):
    logits = torch.arange(12 * members, dtype=torch.float32).reshape(6, members, 2) / 10
    a, b = torch.tensor([0, 1, 0, 1, 1, 0]), torch.tensor([1, 0, 1, 0, 0, 1])
    wa, wb, cw = (
        torch.arange(1.0, 7.0),
        torch.arange(6.0, 0.0, -1.0),
        torch.tensor([0.7, 1.3]),
    )
    fraction = 0.3
    probabilities = logits.log_softmax(-1)
    ca = -torch.stack([probabilities[i, :, a[i]].mean() for i in range(6)]) * cw[a]
    cb = -torch.stack([probabilities[i, :, b[i]].mean() for i in range(6)]) * cw[b]
    expected = (fraction * wa * ca + (1 - fraction) * wb * cb).sum() / (
        fraction * wa * cw[a] + (1 - fraction) * wb * cw[b]
    ).sum()
    actual = mixed_loss(
        logits if members > 1 else logits[:, 0], a, b, wa, wb, cw, fraction
    )
    torch.testing.assert_close(actual, expected)
    endpoint = mixed_loss(logits[:, 0], a, b, wa, wb, cw, 1)
    torch.testing.assert_close(
        endpoint,
        (F.cross_entropy(logits[:, 0], a, weight=cw, reduction="none") * wa).sum()
        / (cw[a] * wa).sum(),
    )


def test_zero_alpha_delegates_unchanged_legacy_trainer(monkeypatch):
    seen = []
    monkeypatch.setattr(
        module, "train_neural", lambda *args: seen.append(args) or "legacy"
    )
    result = module.train_mixup(
        "single_row",
        {"train": [], "val": []},
        {"train": [], "val": []},
        42,
        SimpleNamespace(mixup_alpha=0),
        "out",
        None,
        None,
    )
    assert result == "legacy" and len(seen) == 1


@pytest.mark.parametrize("alpha", [0, 0.2])
def test_rejects_test_arrays_before_model_build(alpha):
    with pytest.raises(ValueError, match="held-out"):
        module.train_mixup(
            "single_row",
            {"train": [], "val": [], "test": []},
            {},
            42,
            SimpleNamespace(mixup_alpha=alpha),
            "out",
            None,
            [],
        )


def test_middle_group_exclusion_and_uniform_allowed_rows():
    groups = np.array(["WELL-A"] * 2 + ["WELL-B"] * 3 + ["WELL-C"] * 5)
    sampler = CrossGroupPartners(groups)
    paired = sampler.sample(np.full(20000, 3), np.random.default_rng(43))
    counts = np.bincount(paired, minlength=len(groups))
    assert counts[2:5].sum() == 0
    np.testing.assert_allclose(counts[[0, 1, 5, 6, 7, 8, 9]] / 20000, 1 / 7, atol=0.015)


def test_soft_weighted_loss_has_finite_gradients():
    logits = torch.tensor([[0.5, -0.1], [-0.2, 0.4]], requires_grad=True)
    labels = torch.tensor([0, 1])
    value = mixed_loss(
        logits,
        labels,
        labels.flip(0),
        torch.ones(2),
        torch.ones(2),
        torch.tensor([0.6, 1.4]),
        0.2,
    )
    value.backward()
    assert torch.isfinite(logits.grad).all()
