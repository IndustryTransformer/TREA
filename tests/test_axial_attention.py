"""Unit tests for the harvested axial (intra-row + inter-row) attention.

These lock the three properties that distinguish a correct harvest from a verbatim
copy of TabNCT's encoder (whose temporal attention silently dropped out at T>64 and
whose NaN handling bypassed the mask channel):

- feature-axis attention is permutation-equivariant (set-valued, transfer-safe);
- the learned feature-identity embedding is what breaks that symmetry;
- causal time attention does not leak the future;
- the value+mask triple-encoding survives end-to-end (NaN != real zero).
"""

import pytest
import torch

from trea.models.axial_attention import (
    AxialAttentionBlock,
    AxialEncoder,
    AxialTransformer,
)


torch.manual_seed(0)


class TestAxialAttentionBlock:
    def test_output_shape(self):
        block = AxialAttentionBlock(d_model=16, n_head=2, d_ff=32).eval()
        x = torch.randn(3, 8, 4, 16)  # [B, T, F, d]
        out = block(x)
        assert out.shape == x.shape

    def test_feature_permutation_equivariance(self):
        """Permuting features then attending == attending then permuting.

        The bare block has no per-feature identity, so feature attention must be
        order-agnostic (this is what keeps the architecture transfer-compatible).
        """
        block = AxialAttentionBlock(d_model=16, n_head=2, d_ff=32).eval()
        x = torch.randn(2, 6, 5, 16)
        perm = torch.tensor([2, 0, 4, 1, 3])

        out_then_perm = block(x)[:, :, perm, :]
        perm_then_out = block(x[:, :, perm, :])
        assert torch.allclose(out_then_perm, perm_then_out, atol=1e-5)

    def test_causal_blocks_future(self):
        """Under causal masking, the t=0 output must not depend on later inputs."""
        block = AxialAttentionBlock(d_model=16, n_head=2, d_ff=32).eval()
        x = torch.randn(2, 7, 3, 16)
        out_a = block(x, causal=True)

        x2 = x.clone()
        x2[:, -1] += 5.0  # perturb the last timestep only
        out_b = block(x2, causal=True)

        # Earliest timestep is upstream of the change; later ones may differ.
        assert torch.allclose(out_a[:, 0], out_b[:, 0], atol=1e-5)
        assert not torch.allclose(out_a[:, -1], out_b[:, -1], atol=1e-5)


class TestAxialEncoder:
    def test_pooled_and_token_shapes(self):
        enc = AxialEncoder(
            num_features=4, T=8, d_model=16, n_head=2, num_layers=2
        ).eval()
        x_val = torch.randn(3, 4, 8)
        m_nan = torch.zeros(3, 4, 8)
        assert enc(x_val, m_nan, pool=True).shape == (3, 16)
        assert enc(x_val, m_nan, pool=False).shape == (3, 8, 4, 16)

    def test_feature_identity_breaks_symmetry(self):
        """With feature-identity on, permuting input features changes the output."""
        enc = AxialEncoder(
            num_features=5,
            T=6,
            d_model=16,
            n_head=2,
            num_layers=2,
            use_feature_id_embedding=True,
        ).eval()
        x_val = torch.randn(2, 5, 6)
        m_nan = torch.zeros(2, 5, 6)
        perm = torch.tensor([2, 0, 4, 1, 3])

        base = enc(x_val, m_nan, pool=True)
        permed = enc(x_val[:, perm, :], m_nan[:, perm, :], pool=True)
        assert not torch.allclose(base, permed, atol=1e-4)

    def test_time_patching_reduces_steps(self):
        enc = AxialEncoder(
            num_features=3, T=12, d_model=16, n_head=2, num_layers=1, time_patch_len=4
        ).eval()
        tokens = enc(torch.randn(2, 3, 12), torch.zeros(2, 3, 12), pool=False)
        assert tokens.shape == (2, 3, 3, 16)  # 12 / 4 = 3 steps

    def test_validates_divisibility(self):
        with pytest.raises(ValueError):
            AxialEncoder(num_features=3, T=10, d_model=16, n_head=3)  # 16 % 3
        with pytest.raises(ValueError):
            AxialEncoder(num_features=3, T=10, d_model=16, n_head=2, time_patch_len=3)


class TestAxialTransformer:
    def _model(self, **kw):
        params = dict(
            C_num=4,
            C_cat=0,
            cat_cardinalities=[],
            T=8,
            d_model=16,
            num_classes=3,
            n_head=2,
            num_layers=2,
        )
        params.update(kw)
        return AxialTransformer(**params).eval()

    def test_classification_forward(self):
        model = self._model()
        out = model(torch.randn(5, 4, 8), torch.empty(5, 0, 8))
        assert out.shape == (5, 3)

    def test_regression_forward(self):
        model = self._model(task="regression", num_classes=None)
        out = model(torch.randn(5, 4, 8), torch.empty(5, 0, 8))
        assert out.shape == (5, 1)

    def test_categorical_features(self):
        model = self._model(C_cat=2, cat_cardinalities=[3, 4])
        x_cat = torch.randint(0, 3, (5, 2, 8))
        out = model(torch.randn(5, 4, 8), x_cat)
        assert out.shape == (5, 3)

    def test_nan_mask_is_alive(self):
        """A NaN input must produce a different token than a real zero.

        This is the triple-encoding guarantee: missingness flows through the mask
        channel, not a magic sentinel. If NaN were merely zeroed, the two inputs
        below would be identical and the outputs would match.
        """
        model = self._model()
        x = torch.randn(4, 4, 8)
        x_zero = x.clone()
        x_zero[:, 0, 0] = 0.0
        x_nan = x.clone()
        x_nan[:, 0, 0] = float("nan")

        out_zero = model(x_zero, torch.empty(4, 0, 8))
        out_nan = model(x_nan, torch.empty(4, 0, 8))
        assert torch.isfinite(out_nan).all()
        assert not torch.allclose(out_zero, out_nan, atol=1e-5)

    def test_gradients_flow(self):
        model = self._model().train()
        out = model(torch.randn(4, 4, 8), torch.empty(4, 0, 8))
        out.sum().backward()
        grads = [
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.requires_grad
        ]
        assert all(grads) and len(grads) > 0
