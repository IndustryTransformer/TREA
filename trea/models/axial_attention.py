"""Axial (intra-row + inter-row) attention for tabular time series.

Harvested from TabNCT's hybrid intra/inter-row attention idea, re-implemented as a
clean, correct, ablatable component for TREA. Three deliberate departures from the
original ``tabnct/encoder.py``:

1. **Inter-row (temporal) attention actually runs.** The original routed any
   ``seq_len > chunk_size`` (default 64) to a chunked path that set
   ``inter_row = None`` — so at TREA's ``T=96`` windows it silently degraded to
   feature-attention only. It also collapsed the temporal heads to a single head
   with zeroed weights. Here both axes use all heads, every layer, at any length.
2. **Triple-encoding is preserved.** The original used a ``-999.0`` NaN sentinel +
   a ``[NUMERIC_NAN]`` token. We keep TREA's value+mask encoding: each per-feature
   token is built from ``[value, missing-mask]`` so the mask channel survives.
3. **Feature tokens stay alive end-to-end.** Tokens are ``[B, T, F, d]`` through
   the whole stack and are only pooled at the very end — fixing the mean-pool
   bottleneck (LESSONS §4) where a single per-timestep vector cannot track one
   sensor's trajectory. Feature mixing is attention-based (set-valued), so it stays
   transfer-compatible (no ``F·d → d`` concat-project, which LESSONS §4 rejects).

The encoder is exposed two ways:
- ``AxialEncoder`` — a reusable ``nn.Module`` producing ``[B, T, F, d]`` tokens (or a
  pooled ``[B, d]`` summary), suitable for classification heads or SSL objectives.
- ``AxialTransformer`` — a ``LightningModule`` mirroring ``TriplePatchTransformer``'s
  constructor + ``forward(x_num, x_cat)`` contract, so it is a drop-in ablation in
  ``train_w3.py`` and the benchmark harness.
"""

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

from .embeddings import SemanticColumnEmbedder


class GatedFeatureMixer(nn.Module):
    """Set-equivariant nonlinear mixer across features at each timestep."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        context = h.mean(dim=2, keepdim=True).expand_as(h)
        pair = torch.cat([h, context], dim=-1)
        return x + self.gate(pair) * self.update(pair)


class BilinearFeatureMixer(nn.Module):
    """Set-equivariant low-rank nonlinear mixer across features."""

    def __init__(self, d_model: int, rank: int | None = None, dropout: float = 0.1):
        super().__init__()
        rank = rank or max(4, d_model // 4)
        self.norm = nn.LayerNorm(d_model)
        self.left = nn.Linear(d_model, rank)
        self.right = nn.Linear(d_model, rank)
        self.out = nn.Sequential(
            nn.Linear(rank, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.gate = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        context = h.mean(dim=2, keepdim=True).expand_as(h)
        product = self.left(h) * self.right(context)
        return x + self.gate(torch.cat([h, context], dim=-1)) * self.out(product)


class AxialAttentionBlock(nn.Module):
    """One pre-norm axial block: feature-axis attention, then time-axis, then FFN.

    Operates on ``[B, T, F, d]`` tokens. The feature ("intra-row") attention mixes
    features within each timestep independently; the time ("inter-row") attention
    mixes timesteps within each feature track independently. Both use all heads.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_ff: int,
        dropout: float = 0.1,
        mixer: str = "none",
    ):
        super().__init__()
        if mixer not in {"none", "gated_feature", "bilinear_feature"}:
            raise ValueError(f"Unknown axial mixer: {mixer}")
        self.feat_norm = nn.LayerNorm(d_model)
        self.feat_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        if mixer == "gated_feature":
            self.feature_mixer = GatedFeatureMixer(d_model, dropout)
        elif mixer == "bilinear_feature":
            self.feature_mixer = BilinearFeatureMixer(d_model, dropout=dropout)
        else:
            self.feature_mixer = None
        self.time_norm = nn.LayerNorm(d_model)
        self.time_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        """Args: x ``[B, T, F, d]``. Returns the same shape."""
        B, T, Fdim, d = x.shape

        # Feature (intra-row) attention: independent per (batch, timestep).
        h = self.feat_norm(x).reshape(B * T, Fdim, d)
        attn, _ = self.feat_attn(h, h, h, need_weights=False)
        x = x + self.dropout(attn.reshape(B, T, Fdim, d))
        if self.feature_mixer is not None:
            x = self.feature_mixer(x)

        # Time (inter-row) attention: independent per (batch, feature).
        h = self.time_norm(x).permute(0, 2, 1, 3).reshape(B * Fdim, T, d)
        attn_mask = None
        if causal and T > 1:
            # True = blocked. Upper triangle (strictly future) is masked out.
            attn_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
        attn, _ = self.time_attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        attn = attn.reshape(B, Fdim, T, d).permute(0, 2, 1, 3)
        x = x + self.dropout(attn)

        # Position-wise feed-forward.
        x = x + self.dropout(self.ff(self.ff_norm(x)))
        return x


class AxialEncoder(nn.Module):
    """Stack of axial blocks over per-(feature, timestep) tokens.

    Tokens are built from the triple encoding: ``Linear([value, mask]) +
    feature-identity + optional semantic-column identity + time position``. Feature
    identity (a learned per-column embedding) is the validated single biggest lever
    (SEMANTIC_COLUMNS_SUMMARY: macro-F1 0.464 → 0.678); without it the feature axis
    is permutation-symmetric and cannot tell sensors apart.
    """

    def __init__(
        self,
        num_features: int,
        T: int,
        d_model: int = 128,
        n_head: int = 8,
        num_layers: int = 4,
        d_ff: int | None = None,
        dropout: float = 0.1,
        use_feature_id_embedding: bool = True,
        semantic_embedder: SemanticColumnEmbedder | None = None,
        num_semantic_features: int | None = None,
        time_patch_len: int = 1,
        value_num_features: int | None = None,
        tokenizer: str = "cell_linear",
        tokenizer_kernel_size: int = 5,
        mixer: str = "none",
        pooling: str = "mean",
        use_moment_tokens: bool = False,
        use_axis_patch_tokens: bool = False,
        feature_patch_size: int | None = None,
        num_global_tokens: int = 0,
    ):
        super().__init__()
        if tokenizer not in {
            "cell_linear",
            "temporal_conv",
            "patch_mlp",
            "conv_patch_mlp",
        }:
            raise ValueError(f"Unknown axial tokenizer: {tokenizer}")
        if mixer not in {"none", "gated_feature", "bilinear_feature"}:
            raise ValueError(f"Unknown axial mixer: {mixer}")
        if pooling not in {"mean", "query", "latent_mean"}:
            raise ValueError(f"Unknown axial pooling: {pooling}")
        if pooling == "latent_mean" and num_global_tokens < 1:
            raise ValueError("pooling='latent_mean' requires num_global_tokens > 0")
        if feature_patch_size is not None and feature_patch_size < 1:
            raise ValueError("feature_patch_size must be positive")
        if tokenizer_kernel_size % 2 == 0:
            raise ValueError("tokenizer_kernel_size must be odd")
        if d_model % n_head != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_head ({n_head})"
            )
        patch_tokenizer = tokenizer in {"patch_mlp", "conv_patch_mlp"}
        if not patch_tokenizer and T % time_patch_len != 0:
            raise ValueError(
                f"T ({T}) must be divisible by time_patch_len ({time_patch_len})"
            )
        d_ff = d_ff or 4 * d_model
        self.num_features = num_features
        self.value_num_features = value_num_features or num_features
        self.d_model = d_model
        self.time_patch_len = time_patch_len
        self.tokenizer = tokenizer
        self.pooling = pooling
        self.use_moment_tokens = use_moment_tokens
        self.use_axis_patch_tokens = use_axis_patch_tokens
        self.feature_patch_size = feature_patch_size or max(
            1, min(4, self.value_num_features)
        )
        self.num_feature_patches = (
            self.value_num_features + self.feature_patch_size - 1
        ) // self.feature_patch_size
        self.num_global_tokens = num_global_tokens
        self.num_steps = (
            (T + time_patch_len - 1) // time_patch_len
            if patch_tokenizer
            else T // time_patch_len
        )

        # Per-feature token from the [value, mask] pair (the value+mask leg).
        self.value_proj = nn.Linear(2, d_model)
        self.patch_proj = (
            nn.Sequential(
                nn.LayerNorm(2 * time_patch_len),
                nn.Linear(2 * time_patch_len, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            if patch_tokenizer
            else None
        )
        conv_hidden = 4 * self.value_num_features
        self.temporal_tokenizer = (
            nn.Sequential(
                nn.Conv1d(
                    2 * self.value_num_features,
                    conv_hidden,
                    kernel_size=tokenizer_kernel_size,
                    padding=tokenizer_kernel_size // 2,
                    groups=self.value_num_features,
                ),
                nn.GELU(),
                nn.Conv1d(
                    conv_hidden,
                    2 * self.value_num_features,
                    kernel_size=tokenizer_kernel_size,
                    padding=tokenizer_kernel_size // 2,
                    groups=self.value_num_features,
                ),
            )
            if tokenizer in {"temporal_conv", "conv_patch_mlp"}
            else None
        )
        self.moment_proj = (
            nn.Sequential(
                nn.LayerNorm(4),
                nn.Linear(4, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            if use_moment_tokens
            else None
        )
        self.moment_type = (
            nn.Parameter(torch.randn(1, 1, 1, d_model) * 0.02)
            if use_moment_tokens
            else None
        )
        self.axis_patch_proj = (
            nn.Sequential(
                nn.LayerNorm(6),
                nn.Linear(6, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            if use_axis_patch_tokens
            else None
        )
        self.axis_patch_type = (
            nn.Parameter(torch.randn(1, 1, 1, d_model) * 0.02)
            if use_axis_patch_tokens
            else None
        )
        self.axis_patch_id = (
            nn.Embedding(self.num_feature_patches, d_model)
            if use_axis_patch_tokens
            else None
        )

        # Learned per-feature ("index") identity — the strongest single-dataset lever.
        self.feature_id = (
            nn.Embedding(num_features, d_model) if use_feature_id_embedding else None
        )

        # Optional frozen-text semantic identity (transfer across schemas). Applies to
        # the first ``num_semantic_features`` columns (the numeric ones).
        self.semantic_embedder = semantic_embedder
        self.semantic_norm = nn.LayerNorm(d_model) if semantic_embedder else None
        self.num_semantic_features = num_semantic_features

        # Time position over (possibly patched) steps.
        self.time_pos = nn.Embedding(self.num_steps, d_model)
        self.global_tokens = (
            nn.Parameter(torch.randn(1, 1, num_global_tokens, d_model) * 0.02)
            if num_global_tokens > 0
            else None
        )

        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                AxialAttentionBlock(d_model, n_head, d_ff, dropout, mixer=mixer)
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.pool_query = (
            nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            if pooling == "query"
            else None
        )
        self.pool_attn = (
            nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
            if pooling == "query"
            else None
        )

    def _pad_time(self, x: torch.Tensor, value: float = 0.0) -> torch.Tensor:
        pad = self.num_steps * self.time_patch_len - x.shape[-1]
        if pad > 0:
            x = Fnn.pad(x, (0, pad), value=value)
        return x

    def _patch_view(
        self, x_val: torch.Tensor, m_nan: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, Fdim, _T = x_val.shape
        x_patch = self._pad_time(x_val).reshape(
            B, Fdim, self.num_steps, self.time_patch_len
        )
        m_patch = self._pad_time(m_nan, value=1.0).reshape(
            B, Fdim, self.num_steps, self.time_patch_len
        )
        return x_patch, m_patch

    def _moment_tokens(self, x_val: torch.Tensor, m_nan: torch.Tensor) -> torch.Tensor:
        assert self.moment_proj is not None
        x_patch, m_patch = self._patch_view(x_val, m_nan)
        valid = (1.0 - m_patch).clamp(0.0, 1.0)
        denom = valid.sum(dim=-1).clamp(min=1.0)
        mean = (x_patch * valid).sum(dim=-1) / denom
        centered = (x_patch - mean.unsqueeze(-1)) * valid
        std = torch.sqrt((centered.square().sum(dim=-1) / denom).clamp(min=1e-6))
        delta = (x_patch[..., -1] - x_patch[..., 0]) * (valid[..., -1] * valid[..., 0])
        if self.time_patch_len > 1:
            diff = (x_patch[..., 1:] - x_patch[..., :-1]).abs()
            diff_valid = valid[..., 1:] * valid[..., :-1]
            absdiff = (diff * diff_valid).sum(dim=-1) / diff_valid.sum(dim=-1).clamp(
                min=1.0
            )
        else:
            absdiff = torch.zeros_like(mean)
        stats = torch.stack([mean, std, delta, absdiff], dim=-1).permute(0, 2, 1, 3)
        tokens = self.moment_proj(stats)
        assert self.moment_type is not None
        return tokens + self.moment_type

    def _axis_patch_tokens(
        self, x_val: torch.Tensor, m_nan: torch.Tensor
    ) -> torch.Tensor:
        assert self.axis_patch_proj is not None
        assert self.axis_patch_type is not None
        assert self.axis_patch_id is not None

        x_patch, m_patch = self._patch_view(x_val, m_nan)
        B, Fdim, S, P = x_patch.shape
        feature_pad = self.num_feature_patches * self.feature_patch_size - Fdim
        if feature_pad > 0:
            x_patch = Fnn.pad(x_patch, (0, 0, 0, 0, 0, feature_pad), value=0.0)
            m_patch = Fnn.pad(m_patch, (0, 0, 0, 0, 0, feature_pad), value=1.0)

        x_group = x_patch.reshape(
            B, self.num_feature_patches, self.feature_patch_size, S, P
        )
        m_group = m_patch.reshape(
            B, self.num_feature_patches, self.feature_patch_size, S, P
        )
        valid = (1.0 - m_group).clamp(0.0, 1.0)
        denom = valid.sum(dim=(2, 4)).clamp(min=1.0)

        mean = (x_group * valid).sum(dim=(2, 4)) / denom
        centered = (x_group - mean.unsqueeze(2).unsqueeze(-1)) * valid
        std = torch.sqrt((centered.square().sum(dim=(2, 4)) / denom).clamp(min=1e-6))

        start_valid = valid[..., 0]
        end_valid = valid[..., -1]
        start = (x_group[..., 0] * start_valid).sum(dim=2) / start_valid.sum(
            dim=2
        ).clamp(min=1.0)
        end = (x_group[..., -1] * end_valid).sum(dim=2) / end_valid.sum(dim=2).clamp(
            min=1.0
        )
        delta = end - start

        if self.time_patch_len > 1:
            diff = (x_group[..., 1:] - x_group[..., :-1]).abs()
            diff_valid = valid[..., 1:] * valid[..., :-1]
            absdiff = (diff * diff_valid).sum(dim=(2, 4)) / diff_valid.sum(
                dim=(2, 4)
            ).clamp(min=1.0)
        else:
            absdiff = torch.zeros_like(mean)

        feature_denom = valid.sum(dim=-1).clamp(min=1.0)
        feature_means = (x_group * valid).sum(dim=-1) / feature_denom
        active_features = (feature_denom > 0).float()
        active_count = active_features.sum(dim=2).clamp(min=1.0)
        group_feature_mean = (feature_means * active_features).sum(dim=2) / active_count
        col_spread = torch.sqrt(
            (
                ((feature_means - group_feature_mean.unsqueeze(2)) * active_features)
                .square()
                .sum(dim=2)
                / active_count
            ).clamp(min=1e-6)
        )
        missing_rate = 1.0 - valid.mean(dim=(2, 4))

        stats = torch.stack(
            [mean, std, delta, absdiff, col_spread, missing_rate], dim=-1
        ).permute(0, 2, 1, 3)
        tokens = self.axis_patch_proj(stats)
        patch_ids = torch.arange(self.num_feature_patches, device=x_val.device)
        steps = torch.arange(self.num_steps, device=x_val.device)
        return (
            tokens
            + self.axis_patch_type
            + self.axis_patch_id(patch_ids).view(1, 1, -1, self.d_model)
            + self.time_pos(steps).view(1, self.num_steps, 1, self.d_model)
        )

    def _add_feature_time_identity(
        self, tokens: torch.Tensor, Fdim: int, device: torch.device
    ) -> torch.Tensor:
        if self.feature_id is not None:
            ids = torch.arange(Fdim, device=device)
            tokens = tokens + self.feature_id(ids).view(1, 1, Fdim, self.d_model)

        if self.semantic_embedder is not None:
            assert self.semantic_norm is not None
            sem = self.semantic_norm(self.semantic_embedder.get_embeddings())
            n_sem = self.num_semantic_features or sem.shape[0]
            pad = torch.zeros(Fdim, self.d_model, device=device, dtype=sem.dtype)
            pad[:n_sem] = sem[:n_sem]
            tokens = tokens + pad.view(1, 1, Fdim, self.d_model)

        steps = torch.arange(self.num_steps, device=device)
        return tokens + self.time_pos(steps).view(1, self.num_steps, 1, self.d_model)

    def build_tokens(self, x_val: torch.Tensor, m_nan: torch.Tensor) -> torch.Tensor:
        """Build ``[B, num_steps, F, d]`` tokens from value/mask ``[B, F, T]``."""
        B, Fdim, T = x_val.shape
        if Fdim != self.value_num_features:
            raise ValueError(
                f"Expected {self.value_num_features} value features, got {Fdim}"
            )

        if self.temporal_tokenizer is not None:
            vm_channels = torch.cat([x_val, m_nan], dim=1)
            delta = self.temporal_tokenizer(vm_channels)
            dx, dm = delta.split(Fdim, dim=1)
            x_val = x_val + dx
            m_nan = m_nan + dm

        if self.patch_proj is not None:
            x_patch, m_patch = self._patch_view(x_val, m_nan)
            vm_patch = torch.cat([x_patch, m_patch], dim=-1).permute(0, 2, 1, 3)
            tokens = self.patch_proj(vm_patch)
        elif self.time_patch_len > 1:
            # Average-pool value and mask within each time patch.
            x_val = x_val.reshape(B, Fdim, self.num_steps, self.time_patch_len).mean(-1)
            m_nan = m_nan.reshape(B, Fdim, self.num_steps, self.time_patch_len).mean(-1)
            vm = torch.stack([x_val, m_nan], dim=-1).permute(0, 2, 1, 3)
            tokens = self.value_proj(vm)  # [B, num_steps, F, d]
        else:
            # [B, num_steps, F, 2] -> project to d_model.
            vm = torch.stack([x_val, m_nan], dim=-1).permute(0, 2, 1, 3)
            tokens = self.value_proj(vm)  # [B, num_steps, F, d]

        token_groups = [tokens]
        if self.use_moment_tokens:
            token_groups.append(self._moment_tokens(x_val, m_nan))
        tokens = torch.cat(
            [
                self._add_feature_time_identity(group, Fdim, x_val.device)
                for group in token_groups
            ],
            dim=2,
        )
        if self.use_axis_patch_tokens:
            tokens = torch.cat(
                [tokens, self._axis_patch_tokens(x_val, m_nan)],
                dim=2,
            )
        if self.global_tokens is not None:
            global_tokens = self.global_tokens.expand(B, self.num_steps, -1, -1)
            steps = torch.arange(self.num_steps, device=x_val.device)
            global_tokens = global_tokens + self.time_pos(steps).view(
                1, self.num_steps, 1, self.d_model
            )
            tokens = torch.cat([tokens, global_tokens], dim=2)

        return self.input_dropout(self.input_norm(tokens))

    def forward(
        self,
        x_val: torch.Tensor,
        m_nan: torch.Tensor,
        extra_feature_tokens: torch.Tensor | None = None,
        causal: bool = False,
        pool: bool = True,
    ) -> torch.Tensor:
        """Encode value/mask into tokens and run the axial stack.

        Args:
            x_val: Value channel ``[B, F_num, T]`` (NaNs already replaced by 0).
            m_nan: Missing-mask channel ``[B, F_num, T]`` (1 where the value was NaN).
            extra_feature_tokens: Optional categorical tokens ``[B, num_steps, F_cat,
                d]`` concatenated along the feature axis.
            causal: If True, the time axis uses causal masking (no future leakage).
            pool: If True, mean-pool over time & features and return ``[B, d]``;
                otherwise return the full ``[B, num_steps, F, d]`` token tensor.
        """
        tokens = self.build_tokens(x_val, m_nan)
        if extra_feature_tokens is not None:
            tokens = torch.cat([tokens, extra_feature_tokens], dim=2)

        for block in self.blocks:
            tokens = block(tokens, causal=causal)
        tokens = self.out_norm(tokens)

        if pool:
            if self.pooling == "mean":
                return tokens.mean(dim=(1, 2))
            if self.pooling == "latent_mean":
                return tokens[:, :, -self.num_global_tokens :, :].mean(dim=(1, 2))
            assert self.pool_query is not None
            assert self.pool_attn is not None
            flat = tokens.reshape(tokens.shape[0], -1, tokens.shape[-1])
            query = self.pool_query.expand(tokens.shape[0], -1, -1)
            pooled, _ = self.pool_attn(query, flat, flat, need_weights=False)
            return pooled.squeeze(1)
        return tokens


class AxialTransformer(pl.LightningModule):
    """Axial-attention classifier/regressor; drop-in for ``TriplePatchTransformer``.

    Mirrors ``TriplePatchTransformer``'s constructor and ``forward(x_num, x_cat)``
    contract so the benchmark and ``train_w3.py`` can swap it in by changing one line.
    Unlike the patch model, it keeps per-feature tokens alive and mixes features with
    attention instead of a patch-flattening ``Linear`` (transfer-compatible).
    """

    def __init__(
        self,
        C_num: int,
        C_cat: int,
        cat_cardinalities: list[int],
        T: int,
        d_model: int = 128,
        task: str = "classification",
        num_classes: int | None = 3,
        n_head: int = 8,
        num_layers: int = 4,
        lr: float = 1e-3,
        dropout: float = 0.1,
        d_ff: int | None = None,
        time_patch_len: int = 1,
        use_feature_id_embedding: bool = True,
        causal: bool = False,
        column_names: list[str] | None = None,
        use_semantic_columns: bool = False,
        column_descriptions: list[str] | None = None,
        semantic_bert_model: str = "bert-base-uncased",
        tokenizer: str = "cell_linear",
        tokenizer_kernel_size: int = 5,
        mixer: str = "none",
        pooling: str = "mean",
        use_moment_tokens: bool = False,
        use_axis_patch_tokens: bool = False,
        feature_patch_size: int | None = None,
        num_global_tokens: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        if len(cat_cardinalities) != C_cat:
            raise ValueError(
                f"Length of cat_cardinalities ({len(cat_cardinalities)}) "
                f"must match C_cat ({C_cat})"
            )
        if task == "classification" and num_classes is None:
            raise ValueError("num_classes must be specified for classification task")

        self.task = task
        self.lr = lr
        self.C_num = C_num
        self.C_cat = C_cat
        self.causal = causal

        # Categorical features become extra feature tokens (time-varying).
        self.cat_embs = nn.ModuleList(
            [nn.Embedding(card, d_model) for card in cat_cardinalities]
        )

        # Optional semantic identity for the numeric columns.
        semantic_embedder = None
        if use_semantic_columns:
            descriptions = column_descriptions or column_names
            if descriptions is None:
                raise ValueError(
                    "column_descriptions (or column_names) must be provided when "
                    "use_semantic_columns=True"
                )
            if len(descriptions) != C_num:
                raise ValueError(
                    f"Length of column_descriptions ({len(descriptions)}) "
                    f"must match C_num ({C_num})"
                )
            semantic_embedder = SemanticColumnEmbedder(
                descriptions=descriptions,
                d_model=d_model,
                bert_model=semantic_bert_model,
            )

        self.encoder = AxialEncoder(
            num_features=C_num + C_cat,
            T=T,
            d_model=d_model,
            n_head=n_head,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
            use_feature_id_embedding=use_feature_id_embedding,
            semantic_embedder=semantic_embedder,
            num_semantic_features=C_num,
            time_patch_len=time_patch_len,
            value_num_features=C_num,
            tokenizer=tokenizer,
            tokenizer_kernel_size=tokenizer_kernel_size,
            mixer=mixer,
            pooling=pooling,
            use_moment_tokens=use_moment_tokens,
            use_axis_patch_tokens=use_axis_patch_tokens,
            feature_patch_size=feature_patch_size,
            num_global_tokens=num_global_tokens,
        )

        if task == "classification":
            assert num_classes is not None
            out_dim = num_classes
        else:
            out_dim = 1
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, out_dim),
        )
        self.loss_fn = (
            nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
        )

    def _cat_tokens(self, x_cat: torch.Tensor) -> torch.Tensor | None:
        """Embed categorical channels into ``[B, num_steps, C_cat, d]`` tokens."""
        if len(self.cat_embs) == 0 or x_cat.numel() == 0:
            return None
        B, C_cat, T = x_cat.shape
        # [B, T, C_cat, d] via per-channel embedding, then time-patch pooled to steps.
        toks = torch.stack(
            [emb(x_cat[:, j].long()) for j, emb in enumerate(self.cat_embs)], dim=2
        )  # [B, T, C_cat, d]
        patch = self.encoder.time_patch_len
        if patch > 1:
            pad = self.encoder.num_steps * patch - T
            if pad > 0:
                toks = Fnn.pad(toks, (0, 0, 0, 0, 0, pad))
            toks = toks.reshape(B, self.encoder.num_steps, patch, C_cat, -1).mean(2)
        return toks

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """Args: x_num ``[B, C_num, T]`` (NaNs intact), x_cat ``[B, C_cat, T]``."""
        m_nan = torch.isnan(x_num).float()
        x_val = torch.nan_to_num(x_num, nan=0.0)

        feats = self.encoder(
            x_val,
            m_nan,
            extra_feature_tokens=self._cat_tokens(x_cat),
            causal=self.causal,
            pool=True,
        )
        return self.head(feats)

    def training_step(self, batch: dict, _batch_idx: int) -> torch.Tensor:
        out = self(batch["x_num"], batch["x_cat"])
        if self.task == "classification":
            loss = self.loss_fn(out, batch["y"])
        else:
            loss = self.loss_fn(out.squeeze(), batch["y"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, _batch_idx: int) -> torch.Tensor:
        out = self(batch["x_num"], batch["x_cat"])
        if self.task == "classification":
            loss = self.loss_fn(out, batch["y"])
            acc = (torch.argmax(out, dim=1) == batch["y"]).float().mean()
            self.log("val_acc", acc, prog_bar=True)
        else:
            loss = self.loss_fn(out.squeeze(), batch["y"])
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return [optimizer], [scheduler]

    @classmethod
    def from_config(cls, config: Any, **kwargs: Any) -> "AxialTransformer":
        """Create from a ``DatasetConfig`` (parity with ``TriplePatchTransformer``)."""
        model_params = config.get_model_params()
        model_params.update(kwargs)
        return cls(**model_params)
