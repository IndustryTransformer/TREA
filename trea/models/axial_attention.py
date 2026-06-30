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

from .embeddings import SemanticColumnEmbedder


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
    ):
        super().__init__()
        self.feat_norm = nn.LayerNorm(d_model)
        self.feat_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
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
    ):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_head ({n_head})"
            )
        if T % time_patch_len != 0:
            raise ValueError(
                f"T ({T}) must be divisible by time_patch_len ({time_patch_len})"
            )
        d_ff = d_ff or 4 * d_model
        self.num_features = num_features
        self.d_model = d_model
        self.time_patch_len = time_patch_len
        self.num_steps = T // time_patch_len

        # Per-feature token from the [value, mask] pair (the value+mask leg).
        self.value_proj = nn.Linear(2, d_model)

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

        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                AxialAttentionBlock(d_model, n_head, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(d_model)

    def build_tokens(self, x_val: torch.Tensor, m_nan: torch.Tensor) -> torch.Tensor:
        """Build ``[B, num_steps, F, d]`` tokens from value/mask ``[B, F, T]``."""
        B, Fdim, T = x_val.shape

        if self.time_patch_len > 1:
            # Average-pool value and mask within each time patch.
            x_val = x_val.reshape(B, Fdim, self.num_steps, self.time_patch_len).mean(-1)
            m_nan = m_nan.reshape(B, Fdim, self.num_steps, self.time_patch_len).mean(-1)

        # [B, num_steps, F, 2] -> project to d_model.
        vm = torch.stack([x_val, m_nan], dim=-1).permute(0, 2, 1, 3)
        tokens = self.value_proj(vm)  # [B, num_steps, F, d]

        # Feature identity (broadcast over batch & time).
        if self.feature_id is not None:
            ids = torch.arange(Fdim, device=x_val.device)
            tokens = tokens + self.feature_id(ids).view(1, 1, Fdim, self.d_model)

        # Semantic identity on the numeric feature positions.
        if self.semantic_embedder is not None:
            assert self.semantic_norm is not None
            sem = self.semantic_norm(self.semantic_embedder.get_embeddings())
            n_sem = self.num_semantic_features or sem.shape[0]
            pad = torch.zeros(Fdim, self.d_model, device=x_val.device, dtype=sem.dtype)
            pad[:n_sem] = sem[:n_sem]
            tokens = tokens + pad.view(1, 1, Fdim, self.d_model)

        # Time position (broadcast over batch & features).
        steps = torch.arange(self.num_steps, device=x_val.device)
        tokens = tokens + self.time_pos(steps).view(1, self.num_steps, 1, self.d_model)

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
            return tokens.mean(dim=(1, 2))
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
