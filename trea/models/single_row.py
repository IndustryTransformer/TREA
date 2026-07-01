"""Single-row (non-time-series) tabular model for TREA.

A clean, minimal port of the Hephaestus single-row encoder. The idea: every column
is a token whose *name* is embedded (feature identity); a numeric value is projected
into d_model by a shared dense layer over [value, present] and ADDED to its column
identity (never scalar-times-vector -- that loses identity at value 0), categoricals
embed their value, and a transformer cross-attends column identity against the values.
The column-identity embedding is exactly where semantic / text-derived column
embeddings plug in to enable cross-schema transfer -- the one thing trees cannot do.

Deliberately omitted vs Hephaestus:
  - the enhanced / CrossNet interaction layers (biggest seed-variance source for the
    least benefit at low labels; see docs/LESSONS.md),
  - the BERT/`transformers` reservoir scaffolding (unused by the base model),
  - the hardcoded d_model=128 in the MTM decoder and the nondeterministic token order.

Pieces:
  TabularEncoder        -- columns+values -> per-column representations
  TabularRegressor      -- encoder + attention pool + MLP head -> scalar
  MaskedTabularEncoder  -- encoder + numeric/categorical reconstruction heads (for MTM)
  mask_tensor           -- corrected fixed-probability masking (no batch_idx bug)
"""

import math

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NumericCategoricalData:
    """A single-row batch: numeric features and (optional) categorical features."""

    numeric: torch.Tensor
    categorical: Optional[torch.Tensor] = None

    def to(self, device):
        self.numeric = self.numeric.to(device)
        if self.categorical is not None:
            self.categorical = self.categorical.to(device)
        return self


def initialize_parameters(module):
    """Xavier for linears, scaled-normal for embeddings, standard for layernorm."""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight, gain=1)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(
            module.weight, mean=0.0, std=1.0 / math.sqrt(module.embedding_dim)
        )
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class TransformerEncoderLayer(nn.Module):
    """Standard post-norm transformer block (cross- or self-attention via q,k,v)."""

    def __init__(self, d_model, n_heads, dropout=0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=dropout
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.apply(initialize_parameters)

    def forward(self, q, k, v):
        attn_out, _ = self.attn(q, k, v)
        out1 = self.norm1(q + attn_out)
        return self.norm2(out1 + self.feed_forward(out1))


class TabularEncoder(nn.Module):
    """Embed columns + values, then cross-attend column identity against values.

    Layer 1 attends column-name embeddings (query) over the value embeddings
    (key/value); subsequent layers are residual self-attention. Output is one
    d_model vector per column token (order: categorical columns, then numeric).
    """

    def __init__(
        self,
        config,
        d_model=128,
        n_heads=4,
        n_layers=4,
        dropout=0.2,
        col_embedder=None,
        column_descriptions=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_numeric_cols = config.n_numeric_cols
        self.n_cat_cols = config.n_cat_cols

        self.embeddings = nn.Embedding(config.n_tokens, d_model)
        self.register_buffer(
            "cat_mask_token", torch.tensor(config.token_dict["[MASK]"])
        )
        self.register_buffer(
            "numeric_mask_token", torch.tensor(config.token_dict["[NUMERIC_MASK]"])
        )
        # Column-identity token indices, in the order the model consumes them.
        col_tokens = config.categorical_col_tokens + config.numeric_col_tokens
        self.register_buffer(
            "col_indices",
            torch.tensor(
                [config.tokens.index(c) for c in col_tokens], dtype=torch.long
            ),
        )
        self.register_buffer(
            "numeric_indices",
            torch.tensor(
                [config.tokens.index(c) for c in config.numeric_col_tokens],
                dtype=torch.long,
            ),
        )

        # Optional name-keyed column-identity embedder (semantic transfer). When set,
        # it supplies the per-column identity vectors keyed by column *name*. The
        # internal `embeddings` table still serves categorical value tokens (not names).
        # `column_descriptions` maps a raw column token -> the text actually embedded;
        # feed descriptions, not terse codes (see column_embeddings.py).
        self.col_embedder = col_embedder
        descr = column_descriptions or {}
        self.col_texts = [descr.get(c, c) for c in col_tokens]
        self.numeric_texts = [descr.get(c, c) for c in config.numeric_col_tokens]

        # Numeric value encoder: project [value, present_flag] into d_model with a
        # shared dense layer, then ADD the column identity. Never scale the identity
        # vector by the scalar value -- that collapses to 0 at value 0 (identity lost)
        # and is scale-sensitive. Shared across columns => transfers across schemas.
        self.num_value_proj = nn.Linear(2, d_model)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, n_heads, dropout)
                for _ in range(n_layers)
            ]
        )
        self.apply(initialize_parameters)

    def forward(self, num_inputs, cat_inputs):
        if num_inputs.dim() == 1:
            num_inputs = num_inputs.unsqueeze(0)
            if cat_inputs is not None:
                cat_inputs = cat_inputs.unsqueeze(0)
        b = num_inputs.size(0)

        if self.col_embedder is not None:
            col_emb = self.col_embedder(self.col_texts).unsqueeze(0).expand(b, -1, -1)
        else:
            col_emb = self.embeddings(self.col_indices.unsqueeze(0).expand(b, -1))

        # Numeric value token = column identity + dense projection of [value, present].
        # Any NON-FINITE input (NaN or +/-inf) is treated as MISSING: present=0, value 0,
        # so missingness is learned by the projection (no special token) and identity is
        # never lost. Raw NaN from data is handled natively -- no pre-conversion required.
        missing = ~torch.isfinite(num_inputs)  # [b, n_num]: NaN or inf -> missing
        present = (~missing).to(num_inputs.dtype)
        vals = torch.where(missing, torch.zeros_like(num_inputs), num_inputs)
        value_vec = self.num_value_proj(
            torch.stack([vals, present], dim=-1)
        )  # [b, n_num, d]

        if self.col_embedder is not None:
            num_col_emb = (
                self.col_embedder(self.numeric_texts).unsqueeze(0).expand(b, -1, -1)
            )
        else:
            num_col_emb = self.embeddings(
                self.numeric_indices.unsqueeze(0).expand(b, -1)
            )
        base_numeric = num_col_emb + value_vec

        if cat_inputs is not None and self.n_cat_cols > 0:
            cat_emb = self.embeddings(cat_inputs.long())
            values = torch.cat([cat_emb, base_numeric], dim=1)
        else:
            values = base_numeric

        x = self.layers[0](col_emb, values, values)
        for layer in self.layers[1:]:
            # The block already carries its own attention/FFN residuals; use its output
            # directly (a standard post-norm stack). An extra outer `x +` here would
            # double the skip path and let the residual stream grow with depth.
            x = layer(x, x, x)
        return x


class AttentionPooling(nn.Module):
    """Pool [B, T, d] -> [B, d] with a learned query."""

    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model))
        self.key_proj = nn.Linear(d_model, d_model)
        self.scale = d_model**-0.5

    def forward(self, x):
        keys = self.key_proj(x)
        query = self.query.unsqueeze(0).unsqueeze(0)
        scores = torch.matmul(query, keys.transpose(-2, -1)) * self.scale
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, x).squeeze(1)


class TabularRegressor(nn.Module):
    """Encoder + attention pool + MLP head -> scalar prediction."""

    def __init__(
        self,
        config,
        d_model=128,
        n_heads=4,
        n_layers=4,
        dropout=0.2,
        col_embedder=None,
        column_descriptions=None,
    ):
        super().__init__()
        self.tabular_encoder = TabularEncoder(
            config,
            d_model,
            n_heads,
            n_layers,
            dropout,
            col_embedder,
            column_descriptions,
        )
        self.pooling = AttentionPooling(d_model)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, 1),
        )

    def forward(self, num_inputs, cat_inputs):
        out = self.tabular_encoder(num_inputs, cat_inputs)
        out = self.dropout(self.pooling(out))
        return self.regressor(out)


class TabularClassifier(nn.Module):
    """Encoder + attention pool + MLP head -> class logits.

    Same body as TabularRegressor (so a pretrained encoder transfers to either task);
    only the final layer differs (``num_classes`` logits instead of a scalar).
    """

    def __init__(
        self,
        config,
        num_classes,
        d_model=128,
        n_heads=4,
        n_layers=4,
        dropout=0.2,
        col_embedder=None,
        column_descriptions=None,
    ):
        super().__init__()
        self.tabular_encoder = TabularEncoder(
            config,
            d_model,
            n_heads,
            n_layers,
            dropout,
            col_embedder,
            column_descriptions,
        )
        self.pooling = AttentionPooling(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, num_classes),
        )

    def forward(self, num_inputs, cat_inputs):
        out = self.tabular_encoder(num_inputs, cat_inputs)
        out = self.dropout(self.pooling(out))
        return self.head(out)


class MaskedTabularEncoder(nn.Module):
    """Encoder + reconstruction heads for masked tabular modeling (MTM pretraining)."""

    def __init__(
        self,
        config,
        d_model=128,
        n_heads=4,
        n_layers=4,
        dropout=0.2,
        col_embedder=None,
        column_descriptions=None,
    ):
        super().__init__()
        self.config = config
        self.tabular_encoder = TabularEncoder(
            config,
            d_model,
            n_heads,
            n_layers,
            dropout,
            col_embedder,
            column_descriptions,
        )
        flat = config.n_columns * d_model
        self.cat_decoder = nn.Linear(flat, config.n_cat_cols * config.n_tokens)
        self.num_decoder = nn.Sequential(
            nn.Linear(flat, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, config.n_numeric_cols),
        )

    @property
    def cat_mask_token(self):
        return self.tabular_encoder.cat_mask_token

    def forward(self, num_inputs, cat_inputs):
        out = self.tabular_encoder(num_inputs, cat_inputs)
        flat = out.reshape(out.size(0), -1)
        cat_out = self.cat_decoder(flat).view(
            out.size(0), self.config.n_cat_cols, self.config.n_tokens
        )
        num_out = self.num_decoder(flat)
        return NumericCategoricalData(numeric=num_out, categorical=cat_out)


def mask_tensor(tensor, encoder, keep_prob=0.7):
    """Corrected fixed-probability masking: mask each cell where rand > keep_prob.

    Numeric cells become -inf (the encoder maps that to the numeric-mask embedding);
    categorical cells become the categorical mask token. Fixes the Hephaestus bug
    where `probability` was bound to batch_idx (degenerate, near-no-op masking).
    """
    tensor = tensor.clone()
    bit_mask = torch.rand(tensor.shape, device=tensor.device) > keep_prob
    if tensor.dtype == torch.float32:
        tensor[bit_mask] = float("-inf")
    else:
        tensor[bit_mask] = encoder.cat_mask_token.to(tensor.device)
    return tensor
