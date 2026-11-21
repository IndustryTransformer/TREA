"""Schema-free encoder for cross-schema column-name transfer.

Every weight here is shared across schemas. A table enters only through its column
*texts* (descriptions), which the column embedder turns into identity vectors, and
its numeric values. There is no token vocabulary and no ``SingleRowConfig``, so one
encoder consumes tables of any width and column order.

The configuration is pinned to the run that passed the schema-scaling gate
(docs/LESSONS.md §3b; layer code at commit cb992a0): additive fusion
(``identity + Linear([value, present])``), post-norm layers without QK-norm,
3 layers, dropout 0.1. The single-row defaults that were later tuned on NOx
(product fusion, pre-norm, QK-norm, 2 layers, dropout 0) were never tested for
transfer and are deliberately not inherited.
"""

import torch
import torch.nn as nn

from trea.models.single_row import (
    AttentionPooling,
    TransformerEncoderLayer,
    initialize_parameters,
)


class SchemaFreeEncoder(nn.Module):
    """Numeric cells + column texts -> one d_model vector per column.

    Layer 1 cross-attends column identities (query) over cell tokens (key/value);
    later layers are self-attention. Non-finite inputs (NaN, inf) are missing: the
    cell token keeps the column identity and gets ``present = 0``.
    """

    def __init__(
        self,
        col_embedder: nn.Module,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_value_proj = nn.Linear(2, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model, n_heads, dropout, qk_norm=False, pre_norm=False
                )
                for _ in range(n_layers)
            ]
        )
        self.apply(initialize_parameters)
        # Assigned after init so the embedder keeps its own initialization.
        self.col_embedder = col_embedder

    def forward(self, x: torch.Tensor, texts: list[str]) -> torch.Tensor:
        if x.dim() != 2 or x.size(1) != len(texts):
            raise ValueError(
                f"expected x of shape [batch, {len(texts)}], got {tuple(x.shape)}"
            )
        b = x.size(0)
        col_emb = self.col_embedder(texts).unsqueeze(0).expand(b, -1, -1)
        missing = ~torch.isfinite(x)
        present = (~missing).to(x.dtype)
        vals = torch.where(missing, torch.zeros_like(x), x)
        cells = col_emb + self.num_value_proj(torch.stack([vals, present], dim=-1))
        h = self.layers[0](col_emb, cells, cells)
        for layer in self.layers[1:]:
            h = layer(h, h, h)
        return h


def regression_head(d_model: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Linear(d_model, 2 * d_model),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(2 * d_model, 1),
    )


class SchemaTransferNet(nn.Module):
    """Shared encoder + attention pool, with one regression head per table."""

    def __init__(
        self,
        col_embedder: nn.Module,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.dropout_p = dropout
        self.encoder = SchemaFreeEncoder(
            col_embedder, d_model, n_heads, n_layers, dropout
        )
        self.pooling = AttentionPooling(d_model)
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleDict()

    def add_head(self, name: str) -> None:
        """Create (or replace) the head for ``name`` with fresh weights."""
        device = self.pooling.query.device
        self.heads[name] = regression_head(self.d_model, self.dropout_p).to(device)

    def forward(self, head: str, x: torch.Tensor, texts: list[str]) -> torch.Tensor:
        pooled = self.pooling(self.encoder(x, texts))
        return self.heads[head](self.drop(pooled))
