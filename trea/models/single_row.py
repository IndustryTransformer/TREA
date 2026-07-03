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

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NumericCategoricalData:
    """A single-row batch: numeric features and (optional) categorical features."""

    numeric: torch.Tensor
    categorical: torch.Tensor | None = None

    def to(self, device):
        self.numeric = self.numeric.to(device)
        if self.categorical is not None:
            self.categorical = self.categorical.to(device)
        return self


def initialize_parameters(module):
    """Xavier for linears, scaled-normal for embeddings, standard for layernorm."""
    if isinstance(module, nn.Linear | nn.Conv2d):
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
    """Transformer block with optional pre-norm and QK-normalized attention."""

    def __init__(
        self,
        d_model,
        n_heads,
        dropout=0.2,
        qk_norm: bool = True,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        if self.head_dim * n_heads != d_model:
            raise ValueError("d_model must be divisible by n_heads")
        self.qk_norm = qk_norm
        self.pre_norm = pre_norm
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=dropout
        )
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.last_attn_entropy: torch.Tensor | None = None
        self.apply(initialize_parameters)

    def _attention(self, q, k, v):
        if not self.qk_norm:
            attn_out, attn_weights = self.attn(
                q, k, v, need_weights=True, average_attn_weights=False
            )
            probs = attn_weights.detach().clamp_min(1e-12)
            self.last_attn_entropy = -(probs * probs.log()).sum(dim=-1).mean(dim=(0, 2))
            return attn_out

        b = q.size(0)
        q_proj, k_proj, v_proj = F._in_projection_packed(
            q, k, v, self.attn.in_proj_weight, self.attn.in_proj_bias
        )

        def split_heads(x):
            return x.view(b, x.size(1), self.n_heads, self.head_dim).transpose(1, 2)

        qh = self.q_norm(split_heads(q_proj))
        kh = self.k_norm(split_heads(k_proj))
        vh = split_heads(v_proj)
        scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.head_dim)
        probs = F.softmax(scores, dim=-1)
        safe_probs = probs.detach().clamp_min(1e-12)
        self.last_attn_entropy = (
            -(safe_probs * safe_probs.log()).sum(dim=-1).mean(dim=(0, 2))
        )
        probs = F.dropout(probs, p=self.attn.dropout, training=self.training)
        out = torch.matmul(probs, vh)
        out = out.transpose(1, 2).contiguous().view(b, q.size(1), -1)
        return self.attn.out_proj(out)

    def forward(self, q, k, v):
        if self.pre_norm:
            qn = self.norm1(q)
            kn = self.norm1(k)
            vn = self.norm1(v)
            out1 = q + self._attention(qn, kn, vn)
            return out1 + self.feed_forward(self.norm2(out1))

        attn_out = self._attention(q, k, v)
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
        qk_norm: bool = True,
        pre_norm: bool = True,
        value_encoder="identity",
        fusion="additive",
        ensemble_k=0,
    ):
        super().__init__()
        self.config = config
        self.d_model = d_model
        self.n_cat_cols = config.n_cat_cols
        self.n_numeric_cols = config.n_numeric_cols
        self.value_encoder = value_encoder
        self.fusion = fusion

        # TabM-mini BatchEnsemble (arXiv:2410.24210): k members share every weight
        # except a single per-member multiplicative vector adapter applied to the
        # token embeddings right after the embedding layer. Members diverge from the
        # random-sign init of that adapter alone; the batch is expanded k-fold in a
        # single reshaped forward (no Python loop), so the transformer stack runs once.
        # ensemble_k=0 disables the mechanism (behaviour identical to a single model).
        self.ensemble_k = ensemble_k
        if ensemble_k > 0:
            # First (and only) adapter -> random {-1, +1} signs per the paper.
            signs = torch.randint(0, 2, (ensemble_k, d_model)).float() * 2 - 1
            self.ensemble_adapter = nn.Parameter(signs)

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

        self.col_embedder = col_embedder
        descr = column_descriptions or {}
        self.col_texts = [descr.get(c, c) for c in col_tokens]
        self.numeric_texts = [descr.get(c, c) for c in config.numeric_col_tokens]

        if self.value_encoder == "dual":
            self.n_bins = 64
            self.ple_bank = nn.Embedding(self.n_bins, d_model)
            self.periodic_freqs = nn.Parameter(torch.randn(16))
            self.periodic_bias = nn.Parameter(torch.randn(16))
            self.num_value_proj = nn.Linear(2 + d_model + 16, d_model)
            self.missing_vector = nn.Parameter(torch.zeros(d_model + 16))
            qs = (
                config.numeric_quantiles
                if getattr(config, "numeric_quantiles", None) is not None
                else torch.zeros(config.n_numeric_cols, 65)
            )
            self.register_buffer("numeric_quantiles", qs)
        else:
            self.num_value_proj = nn.Linear(2, d_model)

        if self.fusion == "hadamard":
            self.fusion_mlp = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
            )

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model,
                    n_heads,
                    dropout,
                    qk_norm=qk_norm,
                    pre_norm=pre_norm,
                )
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
        # Any NON-FINITE input (NaN or +/-inf) is treated as MISSING:
        # present=0, value 0, so missingness is learned by the projection
        # (no special token) and identity is never lost. Raw NaN from data is
        # handled natively -- no pre-conversion required.
        missing = ~torch.isfinite(num_inputs)  # [b, n_num]: NaN or inf -> missing
        present = (~missing).to(num_inputs.dtype)
        vals = torch.where(missing, torch.zeros_like(num_inputs), num_inputs)

        if self.value_encoder == "dual":
            continuous = torch.stack([vals, present], dim=-1)
            # soft PLE
            x_exp = vals.unsqueeze(-1)
            qs = self.numeric_quantiles
            widths = qs[:, 1:] - qs[:, :-1]
            ple_v = torch.clamp(
                (x_exp - qs[:, :-1]) / (widths + 1e-8), min=0.0, max=1.0
            )
            ple_emb = torch.matmul(ple_v, self.ple_bank.weight)
            # periodic
            freqs = self.periodic_freqs.view(1, 1, -1)
            bias = self.periodic_bias.view(1, 1, -1)
            periodic_emb = torch.cos(2 * math.pi * freqs * x_exp + bias)
            # missing fallback
            discrete = torch.cat([ple_emb, periodic_emb], dim=-1)
            discrete = torch.where(
                present.unsqueeze(-1).bool(), discrete, self.missing_vector
            )
            concat_all = torch.cat([continuous, discrete], dim=-1)
            value_vec = self.num_value_proj(concat_all)
        else:
            value_vec = self.num_value_proj(torch.stack([vals, present], dim=-1))

        if self.col_embedder is not None:
            num_col_emb = (
                self.col_embedder(self.numeric_texts).unsqueeze(0).expand(b, -1, -1)
            )
        else:
            num_col_emb = self.embeddings(
                self.numeric_indices.unsqueeze(0).expand(b, -1)
            )

        if self.fusion == "hadamard":
            base_numeric = num_col_emb * self.fusion_mlp(value_vec) + value_vec
        else:
            base_numeric = num_col_emb + value_vec

        if cat_inputs is not None and self.n_cat_cols > 0:
            cat_emb = self.embeddings(cat_inputs.long())
            values = torch.cat([cat_emb, base_numeric], dim=1)
        else:
            values = base_numeric

        if self.ensemble_k > 0:
            # Expand [b, T, d] -> [b*k, T, d] and scale every token embedding by the
            # member's adapter r_k. Flat layout is (b, k): row index = b_idx*k + member,
            # which the heads reshape back to [b, k, out]. Both the identity query and
            # the value tokens are scaled so each member is a distinct forward pass.
            k = self.ensemble_k
            t_col, t_val, d = col_emb.size(1), values.size(1), col_emb.size(2)
            r = self.ensemble_adapter.view(1, k, 1, d)
            col_emb = (col_emb.unsqueeze(1) * r).reshape(b * k, t_col, d)
            values = (values.unsqueeze(1) * r).reshape(b * k, t_val, d)

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
        qk_norm: bool = True,
        pre_norm: bool = True,
        output_dim=1,
        value_encoder="identity",
        fusion="additive",
        ensemble_k=0,
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
            qk_norm=qk_norm,
            pre_norm=pre_norm,
            value_encoder=value_encoder,
            fusion=fusion,
            ensemble_k=ensemble_k,
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
            nn.Linear(2 * d_model, output_dim),
        )

    def forward(self, num_inputs, cat_inputs):
        out = self.tabular_encoder(num_inputs, cat_inputs)
        out = self.dropout(self.pooling(out))
        pred = self.regressor(out)
        k = self.tabular_encoder.ensemble_k
        if k > 0:
            # [b*k, out] -> [b, k, out]; the member dim is explicit for the ensembled
            # loss (mean over members) and averaged away at inference by ``predict``.
            pred = pred.view(-1, k, pred.size(-1))
        return pred

    def predict(self, num_inputs, cat_inputs):
        """Ensemble-averaged point prediction [b, out] (mean over members)."""
        out = self.forward(num_inputs, cat_inputs)
        return out.mean(dim=1) if out.dim() == 3 else out


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
        qk_norm: bool = True,
        pre_norm: bool = True,
        value_encoder="identity",
        fusion="additive",
        ensemble_k=0,
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
            qk_norm=qk_norm,
            pre_norm=pre_norm,
            value_encoder=value_encoder,
            fusion=fusion,
            ensemble_k=ensemble_k,
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
        logits = self.head(out)
        k = self.tabular_encoder.ensemble_k
        if k > 0:
            # [b*k, C] -> [b, k, C]; explicit member dim for the ensembled CE loss.
            logits = logits.view(-1, k, logits.size(-1))
        return logits

    def predict(self, num_inputs, cat_inputs):
        """Ensemble-averaged class probabilities [b, C] (mean of member softmaxes)."""
        out = self.forward(num_inputs, cat_inputs)
        if out.dim() == 3:
            return out.softmax(dim=-1).mean(dim=1)
        return out.softmax(dim=-1)


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
        qk_norm: bool = True,
        pre_norm: bool = True,
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
            qk_norm=qk_norm,
            pre_norm=pre_norm,
        )
        flat = config.n_columns * d_model
        if config.n_cat_cols > 0:
            self.cat_decoder = nn.Linear(flat, config.n_cat_cols * config.n_tokens)
        else:
            self.cat_decoder = None
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
        if self.config.n_cat_cols > 0 and self.cat_decoder is not None:
            cat_out = self.cat_decoder(flat).view(
                out.size(0), self.config.n_cat_cols, self.config.n_tokens
            )
        else:
            cat_out = None
        num_out = self.num_decoder(flat)
        return NumericCategoricalData(numeric=num_out, categorical=cat_out)


def mask_tensor(tensor, encoder, keep_prob=0.7, return_mask=False):
    """Corrected fixed-probability masking: mask each cell where rand > keep_prob.

    Numeric cells become -inf (the encoder maps that to the numeric-mask embedding);
    categorical cells become the categorical mask token. Fixes the Hephaestus bug
    where `probability` was bound to batch_idx (degenerate, near-no-op masking).
    """
    if tensor is None:
        if return_mask:
            return None, None
        return None
    tensor = tensor.clone()
    bit_mask = torch.rand(tensor.shape, device=tensor.device) > keep_prob
    if tensor.is_floating_point():
        tensor[bit_mask] = float("-inf")
    else:
        tensor[bit_mask] = encoder.cat_mask_token.to(tensor.device)
    if return_mask:
        return tensor, bit_mask
    return tensor
