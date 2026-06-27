"""Triple-Patch Transformer model implementation."""

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn

from .embeddings import ColumnEmbedding, SemanticColumnEmbedder


class TriplePatchTransformer(pl.LightningModule):
    """Triple-Patch Transformer for time series with numeric and categorical features.

    Handles missing values efficiently by encoding them in a triple-patch format:
    value channels, mask channels, and optional column embeddings.
    """

    def __init__(
        self,
        C_num: int,
        C_cat: int,
        cat_cardinalities: list[int],
        T: int,
        d_model: int = 64,
        task: str = "classification",
        num_classes: int | None = 3,
        n_head: int = 4,
        num_layers: int = 2,
        lr: float = 1e-3,
        pooling: str = "mean",
        dropout: float = 0.1,
        patch_len: int = 16,
        stride: int = 8,
        column_names: list[str] | None = None,
        use_column_embeddings: bool = False,
        column_embedding_config: dict[str, Any] | None = None,
        use_pre_patch_feature_attention: bool = False,
        feature_attention_dim: int = 32,
        feature_attention_heads: int = 4,
        use_stat_tokens: bool = False,
        use_semantic_columns: bool = False,
        column_descriptions: list[str] | None = None,
        semantic_bert_model: str = "bert-base-uncased",
    ):
        """Initialize the Triple-Patch Transformer.

        Args:
            C_num: Number of numeric channels
            C_cat: Number of categorical channels
            cat_cardinalities: List of unique values per categorical channel
            T: Number of time steps
            d_model: Model dimension
            task: 'classification' or 'regression'
            num_classes: Number of classes for classification
                (required if task='classification')
            n_head: Number of attention heads
            num_layers: Number of transformer layers
            lr: Learning rate
            pooling: Pooling strategy ('mean', 'last', 'cls')
            dropout: Dropout rate
            patch_len: Length of each patch
            stride: Stride for patch creation
            column_names: List of column names for semantic embeddings
            use_column_embeddings: Whether to use column semantic embeddings
            column_embedding_config: Configuration for column embeddings
            use_pre_patch_feature_attention: Apply attention across feature
                vectors at each time step before patch embedding.
            feature_attention_dim: Hidden dimension used for pre-patch
                feature attention tokens.
            feature_attention_heads: Number of heads for pre-patch feature
                attention.
            use_stat_tokens: Add per-feature statistical summary tokens
                (mean/std/nan-rate/slope/energy/min/max) before transformer.
            use_semantic_columns: Add a wide (d_model) semantic identity to each
                per-feature token, derived from a frozen text encoding of
                ``column_descriptions`` plus a learned projection. Supersedes the
                1-D ``use_column_embeddings`` channel; the two can also be combined.
            column_descriptions: Natural-language description per numeric feature
                (ordered to match the data feature axis). Required when
                ``use_semantic_columns=True``; falls back to ``column_names``.
            semantic_bert_model: Text encoder used for semantic column embeddings.
        """
        super().__init__()
        self.save_hyperparameters()

        self.task = task
        self.d_model = d_model
        self.lr = lr
        self.pooling = pooling
        self.T = T
        self.patch_len = patch_len
        self.stride = stride
        self.use_column_embeddings = use_column_embeddings
        self.use_pre_patch_feature_attention = use_pre_patch_feature_attention
        self.use_stat_tokens = use_stat_tokens
        self.use_semantic_columns = use_semantic_columns
        self.column_names = column_names
        self.C_num = C_num

        # Calculate number of patches
        self.num_patches = (T - patch_len) // stride + 1

        # Validate inputs
        if len(cat_cardinalities) != C_cat:
            raise ValueError(
                f"Length of cat_cardinalities ({len(cat_cardinalities)}) "
                f"must match C_cat ({C_cat})"
            )
        if task == "classification" and num_classes is None:
            raise ValueError("num_classes must be specified for classification task")
        if pooling not in ["mean", "last", "cls"]:
            raise ValueError(
                f"pooling must be one of ['mean', 'last', 'cls'], got {pooling}"
            )
        if use_column_embeddings and column_names is None:
            raise ValueError(
                "column_names must be provided when use_column_embeddings=True"
            )
        if use_column_embeddings:
            # Type narrowing: we validated column_names is not None above
            assert column_names is not None
            if len(column_names) != C_num:
                raise ValueError(
                    "Length of column_names ({len(column_names)}) "
                    "must match C_num ({C_num})"
                )
        if task == "classification" and num_classes is None:
            raise ValueError("num_classes must be provided for classification tasks")
        if use_pre_patch_feature_attention and (
            feature_attention_dim % feature_attention_heads != 0
        ):
            raise ValueError(
                "feature_attention_dim must be divisible by feature_attention_heads"
            )
        # Semantic columns need one description per numeric feature. Fall back to
        # the raw column_names if no descriptions are supplied (weaker signal).
        semantic_descriptions = column_descriptions or column_names
        if use_semantic_columns:
            if semantic_descriptions is None:
                raise ValueError(
                    "column_descriptions (or column_names) must be provided when "
                    "use_semantic_columns=True"
                )
            if len(semantic_descriptions) != C_num:
                raise ValueError(
                    f"Length of column_descriptions ({len(semantic_descriptions)}) "
                    f"must match C_num ({C_num})"
                )

        # Categorical embeddings (time-varying) with variable cardinalities
        self.cat_embs = nn.ModuleList(
            [nn.Embedding(cardinality, d_model) for cardinality in cat_cardinalities]
        )

        # Column embeddings (semantic information from column names)
        self.column_embedder = None
        if use_column_embeddings:
            column_config = column_embedding_config or {}
            self.column_embedder = ColumnEmbedding(
                column_names=column_names,
                target_dim=1,  # Match value/mask dimensionality
                **column_config,
            )

        # Patch embedding: 2× or 3× input channels based on column embeddings
        input_channels = C_num * 3 if use_column_embeddings else C_num * 2

        # Optional per-time-step feature attention before patching.
        # This lets the model learn cross-feature interactions explicitly before
        # flattening vectors into patch embeddings.
        if use_pre_patch_feature_attention:
            self.feature_token_proj = nn.Linear(1, feature_attention_dim)
            self.feature_attention = nn.MultiheadAttention(
                embed_dim=feature_attention_dim,
                num_heads=feature_attention_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.feature_attention_norm = nn.LayerNorm(feature_attention_dim)
            self.feature_attention_out = nn.Linear(feature_attention_dim, 1)

        # Optional statistical tokens: one token per feature channel.
        # This adds a strong inductive bias similar to common engineered features.
        self.stat_token_proj = None
        self.stat_token_norm = None
        if use_stat_tokens:
            self.stat_token_proj = nn.Linear(7, d_model)
            self.stat_token_norm = nn.LayerNorm(d_model)

        # Optional wide semantic identity per feature: a frozen text encoding of
        # each column's description + a learned projection into d_model, added to
        # the per-feature tokens so every feature's token carries its identity.
        self.semantic_embedder = None
        self.semantic_norm = None
        if use_semantic_columns:
            assert semantic_descriptions is not None
            self.semantic_embedder = SemanticColumnEmbedder(
                descriptions=semantic_descriptions,
                d_model=d_model,
                bert_model=semantic_bert_model,
            )
            self.semantic_norm = nn.LayerNorm(d_model)

        self.patch_embedding = nn.Linear(patch_len * input_channels, d_model)

        # Positional embeddings for patches
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))

        # Add CLS token if using cls pooling
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Dropout layer for after positional embeddings
        self.dropout = nn.Dropout(dropout)

        # Temporal encoder with pre-normalization for better stability
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            batch_first=True,
            dropout=dropout,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Output heads
        if task == "classification":
            # Type narrowing: we validated num_classes is not None above
            assert num_classes is not None
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, num_classes),
            )
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )
            self.loss_fn = nn.MSELoss()

    def create_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Create patches from input tensor using stride.

        Args:
            x: Input tensor [B, C, T]

        Returns:
            Patches [B, num_patches, C * patch_len]
        """
        B, C, T = x.shape
        patches = []

        for i in range(0, T - self.patch_len + 1, self.stride):
            patch = x[:, :, i : i + self.patch_len]  # [B, C, patch_len]
            patch = patch.reshape(B, -1)  # [B, C * patch_len]
            patches.append(patch)

        patches = torch.stack(patches, dim=1)  # [B, num_patches, C * patch_len]
        return patches

    def apply_pre_patch_feature_attention(self, x_val: torch.Tensor) -> torch.Tensor:
        """Apply feature-wise attention at each time step before patch creation.

        Args:
            x_val: Value tensor [B, C_num, T]

        Returns:
            Mixed value tensor [B, C_num, T]
        """
        B, C_num, T = x_val.shape

        # Treat each time step independently: tokens are feature values.
        feature_tokens = x_val.permute(0, 2, 1).reshape(B * T, C_num, 1)
        feature_tokens = self.feature_token_proj(feature_tokens)
        attn_out, _ = self.feature_attention(
            feature_tokens, feature_tokens, feature_tokens, need_weights=False
        )
        feature_tokens = self.feature_attention_norm(feature_tokens + attn_out)
        mixed = self.feature_attention_out(feature_tokens).squeeze(-1)
        mixed = mixed.reshape(B, T, C_num).permute(0, 2, 1)
        return x_val + mixed

    def compute_stat_tokens(
        self, x_val: torch.Tensor, m_nan: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-feature statistical summary tokens.

        Args:
            x_val: NaN-filled numeric values [B, C_num, T]
            m_nan: Missing-value mask [B, C_num, T] (1 = missing)

        Returns:
            Statistical tokens [B, C_num, d_model]
        """
        if self.stat_token_proj is None or self.stat_token_norm is None:
            raise RuntimeError("Stat token layers are not initialized")

        valid = 1.0 - m_nan
        count = valid.sum(dim=2).clamp(min=1.0)

        mean = (x_val * valid).sum(dim=2) / count
        centered = (x_val - mean.unsqueeze(-1)) * valid
        var = (centered * centered).sum(dim=2) / count
        std = torch.sqrt(var + 1e-6)
        nan_rate = m_nan.mean(dim=2)

        half = x_val.shape[2] // 2
        if half > 0:
            valid_first = valid[:, :, :half]
            valid_second = valid[:, :, half:]
            count_first = valid_first.sum(dim=2).clamp(min=1.0)
            count_second = valid_second.sum(dim=2).clamp(min=1.0)
            mean_first = (x_val[:, :, :half] * valid_first).sum(dim=2) / count_first
            mean_second = (x_val[:, :, half:] * valid_second).sum(dim=2) / count_second
            slope = mean_second - mean_first
        else:
            slope = torch.zeros_like(mean)

        energy = ((x_val * x_val) * valid).sum(dim=2) / count

        x_pos_inf = torch.full_like(x_val, float("inf"))
        x_neg_inf = torch.full_like(x_val, float("-inf"))
        min_v = torch.where(valid.bool(), x_val, x_pos_inf).min(dim=2).values
        max_v = torch.where(valid.bool(), x_val, x_neg_inf).max(dim=2).values
        has_valid = valid.sum(dim=2) > 0
        min_v = torch.where(has_valid, min_v, torch.zeros_like(min_v))
        max_v = torch.where(has_valid, max_v, torch.zeros_like(max_v))

        stats = torch.stack(
            [mean, std, nan_rate, slope, energy, min_v, max_v],
            dim=-1,
        )
        stat_tokens = self.stat_token_proj(stats)
        return self.stat_token_norm(stat_tokens)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """Forward pass with patch-based processing.

        Args:
            x_num: Numeric features [B, C_num, T]
            x_cat: Categorical features [B, C_cat, T]

        Returns:
            Model output [B, num_classes] or [B, 1]
        """
        B, C_num, T = x_num.shape

        # 1. Handle missing values: dual-patch encoding (value + mask)
        m_nan = torch.isnan(x_num).float()  # [B, C_num, T]
        x_val = torch.nan_to_num(x_num, nan=0.0)  # [B, C_num, T]

        # 1b. Optional cross-feature attention before row/patch embedding
        if self.use_pre_patch_feature_attention:
            x_val = self.apply_pre_patch_feature_attention(x_val)

        # 2. Create triple-patch or dual-patch based on configuration
        if self.use_column_embeddings:
            # Get column embeddings [B, C_num, T]
            col_emb = self.column_embedder(B, T)  # [B, C_num, T]

            # Stack value, mask, and column channels (triple-patch)
            x_num_processed = torch.cat(
                [x_val, m_nan, col_emb], dim=1
            )  # [B, 3·C_num, T]
        else:
            # Stack value & mask channels (dual-patch)
            x_num_processed = torch.cat([x_val, m_nan], dim=1)  # [B, 2·C_num, T]

        # 3. Create patches from the processed input
        patches = self.create_patches(
            x_num_processed
        )  # [B, num_patches, (2 or 3)·C_num·patch_len]

        # 4. Embed patches
        z = self.patch_embedding(patches)  # [B, num_patches, d_model]

        # 5. Add positional embeddings
        z = z + self.pos_embedding  # [B, num_patches, d_model]

        # Apply dropout after positional embeddings for regularization
        z = self.dropout(z)

        # 6. Add categorical embeddings if present
        if len(self.cat_embs) > 0 and x_cat.numel() > 0:
            # For each patch, aggregate categorical info
            cat_patches = []
            for i in range(0, T - self.patch_len + 1, self.stride):
                cat_slice = x_cat[:, :, i : i + self.patch_len]  # [B, C_cat, patch_len]
                # Use mean embedding over the patch
                cat_vecs = [
                    emb(cat_slice[:, j].long()) for j, emb in enumerate(self.cat_embs)
                ]
                cat_vec = torch.stack(cat_vecs, dim=0).sum(
                    dim=0
                )  # [B, patch_len, d_model]
                cat_vec = cat_vec.mean(dim=1)  # [B, d_model] - average over patch
                cat_patches.append(cat_vec)

            cat_patches = torch.stack(cat_patches, dim=1)  # [B, num_patches, d_model]
            z = z + cat_patches

        patch_token_count = z.shape[1]

        # 6b. Append per-feature tokens (one token per numeric channel): a
        # statistical summary and/or a wide semantic identity. When both are
        # active, each feature's token carries its stats plus its identity.
        feature_tokens = None
        if self.use_stat_tokens:
            feature_tokens = self.compute_stat_tokens(x_val=x_val, m_nan=m_nan)
        if self.use_semantic_columns:
            assert self.semantic_embedder is not None
            assert self.semantic_norm is not None
            sem = self.semantic_norm(self.semantic_embedder.get_embeddings())
            sem = sem.unsqueeze(0).expand(B, -1, -1)  # [B, C_num, d_model]
            feature_tokens = sem if feature_tokens is None else feature_tokens + sem
        if feature_tokens is not None:
            z = torch.cat([z, feature_tokens], dim=1)

        # 7. Add CLS token if using cls pooling
        if self.pooling == "cls":
            cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
            z = torch.cat([cls_tokens, z], dim=1)  # [B, num_patches+1, d_model]

        # 8. Transformer encoding
        z = self.transformer(z)

        # 9. Pooling
        if self.pooling == "cls":
            z = z[:, 0, :]  # CLS token
        elif self.pooling == "mean":
            z = z[:, :patch_token_count, :].mean(dim=1)
        elif self.pooling == "last":
            z = z[:, patch_token_count - 1, :]

        return self.head(z)

    def training_step(self, batch: dict, _batch_idx: int) -> torch.Tensor:
        """Training step."""
        out = self(batch["x_num"], batch["x_cat"])

        # Handle different output shapes for classification vs regression
        if self.task == "classification":
            # Classification: out is [B, num_classes], y is [B]
            loss = self.loss_fn(out, batch["y"])
        else:
            # Regression: out is [B, 1], y is [B, 1] or [B]
            loss = self.loss_fn(out.squeeze(), batch["y"])

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, _batch_idx: int) -> torch.Tensor:
        """Validation step."""
        out = self(batch["x_num"], batch["x_cat"])

        # Handle different output shapes for classification vs regression
        if self.task == "classification":
            # Classification: out is [B, num_classes], y is [B]
            loss = self.loss_fn(out, batch["y"])

            # Calculate accuracy
            preds = torch.argmax(out, dim=1)
            acc = (preds == batch["y"]).float().mean()
            self.log("val_acc", acc, prog_bar=True)
        else:
            # Regression: out is [B, 1], y is [B, 1] or [B]
            loss = self.loss_fn(out.squeeze(), batch["y"])

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch: dict, _batch_idx: int) -> torch.Tensor:
        """Test step."""
        out = self(batch["x_num"], batch["x_cat"])

        # Handle different output shapes for classification vs regression
        if self.task == "classification":
            # Classification: out is [B, num_classes], y is [B]
            loss = self.loss_fn(out, batch["y"])

            # Calculate accuracy
            preds = torch.argmax(out, dim=1)
            acc = (preds == batch["y"]).float().mean()
            self.log("test_acc", acc, prog_bar=True)
        else:
            # Regression: out is [B, 1], y is [B, 1] or [B]
            loss = self.loss_fn(out.squeeze(), batch["y"])

        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer with learning rate scheduler."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return [optimizer], [scheduler]

    @classmethod
    def from_config(cls, config, **kwargs):
        """Create model from DatasetConfig.

        Args:
            config: DatasetConfig instance
            **kwargs: Additional model parameters

        Returns:
            TriplePatchTransformer instance
        """
        model_params = config.get_model_params()
        model_params.update(kwargs)
        return cls(**model_params)
