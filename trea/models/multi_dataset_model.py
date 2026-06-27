"""Consolidated multi-dataset transformer model with configurable modes.

This module provides a unified MultiDatasetModel class that combines functionality from:
1. VariableFeatureTREA - Handles different numbers of features across datasets
2. MultiDatasetTREA - Multi-dataset training with column embeddings
3. PretrainTREA - Self-supervised pretraining with SSL objectives

The model supports three configurable modes:
- 'standard': Basic multi-dataset training with column embeddings
- 'variable_features': Handles datasets with different feature counts
- 'pretrain': Adds self-supervised learning objectives for pretraining

Key features:
- Unified feature space with feature masking for variable schemas
- Multiple column embedding strategies (BERT, auto-expanding, simple)
- Self-supervised pretraining with masked patch prediction, temporal order,
  and contrastive learning
- Dual-patch NaN handling with value + mask channels
- Support for both numeric and categorical features
- Compatible with both classification and regression tasks
"""

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import create_multi_dataset_embedder
from .ssl_objectives import SSLObjectives


class MultiDatasetModel(pl.LightningModule):
    """Unified multi-dataset transformer model with configurable modes.

    This model combines all functionality from the three source models:
    - Variable feature handling across datasets
    - Column embedding strategies for semantic understanding
    - Self-supervised pretraining objectives

    Modes:
        'standard': Basic multi-dataset training with column embeddings
        'variable_features': Handles datasets with different feature counts
        'pretrain': Adds SSL objectives for pretraining
    """

    def __init__(
        self,
        # Core model parameters
        max_numeric_features: int,
        max_categorical_features: int = 0,
        num_classes: int = 2,
        mode: str = "variable_features",
        # Architecture parameters
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_head: int = 8,
        num_layers: int = 3,
        max_sequence_length: int = 1024,
        # Training parameters
        lr: float = 1e-3,
        task: str = "classification",
        # Feature handling
        feature_padding_value: float = 0.0,
        use_feature_masks: bool = True,
        # Column embedding parameters
        column_embedding_strategy: str = "auto_expanding",
        bert_model: str = "bert-base-uncased",
        column_embedding_dim: int = 16,
        initial_vocab_size: int = 1000,
        # Categorical embedding parameters
        categorical_cardinalities: list[int] | None = None,
        categorical_embedding_dim: int = 16,
        # SSL parameters (for pretrain mode)
        ssl_objectives: dict[str, bool] | None = None,
        mask_ratio: float = 0.15,
        temporal_shuffle_ratio: float = 0.3,
        contrastive_temperature: float = 0.1,
        augmentation_strength: float = 0.1,
        lambda_masked: float = 1.0,
        lambda_temporal: float = 0.5,
        lambda_contrastive: float = 0.3,
        lambda_causal: float = 1.0,
        lambda_supervised: float = 0.1,
        **embedder_kwargs,
    ):
        """Initialize MultiDatasetModel.

        Args:
            max_numeric_features: Maximum number of numeric features across datasets
            max_categorical_features: Maximum number of categorical features
            num_classes: Number of output classes
            mode: Model mode - 'standard', 'variable_features', or 'pretrain'
            patch_len: Length of each patch
            stride: Stride for patch creation
            d_model: Model dimension
            n_head: Number of attention heads
            num_layers: Number of transformer layers
            max_sequence_length: Maximum sequence length for positional encoding
            lr: Learning rate
            task: 'classification' or 'regression'
            feature_padding_value: Value for padding missing features
            use_feature_masks: Whether to use explicit feature presence masks
            column_embedding_strategy: 'frozen_bert', 'auto_expanding', or 'none'
            bert_model: BERT model for frozen embeddings
            column_embedding_dim: Embedding dimension for auto-expanding
            initial_vocab_size: Initial vocab size for auto-expanding
            categorical_cardinalities: List of cardinalities for categorical features
            categorical_embedding_dim: Embedding dimension for categorical features
            ssl_objectives: Dict specifying SSL objectives for pretrain mode
            mask_ratio: Ratio of patches to mask for MPP
            temporal_shuffle_ratio: Ratio for temporal order prediction
            contrastive_temperature: Temperature for contrastive learning
            augmentation_strength: Strength of data augmentation
            lambda_masked: Weight for masked patch prediction loss
            lambda_temporal: Weight for temporal order prediction loss
            lambda_contrastive: Weight for contrastive loss
            lambda_causal: Weight for causal next-row prediction loss
            lambda_supervised: Weight for supervised loss during pretraining
            **embedder_kwargs: Additional embedder arguments
        """
        super().__init__()
        self.save_hyperparameters()

        # Validate mode
        valid_modes = ["standard", "variable_features", "pretrain"]
        if mode not in valid_modes:
            raise ValueError(f"Mode must be one of {valid_modes}, got {mode}")

        self.mode = mode
        self.task = task
        self.column_embedding_strategy = column_embedding_strategy
        self.max_numeric_features = max_numeric_features
        self.max_categorical_features = max_categorical_features
        self.feature_padding_value = feature_padding_value
        self.use_feature_masks = use_feature_masks
        self.categorical_embedding_dim = categorical_embedding_dim

        # Patching parameters
        self.patch_len = patch_len
        self.stride = stride
        self.max_sequence_length = max_sequence_length

        # Current dataset configuration (for variable_features and pretrain modes)
        self.current_dataset_info = {
            "numeric_features": max_numeric_features,
            "categorical_features": max_categorical_features,
            "column_names": [],
            "feature_mapping": None,
        }

        # Categorical embeddings
        self.categorical_embeddings = None
        self.categorical_cardinalities = categorical_cardinalities or []
        if max_categorical_features > 0:
            if categorical_cardinalities is None:
                categorical_cardinalities = [100] * max_categorical_features
            elif len(categorical_cardinalities) != max_categorical_features:
                raise ValueError(
                    f"categorical_cardinalities length "
                    f"({len(categorical_cardinalities)}) must match "
                    f"max_categorical_features ({max_categorical_features})"
                )
            self.categorical_cardinalities = categorical_cardinalities

            self.categorical_embeddings = nn.ModuleList(
                [
                    nn.Embedding(cardinality, categorical_embedding_dim)
                    for cardinality in categorical_cardinalities
                ]
            )

        # Column embeddings
        self.use_column_embeddings = column_embedding_strategy != "none"
        self.column_embedder = None

        if self.use_column_embeddings:
            embedder_args = {"target_dim": 1, **embedder_kwargs}

            if column_embedding_strategy == "frozen_bert":
                embedder_args.update({"bert_model": bert_model})
            elif column_embedding_strategy == "auto_expanding":
                embedder_args.update(
                    {
                        "embedding_dim": column_embedding_dim,
                        "initial_vocab_size": initial_vocab_size,
                    }
                )

            self.column_embedder = create_multi_dataset_embedder(
                strategy=column_embedding_strategy, **embedder_args
            )

        # Calculate input channels based on mode
        if mode in ["variable_features", "pretrain"]:
            # Variable features mode: unified feature space
            numeric_channels = 2  # value + nan_mask
            if use_feature_masks:
                numeric_channels += 1  # + feature_mask
            if self.use_column_embeddings:
                numeric_channels += 1  # + column_emb

            categorical_channels = categorical_embedding_dim
            if use_feature_masks:
                categorical_channels += 1  # + feature_mask
            if self.use_column_embeddings:
                categorical_channels += 1  # + column_emb

            total_input_channels = (
                max_numeric_features * numeric_channels
                + max_categorical_features * categorical_channels
            )
        else:
            # Standard mode: fixed feature space
            input_channels_per_feature = 3 if self.use_column_embeddings else 2
            total_input_channels = patch_len * (
                input_channels_per_feature * max_numeric_features
            )

        # Patch embedding
        if mode in ["variable_features", "pretrain"]:
            self.patch_embedding = nn.Linear(patch_len * total_input_channels, d_model)
        else:
            self.patch_embedding = nn.Linear(total_input_channels, d_model)

        # Dynamic positional encoding
        max_patches = (max_sequence_length - patch_len) // stride + 1
        self.pos_embedding = nn.Parameter(torch.randn(1, max_patches, d_model))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head
        if task == "classification":
            self.head = nn.Linear(d_model, num_classes)
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.head = nn.Linear(d_model, 1)
            self.loss_fn = nn.MSELoss()

        # SSL components (for pretrain mode)
        if mode == "pretrain":
            self._setup_ssl_components(
                ssl_objectives,
                mask_ratio,
                temporal_shuffle_ratio,
                contrastive_temperature,
                augmentation_strength,
                lambda_masked,
                lambda_temporal,
                lambda_contrastive,
                lambda_causal,
            )
            self.lambda_supervised = lambda_supervised

            # Set default dataset schema for SSL
            self.set_dataset_schema(max_numeric_features, max_categorical_features)

    def _setup_ssl_components(
        self,
        ssl_objectives: dict[str, bool] | None,
        mask_ratio: float,
        temporal_shuffle_ratio: float,
        contrastive_temperature: float,
        augmentation_strength: float,
        lambda_masked: float,
        lambda_temporal: float,
        lambda_contrastive: float,
        lambda_causal: float,
    ):
        """Setup SSL components for pretrain mode."""
        if ssl_objectives is None:
            ssl_objectives = {
                "masked_patch": True,
                "temporal_order": True,
                "contrastive": True,
                "causal_next_row": True,
            }

        self.ssl_objectives_config = ssl_objectives

        # Initialize SSL objectives
        self.ssl_objectives = SSLObjectives(
            mask_ratio=mask_ratio,
            temporal_shuffle_ratio=temporal_shuffle_ratio,
            contrastive_temperature=contrastive_temperature,
            augmentation_strength=augmentation_strength,
            lambda_masked=lambda_masked,
            lambda_temporal=lambda_temporal,
            lambda_contrastive=lambda_contrastive,
        )
        self.lambda_causal = lambda_causal

        # SSL prediction heads
        self.ssl_heads = nn.ModuleDict()

        # Use the model's actual patch input size to avoid drift between
        # configuration math and feature-processing implementation.
        patch_dim = self.patch_embedding.in_features

        if ssl_objectives.get("masked_patch", False):
            self.ssl_heads["reconstruction"] = nn.Sequential(
                nn.Linear(self.hparams.d_model, self.hparams.d_model),
                nn.ReLU(),
                nn.Linear(self.hparams.d_model, patch_dim),
            )

        if ssl_objectives.get("temporal_order", False):
            self.ssl_heads["temporal_order"] = nn.Sequential(
                nn.Linear(self.hparams.d_model, self.hparams.d_model),
                nn.ReLU(),
                nn.Linear(self.hparams.d_model, self.hparams.d_model),
            )

        if ssl_objectives.get("contrastive", False):
            self.ssl_heads["contrastive"] = nn.Sequential(
                nn.Linear(self.hparams.d_model, self.hparams.d_model),
                nn.ReLU(),
                nn.Linear(self.hparams.d_model, 128),
            )

        if ssl_objectives.get("causal_next_row", False):
            self.ssl_heads["causal_numeric"] = nn.Sequential(
                nn.Linear(self.hparams.d_model, self.hparams.d_model),
                nn.ReLU(),
                nn.Linear(self.hparams.d_model, self.max_numeric_features),
            )

            if self.max_categorical_features > 0:
                self.ssl_heads["causal_categorical"] = nn.ModuleList(
                    [
                        nn.Linear(self.hparams.d_model, int(cardinality))
                        for cardinality in self.categorical_cardinalities
                    ]
                )

    def set_dataset_schema(
        self,
        numeric_features: int,
        categorical_features: int = 0,
        column_names: list[str] | None = None,
    ) -> None:
        """Set the schema for the current dataset.

        For variable_features and pretrain modes.

        Args:
            numeric_features: Number of numeric features in this dataset
            categorical_features: Number of categorical features
            column_names: Optional list of column names
        """
        if self.mode == "standard":
            raise RuntimeError("set_dataset_schema not supported in standard mode")

        if numeric_features > self.max_numeric_features:
            raise ValueError(
                f"Dataset has {numeric_features} numeric features, "
                f"but model supports max {self.max_numeric_features}"
            )

        if categorical_features > self.max_categorical_features:
            raise ValueError(
                f"Dataset has {categorical_features} categorical features, "
                f"but model supports max {self.max_categorical_features}"
            )

        # Update current dataset info
        self.current_dataset_info = {
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "column_names": column_names
            or [f"feature_{i}" for i in range(numeric_features + categorical_features)],
            "feature_mapping": self._create_feature_mapping(
                numeric_features, categorical_features
            ),
        }

        # Set column embeddings if enabled
        if self.use_column_embeddings and self.column_embedder is not None:
            if self.mode in ["variable_features", "pretrain"]:
                full_column_names = self._get_padded_column_names()
                self.column_embedder.set_columns(full_column_names)
            else:
                self.column_embedder.set_columns(
                    self.current_dataset_info["column_names"]
                )

        print(
            f"Set dataset schema: {numeric_features} numeric + "
            f"{categorical_features} categorical features"
        )
        if column_names:
            print(f"Column names: {column_names}")

    def set_dataset_columns(self, column_names: list[str]) -> None:
        """Set column names for the current dataset (standard mode).

        Args:
            column_names: List of column names for the current dataset
        """
        if self.mode != "standard":
            raise RuntimeError("set_dataset_columns only supported in standard mode")

        if self.column_embedder is not None:
            self.column_embedder.set_columns(column_names)
            print(f"Set columns for dataset: {column_names}")
        else:
            print(f"No column embedder - ignoring column names: {column_names}")

    def _create_feature_mapping(
        self, numeric_features: int, categorical_features: int
    ) -> dict[str, Any]:
        """Create mapping for current dataset features to unified space."""
        return {
            "numeric_start": 0,
            "numeric_end": numeric_features,
            "categorical_start": self.max_numeric_features,
            "categorical_end": self.max_numeric_features + categorical_features,
            "total_active_features": numeric_features + categorical_features,
        }

    def _get_padded_column_names(self) -> list[str]:
        """Get column names padded to full feature space."""
        column_names = self.current_dataset_info["column_names"]
        mapping = self.current_dataset_info["feature_mapping"]

        # Create full column name list
        full_names = ["<UNUSED>"] * (
            self.max_numeric_features + self.max_categorical_features
        )

        # Fill in actual column names
        actual_names = column_names[: mapping["total_active_features"]]

        # Place numeric features
        for i, name in enumerate(
            actual_names[: self.current_dataset_info["numeric_features"]]
        ):
            full_names[i] = name

        # Place categorical features
        for i, name in enumerate(
            actual_names[self.current_dataset_info["numeric_features"] :]
        ):
            full_names[self.max_numeric_features + i] = name

        return full_names

    def pad_features_to_unified_space(
        self, x_num: torch.Tensor, x_cat: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad features to unified space with categorical embeddings.

        For variable_features/pretrain modes.

        Args:
            x_num: Numeric features [B, C_num, T]
            x_cat: Categorical features [B, C_cat, T] (optional)

        Returns:
            Tuple of (x_num_padded, x_cat_embedded, feature_mask)
        """
        B, _, T = x_num.shape
        mapping = self.current_dataset_info["feature_mapping"]

        # Pad numeric features
        x_num_padded = torch.full(
            (B, self.max_numeric_features, T),
            self.feature_padding_value,
            dtype=x_num.dtype,
            device=x_num.device,
        )

        # Create feature mask
        total_max_features = self.max_numeric_features + self.max_categorical_features
        feature_mask = torch.zeros(
            (B, total_max_features, T), dtype=torch.float, device=x_num.device
        )

        # Place numeric features
        num_numeric = mapping["numeric_end"] - mapping["numeric_start"]
        if num_numeric > 0:
            num_numeric_input = min(x_num.shape[1], num_numeric)
            x_num_padded[:, :num_numeric_input, :] = x_num[:, :num_numeric_input, :]
            feature_mask[:, :num_numeric_input, :] = 1.0

        # Handle categorical features with embeddings
        x_cat_embedded = torch.zeros(
            (B, self.max_categorical_features, self.categorical_embedding_dim, T),
            dtype=torch.float,
            device=x_num.device,
        )

        if x_cat is not None and self.categorical_embeddings is not None:
            num_categorical = mapping["categorical_end"] - mapping["categorical_start"]
            if num_categorical > 0:
                num_categorical_input = min(x_cat.shape[1], num_categorical)
                for i in range(num_categorical_input):
                    cat_indices = x_cat[:, i, :].long()  # [B, T]
                    embedded = self.categorical_embeddings[i](
                        cat_indices
                    )  # [B, T, embedding_dim]
                    x_cat_embedded[:, i, :, :] = embedded.permute(
                        0, 2, 1
                    )  # [B, embedding_dim, T]

                # Mark categorical features as present
                feature_mask[
                    :,
                    self.max_numeric_features : self.max_numeric_features
                    + num_categorical_input,
                    :,
                ] = 1.0

        return x_num_padded, x_cat_embedded, feature_mask

    def create_patches(self, x):
        """Create patches from input tensor with dynamic sequence length."""
        batch_size, channels, seq_len = x.shape

        # Calculate number of patches dynamically
        num_patches = (seq_len - self.patch_len) // self.stride + 1

        if num_patches <= 0:
            raise ValueError(
                f"Sequence length {seq_len} too short for patch_len={self.patch_len}"
            )

        # Create patches using unfold
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        # patches: [batch_size, channels, num_patches, patch_len]

        # Reshape to [batch_size, num_patches, channels * patch_len]
        patches = patches.permute(0, 2, 1, 3).contiguous()
        patches = patches.view(batch_size, num_patches, channels * self.patch_len)

        return patches

    def get_positional_encoding(self, num_patches: int):
        """Get positional encoding for the given number of patches."""
        if num_patches > self.pos_embedding.shape[1]:
            raise ValueError(
                f"Sequence too long: {num_patches} patches exceeds maximum "
                f"{self.pos_embedding.shape[1]}. Increase max_sequence_length."
            )

        return self.pos_embedding[:, :num_patches, :]

    def forward(self, x_num, x_cat=None):
        """Forward pass with mode-specific processing.

        Args:
            x_num: Numeric data [batch_size, c_num, sequence_length]
            x_cat: Categorical data [batch_size, c_cat, sequence_length] (optional)

        Returns:
            Model output
        """
        B, C, T = x_num.shape

        if self.mode == "standard":
            return self._forward_standard(x_num, x_cat)
        else:
            return self._forward_variable_features(x_num, x_cat)

    def _forward_standard(self, x_num, x_cat=None):
        """Standard mode forward pass (fixed feature space)."""
        B, C, T = x_num.shape

        # Dual-patch NaN handling
        m_nan = torch.isnan(x_num).float()  # [B, C, T]
        x_val = torch.nan_to_num(x_num, nan=0.0)  # [B, C, T]

        # Triple-patch: value + mask + column (if enabled)
        if self.use_column_embeddings and self.column_embedder is not None:
            col_emb = self.column_embedder(B, T)  # [B, C, T]
            x_processed = torch.cat([x_val, m_nan, col_emb], dim=1)  # [B, 3*C, T]
        else:
            x_processed = torch.cat([x_val, m_nan], dim=1)  # [B, 2*C, T]

        # Create patches
        patches = self.create_patches(x_processed)
        num_patches = patches.shape[1]

        # Patch embedding
        patch_embeddings = self.patch_embedding(patches)  # [B, num_patches, d_model]

        # Add positional encoding
        patch_embeddings = patch_embeddings + self.get_positional_encoding(num_patches)

        # Transformer
        transformer_out = self.transformer(
            patch_embeddings
        )  # [B, num_patches, d_model]

        # Global average pooling
        pooled = transformer_out.mean(dim=1)  # [B, d_model]

        # Output head
        output = self.head(pooled)
        return output

    def _forward_variable_features(self, x_num, x_cat=None):
        """Variable features mode forward pass (unified feature space)."""
        B, _, T = x_num.shape

        # Pad features to unified space with categorical embeddings
        x_num_padded, x_cat_embedded, feature_mask = self.pad_features_to_unified_space(
            x_num, x_cat
        )

        # Handle numeric features with dual-patch NaN handling
        m_nan = torch.isnan(x_num_padded).float()  # [B, max_numeric_features, T]
        x_num_val = torch.nan_to_num(
            x_num_padded, nan=0.0
        )  # [B, max_numeric_features, T]

        # Prepare channels for concatenation
        channels_to_cat = []

        # Add numeric channels: [value, nan_mask, feature_mask?, column_emb?]
        channels_to_cat.append(x_num_val)
        channels_to_cat.append(m_nan)

        if self.use_feature_masks:
            numeric_feature_mask = feature_mask[:, : self.max_numeric_features, :]
            channels_to_cat.append(numeric_feature_mask)

        if self.use_column_embeddings and self.column_embedder is not None:
            col_emb = self.column_embedder(B, T)  # [B, max_total_features, T]
            numeric_col_emb = col_emb[:, : self.max_numeric_features, :]
            channels_to_cat.append(numeric_col_emb)

        # Add categorical channels: [embedding_dims, feature_mask?, column_emb?]
        # Reshape categorical embeddings from
        # [B, max_categorical_features, embedding_dim, T]
        # to [B, max_categorical_features * embedding_dim, T]
        x_cat_reshaped = x_cat_embedded.view(
            B, self.max_categorical_features * self.categorical_embedding_dim, T
        )
        channels_to_cat.append(x_cat_reshaped)

        if self.use_feature_masks:
            categorical_feature_mask = feature_mask[:, self.max_numeric_features :, :]
            channels_to_cat.append(categorical_feature_mask)

        if self.use_column_embeddings and self.column_embedder is not None:
            categorical_col_emb = col_emb[:, self.max_numeric_features :, :]
            channels_to_cat.append(categorical_col_emb)

        # Concatenate all channels
        x_processed = torch.cat(channels_to_cat, dim=1)  # [B, total_input_channels, T]

        # Create patches
        patches = self.create_patches(x_processed)
        num_patches = patches.shape[1]

        # Patch embedding
        patch_embeddings = self.patch_embedding(patches)  # [B, num_patches, d_model]

        # Add positional encoding
        pos_enc = self.get_positional_encoding(num_patches)
        patch_embeddings = patch_embeddings + pos_enc

        # Transformer
        transformer_out = self.transformer(
            patch_embeddings
        )  # [B, num_patches, d_model]

        # Global average pooling
        pooled = transformer_out.mean(dim=1)  # [B, d_model]

        # Output head
        output = self.head(pooled)
        return output

    def create_patches_with_features(
        self,
        x_num: torch.Tensor,
        feature_mask: torch.Tensor,
        x_cat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Create patches with all feature processing for SSL (pretrain mode only)."""
        if self.mode != "pretrain":
            raise RuntimeError(
                "create_patches_with_features only available in pretrain mode"
            )

        B, _, T = x_num.shape

        # Use the same feature padding approach as the forward method
        x_padded, x_cat_embedded, feature_mask_padded = (
            self.pad_features_to_unified_space(x_num, x_cat)
        )

        # Handle NaNs on the padded space
        m_nan = torch.isnan(x_padded).float()
        x_val = torch.nan_to_num(x_padded, nan=0.0)

        # Combine channels to match _forward_variable_features exactly.
        channels_to_cat = [x_val, m_nan]

        if self.use_feature_masks:
            numeric_feature_mask = feature_mask_padded[
                :, : self.max_numeric_features, :
            ]
            channels_to_cat.append(numeric_feature_mask)

        if self.use_column_embeddings and self.column_embedder is not None:
            col_emb = self.column_embedder(B, T)
            numeric_col_emb = col_emb[:, : self.max_numeric_features, :]
            channels_to_cat.append(numeric_col_emb)

        # Add categorical channels.
        x_cat_reshaped = x_cat_embedded.view(
            B, self.max_categorical_features * self.categorical_embedding_dim, T
        )
        channels_to_cat.append(x_cat_reshaped)

        if self.use_feature_masks:
            categorical_feature_mask = feature_mask_padded[
                :, self.max_numeric_features :, :
            ]
            channels_to_cat.append(categorical_feature_mask)

        if self.use_column_embeddings and self.column_embedder is not None:
            categorical_col_emb = col_emb[:, self.max_numeric_features :, :]
            channels_to_cat.append(categorical_col_emb)

        x_processed = torch.cat(channels_to_cat, dim=1)

        # Create patches using base class method
        patches = self.create_patches(x_processed)

        return patches

    def get_global_embedding(self, patches: torch.Tensor) -> torch.Tensor:
        """Get global embedding from patches for contrastive learning.

        Pretrain mode only.
        """
        if self.mode != "pretrain":
            raise RuntimeError("get_global_embedding only available in pretrain mode")

        transformer_out = self._encode_patches(patches, causal=False)

        # Global pooling
        global_embedding = transformer_out.mean(dim=1)

        return global_embedding

    def _build_causal_attention_mask(
        self, num_patches: int, device: torch.device
    ) -> torch.Tensor:
        """Build upper-triangular mask for causal patch attention."""
        mask = torch.full(
            (num_patches, num_patches), float("-inf"), dtype=torch.float, device=device
        )
        mask = torch.triu(mask, diagonal=1)
        return mask

    def _encode_patches(
        self, patches: torch.Tensor, causal: bool = False
    ) -> torch.Tensor:
        """Encode patch sequence with optional causal attention mask."""
        B, num_patches, _ = patches.shape

        patch_embeddings = self.patch_embedding(patches)
        pos_enc = self.get_positional_encoding(num_patches)
        patch_embeddings = patch_embeddings + pos_enc

        if causal:
            causal_mask = self._build_causal_attention_mask(num_patches, patches.device)
            return self.transformer(patch_embeddings, mask=causal_mask)

        return self.transformer(patch_embeddings)

    def _pad_categorical_indices_to_unified_space(
        self,
        x_cat: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Pad categorical indices to model's max categorical feature space."""
        x_cat_padded = torch.full(
            (batch_size, self.max_categorical_features, seq_len),
            -100,
            dtype=torch.long,
            device=device,
        )

        if x_cat is None or self.max_categorical_features == 0:
            return x_cat_padded

        mapping = self.current_dataset_info["feature_mapping"]
        num_categorical = mapping["categorical_end"] - mapping["categorical_start"]
        if num_categorical <= 0:
            return x_cat_padded

        num_categorical_input = min(x_cat.shape[1], num_categorical)
        x_cat_padded[:, :num_categorical_input, :] = x_cat[
            :, :num_categorical_input, :
        ].long()
        return x_cat_padded

    def forward_ssl(
        self,
        x_num: torch.Tensor,
        feature_mask: torch.Tensor,
        x_cat: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for SSL objectives (pretrain mode only)."""
        if self.mode != "pretrain":
            raise RuntimeError("forward_ssl only available in pretrain mode")

        B, C, T = x_num.shape

        # Create patches
        patches = self.create_patches_with_features(x_num, feature_mask, x_cat)
        num_patches = patches.shape[1]

        ssl_outputs = {}

        # Masked Patch Prediction
        if self.ssl_objectives_config.get("masked_patch", False):
            mask = self.ssl_objectives.masked_predictor.create_mask(
                num_patches, x_num.device
            )
            masked_patches, targets = (
                self.ssl_objectives.masked_predictor.apply_masking(patches, mask)
            )

            # Forward through transformer
            transformer_out = self._encode_patches(masked_patches, causal=False)

            # Reconstruction
            reconstructed = self.ssl_heads["reconstruction"](transformer_out)

            ssl_outputs["masked_patch"] = {
                "predictions": reconstructed,
                "targets": targets,
                "mask": mask,
            }

        # Temporal Order Prediction
        if self.ssl_objectives_config.get("temporal_order", False):
            shuffled_patches, order_targets = (
                self.ssl_objectives.temporal_predictor.create_shuffled_sequence(patches)
            )

            # Forward through transformer
            transformer_out = self._encode_patches(shuffled_patches, causal=False)

            # Temporal order prediction with dynamic projection
            temporal_features = self.ssl_heads["temporal_order"](
                transformer_out.mean(dim=1)
            )

            # Dynamic projection to num_patches
            if (
                not hasattr(self, "temporal_projection")
                or self.temporal_projection.out_features != num_patches
            ):
                self.temporal_projection = nn.Linear(
                    self.hparams.d_model, num_patches
                ).to(temporal_features.device)
                torch.nn.init.xavier_uniform_(self.temporal_projection.weight)
                torch.nn.init.zeros_(self.temporal_projection.bias)

            order_logits = self.temporal_projection(temporal_features)

            # Check for NaN in temporal components
            if torch.isnan(temporal_features).any() or torch.isnan(order_logits).any():
                print("NaN in temporal order prediction!")
                order_logits = torch.zeros_like(order_logits)

            ssl_outputs["temporal_order"] = {
                "predictions": order_logits,
                "targets": order_targets,
            }

        # Contrastive Learning
        if self.ssl_objectives_config.get("contrastive", False):
            # Original embedding
            orig_embedding = self.get_global_embedding(patches)
            orig_projection = self.ssl_heads["contrastive"](orig_embedding)

            # Augmented embedding
            x_aug = self.ssl_objectives.contrastive_learner.augment_time_series(x_num)
            aug_patches = self.create_patches_with_features(x_aug, feature_mask, x_cat)
            aug_embedding = self.get_global_embedding(aug_patches)
            aug_projection = self.ssl_heads["contrastive"](aug_embedding)

            ssl_outputs["contrastive"] = {
                "original": orig_projection,
                "augmented": aug_projection,
            }

        # Causal Next-Row Prediction
        if self.ssl_objectives_config.get("causal_next_row", False):
            if num_patches > 1:
                causal_transformer_out = self._encode_patches(patches, causal=True)
                current_states = causal_transformer_out[:, :-1, :]  # [B, P-1, D]

                patch_starts = torch.arange(
                    0, T - self.patch_len + 1, self.stride, device=x_num.device
                )  # [P]
                next_steps = patch_starts[:-1] + self.patch_len  # [P-1]
                valid_next = next_steps < T

                if valid_next.any():
                    current_states = current_states[:, valid_next, :]  # [B, V, D]
                    target_steps = next_steps[valid_next]  # [V]

                    x_num_padded, _x_cat_embedded, feature_mask_padded = (
                        self.pad_features_to_unified_space(x_num, x_cat)
                    )
                    x_cat_padded = self._pad_categorical_indices_to_unified_space(
                        x_cat=x_cat, batch_size=B, seq_len=T, device=x_num.device
                    )

                    numeric_predictions = self.ssl_heads["causal_numeric"](
                        current_states
                    )
                    numeric_targets = (
                        x_num_padded[:, :, target_steps].permute(0, 2, 1).contiguous()
                    )  # [B, V, max_numeric]
                    numeric_present = (
                        feature_mask_padded[
                            :, : self.max_numeric_features, target_steps
                        ]
                        .permute(0, 2, 1)
                        .contiguous()
                        > 0
                    )
                    numeric_valid = numeric_present & (~torch.isnan(numeric_targets))

                    causal_output = {
                        "numeric_predictions": numeric_predictions,
                        "numeric_targets": numeric_targets,
                        "numeric_valid_mask": numeric_valid,
                    }

                    if (
                        self.max_categorical_features > 0
                        and "causal_categorical" in self.ssl_heads
                    ):
                        categorical_targets = (
                            x_cat_padded[:, :, target_steps]
                            .permute(0, 2, 1)
                            .contiguous()
                        )  # [B, V, max_categorical]
                        categorical_present = (
                            feature_mask_padded[
                                :, self.max_numeric_features :, target_steps
                            ]
                            .permute(0, 2, 1)
                            .contiguous()
                            > 0
                        )
                        categorical_logits = [
                            head(current_states)
                            for head in self.ssl_heads["causal_categorical"]
                        ]
                        causal_output["categorical_logits"] = categorical_logits
                        causal_output["categorical_targets"] = categorical_targets
                        causal_output["categorical_valid_mask"] = categorical_present

                    ssl_outputs["causal_next_row"] = causal_output

        return ssl_outputs

    def compute_ssl_losses(
        self, ssl_outputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Compute SSL losses from SSL outputs (pretrain mode only)."""
        if self.mode != "pretrain":
            raise RuntimeError("compute_ssl_losses only available in pretrain mode")

        losses = {}

        # Masked Patch Prediction Loss
        if "masked_patch" in ssl_outputs:
            outputs = ssl_outputs["masked_patch"]
            predictions = outputs["predictions"]
            targets = outputs["targets"]
            mask = outputs["mask"]

            # Reshape for loss computation
            B, num_patches, patch_dim = predictions.shape
            predictions_flat = predictions.view(B, num_patches, -1)
            targets_flat = targets.view(B, num_patches, -1)

            # Apply mask
            mask_expanded = mask.unsqueeze(0).unsqueeze(-1).expand_as(predictions_flat)

            if mask_expanded.sum() > 0:  # Ensure we have masked patches
                pred_masked = predictions_flat[mask_expanded]
                targ_masked = targets_flat[mask_expanded]

                # Check for NaN/Inf values
                if (
                    torch.isnan(pred_masked).any()
                    or torch.isnan(targ_masked).any()
                    or torch.isinf(pred_masked).any()
                    or torch.isinf(targ_masked).any()
                ):
                    reconstruction_loss = torch.tensor(0.0, device=predictions.device)
                else:
                    reconstruction_loss = F.mse_loss(pred_masked, targ_masked)
                    reconstruction_loss = torch.clamp(reconstruction_loss, 0, 100.0)

                losses["reconstruction"] = (
                    reconstruction_loss * self.ssl_objectives.lambda_masked
                )

        # Temporal Order Prediction Loss
        if "temporal_order" in ssl_outputs:
            outputs = ssl_outputs["temporal_order"]
            predictions = outputs["predictions"]
            targets = outputs["targets"]

            # Check for NaN/Inf values
            if (
                torch.isnan(predictions).any()
                or torch.isnan(targets).any()
                or torch.isinf(predictions).any()
                or torch.isinf(targets).any()
            ):
                temporal_loss = torch.tensor(0.0, device=predictions.device)
            else:
                temporal_loss = F.mse_loss(predictions, targets.float())
                temporal_loss = torch.clamp(temporal_loss, 0, 100.0)

            losses["temporal_order"] = (
                temporal_loss * self.ssl_objectives.lambda_temporal
            )

        # Contrastive Loss
        if "contrastive" in ssl_outputs:
            outputs = ssl_outputs["contrastive"]
            original = outputs["original"]
            augmented = outputs["augmented"]

            # Check for NaN/Inf values
            if (
                torch.isnan(original).any()
                or torch.isnan(augmented).any()
                or torch.isinf(original).any()
                or torch.isinf(augmented).any()
            ):
                contrastive_loss = torch.tensor(0.0, device=original.device)
            else:
                contrastive_loss = (
                    self.ssl_objectives.contrastive_learner.contrastive_loss(
                        original, augmented
                    )
                )
                contrastive_loss = torch.clamp(contrastive_loss, 0, 10.0)

            losses["contrastive"] = (
                contrastive_loss * self.ssl_objectives.lambda_contrastive
            )

        # Causal Next-Row Loss
        if "causal_next_row" in ssl_outputs:
            outputs = ssl_outputs["causal_next_row"]
            numeric_predictions = outputs["numeric_predictions"]
            numeric_targets = outputs["numeric_targets"]
            numeric_valid_mask = outputs["numeric_valid_mask"]

            if numeric_valid_mask.sum() > 0:
                pred_num = numeric_predictions[numeric_valid_mask]
                targ_num = numeric_targets[numeric_valid_mask]
                if (
                    torch.isnan(pred_num).any()
                    or torch.isnan(targ_num).any()
                    or torch.isinf(pred_num).any()
                    or torch.isinf(targ_num).any()
                ):
                    numeric_loss = torch.tensor(0.0, device=numeric_predictions.device)
                else:
                    numeric_loss = F.mse_loss(pred_num, targ_num)
                    numeric_loss = torch.clamp(numeric_loss, 0, 100.0)
            else:
                numeric_loss = torch.tensor(0.0, device=numeric_predictions.device)

            categorical_loss = torch.tensor(0.0, device=numeric_predictions.device)
            cat_loss_terms = 0

            if (
                "categorical_logits" in outputs
                and "categorical_targets" in outputs
                and "categorical_valid_mask" in outputs
            ):
                categorical_targets = outputs["categorical_targets"]
                categorical_valid_mask = outputs["categorical_valid_mask"]
                for i, logits in enumerate(outputs["categorical_logits"]):
                    targets_i = categorical_targets[:, :, i].clone().long()
                    valid_i = categorical_valid_mask[:, :, i]
                    if valid_i.sum() == 0:
                        continue

                    targets_i = targets_i.masked_fill(~valid_i, -100)
                    cat_i_loss = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        targets_i.reshape(-1),
                        ignore_index=-100,
                    )

                    if not torch.isnan(cat_i_loss) and not torch.isinf(cat_i_loss):
                        categorical_loss = categorical_loss + cat_i_loss
                        cat_loss_terms += 1

            if cat_loss_terms > 0:
                categorical_loss = categorical_loss / cat_loss_terms

            causal_loss = numeric_loss + categorical_loss
            losses["causal_next_row"] = causal_loss * self.lambda_causal

        # Total SSL loss
        if losses:
            total_ssl_loss = sum(losses.values())
            losses["total_ssl"] = total_ssl_loss

        return losses

    def training_step(self, batch, batch_idx):
        """Training step with mode-specific processing."""

        if self.mode == "pretrain":
            return self._training_step_pretrain(batch, batch_idx)
        else:
            return self._training_step_standard(batch, batch_idx)

    def _training_step_standard(self, batch, batch_idx):
        """Standard training step for standard and variable_features modes."""
        x_num = batch["x_num"]
        x_cat = batch.get("x_cat", None)
        y = batch["y"]

        y_hat = self(x_num, x_cat)

        if self.task == "classification":
            if y.ndim == 2:
                y = torch.argmax(y, dim=1)
            y = y.long()
            loss = self.loss_fn(y_hat, y)
        else:
            loss = self.loss_fn(y_hat.squeeze(), y.float())

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def _training_step_pretrain(self, batch, batch_idx):
        """Training step for pretrain mode with SSL + supervised learning."""
        x_num = batch["x_num"]
        x_cat = batch.get("x_cat", None)
        y = batch.get("y", None)
        feature_mask = batch.get("feature_mask", None)

        # Fallback feature mask: mark provided feature slots as active.
        if feature_mask is None:
            B, c_num, T = x_num.shape
            total_features = self.max_numeric_features + self.max_categorical_features
            feature_mask = torch.zeros(
                B, total_features, T, device=x_num.device, dtype=torch.float
            )
            active_num = min(c_num, self.max_numeric_features)
            feature_mask[:, :active_num, :] = 1.0
            if x_cat is not None and self.max_categorical_features > 0:
                active_cat = min(x_cat.shape[1], self.max_categorical_features)
                feature_mask[
                    :,
                    self.max_numeric_features : self.max_numeric_features + active_cat,
                    :,
                ] = 1.0

        # SSL forward pass
        ssl_outputs = self.forward_ssl(x_num, feature_mask, x_cat)
        ssl_losses = self.compute_ssl_losses(ssl_outputs)

        # Optional supervised term (can be disabled for pure SSL pretraining).
        supervised_losses = []
        if y is not None and self.lambda_supervised > 0:
            y_hat = self(x_num, x_cat)
            B = x_num.shape[0]
            num_classes_batch = batch.get("num_classes", None)

            if num_classes_batch is None:
                num_classes_batch = torch.full(
                    (B,),
                    int(self.head.out_features),
                    device=x_num.device,
                    dtype=torch.long,
                )
            elif not torch.is_tensor(num_classes_batch):
                num_classes_batch = torch.tensor(
                    num_classes_batch, device=x_num.device, dtype=torch.long
                )

            for i in range(B):
                sample_y = y[i : i + 1]
                sample_y_hat = y_hat[i : i + 1]
                sample_num_classes = int(num_classes_batch[i].item())
                sample_y_hat_adj = sample_y_hat[:, :sample_num_classes]

                if sample_y.ndim == 2:
                    sample_y = torch.argmax(sample_y, dim=1)
                sample_y = sample_y.long()

                # Skip missing labels and out-of-range labels.
                if sample_y.item() < 0 or sample_y.item() >= sample_num_classes:
                    continue

                loss = self.loss_fn(sample_y_hat_adj, sample_y)
                supervised_losses.append(loss)

        # Average supervised loss
        if supervised_losses:
            supervised_loss = torch.stack(supervised_losses).mean()
        else:
            supervised_loss = torch.tensor(0.0, device=x_num.device)

        # Total loss
        total_ssl_loss = ssl_losses.get(
            "total_ssl", torch.tensor(0.0, device=x_num.device)
        )
        total_loss = total_ssl_loss + self.lambda_supervised * supervised_loss

        # Check for NaN in individual components
        if (
            torch.isnan(total_ssl_loss)
            or torch.isnan(supervised_loss)
            or torch.isnan(total_loss)
        ):
            print("NaN detected!")
            print(f"  SSL loss: {total_ssl_loss}")
            print(f"  Supervised loss: {supervised_loss}")
            print(f"  Total loss: {total_loss}")
            total_loss = torch.tensor(1.0, device=x_num.device, requires_grad=True)

        # Logging
        self.log("train/total_loss", total_loss, prog_bar=True)
        self.log("train/ssl_loss", total_ssl_loss)
        self.log("train/supervised_loss", supervised_loss)

        for loss_name, loss_value in ssl_losses.items():
            if loss_name != "total_ssl":
                self.log(f"train/{loss_name}_loss", loss_value)

        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Validation step with mode-specific processing."""
        x_num = batch["x_num"]
        x_cat = batch.get("x_cat", None)
        y = batch.get("y", None)

        if y is None:
            # Pure SSL pretraining can run without labels.
            return torch.tensor(0.0, device=x_num.device)

        y_hat = self(x_num, x_cat)

        if self.task == "classification":
            if y.ndim == 2:
                y = torch.argmax(y, dim=1)
            y = y.long()

            if self.mode == "pretrain":
                # Handle multi-class datasets
                dataset_names = batch.get("dataset_name", ["dataset"])
                dataset_name = dataset_names[0]
                num_classes_batch = batch.get("num_classes", None)
                if num_classes_batch is None:
                    num_classes = int(self.head.out_features)
                elif torch.is_tensor(num_classes_batch):
                    num_classes = int(num_classes_batch[0].item())
                else:
                    num_classes = int(num_classes_batch[0])
                y_hat_adj = y_hat[:, :num_classes]
                loss = self.loss_fn(y_hat_adj, y)
                self.log(f"val/{dataset_name}_loss", loss, add_dataloader_idx=False)
            else:
                loss = self.loss_fn(y_hat, y)
                self.log("val_loss", loss, prog_bar=True)
        else:
            loss = self.loss_fn(y_hat.squeeze(), y.float())
            self.log("val_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def get_embedding_stats(self):
        """Get statistics about the column embedder."""
        if self.column_embedder is None:
            return {"strategy": "none"}

        stats = {"strategy": self.column_embedding_strategy}

        if hasattr(self.column_embedder, "get_cache_stats"):
            stats.update(self.column_embedder.get_cache_stats())

        if hasattr(self.column_embedder, "embedding"):
            stats["vocab_size"] = self.column_embedder.embedding.num_embeddings
            stats["next_idx"] = self.column_embedder.next_idx

        return stats

    @classmethod
    def create_for_multi_dataset(
        cls,
        dataset_schemas: dict[str, dict[str, Any]],
        num_classes: int,
        mode: str = "variable_features",
        column_embedding_strategy: str = "auto_expanding",
        categorical_cardinalities: list[int] | None = None,
        **kwargs,
    ) -> "MultiDatasetModel":
        """Factory method for multi-dataset training.

        Args:
            dataset_schemas: Dict mapping dataset names to schema info
            num_classes: Number of output classes
            mode: Model mode
            column_embedding_strategy: Column embedding strategy
            categorical_cardinalities: List of cardinality for each categorical feature
            **kwargs: Additional model arguments

        Returns:
            Configured MultiDatasetModel instance
        """
        # Find maximum feature counts across all datasets
        max_numeric = max(schema["numeric"] for schema in dataset_schemas.values())
        max_categorical = max(
            schema.get("categorical", 0) for schema in dataset_schemas.values()
        )

        print("Multi-dataset model configuration:")
        print(f"  Mode: {mode}")
        print(f"  Max numeric features: {max_numeric}")
        print(f"  Max categorical features: {max_categorical}")
        print("  Dataset schemas:")

        for name, schema in dataset_schemas.items():
            print(
                f"    {name}: {schema['numeric']} numeric + "
                f"{schema.get('categorical', 0)} categorical"
            )

        model = cls(
            max_numeric_features=max_numeric,
            max_categorical_features=max_categorical,
            num_classes=num_classes,
            mode=mode,
            column_embedding_strategy=column_embedding_strategy,
            categorical_cardinalities=categorical_cardinalities,
            **kwargs,
        )

        return model

    @classmethod
    def create_baseline(
        cls,
        max_numeric_features: int,
        max_categorical_features: int = 0,
        num_classes: int = 2,
        mode: str = "variable_features",
        categorical_cardinalities: list[int] | None = None,
        **kwargs,
    ) -> "MultiDatasetModel":
        """Create baseline model without column embeddings."""
        return cls(
            max_numeric_features=max_numeric_features,
            max_categorical_features=max_categorical_features,
            num_classes=num_classes,
            mode=mode,
            column_embedding_strategy="none",
            categorical_cardinalities=categorical_cardinalities,
            **kwargs,
        )

    @classmethod
    def create_for_dataset(
        cls,
        c_in: int,
        seq_len: int,
        num_classes: int,
        column_names: list[str],
        mode: str = "standard",
        column_embedding_strategy: str = "frozen_bert",
        **kwargs,
    ) -> "MultiDatasetModel":
        """Factory method to create model for a specific dataset (standard mode).

        Args:
            c_in: Number of input channels
            seq_len: Sequence length
            num_classes: Number of classes
            column_names: Column names for this dataset
            mode: Model mode (should be 'standard' for this factory method)
            column_embedding_strategy: Column embedding strategy
            **kwargs: Additional model arguments

        Returns:
            Configured MultiDatasetModel instance
        """
        model = cls(
            max_numeric_features=c_in,
            max_categorical_features=0,
            num_classes=num_classes,
            mode=mode,
            column_embedding_strategy=column_embedding_strategy,
            max_sequence_length=seq_len,
            **kwargs,
        )

        # Set columns for this dataset
        model.set_dataset_columns(column_names)

        return model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        num_classes: int,
        freeze_encoder: bool = True,
    ) -> "MultiDatasetModel":
        """Load pre-trained model for fine-tuning.

        Args:
            pretrained_path: Path to pre-trained checkpoint
            num_classes: Number of classes for fine-tuning task
            freeze_encoder: Whether to freeze encoder layers

        Returns:
            Model loaded from pre-trained weights
        """
        # Load checkpoint
        checkpoint = torch.load(pretrained_path, map_location="cpu")

        # Extract hyperparameters
        hparams = checkpoint.get("hyper_parameters", {})

        # Create model
        excluded_keys = {
            "max_numeric_features",
            "max_categorical_features",
            "num_classes",
            "mode",
        }
        passthrough_hparams = {
            k: v for k, v in hparams.items() if k not in excluded_keys
        }

        model = cls(
            max_numeric_features=hparams["max_numeric_features"],
            max_categorical_features=hparams["max_categorical_features"],
            num_classes=num_classes,  # Use target number of classes
            mode="variable_features",  # Default to variable_features for fine-tuning
            **passthrough_hparams,
        )

        # Load state dict (skip SSL heads and mismatched classification head)
        state_dict = checkpoint["state_dict"]

        # Filter out SSL heads and classification head
        encoder_state_dict = {
            k: v
            for k, v in state_dict.items()
            if not k.startswith("ssl_heads.") and not k.startswith("head.")
        }

        # Load encoder weights
        model.load_state_dict(encoder_state_dict, strict=False)

        # Freeze encoder if requested
        if freeze_encoder:
            for name, param in model.named_parameters():
                if not name.startswith("head."):
                    param.requires_grad = False

        print(f"Loaded pre-trained model from {pretrained_path}")
        print(f"Frozen encoder: {freeze_encoder}")

        return model


if __name__ == "__main__":
    # Test the consolidated model in different modes
    print("Testing MultiDatasetModel...")

    # Test 1: Standard mode
    print("\n=== Testing Standard Mode ===")
    model_standard = MultiDatasetModel.create_for_dataset(
        c_in=7,
        seq_len=96,
        num_classes=3,
        column_names=["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
        mode="standard",
        column_embedding_strategy="frozen_bert",
    )

    x_num = torch.randn(4, 7, 96)
    x_num[0, :2, :10] = float("nan")  # Add some NaNs
    output = model_standard(x_num)
    print(f"Standard mode output: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model_standard.parameters()):,}")

    # Test 2: Variable features mode
    print("\n=== Testing Variable Features Mode ===")
    dataset_schemas = {
        "sensor_data": {
            "numeric": 8,
            "categorical": 2,
            "columns": [
                "temp1",
                "temp2",
                "humidity",
                "pressure",
                "wind",
                "rain",
                "light",
                "sound",
                "zone",
                "status",
            ],
        },
        "user_metrics": {
            "numeric": 12,
            "categorical": 1,
            "columns": [
                "clicks",
                "views",
                "time",
                "bounce",
                "conversion",
                "revenue",
                "sessions",
                "pages",
                "downloads",
                "shares",
                "likes",
                "comments",
                "segment",
            ],
        },
    }

    model_variable = MultiDatasetModel.create_for_multi_dataset(
        dataset_schemas=dataset_schemas,
        num_classes=3,
        mode="variable_features",
        column_embedding_strategy="auto_expanding",
        categorical_cardinalities=[10, 5],  # For 2 categorical features max
    )

    # Test with first dataset
    model_variable.set_dataset_schema(8, 2, dataset_schemas["sensor_data"]["columns"])
    x_num = torch.randn(4, 8, 96)
    x_cat = torch.randint(0, 5, (4, 2, 96)).float()
    output = model_variable(x_num, x_cat)
    print(f"Variable features mode output: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model_variable.parameters()):,}")

    # Test 3: Pretrain mode
    print("\n=== Testing Pretrain Mode ===")
    model_pretrain = MultiDatasetModel(
        max_numeric_features=12,
        max_categorical_features=0,
        num_classes=6,
        mode="pretrain",
        ssl_objectives={
            "masked_patch": True,
            "temporal_order": True,
            "contrastive": True,
        },
        d_model=64,
        n_head=4,
        num_layers=2,
    )

    B, C, T = 4, 12, 96
    x_num = torch.randn(B, C, T)
    feature_mask = torch.ones(B, C, T)
    x_num[0, :2, :10] = float("nan")  # Add some NaNs

    # SSL forward pass
    ssl_outputs = model_pretrain.forward_ssl(x_num, feature_mask)
    ssl_losses = model_pretrain.compute_ssl_losses(ssl_outputs)

    print(f"SSL outputs keys: {list(ssl_outputs.keys())}")
    print(f"SSL losses: {[(k, v.item()) for k, v in ssl_losses.items()]}")

    # Standard forward pass
    output = model_pretrain(x_num)
    print(f"Pretrain mode output: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model_pretrain.parameters()):,}")

    print("\n✅ MultiDatasetModel test complete!")
