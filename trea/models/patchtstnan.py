"""PatchTSTNan: A Patch-based Transformer with NaN handling for time series.

This module implements the PatchTSTNan model that combines patch-based processing
with robust NaN handling using dual-channel encoding (value + mask).
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn

from torchmetrics.classification import Accuracy, F1Score


class PatchTSTNan(pl.LightningModule):
    """PatchTST with dual-patch NaN handling.

    This model handles missing values by encoding them in a dual-patch format:
    - Value channel: original reading, NaNs replaced by 0
    - Mask channel: 1 if the reading was missing, else 0

    Args:
        C_num: Number of numeric input channels
        C_cat: Number of categorical input channels
        cat_cardinality: Cardinality of categorical features
        T: Sequence length
        d_model: Model dimension (default: 64)
        patch_size: Size of each patch (default: 8)
        num_classes: Number of output classes for classification (default: 2)
        n_heads: Number of attention heads (default: 4)
        n_layers: Number of transformer layers (default: 2)
        dropout: Dropout rate (default: 0.1)
        learning_rate: Learning rate (default: 1e-3)
        task: Task type, 'classification' or 'regression' (default: 'classification')
    """

    def __init__(
        self,
        C_num: int | None = None,  # Make optional so c_in can be used instead
        C_cat: int = 0,
        cat_cardinality: int = 1,
        T: int | None = None,  # Make optional so seq_len can be used instead
        d_model: int = 64,
        patch_size: int = 8,
        num_classes: int = 2,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        task: str = "classification",
        # Alternative parameter names for compatibility
        c_in: int | None = None,
        seq_len: int | None = None,
        patch_len: int | None = None,
        stride: int | None = None,
        n_head: int | None = None,
        num_layers: int | None = None,
        lr: float | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Handle alternative parameter names for compatibility
        if c_in is not None:
            C_num = c_in
        elif C_num is None:
            raise ValueError("Either C_num or c_in must be provided")

        if seq_len is not None:
            T = seq_len
        elif T is None:
            T = 64  # Default value

        if patch_len is not None:
            patch_size = patch_len
        if n_head is not None:
            n_heads = n_head
        if num_layers is not None:
            n_layers = num_layers
        if lr is not None:
            learning_rate = lr

        self.C_num = C_num
        self.C_cat = C_cat
        self.T = T
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        self.d_model = d_model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.task = task

        # Calculate number of patches
        if stride is not None:
            self.n_patches = (T - patch_size) // stride + 1
        else:
            self.n_patches = T // patch_size

        # Categorical embeddings (if any)
        if C_cat > 0:
            self.cat_embs = nn.ModuleList(
                [nn.Embedding(cat_cardinality, d_model) for _ in range(C_cat)]
            )
        else:
            self.cat_embs = None

        # Numeric projection with missing value handling (double channels for mask)
        self.num_proj = nn.Conv1d(C_num * 2, d_model, kernel_size=1)

        # Patch embedding
        if stride is not None:
            # For stride-based patching (like in examples/compare_models_etth1.py)
            self.patch_embedding = nn.Linear(patch_size * (2 * C_num), d_model)
        else:
            # For standard patching
            self.patch_proj = nn.Linear(patch_size, d_model)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
            norm_first=stride is not None,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output head
        if task == "classification":
            self.head = nn.Linear(d_model, num_classes)
            self.loss_fn = nn.CrossEntropyLoss()
        else:  # regression
            self.head = nn.Linear(d_model, 1)
            self.loss_fn = nn.MSELoss()

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Metrics
        if self.task == "classification":
            self.train_accuracy = Accuracy(
                task="multiclass", num_classes=self.num_classes
            )
            self.val_accuracy = Accuracy(
                task="multiclass", num_classes=self.num_classes
            )
            self.val_f1 = F1Score(task="multiclass", num_classes=self.num_classes)

    def create_patches_standard(self, x):
        """Convert time series to patches (standard method)"""
        B, C, T = x.shape
        # Reshape to patches: [B, C, n_patches, patch_size]
        x_patches = x.view(B, C, self.n_patches, self.patch_size)
        # Flatten spatial dimension: [B, C * n_patches, patch_size]
        x_patches = x_patches.view(B, C * self.n_patches, self.patch_size)
        return x_patches

    def create_patches_stride(self, x):
        """Create patches with stride (alternative method)"""
        B, C, T = x.shape
        patches = []

        for i in range(0, T - self.patch_size + 1, self.stride):
            patch = x[:, :, i : i + self.patch_size]  # [B, C, patch_size]
            patch = patch.reshape(B, -1)  # [B, C * patch_size]
            patches.append(patch)

        patches = torch.stack(patches, dim=1)  # [B, num_patches, C * patch_size]
        return patches

    def forward(self, x_num, x_cat=None):
        B = x_num.shape[0]

        # Handle missing values in numeric data
        m_nan = torch.isnan(x_num).float()  # [B, C_num, T]
        x_val = torch.nan_to_num(x_num, nan=0.0)  # [B, C_num, T]

        # Stack value and mask channels (dual-patch)
        x_num_with_mask = torch.cat([x_val, m_nan], dim=1)  # [B, 2*C_num, T]

        if hasattr(self, "patch_embedding"):
            # Stride-based patching method
            patches = self.create_patches_stride(x_num_with_mask)
            z_patches = self.patch_embedding(patches)  # [B, n_patches, d_model]
        else:
            # Standard patching method
            # Project numeric features first
            z_num = self.num_proj(x_num_with_mask)  # [B, d_model, T]

            # Add categorical embeddings (skip if no categorical features)
            if self.C_cat > 0 and x_cat is not None and x_cat.numel() > 0:
                cat_emb = sum(emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embs))
                z_num = z_num + cat_emb.permute(0, 2, 1)

            # Create patches
            z_patches = self.create_patches_standard(
                z_num
            )  # [B, C*n_patches, patch_size]

            # Project patches to model dimension
            z_patches = self.patch_proj(z_patches)  # [B, C*n_patches, d_model]

            # Reshape for transformer
            n_channel_patches = z_patches.shape[1]
            z_patches = z_patches.view(B, n_channel_patches, self.d_model)

        # Add positional embeddings
        if hasattr(self, "patch_embedding"):
            # For stride-based method, pos_embed shape matches directly
            z_patches = z_patches + self.pos_embed
        else:
            # For standard method, broadcast across channels
            n_channel_patches = z_patches.shape[1]
            pos_embed_expanded = self.pos_embed.repeat(
                1, n_channel_patches // self.n_patches, 1
            )
            z_patches = z_patches + pos_embed_expanded

        # Apply dropout
        z_patches = self.dropout(z_patches)

        # Transformer encoding
        z_encoded = self.transformer(z_patches)

        # Global average pooling
        z_pooled = z_encoded.mean(dim=1)  # [B, d_model]

        # Output head
        output = self.head(z_pooled)

        if self.task == "regression":
            output = output.squeeze(-1)  # Remove last dimension for regression

        return output

    def training_step(self, batch, batch_idx):
        output = self(batch["x_num"], batch.get("x_cat"))

        # Handle different label formats
        labels = batch["y"]
        if self.task == "classification":
            # Ensure labels are in the right format
            if labels.ndim == 2:
                labels = torch.argmax(labels, dim=1)
            labels = labels.long()
        else:
            labels = labels.float()

        loss = self.loss_fn(output, labels)

        # Calculate metrics
        if self.task == "classification":
            self.train_accuracy.update(output, labels)
            self.log(
                "train_acc",
                self.train_accuracy,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch["x_num"], batch.get("x_cat"))

        labels = batch["y"]
        if self.task == "classification":
            if labels.ndim == 2:
                labels = torch.argmax(labels, dim=1)
            labels = labels.long()
        else:
            labels = labels.float()

        loss = self.loss_fn(output, labels)

        # Calculate metrics
        if self.task == "classification":
            self.val_accuracy.update(output, labels)
            self.val_f1.update(output, labels)
            self.log("val_acc", self.val_accuracy, on_epoch=True, prog_bar=True)
            self.log("val_f1", self.val_f1, on_epoch=True, prog_bar=True)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return [optimizer], [scheduler]
