"""Lightning modules for the single-row tabular model.

SingleRowRegressor  -- supervised regression head on the tabular encoder.
SingleRowMTM        -- masked tabular modeling (SSL pretraining), corrected
                       fixed-probability masking (no batch_idx bug).
transfer_encoder    -- copy a pretrained encoder into a fresh regressor.
build_trainer       -- Trainer with early-stopping + best-val checkpointing baked in
                       (we evaluate best-val weights, not last-epoch weights -- the
                       lesson from the variance diagnostic).
"""

import os
from typing import Optional

import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn

from trea.models.single_row import (
    MaskedTabularEncoder,
    TabularClassifier,
    TabularRegressor,
    mask_tensor,
)
from trea.utils.single_row_data import InputsTarget


class SingleRowRegressor(L.LightningModule):
    def __init__(
        self,
        config,
        d_model=128,
        n_heads=4,
        n_layers=4,
        lr=8e-4,
        dropout=0.2,
        col_embedder=None,
        column_descriptions=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["config", "col_embedder"])
        self.lr = lr
        self.model = TabularRegressor(
            config,
            d_model,
            n_heads,
            n_layers,
            dropout,
            col_embedder,
            column_descriptions,
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, batch: InputsTarget):
        return self.model(batch.inputs.numeric, batch.inputs.categorical)

    def _step(self, batch, name):
        y_hat = self.model(batch.inputs.numeric, batch.inputs.categorical)
        loss = self.loss_fn(y_hat, batch.target)
        self.log(name, loss, prog_bar=(name == "val_loss"))
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train_loss")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val_loss")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.1, patience=3
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"},
        }


class SingleRowClassifier(L.LightningModule):
    """Supervised classification head on the tabular encoder (macro-F1 the metric).

    Same encoder/pooling as the regressor, so a pretrained encoder transfers to either
    task via ``transfer_encoder``. Targets are class indices (``long``); the collated
    target arrives as ``[B, 1]`` float and is cast/flattened here.
    """

    def __init__(
        self,
        config,
        num_classes,
        d_model=128,
        n_heads=4,
        n_layers=4,
        lr=8e-4,
        dropout=0.2,
        col_embedder=None,
        column_descriptions=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["config", "col_embedder"])
        self.lr = lr
        self.model = TabularClassifier(
            config,
            num_classes,
            d_model,
            n_heads,
            n_layers,
            dropout,
            col_embedder,
            column_descriptions,
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch: InputsTarget):
        return self.model(batch.inputs.numeric, batch.inputs.categorical)

    def _step(self, batch, name):
        logits = self.model(batch.inputs.numeric, batch.inputs.categorical)
        loss = self.loss_fn(logits, batch.target.long().view(-1))
        self.log(name, loss, prog_bar=(name == "val_loss"))
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train_loss")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val_loss")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.1, patience=3
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"},
        }


class SingleRowMTM(L.LightningModule):
    """Masked tabular modeling: reconstruct masked numeric + categorical cells."""

    MASK_KEEP = 0.7  # mask each cell where rand > 0.7  -> ~30% masked

    def __init__(
        self,
        config,
        d_model=128,
        n_heads=4,
        n_layers=4,
        lr=8e-4,
        dropout=0.2,
        col_embedder=None,
        column_descriptions=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["config", "col_embedder"])
        self.lr = lr
        self.model = MaskedTabularEncoder(
            config,
            d_model,
            n_heads,
            n_layers,
            dropout,
            col_embedder,
            column_descriptions,
        )
        self.numeric_loss_fn = nn.MSELoss()
        self.categorical_loss_fn = nn.CrossEntropyLoss()

    def aggregate_loss(self, actual, predicted):
        num_loss = self.numeric_loss_fn(predicted.numeric, actual.numeric)
        b, n_cat, n_tok = predicted.categorical.size()
        cat_loss = self.categorical_loss_fn(
            predicted.categorical.reshape(b * n_cat, n_tok),
            actual.categorical.reshape(-1),
        )
        return num_loss + cat_loss

    def _step(self, batch, name):
        num_masked = mask_tensor(batch.numeric, self.model, self.MASK_KEEP)
        cat_masked = mask_tensor(batch.categorical, self.model, self.MASK_KEEP)
        predicted = self.model(num_masked, cat_masked)
        loss = self.aggregate_loss(batch, predicted)
        self.log(name, loss, prog_bar=(name == "val_loss"))
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train_loss")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val_loss")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.1, patience=3
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"},
        }


def transfer_encoder(mtm: SingleRowMTM, regressor: SingleRowRegressor):
    """Copy the pretrained encoder weights from an MTM module into a regressor."""
    regressor.model.tabular_encoder.load_state_dict(
        mtm.model.tabular_encoder.state_dict()
    )
    return regressor


def copy_encoder_weights(src_encoder, dst_encoder, include_col_embedder: bool = True):
    """Copy a trained ``TabularEncoder``'s weights into another (cross-schema transfer).

    With a SemanticColumnEmbedder, ``include_col_embedder=True`` moves the learned
    name->d_model projection so the source schema's identity->value knowledge applies
    to the destination schema's (differently named, reordered) columns -- the whole
    point. The frozen text features are not in state_dict, so the destination keeps
    its own column descriptions.

    ``include_col_embedder=False`` transfers everything *except* the column identities
    (fresh identities on the destination). This is the charitable index-embedding
    control: index embeddings have no cross-schema name correspondence, so the most
    they can offer is the transferred attention stack + value embeddings.
    """
    sd = src_encoder.state_dict()
    if not include_col_embedder:
        sd = {k: v for k, v in sd.items() if not k.startswith("col_embedder.")}
    dst_encoder.load_state_dict(sd, strict=False)
    return dst_encoder


def build_trainer(
    max_epochs: int = 100,
    patience: Optional[int] = 8,
    ckpt_dir: Optional[str] = None,
    grad_clip: float = 1.0,
):
    """Trainer with early-stopping (if patience set) and best-val checkpointing.

    Returns (trainer, ckpt_callback). After `fit`, call `load_best(module, ckpt)`
    to restore the best-validation weights before evaluating.
    """
    callbacks = []
    if patience is not None:
        callbacks.append(
            EarlyStopping(monitor="val_loss", patience=patience, mode="min")
        )
    ckpt = None
    if ckpt_dir is not None:
        ckpt = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            dirpath=ckpt_dir,
            filename="best",
        )
        callbacks.append(ckpt)
    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=False,
        enable_checkpointing=ckpt_dir is not None,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="auto",
        devices=1,
        gradient_clip_val=grad_clip,
    )
    return trainer, ckpt


def load_best(module: L.LightningModule, ckpt: ModelCheckpoint):
    """Restore best-validation weights saved by `ckpt` into `module` (in place)."""
    if (
        ckpt is None
        or not ckpt.best_model_path
        or not os.path.exists(ckpt.best_model_path)
    ):
        return module
    state = torch.load(ckpt.best_model_path, map_location="cpu", weights_only=False)
    module.load_state_dict(state["state_dict"])
    return module
