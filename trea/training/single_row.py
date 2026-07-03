"""Lightning modules for the single-row tabular model.

SingleRowRegressor  -- supervised regression head on the tabular encoder
                       (MSE or HL-Gauss binned head).
SingleRowClassifier -- supervised classification head (logit-adjusted CE,
                       macro-F1 model selection).
SingleRowMTM        -- masked tabular modeling (SSL pretraining), corrected
                       fixed-probability masking (no batch_idx bug); loss on
                       masked cells only.
transfer_encoder    -- copy a pretrained encoder into a fresh regressor.
build_trainer       -- Trainer with early-stopping + best-val checkpointing baked in
                       (we evaluate best-val weights, not last-epoch weights -- the
                       lesson from the variance diagnostic).
"""

import os

import pytorch_lightning as L
import torch
import torch.nn.functional as F

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from torchmetrics.classification import MulticlassF1Score

from trea.models.single_row import (
    MaskedTabularEncoder,
    TabularClassifier,
    TabularRegressor,
    mask_tensor,
)
from trea.utils.single_row_data import InputsTarget


def adamw_parameter_groups(module: nn.Module, weight_decay: float):
    """AdamW groups: decay regular weights, never embeddings/norms/biases."""
    no_decay_modules = (nn.Embedding, nn.LayerNorm)
    no_decay_ids = set()
    for child in module.modules():
        if isinstance(child, no_decay_modules):
            no_decay_ids.update(id(p) for p in child.parameters(recurse=False))

    decay_params, no_decay_params = [], []
    decay_names, no_decay_names = [], []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or name == "bias" or id(param) in no_decay_ids:
            no_decay_params.append(param)
            no_decay_names.append(name)
        else:
            decay_params.append(param)
            decay_names.append(name)

    return [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
            "names": decay_names,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
            "names": no_decay_names,
        },
    ]


def _configure_adamw(module: L.LightningModule):
    opt = torch.optim.AdamW(
        adamw_parameter_groups(module, module.weight_decay),
        lr=module.lr,
        betas=(0.9, 0.95),
    )
    if module.warmup_frac <= 0:
        return opt
    total_steps = max(1, int(module.trainer.estimated_stepping_batches))
    warmup_steps = max(1, int(total_steps * module.warmup_frac))

    def lr_lambda(step):
        return min(1.0, float(step + 1) / float(warmup_steps))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    return {
        "optimizer": opt,
        "lr_scheduler": {"scheduler": sched, "interval": "step"},
    }


def _log_attention_entropy(module: L.LightningModule, encoder, prefix: str):
    for layer_i, layer in enumerate(encoder.layers):
        entropy = layer.last_attn_entropy
        if entropy is None:
            continue
        for head_i, value in enumerate(entropy):
            module.log(
                f"{prefix}_attn_entropy_l{layer_i}_h{head_i}",
                value,
                on_step=False,
                on_epoch=True,
            )


class HLGaussLoss(nn.Module):
    """Histogram-loss Gaussian target for scalar regression (arXiv:2403.03950)."""

    def __init__(
        self,
        y_train: torch.Tensor | list[float],
        n_bins: int = 100,
        sigma_bins: float = 2.0,
        bin_spacing: str = "uniform",
        symlog: bool = False,
    ):
        super().__init__()
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        self.symlog = symlog
        y_t = torch.as_tensor(y_train, dtype=torch.float32).view(-1)
        if symlog:
            y_t = torch.sign(y_t) * torch.log(torch.abs(y_t) + 1.0)

        if bin_spacing == "quantile":
            centers = torch.quantile(
                y_t, torch.linspace(0, 1, n_bins, device=y_t.device)
            )
            width = float(centers[-1] - centers[0]) / max(1, n_bins - 1)
        else:
            y_min, y_max = y_t.min().item(), y_t.max().item()
            width = (y_max - y_min) / max(1, n_bins - 1)
            pad = 3 * sigma_bins * width
            centers = torch.linspace(y_min - pad, y_max + pad, n_bins)
            width = float(centers[-1] - centers[0]) / max(1, n_bins - 1)

        self.register_buffer("centers", centers)
        self.sigma = max(1e-6, float(sigma_bins * width))

    def target_distribution(self, target: torch.Tensor):
        y = target.view(-1, 1).to(self.centers.device)
        if self.symlog:
            y = torch.sign(y) * torch.log(torch.abs(y) + 1.0)
        weights = torch.exp(-0.5 * ((self.centers.view(1, -1) - y) / self.sigma) ** 2)
        return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        dist = self.target_distribution(target)
        return -(dist * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

    def expected_value(self, logits: torch.Tensor):
        probs = F.softmax(logits, dim=1)
        val = torch.matmul(probs, self.centers.to(logits.device)).unsqueeze(-1)
        if self.symlog:
            val = torch.sign(val) * (torch.exp(torch.abs(val)) - 1.0)
        return val


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
        head: str = "mse",
        y_train: torch.Tensor | list[float] | None = None,
        hl_gauss_bins: int = 100,
        hl_gauss_sigma_bins: float = 2.0,
        bin_spacing: str = "uniform",
        symlog: bool = False,
        qk_norm: bool = True,
        pre_norm: bool = True,
        value_encoder: str = "identity",
        fusion: str = "additive",
        warmup_frac: float = 0.05,
        weight_decay: float = 3e-4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["config", "col_embedder"])
        self.lr = lr
        self.warmup_frac = warmup_frac
        self.weight_decay = weight_decay
        output_dim = hl_gauss_bins if head == "hl_gauss" else 1
        self.model = TabularRegressor(
            config,
            d_model,
            n_heads,
            n_layers,
            dropout,
            col_embedder,
            column_descriptions,
            qk_norm=qk_norm,
            pre_norm=pre_norm,
            output_dim=output_dim,
            value_encoder=value_encoder,
            fusion=fusion,
        )
        if head == "hl_gauss":
            if y_train is None:
                raise ValueError("y_train is required for hl_gauss head")
            self.loss_fn = HLGaussLoss(
                y_train=y_train,
                n_bins=hl_gauss_bins,
                sigma_bins=hl_gauss_sigma_bins,
                bin_spacing=bin_spacing,
                symlog=symlog,
            )
        else:
            self.loss_fn = nn.MSELoss()

    def forward(self, batch: InputsTarget):
        return self.model(batch.inputs.numeric, batch.inputs.categorical)

    def _step(self, batch, name):
        out = self.model(batch.inputs.numeric, batch.inputs.categorical)
        loss = self.loss_fn(out, batch.target)
        self.log(name, loss, prog_bar=(name == "val_loss"))
        if name == "val_loss":
            _log_attention_entropy(self, self.model.tabular_encoder, "val")
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train_loss")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val_loss")

    def configure_optimizers(self):
        return _configure_adamw(self)

    def predict_value(self, batch: InputsTarget):
        out = self.model(batch.inputs.numeric, batch.inputs.categorical)
        if isinstance(self.loss_fn, HLGaussLoss):
            return self.loss_fn.expected_value(out)
        return out


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
        class_priors=None,
        logit_adjustment_tau: float = 1.0,
        label_smoothing: float = 0.0,
        qk_norm: bool = True,
        pre_norm: bool = True,
        value_encoder: str = "identity",
        fusion: str = "additive",
        warmup_frac: float = 0.05,
        weight_decay: float = 3e-4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["config", "col_embedder"])
        self.lr = lr
        self.warmup_frac = warmup_frac
        self.weight_decay = weight_decay
        self.logit_adjustment_tau = logit_adjustment_tau
        self.model = TabularClassifier(
            config,
            num_classes,
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
        )
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        priors = (
            torch.as_tensor(class_priors, dtype=torch.float32)
            if class_priors is not None
            else torch.ones(num_classes, dtype=torch.float32) / num_classes
        )
        priors = priors.clamp_min(1e-12)
        priors = priors / priors.sum()
        self.register_buffer("class_log_priors", priors.log())
        self.val_macro_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.register_buffer(
            "val_pred_counts", torch.zeros(num_classes, dtype=torch.float32)
        )

    def forward(self, batch: InputsTarget):
        return self.model(batch.inputs.numeric, batch.inputs.categorical)

    def training_logits(self, logits):
        if self.logit_adjustment_tau == 0:
            return logits
        return logits + self.logit_adjustment_tau * self.class_log_priors

    def posthoc_adjust_logits(self, logits, tau: float | None = None):
        tau = self.logit_adjustment_tau if tau is None else tau
        if tau == 0:
            return logits
        return logits - tau * self.class_log_priors

    def _step(self, batch, name):
        logits = self.model(batch.inputs.numeric, batch.inputs.categorical)
        target = batch.target.long().view(-1)
        loss = self.loss_fn(self.training_logits(logits), target)
        self.log(name, loss, prog_bar=(name == "val_loss"))
        if name == "val_loss":
            preds = logits.argmax(dim=1)
            self.val_macro_f1.update(preds, target)
            self.log(
                "val_macro_f1",
                self.val_macro_f1,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.val_pred_counts += torch.bincount(
                preds, minlength=self.val_pred_counts.numel()
            ).to(self.val_pred_counts)
            _log_attention_entropy(self, self.model.tabular_encoder, "val")
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train_loss")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val_loss")

    def configure_optimizers(self):
        return _configure_adamw(self)

    def on_validation_epoch_start(self):
        self.val_pred_counts.zero_()

    def on_validation_epoch_end(self):
        counts = self.val_pred_counts
        total = counts.sum().clamp_min(1.0)
        probs = counts / total
        pred_entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum()
        self.log("val_pred_entropy", pred_entropy, on_epoch=True)
        for class_i, frac in enumerate(probs):
            self.log(f"val_pred_frac_class_{class_i}", frac, on_epoch=True)


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
        qk_norm: bool = True,
        pre_norm: bool = True,
        warmup_frac: float = 0.05,
        weight_decay: float = 3e-4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["config", "col_embedder"])
        self.lr = lr
        self.warmup_frac = warmup_frac
        self.weight_decay = weight_decay
        self.model = MaskedTabularEncoder(
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
        self.numeric_loss_fn = nn.MSELoss(reduction="none")
        self.categorical_loss_fn = nn.CrossEntropyLoss(reduction="none")

    def aggregate_loss(self, actual, predicted, num_mask, cat_mask):
        losses = []
        valid_num = num_mask & torch.isfinite(actual.numeric)
        if valid_num.any():
            num_loss = self.numeric_loss_fn(predicted.numeric, actual.numeric)
            losses.append(num_loss[valid_num].mean())
        if (
            actual.categorical is not None
            and predicted.categorical is not None
            and cat_mask is not None
            and cat_mask.any()
        ):
            b, n_cat, n_tok = predicted.categorical.size()
            cat_loss = self.categorical_loss_fn(
                predicted.categorical.reshape(b * n_cat, n_tok),
                actual.categorical.reshape(-1),
            ).view(b, n_cat)
            losses.append(cat_loss[cat_mask].mean())
        if not losses:
            return predicted.numeric.sum() * 0.0
        return torch.stack(losses).sum()

    def _step(self, batch, name):
        num_masked, num_mask = mask_tensor(
            batch.numeric, self.model, self.MASK_KEEP, return_mask=True
        )
        if batch.categorical is not None:
            cat_masked, cat_mask = mask_tensor(
                batch.categorical, self.model, self.MASK_KEEP, return_mask=True
            )
        else:
            cat_masked, cat_mask = None, None
        predicted = self.model(num_masked, cat_masked)
        loss = self.aggregate_loss(batch, predicted, num_mask, cat_mask)
        self.log(name, loss, prog_bar=(name == "val_loss"))
        if name == "val_loss":
            _log_attention_entropy(self, self.model.tabular_encoder, "val")
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train_loss")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val_loss")

    def configure_optimizers(self):
        return _configure_adamw(self)


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
    patience: int | None = 16,
    ckpt_dir: str | None = None,
    grad_clip: float = 1.0,
    monitor: str = "val_loss",
    mode: str = "min",
):
    """Trainer with early-stopping (if patience set) and best-val checkpointing.

    Returns (trainer, ckpt_callback). After `fit`, call `load_best(module, ckpt)`
    to restore the best-validation weights before evaluating.
    """
    callbacks = []
    if patience is not None:
        callbacks.append(EarlyStopping(monitor=monitor, patience=patience, mode=mode))
    ckpt = None
    if ckpt_dir is not None:
        ckpt = ModelCheckpoint(
            monitor=monitor,
            mode=mode,
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
