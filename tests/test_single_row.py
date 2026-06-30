"""Smoke tests for the single-row tabular model."""

import numpy as np
import pandas as pd
import pytorch_lightning as L
import torch
from torch.utils.data import DataLoader

from trea.models.single_row import mask_tensor
from trea.training.single_row import (
    SingleRowMTM,
    SingleRowRegressor,
    build_trainer,
    load_best,
    transfer_encoder,
)
from trea.utils.single_row_data import (
    SingleRowConfig,
    TabularDS,
    masked_tabular_collate_fn,
    tabular_collate_fn,
)

D_MODEL, N_HEADS, N_LAYERS = 16, 2, 2


def _toy_df(n=64, seed=0):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(
        {
            "a": rng.randn(n).astype("float32"),
            "b": rng.randn(n).astype("float32"),
            "c": rng.randn(n).astype("float32"),
            "cat": rng.choice(["x", "y", "z"], size=n),
        }
    )
    df["target"] = (df["a"] * 2 - df["b"] + rng.randn(n) * 0.1).astype("float32")
    return df


def test_config_shapes():
    df = _toy_df()
    cfg = SingleRowConfig.generate(df, target="target")
    assert cfg.n_numeric_cols == 3  # a, b, c (target excluded)
    assert cfg.n_cat_cols == 1  # cat
    assert cfg.n_columns == 4
    # special tokens + 3 numeric names + 3 cat values + 1 cat name
    assert cfg.n_tokens == 5 + 3 + 3 + 1
    # deterministic token ordering
    assert cfg.tokens == SingleRowConfig.generate(df, target="target").tokens


def test_regressor_forward_and_fit():
    df = _toy_df()
    cfg = SingleRowConfig.generate(df, target="target")
    ds = TabularDS(df, cfg)
    loader = DataLoader(ds, batch_size=16, collate_fn=tabular_collate_fn)

    model = SingleRowRegressor(cfg, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS)
    batch = next(iter(loader))
    out = model(batch)
    assert out.shape == (16, 1)
    assert torch.isfinite(out).all()

    trainer, _ = build_trainer(max_epochs=1, patience=None)
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)


def test_mtm_forward_and_transfer():
    df = _toy_df()
    cfg_mtm = SingleRowConfig.generate(df.drop(columns=["target"]))
    ds = TabularDS(df.drop(columns=["target"]), cfg_mtm)
    loader = DataLoader(ds, batch_size=16, collate_fn=masked_tabular_collate_fn)

    mtm = SingleRowMTM(cfg_mtm, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS)
    batch = next(iter(loader))
    pred = mtm.model(batch.numeric, batch.categorical)
    assert pred.numeric.shape == (16, cfg_mtm.n_numeric_cols)
    assert pred.categorical.shape == (16, cfg_mtm.n_cat_cols, cfg_mtm.n_tokens)

    # encoder transfer into a regressor with the same d_model must succeed
    cfg_reg = SingleRowConfig.generate(df, target="target")
    reg = SingleRowRegressor(cfg_reg, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS)
    before = reg.model.tabular_encoder.layers[0].attn.in_proj_weight.clone()
    transfer_encoder(mtm, reg)
    after = reg.model.tabular_encoder.layers[0].attn.in_proj_weight
    assert torch.allclose(after, mtm.model.tabular_encoder.layers[0].attn.in_proj_weight)
    assert not torch.allclose(before, after)  # weights actually changed


def test_mask_tensor_fraction_and_dtype():
    L.seed_everything(0, verbose=False)
    df = _toy_df()
    cfg = SingleRowConfig.generate(df.drop(columns=["target"]))
    mtm = SingleRowMTM(cfg, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS)

    num = torch.randn(1000, cfg.n_numeric_cols)
    masked = mask_tensor(num, mtm.model, keep_prob=0.7)
    assert masked.dtype == torch.float32
    frac = torch.isinf(masked).float().mean().item()
    assert 0.2 < frac < 0.4  # ~30% masked

    cat = torch.zeros(1000, cfg.n_cat_cols, dtype=torch.long)
    masked_cat = mask_tensor(cat, mtm.model, keep_prob=0.7)
    assert masked_cat.dtype == torch.long
    mask_id = int(mtm.model.cat_mask_token)
    assert (masked_cat == mask_id).float().mean().item() > 0.15
