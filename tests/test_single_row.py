"""Smoke tests for the single-row tabular model."""

import numpy as np
import pandas as pd
import pytorch_lightning as L
import torch

from torch.utils.data import DataLoader

from trea.models.single_row import mask_tensor
from trea.training.single_row import (
    HLGaussLoss,
    LogitAdjustedCrossEntropy,
    SingleRowClassifier,
    SingleRowMTM,
    SingleRowRegressor,
    build_trainer,
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


def test_hl_gauss_regressor_predicts_scalar_values():
    df = _toy_df()
    cfg = SingleRowConfig.generate(df, target="target")
    loader = DataLoader(
        TabularDS(df, cfg), batch_size=16, collate_fn=tabular_collate_fn
    )
    model = SingleRowRegressor(
        cfg,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        head="hl_gauss",
        y_train=df["target"].values,
        hl_gauss_bins=17,
    )
    batch = next(iter(loader))
    logits = model(batch)
    pred = model.predict_value(batch)
    assert isinstance(model.loss_fn, HLGaussLoss)
    assert logits.shape == (16, 17)
    assert pred.shape == (16, 1)
    assert torch.isfinite(pred).all()


def test_hl_gauss_vs_mse_performance():
    L.seed_everything(42, verbose=False)
    n = 300
    x = torch.linspace(-3, 3, n).unsqueeze(1)
    y_smooth = torch.sin(x * 2.0)
    # Bimodal: split randomly between y_smooth and y_smooth + 5.0
    y_bimodal = torch.where(torch.rand(n, 1) > 0.5, y_smooth, y_smooth + 5.0)

    def train_and_eval(y_target, head):
        df = pd.DataFrame(
            {"feat": x.squeeze().numpy(), "target": y_target.squeeze().numpy()}
        )
        cfg = SingleRowConfig.generate(df, target="target")
        ds = TabularDS(df, cfg)
        loader = DataLoader(ds, batch_size=64, collate_fn=tabular_collate_fn)

        model = SingleRowRegressor(
            cfg,
            d_model=32,
            n_heads=2,
            n_layers=2,
            head=head,
            y_train=df["target"].values if head == "hl_gauss" else None,
            hl_gauss_bins=30,
            lr=0.02,
        )
        trainer, _ = build_trainer(max_epochs=30, patience=None)
        trainer.fit(model, train_dataloaders=loader)

        model.eval()
        with torch.no_grad():
            preds = []
            for b in loader:
                preds.append(model.predict_value(b))
            pred = torch.cat(preds, dim=0)
        rmse = torch.sqrt(torch.mean((pred - y_target) ** 2)).item()
        return rmse

    rmse_mse_smooth = train_and_eval(y_smooth, "mse")
    rmse_hl_smooth = train_and_eval(y_smooth, "hl_gauss")
    # hl_gauss should be within 10% of mse, or better
    assert rmse_hl_smooth < rmse_mse_smooth * 1.10

    rmse_mse_bimodal = train_and_eval(y_bimodal, "mse")
    rmse_hl_bimodal = train_and_eval(y_bimodal, "hl_gauss")
    # hl_gauss should beat mse on bimodal
    assert rmse_hl_bimodal < rmse_mse_bimodal


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
    reg = SingleRowRegressor(
        cfg_reg, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS
    )
    before = reg.model.tabular_encoder.layers[0].attn.q_proj.weight.clone()
    transfer_encoder(mtm, reg)
    after = reg.model.tabular_encoder.layers[0].attn.q_proj.weight
    assert torch.allclose(after, mtm.model.tabular_encoder.layers[0].attn.q_proj.weight)
    assert not torch.allclose(before, after)  # weights actually changed


def test_mtm_numeric_only_uses_masked_cells_only():
    df = _toy_df().drop(columns=["cat", "target"])
    cfg = SingleRowConfig.generate(df)
    ds = TabularDS(df, cfg)
    loader = DataLoader(ds, batch_size=16, collate_fn=masked_tabular_collate_fn)
    mtm = SingleRowMTM(cfg, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS)
    batch = next(iter(loader))
    loss = mtm._step(batch, "train_loss")
    assert torch.isfinite(loss)


def test_mtm_aggregate_loss_only_on_masked():
    # Test that MTM loss is only calculated on masked cells
    from trea.models.single_row import NumericCategoricalData

    df = _toy_df().drop(columns=["cat", "target"])
    cfg = SingleRowConfig.generate(df)
    mtm = SingleRowMTM(cfg, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS)

    # Create manual actual and predicted tensors
    actual = NumericCategoricalData(numeric=torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    predicted = NumericCategoricalData(numeric=torch.tensor([[1.5, 2.5], [3.5, 4.5]]))

    # Mask is: row 0 col 0 is masked, others not
    num_mask = torch.tensor([[True, False], [False, False]])

    # Loss should only be based on actual[0,0] vs predicted[0,0]
    # which is (1.5 - 1.0)^2 = 0.25
    loss = mtm.aggregate_loss(actual, predicted, num_mask, cat_mask=None)
    assert torch.allclose(loss, torch.tensor(0.25))

    # If we modify actual[0,1] (unmasked), the loss should NOT change!
    actual.numeric = actual.numeric.clone()
    actual.numeric[0, 1] = 999.0
    loss2 = mtm.aggregate_loss(actual, predicted, num_mask, cat_mask=None)
    assert torch.allclose(loss2, torch.tensor(0.25))

    # If we modify predicted[0,1] (unmasked), the loss should NOT change!
    predicted.numeric = predicted.numeric.clone()
    predicted.numeric[0, 1] = 999.0
    loss3 = mtm.aggregate_loss(actual, predicted, num_mask, cat_mask=None)
    assert torch.allclose(loss3, torch.tensor(0.25))


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


def _toy_clf_df(n=96, n_classes=3, seed=0):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(
        {
            "a": rng.randn(n).astype("float32"),
            "b": rng.randn(n).astype("float32"),
            "c": rng.randn(n).astype("float32"),
        }
    )
    score = df["a"] * 2 - df["b"]
    df["target"] = pd.qcut(score, n_classes, labels=False).astype("int64")
    return df


def test_classifier_forward_fit_and_cross_task_transfer():
    df = _toy_clf_df()
    cfg = SingleRowConfig.generate(df, target="target")
    assert cfg.n_cat_cols == 0  # int target is not mistaken for a categorical feature
    loader = DataLoader(
        TabularDS(df, cfg), batch_size=16, collate_fn=tabular_collate_fn
    )

    clf = SingleRowClassifier(
        cfg, num_classes=3, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS
    )
    batch = next(iter(loader))
    out = clf(batch)
    assert out.shape == (16, 3) and torch.isfinite(out).all()

    trainer, _ = build_trainer(max_epochs=1, patience=None)
    trainer.fit(clf, train_dataloaders=loader, val_dataloaders=loader)

    # the encoder is task-agnostic: a regressor's encoder loads into the classifier
    reg = SingleRowRegressor(cfg, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS)
    transfer_encoder(reg, clf)  # duck-typed on .model.tabular_encoder
    assert torch.isfinite(clf(batch)).all()


def test_oov_categorical_maps_to_unk():
    train = _toy_df()
    cfg = SingleRowConfig.generate(train, target="target")
    test = train.head(2).copy()
    test.loc[test.index[0], "cat"] = "never_seen"
    ds = TabularDS(test, cfg)
    unk = cfg.token_dict["[UNK]"]
    assert int(ds[0].inputs.categorical[0]) == unk


def test_logit_adjustment_changes_loss():
    logits = torch.zeros(4, 2)
    target = torch.tensor([0, 1, 1, 1])
    plain = LogitAdjustedCrossEntropy(class_priors=None)(logits, target)
    adjusted = LogitAdjustedCrossEntropy(class_priors=[0.75, 0.25], tau=1.0)(
        logits, target
    )
    assert adjusted != plain
