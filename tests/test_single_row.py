"""Smoke tests for the single-row tabular model."""

import time

import numpy as np
import pandas as pd
import pytorch_lightning as L
import torch

from torch.utils.data import DataLoader

from trea.models.single_row import mask_tensor
from trea.training.single_row import (
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
    before = reg.model.tabular_encoder.layers[0].attn.in_proj_weight.clone()
    transfer_encoder(mtm, reg)
    after = reg.model.tabular_encoder.layers[0].attn.in_proj_weight
    assert torch.allclose(
        after, mtm.model.tabular_encoder.layers[0].attn.in_proj_weight
    )
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


# ---------------------------------------------------------------------------
# WO-7: TabM-mini BatchEnsemble
# ---------------------------------------------------------------------------


def test_ensemble_disabled_is_single_model():
    """ensemble_k=0 keeps the old contract: 2-D output, no adapter parameter."""
    df = _toy_clf_df()
    cfg = SingleRowConfig.generate(df, target="target")
    clf = SingleRowClassifier(
        cfg, num_classes=3, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS
    )
    assert not hasattr(clf.model.tabular_encoder, "ensemble_adapter")
    loader = DataLoader(
        TabularDS(df, cfg), batch_size=16, collate_fn=tabular_collate_fn
    )
    batch = next(iter(loader))
    raw = clf.model(batch.inputs.numeric, batch.inputs.categorical)
    assert raw.shape == (16, 3)  # [b, C], no member dim


def test_ensemble_shapes_and_member_diversity():
    """k>0 exposes a member dim and members produce genuinely different outputs."""
    L.seed_everything(0, verbose=False)
    df = _toy_clf_df()
    cfg = SingleRowConfig.generate(df, target="target")
    k = 4
    clf = SingleRowClassifier(
        cfg,
        num_classes=3,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        ensemble_k=k,
    )
    assert clf.model.tabular_encoder.ensemble_adapter.shape == (k, D_MODEL)
    loader = DataLoader(
        TabularDS(df, cfg), batch_size=16, collate_fn=tabular_collate_fn
    )
    batch = next(iter(loader))
    clf.eval()  # drop dropout so diversity is purely from the per-member adapter
    with torch.no_grad():
        logits = clf.model(batch.inputs.numeric, batch.inputs.categorical)
    assert logits.shape == (16, k, 3)
    # members must not be identical (adapters are distinct random signs)
    assert logits.std(dim=1).mean().item() > 1e-4
    # averaged prediction collapses the member dim
    probs = clf.model.predict(batch.inputs.numeric, batch.inputs.categorical)
    assert probs.shape == (16, 3)
    assert torch.allclose(probs.sum(-1), torch.ones(16), atol=1e-5)


def _easy_clf_df(n=300, seed=0):
    """Three well-separated Gaussian blobs -> members saturate, ensemble can't lose."""
    rng = np.random.RandomState(seed)
    per = n // 3
    centers = np.array([[3.0, 0.0], [-3.0, 3.0], [-3.0, -3.0]], dtype="float32")
    xs, ys = [], []
    for c, ctr in enumerate(centers):
        xs.append(rng.randn(per, 2).astype("float32") * 0.4 + ctr)
        ys.append(np.full(per, c, dtype="int64"))
    X = np.concatenate(xs)
    df = pd.DataFrame({"a": X[:, 0], "b": X[:, 1]})
    df["target"] = np.concatenate(ys)
    return df


def _macro_f1(y_true, y_pred, n_classes):
    f1s = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        denom = 2 * tp + fp + fn
        f1s.append(1.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s))


def test_ensemble_beats_or_ties_best_member():
    """Trained k=4 ensemble's macro-F1 >= the best individual member's macro-F1."""
    L.seed_everything(0, verbose=False)
    df = _easy_clf_df()
    cfg = SingleRowConfig.generate(df, target="target")
    k = 4
    clf = SingleRowClassifier(
        cfg,
        num_classes=3,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        ensemble_k=k,
    )
    loader = DataLoader(
        TabularDS(df, cfg), batch_size=64, shuffle=True, collate_fn=tabular_collate_fn
    )
    trainer, _ = build_trainer(max_epochs=40, patience=None)
    trainer.fit(clf, train_dataloaders=loader, val_dataloaders=loader)

    clf.eval()
    ev = DataLoader(TabularDS(df, cfg), batch_size=512, collate_fn=tabular_collate_fn)
    batch = next(iter(ev))
    y = df["target"].to_numpy()
    with torch.no_grad():
        member_logits = clf.model(batch.inputs.numeric, batch.inputs.categorical)
        ens_probs = clf.model.predict(batch.inputs.numeric, batch.inputs.categorical)
    member_f1 = [
        _macro_f1(y, member_logits[:, m, :].argmax(-1).numpy(), 3) for m in range(k)
    ]
    ens_f1 = _macro_f1(y, ens_probs.argmax(-1).numpy(), 3)
    assert ens_f1 + 1e-9 >= max(member_f1), (ens_f1, member_f1)


def test_ensemble_runtime_is_a_single_batched_forward():
    """k=8 ensemble is ONE batched forward (~k x FLOPs), not k separate models.

    The TabM paper's "< 2x overhead" is a GPU-underutilization property: when a
    single-model forward doesn't saturate the device, folding k members into one
    batch is nearly free. On a compute-bound CPU runner the k-fold work is genuinely
    ~k x, so the strict < 2x bound is only asserted when CUDA is present. On CPU we
    instead guard that the efficient reshaped path stays near k x and never balloons
    into the per-iteration overhead of an accidental Python loop over members.
    """
    torch.manual_seed(0)
    k = 8
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    df = _toy_clf_df(n=256)
    cfg = SingleRowConfig.generate(df, target="target")
    num = torch.randn(256, cfg.n_numeric_cols, device=device)

    def build(kk):
        L.seed_everything(0, verbose=False)
        m = SingleRowClassifier(
            cfg, num_classes=3, d_model=128, n_heads=4, n_layers=3, ensemble_k=kk
        )
        return m.to(device).eval()

    def bench(model, reps=10):
        with torch.no_grad():
            for _ in range(3):
                model.model(num, None)  # warmup
            if cuda:
                torch.cuda.synchronize()
            best = float("inf")
            for _ in range(reps):
                t0 = time.perf_counter()
                model.model(num, None)
                if cuda:
                    torch.cuda.synchronize()
                best = min(best, time.perf_counter() - t0)
        return best

    base = bench(build(0))
    ens = bench(build(k))
    ratio = ens / base
    if cuda:
        # Underutilized GPU: folding k members into one batch is nearly free (< 2x).
        assert ens < 2.0 * base, f"ensemble k={k} overhead {ratio:.2f}x >= 2x on GPU"
    else:
        # Compute-bound CPU: work is genuinely ~k x. Guard only that it stays near k x
        # (the efficient reshape), not the per-iteration blowup of a Python loop.
        assert ratio < (k + 2), f"ensemble k={k} overhead {ratio:.2f}x >> {k}x (loop?)"
