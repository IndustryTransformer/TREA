"""Regression tests for benchmark fairness and missing-safe numeric bases."""

import numpy as np
import pandas as pd
import pytest
import torch

from sklearn.model_selection import train_test_split

from scripts import tabular_vs_trees as bench
from trea.models.single_row import TabularEncoder
from trea.utils.single_row_data import SingleRowConfig


def test_validation_outliers_do_not_change_fit_scaling():
    df = pd.DataFrame({"x": np.arange(100.0), "__target__": np.arange(100.0) ** 2})
    tr, _ = train_test_split(df, test_size=0.3, random_state=3)
    fit, val = bench.fit_val_indices(tr["__target__"], "regression", 3)
    before = bench.prep(df, ["x"], "regression", 3, fit_only=True)
    df.loc[tr.index[val], ["x", "__target__"]] = 1e10
    after = bench.prep(df, ["x"], "regression", 3, fit_only=True)
    np.testing.assert_array_equal(before[0][fit], after[0][fit])
    np.testing.assert_array_equal(before[1][fit], after[1][fit])
    assert before[4] == after[4]


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_all_trees_receive_only_deep_fit_rows(monkeypatch, task):
    df = pd.DataFrame({"x": np.arange(100.0), "__target__": np.arange(100) % 2})
    monkeypatch.setattr(bench.pd, "read_parquet", lambda _: df)
    seen = {}

    def deep(x, y, *args, **kwargs):
        fit, _ = bench.fit_val_indices(y, task, 0)
        seen["x"], seen["y"] = x[fit], y[fit]
        return {"score": 0.5}

    def tree(kind, x, y, *args):
        np.testing.assert_array_equal(x, seen["x"])
        np.testing.assert_array_equal(y, seen["y"])
        seen[kind] = len(y)
        return 0.5

    monkeypatch.setattr(bench, "deep_score", deep)
    monkeypatch.setattr(bench, "tree_score", tree)
    cfg = dict(
        seeds=1,
        epochs=1,
        patience=1,
        head="mse",
        bin_spacing="uniform",
        symlog=False,
        value_encoder="identity",
        fusion="additive",
        label_smoothing=0,
        logit_adjustment_tau=1,
        warmup_frac=0.05,
        ensemble_k=0,
        tree_jobs=1,
        classification_margin=0.01,
        regression_margin=0.02,
        lr=8e-4,
        dropout=0.2,
    )
    _, result = bench._score_dataset(
        (0, dict(name="toy", task=task, feature_cols=["x"], n_rows=100), cfg)
    )
    assert "__fail__" not in result, result
    assert all(seen[k] == 56 for k in bench.TREE_KINDS)
    assert result["seed_results"][0]["n_val"] == 14


def test_practical_margin_does_not_reward_unstable_seeds():
    row = dict(
        task="classification", deep=0.8, deep_std=0.5, xgb=0.9, rf=0.88, cat=0.89
    )
    assert not bench.practical_comparison(row)["within_margin_or_better"]
    row["deep"] = 0.895
    assert bench.practical_comparison(row)["within_margin_or_better"]
    row.update(task="regression", deep=10.1, xgb=10, rf=11, cat=12)
    assert bench.practical_comparison(row)["within_margin_or_better"]
    row["deep"] = 10.3
    assert not bench.practical_comparison(row)["within_margin_or_better"]


@pytest.mark.parametrize("basis", ["ple", "periodic", "dual"])
def test_numeric_bases_handle_inf_constant_and_missing_columns(basis):
    df = pd.DataFrame(
        {
            "constant": [1.0, 1.0, np.inf, -np.inf],
            "absent": [np.nan] * 4,
            "varying": [0.0, 1.0, 2.0, 3.0],
        }
    )
    cfg = SingleRowConfig.generate(df)
    assert torch.isfinite(cfg.numeric_quantiles).all()
    torch.testing.assert_close(cfg.numeric_quantiles[0], torch.ones(65))
    model = TabularEncoder(
        cfg,
        d_model=8,
        n_heads=2,
        n_layers=1,
        value_encoder=basis,
        fusion="hadamard",
        dropout=0,
    )
    x = torch.tensor(df.to_numpy(), dtype=torch.float32)
    out = model(x, None)
    assert torch.isfinite(out).all()
    missing = x.clone()
    missing[~torch.isfinite(missing)] = float("nan")
    torch.testing.assert_close(out, model(missing, None))
    out.square().mean().backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_invalid_numeric_options_fail_loudly():
    cfg = SingleRowConfig.generate(pd.DataFrame({"x": [1.0, 2.0]}))
    with pytest.raises(ValueError, match="value_encoder"):
        TabularEncoder(cfg, value_encoder="typo")
    with pytest.raises(ValueError, match="fusion"):
        TabularEncoder(cfg, fusion="typo")


@pytest.mark.parametrize("fusion", ["product", "product_sum"])
def test_scalar_product_tokens_and_missingness(fusion):
    cfg = SingleRowConfig.generate(pd.DataFrame({"a": [0.0, 1.0], "b": [2.0, 3.0]}))
    model = TabularEncoder(
        cfg, d_model=8, n_heads=2, n_layers=1, fusion=fusion, dropout=0
    ).eval()
    captured = []
    model.layers[0].register_forward_pre_hook(
        lambda module, args: captured.append(args[1].detach().clone())
    )
    x = torch.tensor([[0.0, 2.0], [float("inf"), float("nan")]])
    y = model(x, None)
    e = model.embeddings(model.numeric_indices)
    expected = x[0, :, None] * e
    if fusion == "product_sum":
        expected = expected + model.product_bias(model.numeric_indices)
        assert not torch.allclose(expected[0], torch.zeros(8))
    else:
        torch.testing.assert_close(expected[0], torch.zeros(8))
    torch.testing.assert_close(captured[0][0], expected)
    torch.testing.assert_close(captured[0][1], e + model.product_missing)
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_scalar_product_rejects_ignored_basis():
    cfg = SingleRowConfig.generate(pd.DataFrame({"x": [1.0, 2.0]}))
    with pytest.raises(ValueError, match="requires"):
        TabularEncoder(cfg, fusion="product", value_encoder="dual")


def test_first_attention_norm_bypass_preserves_other_paths():
    from scripts.nox_encoding_ablation import bypass_norm

    cfg = SingleRowConfig.generate(pd.DataFrame({"a": [0.0, 1.0], "b": [2.0, 3.0]}))
    model = TabularEncoder(
        cfg, d_model=8, n_heads=2, n_layers=2, dropout=0, first_prenormalization=True
    ).eval()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    first = model.layers[0]
    first.norm1.register_forward_hook(bypass_norm)
    counts = {"first_ffn": 0, "q_norm": 0, "k_norm": 0, "later_norm": 0}

    def count(name):
        def hook(module, inputs, output):
            counts[name] += 1

        return hook

    first.norm2.register_forward_hook(count("first_ffn"))
    first.q_norm.register_forward_hook(count("q_norm"))
    first.k_norm.register_forward_hook(count("k_norm"))
    model.layers[1].norm1.register_forward_hook(count("later_norm"))
    captured = []
    first.norm1.register_forward_hook(
        lambda module, inputs, output: captured.append((inputs[0], output))
    )
    out = model(torch.tensor([[0.0, 2.0], [1.0, 3.0]]), None)
    assert len(captured) == 3
    assert all(a is b for a, b in captured)
    assert counts == {"first_ffn": 1, "q_norm": 1, "k_norm": 1, "later_norm": 3}
    out.square().mean().backward()
    assert first.norm1.weight.grad is None
    assert first.norm2.weight.grad is not None
    assert model.layers[1].norm1.weight.grad is not None
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    for key, value in state.items():
        torch.testing.assert_close(value, model.state_dict()[key])


def test_production_default_matches_experimental_norm_bypass():
    from scripts.nox_encoding_ablation import bypass_norm
    from trea.training.single_row import (
        SingleRowClassifier,
        SingleRowMTM,
        SingleRowRegressor,
    )

    df = pd.DataFrame({"a": [0.0, 1.0, 2.0], "target": [0.0, 1.0, 0.0]})
    cfg = SingleRowConfig.generate(df, target="target")
    for factory in (
        SingleRowRegressor,
        SingleRowMTM,
        lambda *a, **k: SingleRowClassifier(*a, num_classes=2, **k),
    ):
        model = factory(cfg, d_model=8, n_heads=2, n_layers=2)
        enc = model.model.tabular_encoder
        assert enc.fusion == "product_sum"
        assert not enc.layers[0].attention_prenorm
        assert enc.layers[0].qk_norm and enc.layers[0].pre_norm
        assert enc.layers[1].attention_prenorm
        assert model.hparams.first_prenormalization is False

    torch.manual_seed(42)
    default = TabularEncoder(cfg, d_model=8, n_heads=2, n_layers=2, dropout=0).eval()
    torch.manual_seed(42)
    reference = TabularEncoder(
        cfg,
        d_model=8,
        n_heads=2,
        n_layers=2,
        dropout=0,
        fusion="product_sum",
        first_prenormalization=True,
    ).eval()
    reference.layers[0].norm1.register_forward_hook(bypass_norm)
    x = torch.tensor([[0.0], [2.0], [float("nan")]])
    torch.testing.assert_close(default(x, None), reference(x, None), rtol=0, atol=0)


def test_compact_defaults_match_explicit_recipe_and_allow_legacy_overrides():
    from trea.training.single_row import (
        SingleRowClassifier,
        SingleRowMTM,
        SingleRowRegressor,
    )

    cfg = SingleRowConfig.generate(pd.DataFrame({"a": [0.0, 1.0, 2.0]}))
    for factory, extra, lr in [
        (SingleRowRegressor, {}, 3e-4),
        (SingleRowClassifier, {"num_classes": 2}, 3e-4),
        (SingleRowMTM, {}, 8e-4),
    ]:
        torch.manual_seed(17)
        default = factory(cfg, **extra)
        torch.manual_seed(17)
        explicit = factory(cfg, n_layers=2, dropout=0.0, lr=lr, **extra)
        assert default.hparams.n_layers == 2
        assert default.hparams.dropout == 0.0
        assert default.hparams.lr == lr
        assert len(default.model.tabular_encoder.layers) == 2
        for name, value in default.state_dict().items():
            torch.testing.assert_close(
                value, explicit.state_dict()[name], rtol=0, atol=0
            )
        legacy = factory(cfg, n_layers=4, dropout=0.2, lr=8e-4, **extra)
        assert len(legacy.model.tabular_encoder.layers) == 4
        assert legacy.hparams.dropout == 0.2
