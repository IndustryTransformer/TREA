"""Tests for the shipped cross-schema transfer API (trea.transfer).

They need no network: the frozen text backbone is replaced by a deterministic
hash embedding, so the semantic path runs without MiniLM. The end-to-end
release gate lives in scripts/schema_scaling.py.
"""

import hashlib
import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from trea.models.column_embeddings import SemanticColumnEmbedder
from trea.models.single_row import TabularEncoder
from trea.transfer import SchemaFreeEncoder, SchemaTransferRegressor, Table
from trea.utils.single_row_data import SingleRowConfig


TEXT_DIM = 24


def _fake_encode(self, names):
    for n in names:
        if n not in self._feat_cache:
            seed = int(hashlib.sha256(n.encode()).hexdigest()[:8], 16)
            g = torch.Generator().manual_seed(seed)
            v = torch.randn(TEXT_DIM, generator=g)
            self._feat_cache[n] = v / v.norm()
    return torch.stack([self._feat_cache[n] for n in names])


@pytest.fixture(autouse=True)
def no_text_model(monkeypatch):
    monkeypatch.setattr(SemanticColumnEmbedder, "_encode", _fake_encode)


def _table(n=120, cols=("a", "b", "c"), seed=0, name=None):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({c: rng.randn(n) for c in cols})
    df["y"] = 3.0 * df[cols[0]] - df[cols[1]] + 0.1 * rng.randn(n) + 50.0
    descr = {c: f"sensor measuring {c}" for c in cols}
    return Table(df, "y", descr, name=name)


def _small(**kw):
    return SchemaTransferRegressor(
        d_model=16, n_heads=2, n_layers=2, batch_size=32, device="cpu", **kw
    )


def test_encoder_handles_any_width_and_nan():
    emb = SemanticColumnEmbedder(["x"], d_model=16)
    enc = SchemaFreeEncoder(emb, d_model=16, n_heads=2, n_layers=2).eval()
    for width in (1, 3, 7):
        x = torch.randn(4, width)
        x[0, 0] = float("nan")
        x[1, -1] = float("inf")
        out = enc(x, [f"col {i}" for i in range(width)])
        assert out.shape == (4, width, 16)
        assert torch.isfinite(out).all()


def test_encoder_rejects_width_mismatch():
    emb = SemanticColumnEmbedder(["x"], d_model=16)
    enc = SchemaFreeEncoder(emb, d_model=16, n_heads=2, n_layers=1)
    with pytest.raises(ValueError):
        enc(torch.randn(2, 3), ["a", "b"])


def test_parity_with_tabular_encoder_in_pinned_config():
    """SchemaFreeEncoder == TabularEncoder(additive, post-norm, no QK-norm)."""
    torch.manual_seed(0)
    df = pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0], "t": [0.0, 1.0]})
    cfg = SingleRowConfig.generate(df, target="t")
    texts = ["first column", "second column"]
    emb = SemanticColumnEmbedder(texts, d_model=16)
    ref = TabularEncoder(
        cfg,
        16,
        2,
        3,
        0.0,
        col_embedder=emb,
        qk_norm=False,
        pre_norm=False,
        fusion="additive",
    ).eval()
    ref.col_texts = ref.numeric_texts = texts
    new = SchemaFreeEncoder(emb, 16, 2, 3, 0.0).eval()
    missing, unexpected = new.load_state_dict(ref.state_dict(), strict=False)
    assert not missing, missing
    x = torch.randn(5, 2)
    x[2, 1] = float("nan")
    assert torch.allclose(new(x, texts), ref(x, None), atol=1e-6)


def test_fit_predict_learns_and_returns_original_units():
    tab = _table(n=300)
    model = _small().fit(tab, epochs=60, patience=60)
    pred = model.predict(tab.df)
    assert pred.shape == (300,)
    rmse = float(np.sqrt(np.mean((pred - tab.df["y"].to_numpy()) ** 2)))
    assert abs(pred.mean() - 50.0) < 2.0  # un-standardized
    assert rmse < 0.6 * tab.df["y"].std()


def test_predict_matches_columns_by_name_and_tolerates_missing():
    tab = _table()
    model = _small().fit(tab, epochs=3)
    base = model.predict(tab.df)
    shuffled = tab.df[["c", "y", "a", "b"]].assign(extra=1.0)
    assert np.allclose(model.predict(shuffled), base, atol=1e-5)
    dropped = model.predict(tab.df.drop(columns=["c"]))
    assert np.isfinite(dropped).all()


def test_pretrain_then_fit_unseen_schema():
    sources = [
        _table(cols=("a", "b"), seed=1, name="src.one"),
        _table(cols=("p", "q", "r"), seed=2, name="src_two"),
    ]
    model = _small().pretrain(sources, epochs=2)
    held = _table(cols=("u", "v", "w", "z"), seed=3, name="held")
    model.fit(held, epochs=3)
    assert set(model.scalers) == {"src.one", "src_two", "held"}
    assert np.isfinite(model.predict(held.df)).all()
    assert np.isfinite(model.predict(sources[0].df, table="src.one")).all()


def test_pretrain_rejects_duplicate_names():
    with pytest.raises(ValueError):
        _small().pretrain([_table(name="x"), _table(name="x")], epochs=1)


def test_index_control_maps_unseen_columns_to_unk():
    model = _small(embedder="index").pretrain(
        [_table(cols=("a", "b"), name="s")], epochs=1
    )
    emb = model.net.encoder.col_embedder
    unseen = emb(["never seen one", "never seen two"])
    assert torch.allclose(unseen[0], unseen[1])


@pytest.mark.parametrize("embedder", ["semantic", "index"])
def test_save_load_round_trip(tmp_path, embedder):
    tab = _table()
    model = _small(embedder=embedder).fit(tab, epochs=3)
    path = tmp_path / "m.pt"
    model.save(path)
    loaded = SchemaTransferRegressor.load(path, device="cpu")
    assert np.allclose(loaded.predict(tab.df), model.predict(tab.df), atol=1e-6)


def test_missing_description_warns():
    df = pd.DataFrame({"a": np.arange(10.0), "y": np.arange(10.0)})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Table(df, "y").texts()
    assert any("description" in str(x.message) for x in w)
