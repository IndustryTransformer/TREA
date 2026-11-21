"""User-facing API for cross-schema column-name transfer.

    model = SchemaTransferRegressor()
    model.pretrain([Table(df_a, "y", descr_a), Table(df_b, "price", descr_b)])
    model.fit(Table(df_new, "target", descr_new))   # columns never seen before
    pred = model.predict(df_new_test)

What the evidence supports (docs/LESSONS.md §3b): pretraining on a few (K≈2-4)
diverse schemas gives a table with *unseen* columns a better few-shot start than
training from scratch, through the column descriptions. It does not beat tuned trees
on most tables, and it adds nothing over index identities when the new table has the
same columns under new names (§3a). Scope: numeric columns, regression.

Feed human-readable descriptions, not raw codes: terse codes embed poorly.
"""

import copy
import warnings

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from trea.models.column_embeddings import (
    DEFAULT_TEXT_MODEL,
    IndexColumnEmbedder,
    SemanticColumnEmbedder,
)
from trea.transfer.model import SchemaTransferNet


CLIP = 8.0  # standardized values are clipped to +-CLIP (NaN is preserved)


@dataclass
class Table:
    """A DataFrame, its target column, and a text description per feature column."""

    df: pd.DataFrame
    target: str
    descriptions: dict[str, str] | None = None
    name: str | None = None

    def __post_init__(self):
        if self.target not in self.df.columns:
            raise ValueError(f"target {self.target!r} is not a column of df")

    @property
    def features(self) -> list[str]:
        return [c for c in self.df.columns if c != self.target]

    def texts(self) -> list[str]:
        descr = self.descriptions or {}
        missing = [c for c in self.features if c not in descr]
        if missing:
            warnings.warn(
                f"No description for {missing}; using the raw column name. "
                "Terse codes embed poorly; pass human-readable descriptions.",
                stacklevel=3,
            )
        return [descr.get(c, c) for c in self.features]


@dataclass
class _Scaler:
    """Per-table standardization, fitted on that table's own rows."""

    features: list[str]
    texts: list[str]
    mu: np.ndarray
    sd: np.ndarray
    y_mu: float
    y_sd: float

    @classmethod
    def from_table(cls, table: Table) -> "_Scaler":
        feats = table.features
        X = table.df[feats].astype(np.float64)
        mu = X.mean().to_numpy()
        sd = X.std().replace(0, 1.0).fillna(1.0).to_numpy()
        y = table.df[table.target].astype(np.float64)
        y_sd = float(y.std())
        if not np.isfinite(y_sd) or y_sd == 0:
            y_sd = 1.0
        return cls(feats, table.texts(), np.nan_to_num(mu), sd, float(y.mean()), y_sd)

    def transform_x(self, df: pd.DataFrame) -> np.ndarray:
        # Match columns by name; an absent column is all-missing (NaN).
        X = df.reindex(columns=self.features).astype(np.float64).to_numpy()
        X = (X - self.mu) / self.sd
        return np.clip(X, -CLIP, CLIP).astype(np.float32)

    def transform_y(self, y) -> np.ndarray:
        return ((np.asarray(y, dtype=np.float64) - self.y_mu) / self.y_sd).astype(
            np.float32
        )

    def inverse_y(self, y: np.ndarray) -> np.ndarray:
        return y * self.y_sd + self.y_mu


def _head_key(name: str) -> str:
    # nn.ModuleDict keys cannot contain "."; keep the mapping readable.
    return name.replace(".", "_")


class SchemaTransferRegressor:
    """Pretrain on several schemas, then few-shot fit a table with unseen columns.

    ``embedder="index"`` is the negative control used by the release gate (an unseen
    column maps to one shared ``[UNK]`` row); it is not a user feature.
    """

    def __init__(
        self,
        embedder: str = "semantic",
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        text_model: str = DEFAULT_TEXT_MODEL,
        batch_size: int = 256,
        lr: float = 5e-4,
        weight_decay: float = 1e-5,
        device: str | None = None,
        seed: int = 0,
    ):
        if embedder not in {"semantic", "index"}:
            raise ValueError("embedder must be 'semantic' or 'index'")
        self.config = dict(
            embedder=embedder,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            text_model=text_model,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            seed=seed,
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net: SchemaTransferNet | None = None
        self.scalers: dict[str, _Scaler] = {}
        self.vocab: list[str] | None = None  # index embedder only
        self.default_table: str | None = None

    # ------------------------------------------------------------------ building
    def _make_embedder(self, texts: list[str]) -> nn.Module:
        c = self.config
        if c["embedder"] == "index":
            self.vocab = list(dict.fromkeys(texts))
            return IndexColumnEmbedder(self.vocab, c["d_model"])
        return SemanticColumnEmbedder(texts, c["d_model"], model_name=c["text_model"])

    def _ensure_net(self, texts: list[str]) -> None:
        if self.net is not None:
            return
        c = self.config
        torch.manual_seed(c["seed"])
        self.net = SchemaTransferNet(
            self._make_embedder(texts),
            c["d_model"],
            c["n_heads"],
            c["n_layers"],
            c["dropout"],
        ).to(self.device)

    def _optimizer(self, lr: float | None):
        assert self.net is not None
        return torch.optim.Adam(
            self.net.parameters(),
            lr=lr or self.config["lr"],
            weight_decay=self.config["weight_decay"],
        )

    def _register(self, table: Table, default_name: str) -> tuple[str, _Scaler]:
        name = table.name or default_name
        scaler = _Scaler.from_table(table)
        self.scalers[name] = scaler
        assert self.net is not None
        self.net.add_head(_head_key(name))
        return name, scaler

    def _step(self, opt, name, texts, X, y, lossf) -> None:
        assert self.net is not None
        loss = lossf(self.net(_head_key(name), X, texts), y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        opt.step()

    # ------------------------------------------------------------------ training
    def pretrain(
        self, sources: list[Table], epochs: int = 40, lr: float | None = None
    ) -> "SchemaTransferRegressor":
        """Train the shared encoder on several labeled source tables (round-robin)."""
        if not sources:
            raise ValueError("pretrain needs at least one source table")
        names = [t.name or f"source_{i}" for i, t in enumerate(sources)]
        if len(set(names)) != len(names):
            raise ValueError(f"source table names must be unique: {names}")
        all_texts = [t for tab in sources for t in tab.texts()]
        self._ensure_net(all_texts)
        assert self.net is not None
        data = {}
        for tab, name in zip(sources, names, strict=True):
            _, sc = self._register(tab, name)
            X = torch.from_numpy(sc.transform_x(tab.df))
            y = torch.from_numpy(sc.transform_y(tab.df[tab.target])).unsqueeze(1)
            data[name] = (X, y, sc.texts)

        gen = torch.Generator().manual_seed(self.config["seed"])
        bs = self.config["batch_size"]
        opt = self._optimizer(lr)
        lossf = nn.MSELoss()
        self.net.train()
        for _ in range(epochs):
            orders = {
                n: torch.randperm(len(d[0]), generator=gen) for n, d in data.items()
            }
            pos = dict.fromkeys(data, 0)
            active = list(data)
            while active:
                for n in list(active):
                    X, y, texts = data[n]
                    i = pos[n]
                    if i >= len(X):
                        active.remove(n)
                        continue
                    idx = orders[n][i : i + bs]
                    pos[n] = i + bs
                    self._step(
                        opt,
                        n,
                        texts,
                        X[idx].to(self.device),
                        y[idx].to(self.device),
                        lossf,
                    )
        return self

    def fit(
        self,
        table: Table,
        epochs: int = 80,
        lr: float | None = None,
        val_fraction: float = 0.25,
        patience: int = 10,
        seed: int | None = None,
    ) -> "SchemaTransferRegressor":
        """Fit (or fine-tune, after ``pretrain``) on one table with a fresh head.

        Holds out ``val_fraction`` of the rows for early stopping and restores the
        best-validation weights. Without ``pretrain`` this is the from-scratch model.
        ``seed`` controls the validation split and batch order (default: the
        constructor seed).
        """
        seed = self.config["seed"] if seed is None else seed
        torch.manual_seed(seed)  # fresh head init + dropout
        self._ensure_net(table.texts())
        assert self.net is not None
        name, sc = self._register(table, "target")
        self.default_table = name
        X = sc.transform_x(table.df)
        y = sc.transform_y(table.df[table.target])

        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(X))
        n_val = int(round(len(X) * val_fraction)) if len(X) > 1 else 0
        val_i, fit_i = perm[:n_val], perm[n_val:]
        Xf = torch.from_numpy(X[fit_i])
        yf = torch.from_numpy(y[fit_i]).unsqueeze(1)
        bs = min(self.config["batch_size"], max(len(fit_i), 1))
        gen = torch.Generator().manual_seed(seed)
        opt = self._optimizer(lr)
        lossf = nn.MSELoss()

        best, best_state, bad = np.inf, None, 0
        for _ in range(epochs):
            self.net.train()
            order = torch.randperm(len(Xf), generator=gen)
            for i in range(0, len(Xf), bs):
                idx = order[i : i + bs]
                self._step(
                    opt,
                    name,
                    sc.texts,
                    Xf[idx].to(self.device),
                    yf[idx].to(self.device),
                    lossf,
                )
            if n_val == 0:
                continue
            val = float(
                np.sqrt(np.mean((self._predict_std(name, X[val_i]) - y[val_i]) ** 2))
            )
            if val < best - 1e-4:
                best, bad = val, 0
                best_state = copy.deepcopy(self.net.state_dict())
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    # ---------------------------------------------------------------- inference
    @torch.no_grad()
    def _predict_std(self, name: str, X: np.ndarray) -> np.ndarray:
        assert self.net is not None
        self.net.eval()
        texts = self.scalers[name].texts
        out = []
        for i in range(0, len(X), 4096):
            xb = torch.from_numpy(X[i : i + 4096]).to(self.device)
            out.append(self.net(_head_key(name), xb, texts).cpu().numpy())
        return np.concatenate(out).ravel() if out else np.zeros(0, np.float32)

    def predict(self, df: pd.DataFrame, table: str | None = None) -> np.ndarray:
        """Predict in the target's original units. Columns are matched by name."""
        name = table or self.default_table
        if name is None or name not in self.scalers:
            raise ValueError(f"unknown table {name!r}; known: {list(self.scalers)}")
        sc = self.scalers[name]
        return sc.inverse_y(self._predict_std(name, sc.transform_x(df)))

    # -------------------------------------------------------------- persistence
    def save(self, path) -> None:
        if self.net is None:
            raise ValueError("nothing to save: call pretrain or fit first")
        torch.save(
            {
                "config": self.config,
                "state_dict": self.net.state_dict(),
                "scalers": {n: vars(s) for n, s in self.scalers.items()},
                "vocab": self.vocab,
                "default_table": self.default_table,
            },
            path,
        )

    @classmethod
    def load(cls, path, device: str | None = None) -> "SchemaTransferRegressor":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(**blob["config"], device=device)
        model.scalers = {n: _Scaler(**s) for n, s in blob["scalers"].items()}
        texts = blob["vocab"] or [t for s in model.scalers.values() for t in s.texts]
        model._ensure_net(texts)
        assert model.net is not None
        for name in model.scalers:
            model.net.add_head(_head_key(name))
        model.net.load_state_dict(blob["state_dict"])
        model.default_table = blob["default_table"]
        return model
