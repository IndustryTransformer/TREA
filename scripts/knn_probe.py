"""kNN / retrieval probes for the single-row tabular encoder (WO-8).

Two cheap diagnostics of whether the encoder learns useful *metric* structure:

  1. Plain kNN on attention-pooled row embeddings ([N, d]) vs the SAME kNN on the raw
     standardized features. If embedding-space kNN LOSES to raw-space kNN, the encoder
     isn't buying you any geometry -- we say so LOUDLY.
  2. (--nca) A ModernNCA-style soft-kNN: a small projection phi trained on the FROZEN
     embeddings with the NCA loss (softmax over -||phi(x)-phi(x_i)||^2, weighted-vote
     labels; candidates subsampled per batch). Reference: arXiv:2407.03257.

Classification -> macro-F1 (higher better). Regression -> RMSE in original target
units (lower better); the harness standardizes the target by `tscale`, so RMSE is
rescaled back.

The encoder is either loaded from a trained checkpoint (--checkpoint), quick-trained
for a few epochs (default, self-contained), or left randomly initialized (--untrained).

Run:
    uv run python scripts/knn_probe.py --synthetic --task classification --epochs 3
    uv run python scripts/knn_probe.py --synthetic --task regression --epochs 3
    uv run python scripts/knn_probe.py --synthetic --nca --epochs 3
    uv run python scripts/knn_probe.py --dataset adult --task classification
    uv run python scripts/knn_probe.py --dataset abalone --checkpoint ckpts/best.ckpt
"""

import argparse
import json
import os
import warnings

from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as L
import torch

from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader


warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
torch.set_float32_matmul_precision("medium")

from trea.training.single_row import (  # noqa: E402
    SingleRowClassifier,
    SingleRowRegressor,
    build_trainer,
    load_best,
)
from trea.utils.single_row_data import (  # noqa: E402
    SingleRowConfig,
    TabularDS,
    tabular_collate_fn,
)


CORPUS = Path(__file__).resolve().parent.parent / "data" / "tabular_corpus"
D_MODEL, N_HEADS, N_LAYERS, LR, BATCH = 128, 4, 3, 8e-4, 256


# --------------------------------------------------------------------------- data


def load_dataset(name, task):
    """Load one corpus dataset by name from the manifest; return (df, feat, task)."""
    man = json.loads((CORPUS / "manifest.json").read_text())
    meta = next((m for m in man if m["name"] == name), None)
    if meta is None:
        raise SystemExit(
            f"dataset '{name}' not in manifest ({CORPUS / 'manifest.json'})"
        )
    meta.setdefault("task", task or "regression")
    df = pd.read_parquet(CORPUS / f"{name}.parquet")
    return df, meta["feature_cols"], meta["task"]


def synthetic(task, seed, n=600, n_feat=8):
    """A small in-memory toy dataset (with some NaNs) so the probe is runnable anywhere.

    Classification: 3 separable Gaussian blobs. Regression: a linear target + noise.
    ~5% of numeric cells are set NaN to exercise the encoder's native missingness path.
    """
    rng = np.random.default_rng(seed)
    feat = [f"f{i}" for i in range(n_feat)]
    if task == "classification":
        n_cls = 3
        centers = rng.normal(0, 4, size=(n_cls, n_feat))
        y = rng.integers(0, n_cls, size=n)
        X = centers[y] + rng.normal(0, 1.0, size=(n, n_feat))
        target = y
    else:
        X = rng.normal(0, 1, size=(n, n_feat))
        w = rng.normal(0, 1, size=n_feat)
        target = X @ w + rng.normal(0, 0.3, size=n)
    df = pd.DataFrame(X, columns=feat)
    nan_mask = rng.random(X.shape) < 0.05
    df[feat] = df[feat].mask(nan_mask)
    df["__target__"] = target
    return df, feat, task


def prep(df, feat, task, seed):
    """Split + standardize (clip +-8, NaN kept); label-encode a classification target.

    Mirrors scripts/tabular_vs_trees.py so the probe sees the harness's exact inputs.
    """
    strat = df["__target__"] if task == "classification" else None
    tr, te = train_test_split(df, test_size=0.3, random_state=seed, stratify=strat)
    mu, sd = tr[feat].mean(), tr[feat].std().replace(0, 1)

    def sx(d):
        return np.clip(((d[feat] - mu) / sd).to_numpy(np.float32), -8.0, 8.0)

    Xtr, Xte = sx(tr), sx(te)
    if task == "regression":
        tmu, tsd = tr["__target__"].mean(), tr["__target__"].std() or 1.0
        ytr = ((tr["__target__"] - tmu) / tsd).to_numpy(np.float32)
        yte = ((te["__target__"] - tmu) / tsd).to_numpy(np.float32)
        return Xtr, ytr, Xte, yte, float(tsd), None
    le = LabelEncoder().fit(tr["__target__"].astype(str))
    ytr = le.transform(tr["__target__"].astype(str)).astype(np.int64)
    yte = le.transform(te["__target__"].astype(str)).astype(np.int64)
    return Xtr, ytr, Xte, yte, 1.0, len(le.classes_)


def _frame(X, y, feat):
    df = pd.DataFrame(X, columns=feat)
    df["__target__"] = y
    return df


# ----------------------------------------------------------------------- encoder


def train_encoder(cfg, Xtr, ytr, feat, task, n_classes, seed, epochs):
    """Quick-train a supervised SingleRow model; return module with best-val weights."""
    L.seed_everything(seed, verbose=False)
    fit_i, val_i = train_test_split(
        np.arange(len(Xtr)),
        test_size=0.2,
        random_state=seed,
        stratify=ytr if task == "classification" else None,
    )

    def loader(idx, shuffle):
        ds = TabularDS(_frame(Xtr[idx], ytr[idx], feat), cfg)
        return DataLoader(
            ds, batch_size=BATCH, shuffle=shuffle, collate_fn=tabular_collate_fn
        )

    if task == "regression":
        model = SingleRowRegressor(
            cfg, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, lr=LR
        )
    else:
        model = SingleRowClassifier(
            cfg,
            num_classes=n_classes,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            lr=LR,
        )
    ckdir = Path(os.environ.get("TMPDIR", "/tmp")) / f"knn_probe_{seed}"
    trainer, ckpt = build_trainer(
        max_epochs=epochs, patience=max(2, epochs), ckpt_dir=str(ckdir)
    )
    trainer.fit(
        model,
        train_dataloaders=loader(fit_i, True),
        val_dataloaders=loader(val_i, False),
    )
    load_best(model, ckpt)
    return model


def untrained_encoder(cfg, task, n_classes, seed):
    """A randomly-initialized module (control: does init already carry geometry?)."""
    L.seed_everything(seed, verbose=False)
    if task == "regression":
        return SingleRowRegressor(
            cfg, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, lr=LR
        )
    return SingleRowClassifier(
        cfg,
        num_classes=n_classes,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        lr=LR,
    )


def load_checkpoint(path, cfg, task, n_classes):
    """Load a trained SingleRow{Classifier,Regressor}; config isn't pickled, pass it.

    Checkpoints saved before an architecture param existed (e.g. qk_norm/pre_norm's
    q_norm/k_norm layers) fail a strict state_dict load. Retry with strict=False and
    warn loudly about it -- silently leaving submodules randomly initialized would
    make the probe's verdict describe a different, weaker encoder than intended.
    """
    cls = SingleRowRegressor if task == "regression" else SingleRowClassifier
    kw = {} if task == "regression" else {"num_classes": n_classes}
    try:
        return cls.load_from_checkpoint(path, config=cfg, map_location="cpu", **kw)
    except RuntimeError as e:
        if "Missing key(s)" not in str(e) and "Unexpected key(s)" not in str(e):
            raise
        model = cls.load_from_checkpoint(
            path, config=cfg, map_location="cpu", strict=False, **kw
        )
        print(
            f"  !! WARNING: {path} does not match the current architecture "
            "(likely a pre-qk_norm/pre_norm checkpoint) -- missing weights were left "
            "at random init. Probe results may not reflect the original trained "
            "encoder.",
            flush=True,
        )
        return model


def extract_embeddings(model, X, y, feat, cfg, device):
    """Attention-pooled row embeddings: encoder -> pooling, batched no_grad -> [N,d]."""
    model = model.to(device).eval()
    loader = DataLoader(
        TabularDS(_frame(X, y, feat), cfg),
        batch_size=512,
        collate_fn=tabular_collate_fn,
    )
    out = []
    with torch.no_grad():
        for b in loader:
            num = b.inputs.numeric.to(device)
            cat = (
                b.inputs.categorical.to(device)
                if b.inputs.categorical is not None
                else None
            )
            tokens = model.model.tabular_encoder(num, cat)
            pooled = model.model.pooling(tokens)
            out.append(pooled.cpu().numpy())
    return np.concatenate(out).astype(np.float32)


# --------------------------------------------------------------------------- kNN


def knn_metric(Xtr, ytr, Xte, yte, task, tscale, k):
    """Plain sklearn kNN; macro-F1 (classification) or original-unit RMSE (regress)."""
    if task == "classification":
        m = KNeighborsClassifier(n_neighbors=k).fit(Xtr, ytr)
        return float(f1_score(yte, m.predict(Xte), average="macro"))
    m = KNeighborsRegressor(n_neighbors=k).fit(Xtr, ytr)
    return float(np.sqrt(mean_squared_error(yte, m.predict(Xte)))) * tscale


# --------------------------------------------------------------- ModernNCA probe


class _Phi(torch.nn.Module):
    """2-layer MLP projection on top of frozen embeddings."""

    def __init__(self, d_in, d_hid=64, d_out=32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, d_hid),
            torch.nn.GELU(),
            torch.nn.Linear(d_hid, d_out),
        )

    def forward(self, x):
        return self.net(x)


def nca_probe(
    Etr, ytr, Ete, yte, task, tscale, k, seed, epochs=40, cand_frac=0.4, batch=128
):
    """Soft-kNN over a learned projection phi of the FROZEN embeddings (ModernNCA).

    NCA loss per query: softmax over -||phi(q)-phi(c)||^2 across sampled candidates,
    weighted-vote candidate labels (NLL of true class for classification, MSE of the
    weighted target for regression). Self is masked out. Evaluation is a soft-kNN over
    the full train set in phi-space.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    # Standardize embeddings so phi starts on a sane scale.
    mu, sd = Etr.mean(0), Etr.std(0) + 1e-6
    Etr_s = (Etr - mu) / sd
    Ete_s = (Ete - mu) / sd

    Xtr = torch.tensor(Etr_s)
    Xte = torch.tensor(Ete_s)
    n = len(Xtr)
    n_cand = max(k, int(cand_frac * n))

    if task == "classification":
        yt = torch.tensor(ytr, dtype=torch.long)
        n_cls = int(yt.max().item()) + 1
    else:
        yt = torch.tensor(ytr, dtype=torch.float32)

    phi = _Phi(Xtr.shape[1])
    opt = torch.optim.Adam(phi.parameters(), lr=1e-3, weight_decay=1e-4)

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            q_idx = torch.tensor(order[s : s + batch], dtype=torch.long)
            c_idx = torch.tensor(rng.choice(n, size=n_cand, replace=False))
            pq, pc = phi(Xtr[q_idx]), phi(Xtr[c_idx])
            dist = torch.cdist(pq, pc) ** 2  # [Bq, Bc]
            self_mask = q_idx.unsqueeze(1) == c_idx.unsqueeze(0)
            dist = dist.masked_fill(self_mask, float("inf"))
            w = torch.softmax(-dist, dim=1)  # [Bq, Bc]
            if task == "classification":
                onehot = torch.zeros(len(c_idx), n_cls)
                onehot[torch.arange(len(c_idx)), yt[c_idx]] = 1.0
                p = (w @ onehot).clamp_min(1e-9)  # [Bq, n_cls]
                loss = torch.nn.functional.nll_loss(torch.log(p), yt[q_idx])
            else:
                pred = w @ yt[c_idx]
                loss = torch.nn.functional.mse_loss(pred, yt[q_idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    # Soft-kNN evaluation over the full train set in phi-space.
    phi.eval()
    with torch.no_grad():
        ztr, zte = phi(Xtr), phi(Xte)
        dist = torch.cdist(zte, ztr) ** 2  # [Nte, Ntr]
        # Restrict softmax to each query's k nearest for a genuine kNN vote.
        kk = min(k, ztr.shape[0])
        nd, ni = torch.topk(-dist, kk, dim=1)
        w = torch.softmax(nd, dim=1)  # [Nte, k]
        if task == "classification":
            votes = yt[ni]  # [Nte, k]
            n_cls = int(yt.max().item()) + 1
            scores = torch.zeros(len(zte), n_cls)
            scores.scatter_add_(1, votes, w)
            pred = scores.argmax(1).numpy()
            return float(f1_score(yte, pred, average="macro"))
        pred = (w * yt[ni]).sum(1).numpy()
        return float(np.sqrt(mean_squared_error(yte, pred))) * tscale


# --------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset", type=str, help="corpus dataset name (data/tabular_corpus)"
    )
    ap.add_argument(
        "--synthetic", action="store_true", help="use an in-memory toy dataset"
    )
    ap.add_argument(
        "--task", choices=["classification", "regression"], default="classification"
    )
    ap.add_argument("--checkpoint", type=str, help="trained SingleRow*.ckpt to probe")
    ap.add_argument(
        "--untrained", action="store_true", help="probe a random-init encoder"
    )
    ap.add_argument(
        "--nca", action="store_true", help="also run the ModernNCA soft-kNN"
    )
    ap.add_argument("--epochs", type=int, default=15, help="quick-train epochs")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.synthetic:
        df, feat, task = synthetic(a.task, a.seed)
        name = f"synthetic-{task}"
    elif a.dataset:
        df, feat, task = load_dataset(a.dataset, a.task)
        name = a.dataset
    else:
        raise SystemExit("pass --synthetic or --dataset NAME")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, ytr, Xte, yte, tscale, n_classes = prep(df, feat, task, a.seed)
    cfg = SingleRowConfig.generate(_frame(Xtr, ytr, feat), target="__target__")

    # --- encoder ---
    if a.checkpoint:
        model, src = load_checkpoint(a.checkpoint, cfg, task, n_classes), "checkpoint"
    elif a.untrained:
        model, src = untrained_encoder(cfg, task, n_classes, a.seed), "untrained"
    else:
        model = train_encoder(cfg, Xtr, ytr, feat, task, n_classes, a.seed, a.epochs)
        src = f"quick-trained ({a.epochs}ep)"

    Etr = extract_embeddings(model, Xtr, ytr, feat, cfg, device)
    Ete = extract_embeddings(model, Xte, yte, feat, cfg, device)

    # --- kNN probes ---
    n = len(Xtr)
    k = min(int(10 * np.sqrt(n)), 1000)
    k = max(1, min(k, n - 1))
    raw = knn_metric(np.nan_to_num(Xtr), ytr, np.nan_to_num(Xte), yte, task, tscale, k)
    emb = knn_metric(Etr, ytr, Ete, yte, task, tscale, k)
    rows = [("raw-kNN", raw), ("embed-kNN", emb)]
    if a.nca:
        rows.append(
            ("nca-softkNN", nca_probe(Etr, ytr, Ete, yte, task, tscale, k, a.seed))
        )

    # --- report ---
    higher = task == "classification"
    metric = "macro-F1" if higher else "RMSE"
    print(
        f"\n{name} | task={task} | encoder={src} | N_train={n} k={k} | metric={metric}"
        f" ({'higher' if higher else 'lower'} better)"
    )
    print(f"  {'method':<14} {metric}")
    for label, val in rows:
        print(f"  {label:<14} {val:.4f}")

    embed_loses = (emb < raw) if higher else (emb > raw)
    if embed_loses:
        print(
            "  !! WARNING: embedding-space kNN LOSES to raw-feature kNN "
            f"({emb:.4f} vs {raw:.4f}) -- the encoder is NOT learning useful metric "
            "structure on this dataset."
        )
    else:
        print(f"  OK: embed-kNN ties-or-beats raw-kNN ({emb:.4f} vs {raw:.4f}).")


if __name__ == "__main__":
    main()
