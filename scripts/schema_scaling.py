"""Schema-scaling: does semantic column transfer improve with the NUMBER of
pretraining schemas? This is the honest test the single A->B pair could not reach.

Thesis (Kai): semantic column embeddings are a *meta-learning* bet -- the projection
"column-description -> predictive role" only becomes reusable if fit across MANY column
vocabularies. One source schema has nothing to generalize; the payoff (if any) shows up
as a SLOPE: transfer to a held-out dataset should improve as K (# pretraining datasets)
grows, and semantic should pull ahead of an index embedder that cannot cross vocabularies.

Setup (corpus from scripts/download_tabular_corpus.py):
  - One SHARED encoder (semantic embedder proj + numeric value projection + attention +
    pooling) with a per-dataset regression head. For all-numeric datasets every shared
    weight is schema-independent, so the same encoder consumes any schema by swapping the
    column descriptions -- variable-width transfer for free.
  - Pretrain the shared encoder on K source datasets (round-robin, each standardized on
    its own train). Then few-shot fine-tune on a held-out dataset (unseen columns) and
    measure test RMSE (held-out's own units).

Arms (held-out test RMSE):
  transfer_semantic : shared encoder pretrained on K sources (semantic embedder) -> held-out
  transfer_index    : same, but an index embedder whose vocab = source column names only
                      (held-out's columns are unseen -> [UNK]: the honest floor)
  scratch_semantic  : semantic model trained on the held-out few-shot labels only (K-independent)
  xgb               : XGBoost on the held-out few-shot labels (the tree bar; no transfer)

== PRE-REGISTERED signal (locked before results) ==
Semantic column transfer is VALIDATED iff, on average over held-out datasets:
  (1) transfer_semantic test-RMSE DECREASES as K grows (a real schema-scaling slope), AND
  (2) at the largest K, transfer_semantic < transfer_index (the gain is name-semantics,
      not just the shared numeric/attention backbone), AND
  (3) at the largest K, transfer_semantic < scratch_semantic (pretraining helped at all).
KILL if the slope is flat OR transfer_index >= transfer_semantic at large K -> tabular
column-name semantics carry no transferable signal even across many schemas.

Run (trains many models; progress off; ask before running):
    uv run python scripts/schema_scaling.py
    uv run python scripts/schema_scaling.py --smoke
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
torch.set_float32_matmul_precision("medium")

from trea.models.column_embeddings import (  # noqa: E402
    IndexColumnEmbedder,
    SemanticColumnEmbedder,
)
from trea.models.single_row import AttentionPooling, TabularEncoder  # noqa: E402
from trea.utils.single_row_data import SingleRowConfig  # noqa: E402

CORPUS = Path(__file__).resolve().parent.parent / "data" / "tabular_corpus"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
D_MODEL, N_HEADS, N_LAYERS, DROPOUT, BATCH = 128, 4, 3, 0.1, 256
RESULTS = Path(__file__).resolve().parent.parent / "schema_scaling_results.json"


# --------------------------------------------------------------------------- data
def load_corpus():
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    out = []
    for m in manifest:
        df = pd.read_parquet(CORPUS / f"{m['name']}.parquet")
        feat = m["feature_cols"]
        descr = [m["descriptions"][c] for c in feat]  # cleaned names, in column order
        out.append({"name": m["name"], "df": df, "feat": feat, "descr": descr})
    return out


def standardize(train, other, feat):
    mu, sd = train[feat].mean(), train[feat].std().replace(0, 1)
    tmu, tsd = train["__target__"].mean(), train["__target__"].std() or 1.0

    def sc(d):
        X = ((d[feat] - mu) / sd).to_numpy(np.float32)
        # Tame outliers to avoid attention-logit overflow (clip preserves NaN, which the
        # model treats as missing natively -- no sentinel conversion needed here).
        X = np.clip(X, -8.0, 8.0)
        y = ((d["__target__"] - tmu) / tsd).to_numpy(np.float32)
        return X, y

    return sc(train), sc(other), float(tsd)


def tensors(X, y):
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y).unsqueeze(1))


# -------------------------------------------------------------------------- model
def head(d_model, dropout):
    return nn.Sequential(
        nn.Linear(d_model, 2 * d_model),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(2 * d_model, 1),
    )


class SharedModel(nn.Module):
    """One shared encoder (+ embedder, pooling) with a per-dataset regression head.

    The encoder is built from a dummy config -- its schema-dependent parts (the value/
    mask embedding table and index buffers) are unused for all-numeric inputs with a
    plug-in col_embedder, so forward works for ANY schema by setting the column
    descriptions per batch.
    """

    def __init__(self, embedder, head_names):
        super().__init__()
        dummy = SingleRowConfig.generate(
            pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0], "__t__": [0.0, 1.0]}),
            target="__t__",
        )
        self.encoder = TabularEncoder(
            dummy, D_MODEL, N_HEADS, N_LAYERS, DROPOUT, col_embedder=embedder
        )
        self.pooling = AttentionPooling(D_MODEL)
        self.drop = nn.Dropout(DROPOUT)
        self.heads = nn.ModuleDict({n: head(D_MODEL, DROPOUT) for n in head_names})

    def forward(self, name, num, descr):
        self.encoder.col_texts = descr
        self.encoder.numeric_texts = descr
        h = self.encoder(num, None)
        return self.heads[name](self.drop(self.pooling(h)))


# ------------------------------------------------------------------------ training
def loaders_for(datasets, splits):
    out = {}
    for ds in datasets:
        (Xtr, ytr), _, _ = splits[ds["name"]]
        out[ds["name"]] = DataLoader(
            tensors(Xtr, ytr), batch_size=BATCH, shuffle=True, drop_last=False
        )
    return out


def pretrain(model, sources, splits, descrs, epochs, lr):
    model.to(DEV).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    lossf = nn.MSELoss()
    loaders = loaders_for(sources, splits)
    names = [ds["name"] for ds in sources]
    for _ in range(epochs):
        iters = {n: iter(loaders[n]) for n in names}
        active = list(names)
        while active:
            for n in list(active):
                try:
                    X, y = next(iters[n])
                except StopIteration:
                    active.remove(n)
                    continue
                X, y = X.to(DEV), y.to(DEV)
                loss = lossf(model(n, X, descrs[n]), y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
    return model


@torch.no_grad()
def eval_rmse(model, name, X, y, descr, tscale):
    model.eval()
    preds = []
    for i in range(0, len(X), 4096):
        xb = torch.from_numpy(X[i : i + 4096]).to(DEV)
        preds.append(model(name, xb, descr).cpu().numpy())
    p = np.concatenate(preds).ravel()
    return float(np.sqrt(mean_squared_error(y, p))) * tscale


def finetune_heldout(model, held, splits, descrs, n_labels, epochs, lr, tscale, seed):
    """Fine-tune the (possibly pretrained) model on n_labels of held-out; best-val test."""
    name = held["name"]
    (Xtr, ytr), (Xte, yte), _ = splits[name]
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(Xtr), size=min(n_labels, len(Xtr)), replace=False)
    Xs, ys = Xtr[idx], ytr[idx]
    fit_i, val_i = train_test_split(
        np.arange(len(Xs)), test_size=0.25, random_state=seed
    )
    loader = DataLoader(
        tensors(Xs[fit_i], ys[fit_i]), batch_size=min(BATCH, len(fit_i)), shuffle=True
    )
    model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    lossf = nn.MSELoss()
    best_val, best_state, patience, bad = np.inf, None, 10, 0
    for _ in range(epochs):
        model.train()
        for X, y in loader:
            X, y = X.to(DEV), y.to(DEV)
            loss = lossf(model(name, X, descrs[name]), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val = eval_rmse(model, name, Xs[val_i], ys[val_i], descrs[name], 1.0)
        if val < best_val - 1e-4:
            best_val, bad = val, 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return eval_rmse(model, name, Xte, yte, descrs[name], tscale)


def xgb_rmse(held, splits, n_labels, tscale, seed):
    name = held["name"]
    (Xtr, ytr), (Xte, yte), _ = splits[name]
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(Xtr), size=min(n_labels, len(Xtr)), replace=False)
    m = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=seed,
    )
    m.fit(Xtr[idx], ytr[idx])
    return float(np.sqrt(mean_squared_error(yte, m.predict(Xte)))) * tscale


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-labels", type=int, default=200)
    ap.add_argument(
        "--held-out",
        nargs="+",
        default=["california_housing", "concrete_compressive_strength", "abalone"],
    )
    ap.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--source-seeds", type=int, default=2)
    a = ap.parse_args()

    pre_ep, ft_ep, ft_lr, pre_lr = (
        (2, 3, 1e-3, 1e-3) if a.smoke else (40, 80, 5e-4, 5e-4)
    )
    if a.smoke:
        a.k_values, a.source_seeds, a.held_out = [1, 2], 1, a.held_out[:1]

    corpus = load_corpus()
    by_name = {d["name"]: d for d in corpus}
    descrs = {d["name"]: d["descr"] for d in corpus}
    held_list = [h for h in a.held_out if h in by_name]

    # per-dataset train/test split + standardization (fixed across all arms)
    splits = {}
    for d in corpus:
        tr, te = train_test_split(d["df"], test_size=0.3, random_state=42)
        (Xtr, ytr), (Xte, yte), tsd = standardize(tr, te, d["feat"])
        splits[d["name"]] = ((Xtr, ytr), (Xte, yte), tsd)

    print(
        f"device={DEV} | corpus={len(corpus)} | held_out={held_list} | "
        f"K={a.k_values} | n_labels={a.n_labels} | source_seeds={a.source_seeds}",
        flush=True,
    )

    results = []
    for held in held_list:
        h = by_name[held]
        tscale = splits[held][2]
        pool = [d for d in corpus if d["name"] != held]

        # K-independent baselines (few-shot only, averaged over seeds)
        base = {"scratch_semantic": [], "xgb": []}
        for seed in range(max(a.source_seeds, 2)):
            emb = SemanticColumnEmbedder(descrs[held], D_MODEL)
            sm = SharedModel(emb, [held])
            base["scratch_semantic"].append(
                finetune_heldout(
                    sm, h, splits, descrs, a.n_labels, ft_ep, ft_lr, tscale, seed
                )
            )
            base["xgb"].append(xgb_rmse(h, splits, a.n_labels, tscale, seed))
        b_sem, b_xgb = np.mean(base["scratch_semantic"]), np.mean(base["xgb"])
        print(
            f"\n[{held}] tscale={tscale:.3f} | scratch_semantic={b_sem:.3f} | xgb={b_xgb:.3f}",
            flush=True,
        )

        for K in a.k_values:
            if K > len(pool):
                continue
            for etype in ("semantic", "index"):
                rmses = []
                for seed in range(a.source_seeds):
                    rng = np.random.RandomState(1000 * K + seed)
                    src = [
                        pool[i] for i in rng.choice(len(pool), size=K, replace=False)
                    ]
                    src_descr = sorted({t for d in src for t in d["descr"]})
                    if etype == "semantic":
                        emb = SemanticColumnEmbedder(src_descr, D_MODEL)
                    else:
                        emb = IndexColumnEmbedder(src_descr, D_MODEL)
                    model = SharedModel(emb, [d["name"] for d in src] + [held])
                    pretrain(model, src, splits, descrs, pre_ep, pre_lr)
                    rmses.append(
                        finetune_heldout(
                            model,
                            h,
                            splits,
                            descrs,
                            a.n_labels,
                            ft_ep,
                            ft_lr,
                            tscale,
                            seed,
                        )
                    )
                r = float(np.mean(rmses))
                results.append(
                    {
                        "held_out": held,
                        "K": K,
                        "arm": f"transfer_{etype}",
                        "rmse": r,
                        "std": float(np.std(rmses)),
                        "n_seeds": a.source_seeds,
                    }
                )
                print(
                    f"  K={K:>2} transfer_{etype:<8} RMSE {r:.3f} +- {np.std(rmses):.3f}",
                    flush=True,
                )
            results.append(
                {
                    "held_out": held,
                    "K": K,
                    "arm": "scratch_semantic",
                    "rmse": float(b_sem),
                }
            )
            results.append(
                {"held_out": held, "K": K, "arm": "xgb", "rmse": float(b_xgb)}
            )
            RESULTS.write_text(json.dumps(results, indent=2))  # incremental checkpoint

    # ----- verdict -----
    print("\n=== SCHEMA-SCALING SUMMARY (held-out test RMSE) ===", flush=True)
    verdicts = []
    for held in held_list:
        rows = [r for r in results if r["held_out"] == held]
        ks = sorted({r["K"] for r in rows})

        def g(K, arm):
            v = [r["rmse"] for r in rows if r["K"] == K and r["arm"] == arm]
            return v[0] if v else float("nan")

        print(
            f"\n[{held}]  scratch_semantic={g(ks[0], 'scratch_semantic'):.3f}  xgb={g(ks[0], 'xgb'):.3f}",
            flush=True,
        )
        print(
            f"  {'K':>3} {'transfer_semantic':>18} {'transfer_index':>16}", flush=True
        )
        for K in ks:
            print(
                f"  {K:>3} {g(K, 'transfer_semantic'):>18.3f} {g(K, 'transfer_index'):>16.3f}",
                flush=True,
            )
        kmax = ks[-1]
        slope = g(ks[0], "transfer_semantic") - g(
            kmax, "transfer_semantic"
        )  # >0 => improves with K
        v1 = slope > 0
        v2 = g(kmax, "transfer_semantic") < g(kmax, "transfer_index")
        v3 = g(kmax, "transfer_semantic") < g(ks[0], "scratch_semantic")
        verdicts.append((v1, v2, v3))
        print(
            f"  (1) improves with K: {'PASS' if v1 else 'FAIL'} (slope {slope:+.3f}) | "
            f"(2) sem<idx @K={kmax}: {'PASS' if v2 else 'FAIL'} | "
            f"(3) sem<scratch: {'PASS' if v3 else 'FAIL'}",
            flush=True,
        )

    n_val = sum(1 for v in verdicts if all(v))
    print(
        f"\n=== VERDICT: {n_val}/{len(verdicts)} held-out datasets satisfy ALL of "
        f"(slope>0, sem<idx, sem<scratch) ===",
        flush=True,
    )
    print(
        "VALIDATED (majority)"
        if n_val > len(verdicts) / 2
        else "KILL/INCONCLUSIVE: column-name semantics show no scaling transfer signal",
        flush=True,
    )
    print(f"\nresults -> {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
