"""Head-to-head: single-row deep model vs XGBoost / RandomForest.

The goal-1 bar: is the deep model "as good as trees, in error bands"? Regression -> test
RMSE (original target units), classification -> test macro-F1. Multi-seed mean+-std per
model per dataset, then an aggregate: on how many datasets does the deep model
TIE-OR-BEAT the best tree (within one deep std -> "not significantly worse")?

Notes:
  - Deep = from-scratch SingleRowRegressor / SingleRowClassifier (MTM pretraining is the
    next lever but needs the no-categorical MTM fix; this is the honest scratch
    baseline).
  - Missing cells: the deep model handles NaN natively (present-flag); trees get
    the same standardized features with NaN mean-imputed (post-standardization 0).
    Features clipped to +-8 to tame outliers.
  - Metrics are direction-aware (RMSE lower-better, macro-F1 higher-better).

Run (trains many models; progress off):
    uv run python scripts/tabular_vs_trees.py
    uv run python scripts/tabular_vs_trees.py --smoke
    uv run python scripts/tabular_vs_trees.py --tasks classification --seeds 3
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import shutil
import tempfile
import warnings

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as L
import torch

from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from xgboost import XGBClassifier, XGBRegressor


warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
torch.set_float32_matmul_precision("medium")
# Silence Lightning's per-fit device/dataloader chatter (the "GPU available: ..."
# blocks etc.) -- these go through rank_zero_info under a logger whose name varies
# with import order, so gate everything at INFO globally. The harness prints its
# own results with print(), not logging, so nothing we care about is lost.
logging.disable(logging.INFO)

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
RESULTS = Path(__file__).resolve().parent.parent / "tabular_vs_trees_results.json"
D_MODEL, N_HEADS, N_LAYERS, LR, BATCH = 128, 4, 3, 8e-4, 256


def load_corpus(tasks):
    man = json.loads((CORPUS / "manifest.json").read_text())
    out = []
    for m in man:
        m.setdefault("task", "regression")
        if m["task"] not in tasks:
            continue
        out.append(m)
    return man, out


def prep(df, feat, task, seed):
    """Split + standardize, with NaN kept; label-encode classification target."""
    strat = df["__target__"] if task == "classification" else None
    tr, te = train_test_split(df, test_size=0.3, random_state=seed, stratify=strat)
    mu, sd = tr[feat].mean(), tr[feat].std().replace(0, 1)

    def sx(d):
        X = np.clip(((d[feat] - mu) / sd).to_numpy(np.float32), -8.0, 8.0)
        return X

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


def _class_priors(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    return counts / counts.sum()


def _prediction_entropy(pred, n_classes):
    counts = np.bincount(pred, minlength=n_classes).astype(np.float64)
    probs = counts / max(1.0, counts.sum())
    nz = probs[probs > 0]
    return float(-(nz * np.log(nz)).sum())


def deep_score(
    Xtr,
    ytr,
    Xte,
    yte,
    feat,
    task,
    n_classes,
    tscale,
    seed,
    epochs=100,
    patience=16,
    head="mse",
    bin_spacing="uniform",
    symlog=False,
    value_encoder="identity",
    fusion="additive",
    label_smoothing=0.0,
    logit_adjustment_tau=1.0,
    warmup_frac=0.05,
    ensemble_k=0,
):
    L.seed_everything(seed, verbose=False)
    fit_i, val_i = train_test_split(
        np.arange(len(Xtr)),
        test_size=0.2,
        random_state=seed,
        stratify=ytr if task == "classification" else None,
    )
    cfg = SingleRowConfig.generate(_frame(Xtr, ytr, feat), target="__target__")

    def loader(idx, shuffle):
        ds = TabularDS(_frame(Xtr[idx], ytr[idx], feat), cfg)
        return DataLoader(
            ds, batch_size=BATCH, shuffle=shuffle, collate_fn=tabular_collate_fn
        )

    if task == "regression":
        model = SingleRowRegressor(
            cfg,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            lr=LR,
            head=head,
            y_train=ytr[fit_i],
            bin_spacing=bin_spacing,
            symlog=symlog,
            value_encoder=value_encoder,
            fusion=fusion,
            warmup_frac=warmup_frac,
            ensemble_k=ensemble_k,
        )
    else:
        model = SingleRowClassifier(
            cfg,
            num_classes=n_classes,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            lr=LR,
            class_priors=_class_priors(ytr[fit_i], n_classes),
            logit_adjustment_tau=logit_adjustment_tau,
            label_smoothing=label_smoothing,
            value_encoder=value_encoder,
            fusion=fusion,
            warmup_frac=warmup_frac,
            ensemble_k=ensemble_k,
        )
    # Unique per-call checkpoint dir: parallel workers must NOT share a path
    # (a dir keyed by seed alone let two datasets clobber each other's best.ckpt,
    # so load_best restored a foreign architecture -> state_dict mismatch).
    _tmp = os.environ.get("TMPDIR", "/tmp")
    ckdir = tempfile.mkdtemp(prefix=f"tvt_{seed}_", dir=_tmp)
    trainer, ckpt = build_trainer(
        max_epochs=epochs,
        patience=patience,
        ckpt_dir=str(ckdir),
        monitor="val_macro_f1" if task == "classification" else "val_loss",
        mode="max" if task == "classification" else "min",
    )
    trainer.fit(
        model,
        train_dataloaders=loader(fit_i, True),
        val_dataloaders=loader(val_i, False),
    )
    load_best(model, ckpt)
    shutil.rmtree(ckdir, ignore_errors=True)

    model.eval()
    te_loader = DataLoader(
        TabularDS(_frame(Xte, yte, feat), cfg),
        batch_size=4096,
        collate_fn=tabular_collate_fn,
    )
    preds = []
    with torch.no_grad():
        for b in te_loader:
            b.inputs = b.inputs.to(model.device)
            if task == "regression":
                # predict_value averages ensemble members and applies the HL-Gauss
                # expected value when that head is active (no-op when ensemble_k == 0).
                out = model.predict_value(b)
            else:
                # predict() averages member softmaxes -> [b, C] probabilities
                # (a no-op softmax of the single model when ensemble_k == 0).
                out = model.model.predict(b.inputs.numeric, b.inputs.categorical)
            preds.append(out.cpu().numpy())
    P = np.concatenate(preds)
    if task == "regression":
        # WO-5: clip test-time predictions to the observed train target range.
        pred = np.clip(P.ravel(), float(np.nanmin(ytr)), float(np.nanmax(ytr)))
        return {
            "score": float(np.sqrt(mean_squared_error(yte, pred))) * tscale,
            "collapsed": False,
        }
    pred = P.argmax(1)
    entropy = _prediction_entropy(pred, n_classes)
    return {
        "score": float(f1_score(yte, pred, average="macro")),
        "collapsed": bool(entropy < 1e-8),
        "prediction_entropy": entropy,
        "prediction_histogram": np.bincount(pred, minlength=n_classes).tolist(),
    }


TREE_KINDS = ("xgb", "rf", "cat")  # classical baselines; best-of-these is the bar


def tree_score(kind, Xtr, ytr, Xte, yte, task, tscale, seed, n_jobs=-1):
    Xtr_i = np.nan_to_num(Xtr, nan=0.0)  # trees have no missing channel -> mean-impute
    Xte_i = np.nan_to_num(Xte, nan=0.0)
    if task == "regression":
        if kind == "xgb":
            m = XGBRegressor(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=n_jobs,
                random_state=seed,
            )
        elif kind == "cat":
            m = CatBoostRegressor(
                iterations=400,
                depth=6,
                learning_rate=0.05,
                random_seed=seed,
                thread_count=n_jobs,
                verbose=0,
            )
        else:
            m = RandomForestRegressor(
                n_estimators=300, n_jobs=n_jobs, random_state=seed
            )
        m.fit(Xtr_i, ytr)
        return float(np.sqrt(mean_squared_error(yte, m.predict(Xte_i)))) * tscale
    if kind == "xgb":
        m = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=n_jobs,
            random_state=seed,
        )
    elif kind == "cat":
        # auto_class_weights="Balanced" mirrors RF's class_weight -> fair on imbalance.
        m = CatBoostClassifier(
            iterations=400,
            depth=6,
            learning_rate=0.05,
            random_seed=seed,
            thread_count=n_jobs,
            verbose=0,
            auto_class_weights="Balanced",
        )
    else:
        m = RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=n_jobs, random_state=seed
        )
    m.fit(Xtr_i, ytr)
    return float(f1_score(yte, m.predict(Xte_i), average="macro"))


def _score_dataset(payload):
    """Score one dataset (all seeds, deep + trees). Runs in a worker process.

    Self-contained so ProcessPoolExecutor (spawn) can pickle + dispatch it: all
    it needs are the picklable ``payload`` and this module's globals (re-imported
    in the child). Returns ``(i, rec)`` on success, else ``(i, {"__skip__": msg})``
    or ``(i, {"__fail__": msg})`` so the parent owns printing and the file writes.
    """
    i, m, cfg = payload
    name, task, feat = m["name"], m["task"], m["feature_cols"]
    try:
        df = pd.read_parquet(CORPUS / f"{name}.parquet")
    except Exception as e:
        return i, {"__skip__": f"{name}: {e}"}
    row = {"deep": [], "xgb": [], "rf": [], "cat": [], "seed_results": []}
    try:
        for seed in range(cfg["seeds"]):
            Xtr, ytr, Xte, yte, tscale, nc = prep(df, feat, task, seed)
            deep = deep_score(
                Xtr,
                ytr,
                Xte,
                yte,
                feat,
                task,
                nc,
                tscale,
                seed,
                cfg["epochs"],
                cfg["patience"],
                head=cfg["head"],
                bin_spacing=cfg["bin_spacing"],
                symlog=cfg["symlog"],
                value_encoder=cfg["value_encoder"],
                fusion=cfg["fusion"],
                label_smoothing=cfg["label_smoothing"],
                logit_adjustment_tau=cfg["logit_adjustment_tau"],
                warmup_frac=cfg["warmup_frac"],
                ensemble_k=cfg["ensemble_k"],
            )
            xgb = tree_score(
                "xgb", Xtr, ytr, Xte, yte, task, tscale, seed, cfg["tree_jobs"]
            )
            rf = tree_score(
                "rf", Xtr, ytr, Xte, yte, task, tscale, seed, cfg["tree_jobs"]
            )
            cat = tree_score(
                "cat", Xtr, ytr, Xte, yte, task, tscale, seed, cfg["tree_jobs"]
            )
            row["deep"].append(deep["score"])
            row["xgb"].append(xgb)
            row["rf"].append(rf)
            row["cat"].append(cat)
            row["seed_results"].append(
                {
                    "seed": seed,
                    "deep": deep["score"],
                    "xgb": xgb,
                    "rf": rf,
                    "cat": cat,
                    "collapsed": bool(deep.get("collapsed", False)),
                    "prediction_entropy": deep.get("prediction_entropy"),
                    "prediction_histogram": deep.get("prediction_histogram"),
                }
            )
    except Exception as e:
        return i, {"__fail__": f"{name} ({task}): {type(e).__name__} {str(e)[:50]}"}
    rec = {
        "name": name,
        "task": task,
        "n_rows": m["n_rows"],
        "n_features": len(feat),
    }
    for k in ("deep", "xgb", "rf", "cat"):
        rec[k], rec[k + "_std"] = float(np.mean(row[k])), float(np.std(row[k]))
    rec["seed_results"] = row["seed_results"]
    rec["collapsed"] = any(s["collapsed"] for s in row["seed_results"])
    rec["collapsed_seeds"] = [s["seed"] for s in row["seed_results"] if s["collapsed"]]
    return i, rec


def _best_tree(rec, higher):
    scores = [rec[k] for k in TREE_KINDS]
    return max(scores) if higher else min(scores)


def _print_row(i, rec):
    higher = rec["task"] == "classification"
    best_tree = _best_tree(rec, higher)
    tie = (
        (rec["deep"] >= best_tree - rec["deep_std"])
        if higher
        else (rec["deep"] <= best_tree + rec["deep_std"])
    )
    metric = "F1" if higher else "RMSE"
    print(
        f"[{i:>2}] {rec['name']:34s} {rec['task'][:3]} {metric} "
        f"deep {rec['deep']:.3f}±{rec['deep_std']:.3f} "
        f"| xgb {rec['xgb']:.3f} rf {rec['rf']:.3f} cat {rec['cat']:.3f} "
        f"| {'TIE/WIN' if tie else 'loss'}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["regression", "classification"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--head", choices=["mse", "hl_gauss"], default="mse")
    ap.add_argument("--bin-spacing", choices=["uniform", "quantile"], default="uniform")
    ap.add_argument("--symlog", action="store_true")
    ap.add_argument("--value-encoder", choices=["identity", "dual"], default="identity")
    ap.add_argument("--fusion", choices=["additive", "hadamard"], default="additive")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--logit-adjustment-tau", type=float, default=1.0)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument(
        "--ensemble-k",
        type=int,
        default=0,
        help="TabM-mini BatchEnsemble members (0=off; 32 recommended).",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel dataset workers sharing the GPU (1=serial). The tiny models "
        "leave the GPU ~90%% idle; 6-8 workers usually gives a near-linear speedup.",
    )
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    epochs, patience = (3, 2) if a.smoke else (100, 16)
    if a.smoke:
        a.seeds = 2

    _, datasets = load_corpus(a.tasks)
    if a.smoke:  # a couple of each requested task
        pick = []
        for t in a.tasks:
            pick += [m for m in datasets if m["task"] == t][:2]
        datasets = pick
    # Cap tree threads so W GPU-sharing workers don't oversubscribe the CPUs.
    cpu = os.cpu_count() or 1
    tree_jobs = -1 if a.workers <= 1 else max(1, cpu // a.workers)
    print(
        f"datasets={len(datasets)} | tasks={a.tasks} | seeds={a.seeds} | "
        f"ensemble_k={a.ensemble_k} | workers={a.workers} | device="
        f"{'cuda' if torch.cuda.is_available() else 'cpu'}",
        flush=True,
    )

    cfg = {
        "seeds": a.seeds,
        "epochs": epochs,
        "patience": patience,
        "head": a.head,
        "bin_spacing": a.bin_spacing,
        "symlog": a.symlog,
        "value_encoder": a.value_encoder,
        "fusion": a.fusion,
        "label_smoothing": a.label_smoothing,
        "logit_adjustment_tau": a.logit_adjustment_tau,
        "warmup_frac": a.warmup_frac,
        "ensemble_k": a.ensemble_k,
        "tree_jobs": tree_jobs,
    }
    payloads = [(i, m, cfg) for i, m in enumerate(datasets)]
    by_index = {}

    def _emit(i, res):
        if "__skip__" in res:
            print(f"[{i}] SKIP {res['__skip__']}", flush=True)
            return
        if "__fail__" in res:
            print(f"[{i}] FAIL {res['__fail__']}", flush=True)
            return
        by_index[i] = res
        _print_row(i, res)
        # Write in original dataset order regardless of completion order.
        RESULTS.write_text(
            json.dumps([by_index[j] for j in sorted(by_index)], indent=2)
        )

    if a.workers <= 1:
        for p in payloads:
            i, res = _score_dataset(p)
            _emit(i, res)
    else:
        # Separate processes (not threads): the GIL + Lightning's Trainer make
        # threads useless here; spawned procs each get a CUDA context and the
        # driver time-slices the underused GPU across them.
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx) as ex:
            futures = [ex.submit(_score_dataset, p) for p in payloads]
            for fut in as_completed(futures):
                i, res = fut.result()
                _emit(i, res)

    results = [by_index[j] for j in sorted(by_index)]

    # ---- aggregate ----
    print(
        "\n=== AGGREGATE: deep vs best tree (tie = within one deep std) ===", flush=True
    )
    for task in a.tasks:
        rows = [r for r in results if r["task"] == task]
        if not rows:
            continue
        higher = task == "classification"

        def tie(r, higher=higher):
            bt = _best_tree(r, higher)
            return (
                (r["deep"] >= bt - r["deep_std"])
                if higher
                else (r["deep"] <= bt + r["deep_std"])
            )

        def win(r, higher=higher):
            bt = _best_tree(r, higher)
            return (r["deep"] > bt) if higher else (r["deep"] < bt)

        n = len(rows)
        n_tie = sum(tie(r) for r in rows)
        n_win = sum(win(r) for r in rows)
        print(
            f"  {task:14s} n={n:>2} | deep ties-or-beats best tree on "
            f"{n_tie}/{n} | strictly beats on {n_win}/{n}",
            flush=True,
        )
    print(f"\nresults -> {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
