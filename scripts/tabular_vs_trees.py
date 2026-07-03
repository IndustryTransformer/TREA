"""Head-to-head: single-row deep model vs tree baselines.

The goal-1 bar: is the deep model "as good as trees, in error bands"?
Regression -> test RMSE (original target units), classification -> test macro-F1.
Multi-seed mean+-std per model per dataset, then an aggregate: on how many
datasets does the deep model TIE-OR-BEAT the best tree?

Notes:
  - Deep = from-scratch SingleRowRegressor / SingleRowClassifier.
  - Missing cells: the deep model handles NaN natively (present-flag); trees get
    the same standardized features with NaN mean-imputed after standardization.
  - Metrics are direction-aware (RMSE lower-better, macro-F1 higher-better).

Run (trains many models; progress off):
    uv run python scripts/tabular_vs_trees.py
    uv run python scripts/tabular_vs_trees.py --smoke
    uv run python scripts/tabular_vs_trees.py --tasks classification --seeds 3
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

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from xgboost import XGBClassifier, XGBRegressor


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
    """Split, standardize, and label-encode classification targets."""
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
    epochs=256,
    patience=None,
    head="mse",
    bin_spacing="uniform",
    symlog=False,
    label_smoothing=0.05,
    logit_adjust_tau=1.0,
    warmup_frac=0.1,
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

    train_batches = max(1, int(np.ceil(len(fit_i) / BATCH)))
    warmup_steps = int(train_batches * epochs * warmup_frac)

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
            warmup_steps=warmup_steps,
        )
        monitor, mode = "val_loss", "min"
    else:
        counts = np.bincount(ytr[fit_i], minlength=n_classes).astype(np.float32)
        priors = counts / counts.sum()
        model = SingleRowClassifier(
            cfg,
            num_classes=n_classes,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            lr=LR,
            class_priors=priors,
            logit_adjust_tau=logit_adjust_tau,
            label_smoothing=label_smoothing,
            warmup_steps=warmup_steps,
        )
        monitor, mode = "val_f1_macro", "max"
    ckdir = Path(os.environ.get("TMPDIR", "/tmp")) / f"tvt_{seed}"
    trainer, ckpt = build_trainer(
        max_epochs=epochs,
        patience=patience,
        ckpt_dir=str(ckdir),
        monitor=monitor,
        mode=mode,
    )
    trainer.fit(
        model,
        train_dataloaders=loader(fit_i, True),
        val_dataloaders=loader(val_i, False),
    )
    load_best(model, ckpt)

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
                out = model.predict_value(b)
            else:
                out = model.model(b.inputs.numeric, b.inputs.categorical)
            preds.append(out.cpu().numpy())
    P = np.concatenate(preds)
    if task == "regression":
        pred = np.clip(P.ravel(), float(np.nanmin(ytr)), float(np.nanmax(ytr)))
        score = float(np.sqrt(mean_squared_error(yte, pred))) * tscale
        return score, {"constant_pred": False}
    pred = P.argmax(1)
    return float(f1_score(yte, pred, average="macro")), {
        "constant_pred": bool(np.unique(pred).size <= 1)
    }


def tree_score(kind, Xtr, ytr, Xte, yte, task, tscale, seed):
    Xtr_i = np.nan_to_num(Xtr, nan=0.0)  # trees have no missing channel -> mean-impute
    Xte_i = np.nan_to_num(Xte, nan=0.0)
    if task == "regression":
        m = (
            XGBRegressor(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=seed,
            )
            if kind == "xgb"
            else RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=seed)
        )
        m.fit(Xtr_i, ytr)
        return float(np.sqrt(mean_squared_error(yte, m.predict(Xte_i)))) * tscale
    m = (
        XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=seed,
        )
        if kind == "xgb"
        else RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=seed
        )
    )
    m.fit(Xtr_i, ytr)
    return float(f1_score(yte, m.predict(Xte_i), average="macro"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["regression", "classification"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=256)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--head", choices=["mse", "hl_gauss"], default="mse")
    ap.add_argument("--bin-spacing", choices=["uniform", "quantile"], default="uniform")
    ap.add_argument("--symlog", action="store_true")
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--logit-adjust-tau", type=float, default=1.0)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    epochs, patience = (3, None) if a.smoke else (a.epochs, a.patience)
    if a.smoke:
        a.seeds = 2

    _, datasets = load_corpus(a.tasks)
    if a.smoke:  # a couple of each requested task
        pick = []
        for t in a.tasks:
            pick += [m for m in datasets if m["task"] == t][:2]
        datasets = pick
    print(
        f"datasets={len(datasets)} | tasks={a.tasks} | seeds={a.seeds} | device="
        f"{'cuda' if torch.cuda.is_available() else 'cpu'}",
        flush=True,
    )

    results = []
    for i, m in enumerate(datasets):
        name, task, feat = m["name"], m["task"], m["feature_cols"]
        try:
            df = pd.read_parquet(CORPUS / f"{name}.parquet")
        except Exception as e:
            print(f"[{i}] SKIP {name}: {e}", flush=True)
            continue
        row = {"deep": [], "xgb": [], "rf": [], "deep_constant_pred": []}
        try:
            for seed in range(a.seeds):
                Xtr, ytr, Xte, yte, tscale, nc = prep(df, feat, task, seed)
                deep_metric, deep_diag = deep_score(
                    Xtr,
                    ytr,
                    Xte,
                    yte,
                    feat,
                    task,
                    nc,
                    tscale,
                    seed,
                    epochs,
                    patience,
                    head=a.head,
                    bin_spacing=a.bin_spacing,
                    symlog=a.symlog,
                    label_smoothing=a.label_smoothing,
                    logit_adjust_tau=a.logit_adjust_tau,
                    warmup_frac=a.warmup_frac,
                )
                row["deep"].append(deep_metric)
                row["deep_constant_pred"].append(deep_diag["constant_pred"])
                row["xgb"].append(
                    tree_score("xgb", Xtr, ytr, Xte, yte, task, tscale, seed)
                )
                row["rf"].append(
                    tree_score("rf", Xtr, ytr, Xte, yte, task, tscale, seed)
                )
        except Exception as e:
            print(
                f"[{i}] FAIL {name} ({task}): {type(e).__name__} {str(e)[:50]}",
                flush=True,
            )
            continue
        rec = {
            "name": name,
            "task": task,
            "n_rows": m["n_rows"],
            "n_features": len(feat),
        }
        for k in ("deep", "xgb", "rf"):
            rec[k], rec[k + "_std"] = float(np.mean(row[k])), float(np.std(row[k]))
        rec["deep_constant_pred_seeds"] = int(sum(row["deep_constant_pred"]))
        results.append(rec)
        RESULTS.write_text(json.dumps(results, indent=2))
        higher = task == "classification"
        best_tree = max(rec["xgb"], rec["rf"]) if higher else min(rec["xgb"], rec["rf"])
        tie = (
            (rec["deep"] >= best_tree - rec["deep_std"])
            if higher
            else (rec["deep"] <= best_tree + rec["deep_std"])
        )
        metric = "F1" if higher else "RMSE"
        status = "TIE/WIN" if tie else "loss"
        print(
            f"[{i:>2}] {name:34s} {task[:3]} {metric} "
            f"deep {rec['deep']:.3f}±{rec['deep_std']:.3f} "
            f"| xgb {rec['xgb']:.3f} rf {rec['rf']:.3f} | {status}"
            f" | const_seeds={rec['deep_constant_pred_seeds']}",
            flush=True,
        )

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
            bt = max(r["xgb"], r["rf"]) if higher else min(r["xgb"], r["rf"])
            return (
                (r["deep"] >= bt - r["deep_std"])
                if higher
                else (r["deep"] <= bt + r["deep_std"])
            )

        def win(r, higher=higher):
            bt = max(r["xgb"], r["rf"]) if higher else min(r["xgb"], r["rf"])
            return (r["deep"] > bt) if higher else (r["deep"] < bt)

        n = len(rows)
        n_tie = sum(tie(r) for r in rows)
        n_win = sum(win(r) for r in rows)
        print(
            f"  {task:14s} n={n:>2} | deep ties-or-beats best tree "
            f"on {n_tie}/{n} | strictly beats on {n_win}/{n}",
            flush=True,
        )
    print(f"\nresults -> {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
