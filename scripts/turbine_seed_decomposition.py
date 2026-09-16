"""Decompose deep seed variance on turbine NOx: labeled-subset draw vs model init.

In scripts/turbine_label_efficiency_deep.py one seed sets (a) which rows are
labeled, (b) the fit/val split, (c) weight init + shuffle order, (d) tree random
state, so the logged spread cannot be attributed. Here the two are decoupled:

  for each fraction:
    for subset seed s in SUBSET_SEEDS:      # fixes the labeled rows AND fit/val split
      for init seed i in INIT_SEEDS:        # varies ONLY init / shuffle / tree rng
        xgb, scratch, pre  ->  test RMSE (+ test predictions for the ensemble)

One-way decomposition per arm:
  between  = std of the per-subset means            (data-draw effect)
  within   = mean of the per-subset std over inits  (model / optimizer effect)
  ensemble = RMSE of the mean prediction over the INIT_SEEDS inits, per subset
             (the variance-reduction fix, no seed selection involved)

Same stack / hyperparameters as the v1 turbine curve (lr 8e-4, temporal split),
so the numbers are comparable to LESSONS §3c.

Run:  uv run python scripts/turbine_seed_decomposition.py > logs/turbine_seed_decomp.log 2>&1
"""

import json
import logging
import os
import shutil
import tempfile
import warnings

from pathlib import Path

import numpy as np
import pytorch_lightning as L
import torch

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from scripts.turbine_label_efficiency_deep import (
    BATCH,
    D_MODEL,
    FEATURES,
    LR,
    N_HEADS,
    N_LAYERS,
    TEST_YEARS,
    TRAIN_YEARS,
    load,
    loader,
    pretrain,
)
from trea.training.single_row import (
    SingleRowRegressor,
    build_trainer,
    load_best,
    transfer_encoder,
)
from trea.utils.single_row_data import SingleRowConfig


warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")
logging.disable(logging.INFO)

FRACTIONS = [0.01, 0.05, 0.10]
SUBSET_SEEDS = [0, 1, 2]
INIT_SEEDS = [10, 11, 12, 13, 14]
CKDIR = os.path.join(tempfile.gettempdir(), "trea_seed_decomp_ckpts")
OUT = Path(__file__).resolve().parent.parent / "logs" / "turbine_seed_decomp.json"


def finetune_predict(fit_df, val_df, te, cfg_reg, tag, tscale, mtm=None):
    reg = SingleRowRegressor(
        cfg_reg, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, lr=LR
    )
    if mtm is not None:
        transfer_encoder(mtm, reg)
    ckdir = os.path.join(CKDIR, tag)
    trainer, ckpt = build_trainer(max_epochs=100, patience=8, ckpt_dir=ckdir)
    trainer.fit(
        reg,
        train_dataloaders=loader(fit_df, cfg_reg, shuffle=True),
        val_dataloaders=loader(val_df, cfg_reg),
    )
    model = load_best(reg, ckpt)
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for b in loader(te, cfg_reg):
            preds.append(model.predict_value(b).cpu().numpy())
            ys.append(b.target.cpu().numpy())
    shutil.rmtree(ckdir, ignore_errors=True)
    p, y = np.concatenate(preds).ravel(), np.concatenate(ys).ravel()
    return p, y


def xgb_predict(Xfit, yfit, Xte, seed):
    m = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=seed,
    )
    m.fit(Xfit, yfit)
    return m.predict(Xte)


def rmse(y, p, tscale):
    return float(np.sqrt(mean_squared_error(y, p))) * tscale


def main():
    tr, te, tscale = load()
    cfg_reg = SingleRowConfig.generate(tr, target="target")
    cfg_mtm = SingleRowConfig.generate(tr.drop(columns=["target"]))
    Xtr_all = tr[FEATURES].to_numpy(np.float32)
    ytr_all = tr["target"].to_numpy(np.float32)
    Xte = te[FEATURES].to_numpy(np.float32)
    yte = te["target"].to_numpy(np.float32)
    print(
        f"TEMPORAL split | train {len(tr):,} ({TRAIN_YEARS}) / test {len(te):,} "
        f"({TEST_YEARS}) | target std {tscale:.2f} | subsets={SUBSET_SEEDS} "
        f"inits={INIT_SEEDS} | batch {BATCH}",
        flush=True,
    )
    print("Pretraining encoder ONCE on all unlabeled train rows...", flush=True)
    L.seed_everything(42, verbose=False)
    mtm = pretrain(tr, cfg_mtm)

    ARMS = ("xgb", "scratch", "pre")
    results = {}
    for frac in FRACTIONS:
        n_lab = max(30, int(round(len(tr) * frac)))
        # per arm: {subset_seed: [rmse per init]}, plus ensemble rmse per subset
        per = {a: {} for a in ARMS}
        ens = {a: {} for a in ARMS}
        for s in SUBSET_SEEDS:
            rng = np.random.RandomState(s)
            idx = rng.choice(len(tr), size=n_lab, replace=False)
            sub = tr.iloc[idx]
            fit_df, val_df = train_test_split(sub, test_size=0.2, random_state=s)
            preds = {a: [] for a in ARMS}
            for i in INIT_SEEDS:
                p_x = xgb_predict(Xtr_all[idx], ytr_all[idx], Xte, i)
                preds["xgb"].append(p_x)
                L.seed_everything(i, verbose=False)
                p_s, _ = finetune_predict(
                    fit_df, val_df, te, cfg_reg, f"scr_{frac}_{s}_{i}", tscale
                )
                preds["scratch"].append(p_s)
                L.seed_everything(i, verbose=False)
                p_p, _ = finetune_predict(
                    fit_df, val_df, te, cfg_reg, f"pre_{frac}_{s}_{i}", tscale, mtm=mtm
                )
                preds["pre"].append(p_p)
                print(
                    f"  frac {frac} subset {s} init {i}: xgb {rmse(yte, p_x, tscale):.3f} "
                    f"scratch {rmse(yte, p_s, tscale):.3f} pre {rmse(yte, p_p, tscale):.3f}",
                    flush=True,
                )
            for a in ARMS:
                per[a][s] = [rmse(yte, p, tscale) for p in preds[a]]
                ens[a][s] = rmse(yte, np.mean(preds[a], axis=0), tscale)

        summary = {}
        print(
            f"\n=== frac {frac} (n_lab={n_lab:,}) ===\n"
            f"{'arm':>8} | {'mean':>7} {'between':>8} {'within':>8} | "
            f"{'ensemble':>9} {'single':>7}",
            flush=True,
        )
        for a in ARMS:
            mat = np.array([per[a][s] for s in SUBSET_SEEDS])  # [subsets, inits]
            subset_means = mat.mean(axis=1)
            between = float(subset_means.std())
            within = float(mat.std(axis=1).mean())
            ens_mean = float(np.mean([ens[a][s] for s in SUBSET_SEEDS]))
            single_mean = float(mat.mean())
            summary[a] = {
                "matrix": mat.tolist(),
                "mean": single_mean,
                "between_subset_std": between,
                "within_subset_std": within,
                "ensemble_rmse": ens_mean,
            }
            print(
                f"{a:>8} | {single_mean:>7.3f} {between:>8.3f} {within:>8.3f} | "
                f"{ens_mean:>9.3f} {single_mean:>7.3f}",
                flush=True,
            )
        results[str(frac)] = {"n_lab": n_lab, **summary}
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(results, indent=2))

    print("\n=== READ-OUT ===", flush=True)
    for frac, r in results.items():
        for a in ARMS:
            b, w = r[a]["between_subset_std"], r[a]["within_subset_std"]
            dom = "MODEL(init)" if w > b else "DATA(subset)"
            print(
                f"  frac {frac} {a:>8}: between {b:.3f} vs within {w:.3f} -> {dom}; "
                f"single {r[a]['mean']:.3f} -> ensemble {r[a]['ensemble_rmse']:.3f}",
                flush=True,
            )
    print(f"\nresults -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
