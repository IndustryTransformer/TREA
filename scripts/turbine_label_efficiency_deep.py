"""Turbine NOx label-efficiency curve: trees vs from-scratch vs MTM-pretrained deep.

The thesis test (docs/LESSONS.md §3): on the HONEST temporal split, does an
MTM-pretrained -> finetuned single-row model beat the best tree at LOW label
counts (<=10%)? Self-supervised pretraining uses ALL unlabeled train rows once;
each label fraction then finetunes on a labeled subsample of that same train pool.

  xgb / cat : XGBoost / CatBoost on the labeled subsample (the bar)
  scratch   : SingleRowRegressor, random init, finetuned on the subsample
  pre       : SingleRowRegressor with the pretrained encoder, finetuned on the subsample

RMSE is reported in original NOX units. Split is temporal (train 2011-2013, test
2014-2015) so temporally adjacent rows can't leak. 3 seeds.

Run:  uv run python scripts/turbine_label_efficiency_deep.py
"""

import logging
import os
import tempfile
import warnings

from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as L
import torch

from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from xgboost import XGBRegressor

from trea.training.single_row import (
    SingleRowMTM,
    SingleRowRegressor,
    build_trainer,
    load_best,
    transfer_encoder,
)
from trea.utils.single_row_data import (
    SingleRowConfig,
    TabularDS,
    masked_tabular_collate_fn,
    tabular_collate_fn,
)


warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")
logging.disable(logging.INFO)  # mute Lightning's per-fit device banner

D_MODEL, N_HEADS, N_LAYERS, LR, BATCH = 128, 4, 4, 8e-4, 512
SEEDS = [0, 1, 2]
FRACTIONS = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
NOX_DIR = Path(__file__).resolve().parent.parent / "data" / "nox"
FEATURES = ["at", "ap", "ah", "afdp", "gtep", "tit", "tat", "tey", "cdp"]
TRAIN_YEARS = [2011, 2012, 2013]
TEST_YEARS = [2014, 2015]
CKDIR = os.path.join(tempfile.gettempdir(), "trea_labeleff_ckpts")


def load():
    frames = []
    for f in sorted(NOX_DIR.glob("gt_*.csv")):
        d = pd.read_csv(f)
        d.columns = d.columns.str.lower()
        d["year"] = int(f.stem.split("_")[1])
        frames.append(d)
    df = pd.concat(frames, ignore_index=True).rename(columns={"nox": "target"})
    tr = df[df.year.isin(TRAIN_YEARS)].copy()
    te = df[df.year.isin(TEST_YEARS)].copy()
    # Standardize features + target on TRAIN only (no test leakage).
    mu, sd = tr[FEATURES].mean(), tr[FEATURES].std().replace(0, 1)
    tmu, tsd = tr["target"].mean(), tr["target"].std() or 1.0
    for d in (tr, te):
        d[FEATURES] = (d[FEATURES] - mu) / sd
        d["target"] = (d["target"] - tmu) / tsd
        d["cat_column"] = (
            "category"  # constant categorical, mirrors the port validation
        )
    return tr, te, float(tsd)


def loader(df, cfg, masked=False, shuffle=False):
    fn = masked_tabular_collate_fn if masked else tabular_collate_fn
    return DataLoader(
        TabularDS(df, cfg), batch_size=BATCH, shuffle=shuffle, collate_fn=fn
    )


def pretrain(tr, cfg_mtm):
    pt_tr, pt_val = train_test_split(tr, test_size=0.1, random_state=0)
    mtm = SingleRowMTM(
        cfg_mtm, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, lr=LR
    )
    trainer, _ = build_trainer(max_epochs=60, patience=None)
    trainer.fit(
        mtm,
        train_dataloaders=loader(
            pt_tr.drop(columns=["target"]), cfg_mtm, masked=True, shuffle=True
        ),
        val_dataloaders=loader(pt_val.drop(columns=["target"]), cfg_mtm, masked=True),
    )
    return mtm


def finetune(fit_df, val_df, cfg_reg, tag, mtm=None):
    reg = SingleRowRegressor(
        cfg_reg, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, lr=LR
    )
    if mtm is not None:
        transfer_encoder(mtm, reg)
    trainer, ckpt = build_trainer(
        max_epochs=100, patience=8, ckpt_dir=os.path.join(CKDIR, tag)
    )
    trainer.fit(
        reg,
        train_dataloaders=loader(fit_df, cfg_reg, shuffle=True),
        val_dataloaders=loader(val_df, cfg_reg),
    )
    return load_best(reg, ckpt)


@torch.no_grad()
def deep_rmse(model, df_eval, cfg_reg, tscale):
    model.eval()
    preds, ys = [], []
    for b in loader(df_eval, cfg_reg):
        preds.append(model(b).cpu().numpy())
        ys.append(b.target.cpu().numpy())
    p, y = np.concatenate(preds).ravel(), np.concatenate(ys).ravel()
    return float(np.sqrt(mean_squared_error(y, p))) * tscale


def tree_rmse(kind, Xfit, yfit, Xte, yte, tscale, seed):
    if kind == "xgb":
        m = XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=seed,
        )
    else:
        m = CatBoostRegressor(
            iterations=400,
            depth=6,
            learning_rate=0.05,
            random_seed=seed,
            thread_count=-1,
            verbose=0,
        )
    m.fit(Xfit, yfit)
    return float(np.sqrt(mean_squared_error(yte, m.predict(Xte)))) * tscale


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
        f"({TEST_YEARS}) | target std {tscale:.2f} | seeds={SEEDS}",
        flush=True,
    )

    print(
        "Pretraining encoder ONCE on all unlabeled train rows (corrected MTM)...",
        flush=True,
    )
    L.seed_everything(42, verbose=False)
    mtm = pretrain(tr, cfg_mtm)

    print(
        f"\n{'frac':>6} {'n_lab':>7} | {'xgb':>16} {'cat':>16} "
        f"{'scratch':>16} {'pre':>16}",
        flush=True,
    )
    rows = []
    for frac in FRACTIONS:
        n_lab = max(30, int(round(len(tr) * frac)))
        acc = {k: [] for k in ("xgb", "cat", "scratch", "pre")}
        for seed in SEEDS:
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(tr), size=n_lab, replace=False)
            sub = tr.iloc[idx]
            acc["xgb"].append(
                tree_rmse("xgb", Xtr_all[idx], ytr_all[idx], Xte, yte, tscale, seed)
            )
            acc["cat"].append(
                tree_rmse("cat", Xtr_all[idx], ytr_all[idx], Xte, yte, tscale, seed)
            )
            L.seed_everything(seed, verbose=False)
            fit_df, val_df = train_test_split(sub, test_size=0.2, random_state=seed)
            acc["scratch"].append(
                deep_rmse(
                    finetune(fit_df, val_df, cfg_reg, f"scr_{frac}_{seed}"),
                    te,
                    cfg_reg,
                    tscale,
                )
            )
            acc["pre"].append(
                deep_rmse(
                    finetune(fit_df, val_df, cfg_reg, f"pre_{frac}_{seed}", mtm=mtm),
                    te,
                    cfg_reg,
                    tscale,
                )
            )
        stat = {k: (float(np.mean(v)), float(np.std(v))) for k, v in acc.items()}
        rows.append((frac, n_lab, stat))
        cells = " ".join(
            f"{stat[k][0]:>7.3f}±{stat[k][1]:<5.3f}"
            for k in ("xgb", "cat", "scratch", "pre")
        )
        print(f"{frac:>6.2f} {n_lab:>7,} | {cells}", flush=True)

    # ---- verdict ----
    print("\n=== VERDICT: pretrained vs best tree, per label fraction ===", flush=True)
    low_win = True
    for frac, n_lab, stat in rows:
        best_tree = min(stat["xgb"][0], stat["cat"][0])
        pre = stat["pre"][0]
        edge = best_tree - pre  # positive = deep better (lower RMSE)
        tag = "PRE WINS" if edge > 0 else "tree wins"
        print(
            f"  frac {frac:>4} (n={n_lab:>6,}): pre {pre:.3f} vs best-tree "
            f"{best_tree:.3f}  -> {tag} (Δ {edge:+.3f})",
            flush=True,
        )
        if frac <= 0.10 and edge <= 0:
            low_win = False
    print(
        f"\n  KILL CRITERION (pretrained beats best tree at all fracs <=10%): "
        f"{'PASS' if low_win else 'FAIL'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
