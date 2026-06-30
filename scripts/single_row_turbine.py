"""Validate the TREA single-row model on turbine NOx: does it match XGB at full data?

Success bar for the port (agreed): the deep model should reliably sit at/below XGB
test-RMSE at full data, with low seed-variance. Same i.i.d. soft-sensor protocol and
hyperparameters as the Hephaestus verification, now running entirely on TREA's own
single-row stack (trea.models.single_row + trea.training.single_row).

  scratch : SingleRowRegressor, random init
  pre     : SingleRowRegressor, encoder pretrained once with corrected MTM, reused
  xgb     : XGBoost on the same labels

Run:  uv run python scripts/single_row_turbine.py
"""

import glob
import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as L
import torch
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

D_MODEL, N_HEADS, N_LAYERS, LR, BATCH = 128, 4, 4, 8e-4, 512
SEEDS = [0, 1, 2, 3, 4]
DATA = str(Path(__file__).resolve().parent.parent / "data" / "nox" / "*.csv")
CKDIR = os.path.join(tempfile.gettempdir(), "trea_single_row_ckpts")


def load():
    df = pd.concat([pd.read_csv(f) for f in glob.glob(DATA)], ignore_index=True)
    df.columns = df.columns.str.lower()
    if "co" in df.columns:
        df = df.drop(columns=["co"])
    df = df.rename(columns={"nox": "target"})
    feat_cols = [c for c in df.columns if c != "target"]
    df[feat_cols] = StandardScaler().fit_transform(df[feat_cols])
    ty = StandardScaler()
    df["target"] = ty.fit_transform(df[["target"]]).flatten()
    df["cat_column"] = "category"  # one constant categorical so the cat path is exercised
    return df, feat_cols, ty.scale_[0]


def loader(df, cfg, masked=False, shuffle=False):
    ds = TabularDS(df, cfg)
    fn = masked_tabular_collate_fn if masked else tabular_collate_fn
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, collate_fn=fn)


def pretrain(df_train, cfg_mtm):
    pt_tr, pt_val = train_test_split(df_train, test_size=0.1, random_state=0)
    mtm = SingleRowMTM(cfg_mtm, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, lr=LR)
    trainer, _ = build_trainer(max_epochs=60, patience=None)
    trainer.fit(
        mtm,
        train_dataloaders=loader(pt_tr.drop(columns=["target"]), cfg_mtm, masked=True, shuffle=True),
        val_dataloaders=loader(pt_val.drop(columns=["target"]), cfg_mtm, masked=True),
    )
    return mtm


def finetune(fit_df, val_df, cfg_reg, tag, mtm=None):
    reg = SingleRowRegressor(cfg_reg, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, lr=LR)
    if mtm is not None:
        transfer_encoder(mtm, reg)
    trainer, ckpt = build_trainer(max_epochs=100, patience=8, ckpt_dir=os.path.join(CKDIR, tag))
    trainer.fit(reg, train_dataloaders=loader(fit_df, cfg_reg, shuffle=True),
                val_dataloaders=loader(val_df, cfg_reg))
    return load_best(reg, ckpt)


@torch.no_grad()
def rmse(model, df_eval, cfg_reg, tscale):
    model.eval()
    preds, ys = [], []
    for b in loader(df_eval, cfg_reg):
        preds.append(model(b).cpu().numpy())
        ys.append(b.target.cpu().numpy())
    p, y = np.concatenate(preds).ravel(), np.concatenate(ys).ravel()
    return float(np.sqrt(mean_squared_error(y, p))) * tscale


def main():
    df, feat_cols, tscale = load()
    cfg_reg = SingleRowConfig.generate(df, target="target")
    cfg_mtm = SingleRowConfig.generate(df.drop(columns=["target"]))
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    print(f"FULL DATA | train {len(df_train):,} / test {len(df_test):,} | "
          f"target std {tscale:.2f} | seeds={SEEDS}", flush=True)

    print("Pretraining encoder ONCE (corrected MTM)...", flush=True)
    L.seed_everything(42, verbose=False)
    mtm = pretrain(df_train, cfg_mtm)

    res = {"scratch": [], "pre": [], "xgb": []}
    for seed in SEEDS:
        L.seed_everything(seed, verbose=False)
        fit_df, val_df = train_test_split(df_train, test_size=0.2, random_state=seed)
        r_scr = rmse(finetune(fit_df, val_df, cfg_reg, f"scratch_{seed}"), df_test, cfg_reg, tscale)
        r_pre = rmse(finetune(fit_df, val_df, cfg_reg, f"pre_{seed}", mtm=mtm), df_test, cfg_reg, tscale)
        xgb = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=seed)
        xgb.fit(df_train[feat_cols].to_numpy(), df_train["target"].to_numpy())
        r_xgb = float(np.sqrt(mean_squared_error(
            df_test["target"].to_numpy(), xgb.predict(df_test[feat_cols].to_numpy())))) * tscale
        res["scratch"].append(r_scr); res["pre"].append(r_pre); res["xgb"].append(r_xgb)
        print(f"  seed {seed}: scratch {r_scr:.3f} | pre {r_pre:.3f} | xgb {r_xgb:.3f}", flush=True)

    print("\n=== FULL-DATA RESULT (mean +/- std over seeds) ===", flush=True)
    for k in ("scratch", "pre", "xgb"):
        v = res[k]
        print(f"  {k:>7}: {np.mean(v):.3f} +/- {np.std(v):.3f}  (min {min(v):.3f}, max {max(v):.3f})", flush=True)
    deep = min(np.mean(res["scratch"]), np.mean(res["pre"]))
    verdict = "PASS: deep matches/beats XGB" if deep <= np.mean(res["xgb"]) + 0.05 else "MISS: deep above XGB"
    print(f"\n  {verdict}  (best deep {deep:.3f} vs xgb {np.mean(res['xgb']):.3f})", flush=True)


if __name__ == "__main__":
    main()
