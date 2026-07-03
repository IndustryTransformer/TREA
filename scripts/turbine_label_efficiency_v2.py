"""Turbine NOx label-efficiency curve, protocol v2 (pre-registered LESSONS.md 3c).

v1 (turbine_label_efficiency_deep.py) FAILED as written; three protocol defects were
identified before its final verdict. v2 fixes exactly those and is FINAL for turbine
(no further protocol revisions after seeing results):

  1. Model selection for ALL deep arms on a TEMPORAL val slice (labeled val rows are
     sampled from the last 10% of train time, i.e. late 2013) instead of an i.i.d.
     subsample. Val rows count against the label budget; trees get the same total
     budget as fit rows.
  2. New arm `pre_target`: MTM pretraining on train-period rows PLUS unlabeled
     TEST-period feature rows (transductive; no labels used). This is the industrial
     deployment story: the plant has the drifted sensor data, just not its labels.
  3. At fracs <= 0.05 each pretrained arm runs a frozen-encoder probe (encoder frozen,
     head-only, lr 1e-3) AND a low-LR finetune (lr 8e-5); the arm's score is whichever
     wins on the temporal val slice (never chosen on test).

Arms: xgb / cat (trees, the bar), scratch, pre (SSL on train period only),
pre_target (SSL on train + unlabeled test-period features).

5 seeds at fracs <= 0.10, 3 seeds above. RMSE in original NOx units.

Pre-registered v2 success bar: pre_target beats best-tree at >=3 of the 4 fracs
<= 0.10 AND never loses to scratch at those fracs. Anything less: turbine stays a
tree benchmark and the deep line's continuation rests on the 3W binary gate.

Run:  uv run python scripts/turbine_label_efficiency_v2.py
"""

import json
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

D_MODEL, N_HEADS, N_LAYERS, BATCH = 128, 4, 4, 512
LR_BASE, LR_PROBE, LR_LOWFT = 8e-4, 1e-3, 8e-5
FRACTIONS = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
LOW_FRACS = [f for f in FRACTIONS if f <= 0.10]  # 5 seeds + probe/low-ft protocol
PROBE_FRACS = [f for f in FRACTIONS if f <= 0.05]
SEEDS_LOW, SEEDS_HIGH = [0, 1, 2, 3, 4], [0, 1, 2]
TEMPORAL_VAL_FRAC = 0.10  # last 10% of train time = the val pool (late 2013)
NOX_DIR = Path(__file__).resolve().parent.parent / "data" / "nox"
FEATURES = ["at", "ap", "ah", "afdp", "gtep", "tit", "tat", "tey", "cdp"]
TRAIN_YEARS = [2011, 2012, 2013]
TEST_YEARS = [2014, 2015]
CKDIR = os.path.join(tempfile.gettempdir(), "trea_labeleff_v2_ckpts")
RESULTS = Path(__file__).resolve().parent.parent / "logs" / "turbine_labeleff_v2.json"
ARMS = ("xgb", "cat", "scratch", "pre", "pre_target")


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
        d["cat_column"] = "category"  # constant categorical, mirrors v1
    return tr.reset_index(drop=True), te.reset_index(drop=True), float(tsd)


def loader(df, cfg, masked=False, shuffle=False):
    fn = masked_tabular_collate_fn if masked else tabular_collate_fn
    return DataLoader(
        TabularDS(df, cfg), batch_size=BATCH, shuffle=shuffle, collate_fn=fn
    )


def pretrain(df_unlabeled, cfg_mtm, tag):
    """MTM-pretrain once on the given unlabeled rows (i.i.d. 90/10 val for SSL)."""
    n = len(df_unlabeled)
    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    pt_tr = df_unlabeled.iloc[perm[: int(n * 0.9)]]
    pt_val = df_unlabeled.iloc[perm[int(n * 0.9) :]]
    mtm = SingleRowMTM(
        cfg_mtm, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, lr=LR_BASE
    )
    trainer, _ = build_trainer(max_epochs=60, patience=None)
    trainer.fit(
        mtm,
        train_dataloaders=loader(pt_tr, cfg_mtm, masked=True, shuffle=True),
        val_dataloaders=loader(pt_val, cfg_mtm, masked=True),
    )
    print(f"  [pretrain:{tag}] done on {n:,} unlabeled rows", flush=True)
    return mtm


def fit_deep(fit_df, val_df, cfg_reg, tag, lr, mtm=None, freeze_encoder=False):
    reg = SingleRowRegressor(
        cfg_reg, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, lr=lr
    )
    if mtm is not None:
        transfer_encoder(mtm, reg)
    if freeze_encoder:
        for p in reg.model.tabular_encoder.parameters():
            p.requires_grad = False
    trainer, ckpt = build_trainer(max_epochs=100, ckpt_dir=os.path.join(CKDIR, tag))
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
        preds.append(model.predict_value(b).cpu().numpy())
        ys.append(b.target.cpu().numpy())
    p, y = np.concatenate(preds).ravel(), np.concatenate(ys).ravel()
    return float(np.sqrt(mean_squared_error(y, p))) * tscale


def pretrained_arm_rmse(fit_df, val_df, te, cfg_reg, tscale, tag, frac, mtm):
    """Pretrained arm under the v2 protocol.

    fracs <= 0.05: frozen-encoder probe (lr 1e-3) vs low-LR finetune (lr 8e-5),
    winner picked on the TEMPORAL val slice, never on test. Above: standard finetune.
    """
    if frac in PROBE_FRACS:
        probe = fit_deep(
            fit_df, val_df, cfg_reg, f"{tag}_probe", LR_PROBE, mtm, freeze_encoder=True
        )
        lowft = fit_deep(fit_df, val_df, cfg_reg, f"{tag}_lowft", LR_LOWFT, mtm)
        pv, lv = (
            deep_rmse(probe, val_df, cfg_reg, tscale),
            deep_rmse(lowft, val_df, cfg_reg, tscale),
        )
        chosen, mode = (probe, "probe") if pv <= lv else (lowft, "lowft")
        return deep_rmse(chosen, te, cfg_reg, tscale), mode
    model = fit_deep(fit_df, val_df, cfg_reg, tag, LR_BASE, mtm)
    return deep_rmse(model, te, cfg_reg, tscale), "ft"


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
    # Temporal pools: labels for FITTING come from the first 90% of train time;
    # labels for VALIDATION (deep model selection) from the last 10% (late 2013).
    cut = int(len(tr) * (1 - TEMPORAL_VAL_FRAC))
    fit_pool, val_pool = tr.iloc[:cut], tr.iloc[cut:]
    print(
        f"TEMPORAL split | train {len(tr):,} ({TRAIN_YEARS}) -> "
        f"fit-pool {len(fit_pool):,} + temporal-val-pool {len(val_pool):,} | "
        f"test {len(te):,} ({TEST_YEARS}) | target std {tscale:.2f}",
        flush=True,
    )

    print("Pretraining TWO encoders once each (corrected MTM)...", flush=True)
    L.seed_everything(42, verbose=False)
    mtm_pre = pretrain(tr.drop(columns=["target"]), cfg_mtm, "train-period")
    L.seed_everything(42, verbose=False)
    unl = pd.concat(
        [tr.drop(columns=["target"]), te.drop(columns=["target"])], ignore_index=True
    )
    mtm_target = pretrain(unl, cfg_mtm, "train+target-period")

    header = " ".join(f"{k:>16}" for k in ARMS)
    print(f"\n{'frac':>6} {'n_lab':>7} | {header}", flush=True)
    rows = []
    for frac in FRACTIONS:
        seeds = SEEDS_LOW if frac in LOW_FRACS else SEEDS_HIGH
        n_lab = max(30, int(round(len(tr) * frac)))
        n_val = max(20, int(round(n_lab * 0.2)))
        n_fit = max(10, n_lab - n_val)
        acc = {k: [] for k in ARMS}
        modes = []
        for seed in seeds:
            rng = np.random.RandomState(seed)
            fit_df = fit_pool.iloc[
                rng.choice(len(fit_pool), size=min(n_fit, len(fit_pool)), replace=False)
            ]
            val_df = val_pool.iloc[
                rng.choice(len(val_pool), size=min(n_val, len(val_pool)), replace=False)
            ]
            # Trees get the SAME total label budget (fit + val rows) to fit on.
            lab = pd.concat([fit_df, val_df])
            Xl, yl = (
                lab[FEATURES].to_numpy(np.float32),
                lab["target"].to_numpy(np.float32),
            )
            Xte_, yte_ = (
                te[FEATURES].to_numpy(np.float32),
                te["target"].to_numpy(np.float32),
            )
            acc["xgb"].append(tree_rmse("xgb", Xl, yl, Xte_, yte_, tscale, seed))
            acc["cat"].append(tree_rmse("cat", Xl, yl, Xte_, yte_, tscale, seed))
            L.seed_everything(seed, verbose=False)
            acc["scratch"].append(
                deep_rmse(
                    fit_deep(fit_df, val_df, cfg_reg, f"scr_{frac}_{seed}", LR_BASE),
                    te,
                    cfg_reg,
                    tscale,
                )
            )
            r_pre, m_pre = pretrained_arm_rmse(
                fit_df, val_df, te, cfg_reg, tscale, f"pre_{frac}_{seed}", frac, mtm_pre
            )
            acc["pre"].append(r_pre)
            r_tgt, m_tgt = pretrained_arm_rmse(
                fit_df,
                val_df,
                te,
                cfg_reg,
                tscale,
                f"tgt_{frac}_{seed}",
                frac,
                mtm_target,
            )
            acc["pre_target"].append(r_tgt)
            modes.append(f"{m_pre}/{m_tgt}")
        stat = {k: (float(np.mean(v)), float(np.std(v))) for k, v in acc.items()}
        rows.append(
            {
                "frac": frac,
                "n_lab": n_lab,
                "seeds": len(seeds),
                "stat": stat,
                "modes": modes,
                "raw": {k: list(map(float, v)) for k, v in acc.items()},
            }
        )
        cells = " ".join(f"{stat[k][0]:>7.3f}±{stat[k][1]:<5.3f}" for k in ARMS)
        print(
            f"{frac:>6.2f} {n_lab:>7,} | {cells}  [{','.join(set(modes))}]", flush=True
        )
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(rows, indent=2))

    # ---- pre-registered v2 verdict ----
    print("\n=== V2 VERDICT (pre-registered, FINAL for turbine) ===", flush=True)
    wins_low, never_lose_scratch = 0, True
    for r in rows:
        frac, stat = r["frac"], r["stat"]
        best_tree = min(stat["xgb"][0], stat["cat"][0])
        tgt = stat["pre_target"][0]
        edge = best_tree - tgt
        tag = "PRE_TARGET WINS" if edge > 0 else "tree wins"
        print(
            f"  frac {frac:>4} (n={r['n_lab']:>6,}): pre_target {tgt:.3f} vs "
            f"best-tree {best_tree:.3f} vs scratch {stat['scratch'][0]:.3f} "
            f"-> {tag} (Δ {edge:+.3f})",
            flush=True,
        )
        if frac <= 0.10:
            if edge > 0:
                wins_low += 1
            if tgt > stat["scratch"][0]:
                never_lose_scratch = False
    ok = wins_low >= 3 and never_lose_scratch
    print(
        f"\n  V2 BAR: pre_target beats best-tree on >=3 of the 4 fracs <=10% "
        f"(got {wins_low}/4) AND never loses to scratch there "
        f"({'held' if never_lose_scratch else 'violated'}) -> "
        f"{'PASS' if ok else 'FAIL'}",
        flush=True,
    )
    print(f"\nresults -> {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
