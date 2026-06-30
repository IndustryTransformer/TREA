"""Schema / semantic-column transfer test -- the one thing trees structurally cannot do.

A model trained on Plant A's sensor schema is applied to Plant B, which reports the
*same physical quantities under different names and column order*. The question: do
text-derived (semantic) column embeddings let the encoder reuse what it learned on A
when B's few labels are scarce -- where an index embedding (and any tree) cannot,
because they have no notion that B's "outdoor intake air temperature" is A's "AT"?

Construction (controlled schema shift; isolates the naming variable, no distribution
shift between the two plants' physics):
  - Turbine NOx rows are split disjointly into Plant A (pretrain) and Plant B (target).
  - Plant A keeps canonical sensor descriptions. Plant B uses *paraphrased* descriptions
    of the SAME sensors, with renamed tag codes in shuffled order. The single-row model
    keys on column *identity*, not position, so only name *meaning* can align A<->B.
  - Both plants z-score every sensor on their own train, so a given physical sensor is
    ~N(0,1) in both -> value scale is comparable; only the schema differs.

Arms (test RMSE on Plant B's held-out rows, NOx units, mean over seeds vs B-label budget):
  scratch_index     : index-embedder regressor, trained on B's labels only
  scratch_semantic  : semantic-embedder regressor, trained on B's labels only
  transfer_index    : pretrain index regressor on A (full), copy the attention stack
                      (NOT the column identities -- index has no A<->B name map) -> B
  transfer_semantic : pretrain semantic regressor on A (full), copy the attention stack
                      AND the learned name->d_model projection -> B
  xgb               : XGBoost on B's labels only (the tree bar; cannot use Plant A)

`transfer_index` is the *charitable* index control: it still gets A's pretrained
attention/value weights, just not the (untransferable) column identities. So any gap
transfer_semantic -> transfer_index is attributable to semantic name correspondence,
not to the shared backbone.

== PRE-REGISTERED success / kill criterion (locked before results) ==
Averaged over seeds, at Plant-B label fractions <= 10%, semantic column transfer is
VALIDATED iff ALL hold:
  (a) transfer_semantic < scratch_semantic           (pretraining on A actually helps B)
  (b) transfer_semantic < transfer_index             (the help is name-based, not just the
                                                       transferred attention stack)
  (c) transfer_semantic < xgb at the smallest fraction (beats the tree where trees cannot
                                                       touch Plant A at all)
KILL the semantic-column bet if (a) or (b) fails at every low-label fraction -> column
names carry no transferable signal on this pair and the idea does not earn its complexity.

Run (training script -- progress bars are off; ask before running, it trains ~70 models):
    uv run python scripts/schema_transfer.py
    uv run python scripts/schema_transfer.py --smoke          # fast plumbing check
    uv run python scripts/schema_transfer.py --seeds 5 --fractions 0.01 0.05 0.1 1.0
"""

import argparse
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
from xgboost import XGBRegressor

from trea.models.column_embeddings import IndexColumnEmbedder, SemanticColumnEmbedder
from trea.training.single_row import (
    SingleRowRegressor,
    build_trainer,
    copy_encoder_weights,
    load_best,
)
from trea.utils.single_row_data import SingleRowConfig, TabularDS, tabular_collate_fn


warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
torch.set_float32_matmul_precision("medium")

from torch.utils.data import DataLoader  # noqa: E402


D_MODEL, N_HEADS, N_LAYERS, LR, BATCH = 128, 4, 4, 8e-4, 512
DATA = str(Path(__file__).resolve().parent.parent / "data" / "nox" / "*.csv")
CKDIR = os.path.join(tempfile.gettempdir(), "trea_schema_transfer_ckpts")

# Canonical descriptions for Plant A's sensors (UCI Gas Turbine CO/NOx feature glossary).
A_DESCR = {
    "AT": "ambient air temperature in degrees celsius",
    "AP": "ambient barometric pressure in millibar",
    "AH": "ambient relative humidity in percent",
    "AFDP": "air filter differential pressure in millibar",
    "GTEP": "gas turbine exhaust pressure in millibar",
    "TIT": "turbine inlet temperature in degrees celsius",
    "TAT": "turbine after temperature in degrees celsius",
    "TEY": "turbine energy yield in megawatt hours",
    "CDP": "compressor discharge pressure in millibar",
}
# Plant B: same physical sensors, renamed tag codes + PARAPHRASED descriptions. The codes
# are meaningless; only the descriptions carry semantics, and they never reuse A's wording.
B_SCHEMA = {  # A_code -> (B_tag_code, B_description)
    "AT": ("TAG_017", "outdoor intake air temperature, deg c"),
    "AP": ("TAG_004", "local atmospheric pressure, mbar"),
    "AH": ("TAG_022", "outside air moisture content as relative humidity percent"),
    "AFDP": ("TAG_009", "pressure drop across the inlet air filter, mbar"),
    "GTEP": ("TAG_031", "exhaust gas back pressure at the gas turbine outlet, mbar"),
    "TIT": ("TAG_011", "combustor outlet gas temperature entering the turbine, deg c"),
    "TAT": ("TAG_028", "turbine exit gas temperature after the final stage, deg c"),
    "TEY": ("TAG_002", "net electrical power generated by the turbine, megawatt hours"),
    "CDP": ("TAG_019", "compressor outlet discharge pressure, mbar"),
}
A_CODES = list(A_DESCR)
TARGET = "target"


def load_turbine():
    df = pd.concat([pd.read_csv(f) for f in glob.glob(DATA)], ignore_index=True)
    df.columns = [c.upper() for c in df.columns]
    if "CO" in df.columns:
        df = df.drop(columns=["CO"])
    df = df.rename(columns={"NOX": TARGET})
    return df[A_CODES + [TARGET]]


def standardize(train, *others, cols):
    """Z-score `cols` (and target) using train statistics; return scaled copies + tscale."""
    mu, sd = train[cols].mean(), train[cols].std().replace(0, 1)
    tmu, tsd = train[TARGET].mean(), train[TARGET].std() or 1.0

    def scale(d):
        d = d.copy()
        d[cols] = (d[cols] - mu) / sd
        d[TARGET] = (d[TARGET] - tmu) / tsd
        return d

    return (scale(train), *[scale(o) for o in others], float(tsd))


def make_plant_b(df_a_rows: pd.DataFrame, shuffle_seed: int = 7):
    """Relabel Plant A columns to Plant B's renamed, reordered schema. Returns (df, descr)."""
    rename = {a: B_SCHEMA[a][0] for a in A_CODES}
    descr = {B_SCHEMA[a][0]: B_SCHEMA[a][1] for a in A_CODES}
    df = df_a_rows.rename(columns=rename)
    b_codes = [B_SCHEMA[a][0] for a in A_CODES]
    perm = np.random.RandomState(shuffle_seed).permutation(len(b_codes))
    order = [b_codes[i] for i in perm]  # plain str, not np.str_
    return df[order + [TARGET]], descr, order


def loader(df, cfg, shuffle=False):
    return DataLoader(
        TabularDS(df, cfg),
        batch_size=BATCH,
        shuffle=shuffle,
        collate_fn=tabular_collate_fn,
    )


def make_embedder(strategy, descr, feat_cols):
    names = [descr[c] for c in feat_cols]
    if strategy == "index":
        return IndexColumnEmbedder(names, D_MODEL)
    return SemanticColumnEmbedder(names, D_MODEL)


def fit_regressor(
    fit_df, val_df, cfg, descr, strategy, tag, epochs, patience, src_encoder=None
):
    """Train a SingleRowRegressor; optionally warm-start its encoder from `src_encoder`."""
    embedder = make_embedder(strategy, descr, cfg.numeric_col_tokens)
    reg = SingleRowRegressor(
        cfg,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        lr=LR,
        col_embedder=embedder,
        column_descriptions=descr,
    )
    if src_encoder is not None:
        # Semantic moves the name->d_model projection (the actual transfer); index does
        # not (its column identities have no A<->B correspondence).
        copy_encoder_weights(
            src_encoder,
            reg.model.tabular_encoder,
            include_col_embedder=(strategy == "semantic"),
        )
    trainer, ckpt = build_trainer(
        max_epochs=epochs, patience=patience, ckpt_dir=os.path.join(CKDIR, tag)
    )
    trainer.fit(
        reg,
        train_dataloaders=loader(fit_df, cfg, shuffle=True),
        val_dataloaders=loader(val_df, cfg),
    )
    return load_best(reg, ckpt)


@torch.no_grad()
def rmse(model, df_eval, cfg, tscale):
    model.eval()
    preds, ys = [], []
    for b in loader(df_eval, cfg):
        preds.append(model(b).cpu().numpy())
        ys.append(b.target.cpu().numpy())
    p, y = np.concatenate(preds).ravel(), np.concatenate(ys).ravel()
    return float(np.sqrt(mean_squared_error(y, p))) * tscale


def xgb_rmse(fit_df, test_df, feat_cols, tscale, seed):
    m = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=seed,
    )
    m.fit(fit_df[feat_cols].to_numpy(), fit_df[TARGET].to_numpy())
    pred = m.predict(test_df[feat_cols].to_numpy())
    return float(np.sqrt(mean_squared_error(test_df[TARGET].to_numpy(), pred))) * tscale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument(
        "--fractions", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.1, 0.25, 1.0]
    )
    ap.add_argument("--smoke", action="store_true", help="tiny fast plumbing check")
    a = ap.parse_args()
    if a.smoke:
        a.seeds, a.fractions = 1, [0.1, 1.0]
    seeds = list(range(a.seeds))
    pre_epochs, pre_pat = (3, None) if a.smoke else (60, 8)
    ft_epochs, ft_pat = (3, 2) if a.smoke else (100, 8)

    raw = load_turbine()
    if a.smoke:
        raw = raw.sample(1500, random_state=0).reset_index(drop=True)

    # Disjoint rows: Plant A (pretrain schema) vs Plant B (target schema).
    a_rows, b_rows = train_test_split(raw, test_size=0.5, random_state=42)
    df_b_relabeled, B_DESCR, b_cols = make_plant_b(b_rows)

    # Plant A standardized (single train/val for the one-time pretrain).
    a_tr, a_val, _ = standardize(
        *train_test_split(a_rows, test_size=0.1, random_state=0), cols=A_CODES
    )
    cfg_a = SingleRowConfig.generate(a_tr, target=TARGET)

    # Plant B standardized: split into train-pool and held-out test ONCE; fractions
    # subsample the train-pool, test is fixed across all arms.
    b_pool_raw, b_test_raw = train_test_split(
        df_b_relabeled, test_size=0.3, random_state=42
    )
    b_pool, b_test, tscale_b = standardize(b_pool_raw, b_test_raw, cols=b_cols)
    cfg_b = SingleRowConfig.generate(b_pool, target=TARGET)

    print(f"Plant A: {len(a_rows):,} rows, schema {A_CODES}", flush=True)
    print(
        f"Plant B: {len(df_b_relabeled):,} rows, schema {b_cols} (renamed+reordered)",
        flush=True,
    )
    print(
        f"  B train-pool {len(b_pool):,} / test {len(b_test):,} | NOx std {tscale_b:.2f} | "
        f"seeds={seeds} | fractions={a.fractions}\n",
        flush=True,
    )

    # --- One-time pretrain on Plant A: a semantic and an index regressor. ---
    print(
        "Pretraining on Plant A (full labels): semantic + index encoders...", flush=True
    )
    L.seed_everything(42, verbose=False)
    pre_sem = fit_regressor(
        a_tr, a_val, cfg_a, A_DESCR, "semantic", "preA_sem", pre_epochs, pre_pat
    )
    L.seed_everything(42, verbose=False)
    pre_idx = fit_regressor(
        a_tr, a_val, cfg_a, A_DESCR, "index", "preA_idx", pre_epochs, pre_pat
    )
    enc_sem = pre_sem.model.tabular_encoder
    enc_idx = pre_idx.model.tabular_encoder
    r_pre_on_b = rmse(
        pre_sem, b_test, cfg_b, tscale_b
    )  # zero-shot A->B (no B finetuning)
    print(
        f"  zero-shot semantic A->B (no B labels): RMSE {r_pre_on_b:.3f}\n", flush=True
    )

    arms = [
        "scratch_index",
        "scratch_semantic",
        "transfer_index",
        "transfer_semantic",
        "xgb",
    ]
    res = {f: {arm: [] for arm in arms} for f in a.fractions}

    for frac in a.fractions:
        n_lab = max(BATCH if not a.smoke else 64, int(round(len(b_pool) * frac)))
        n_lab = min(n_lab, len(b_pool))
        for seed in seeds:
            L.seed_everything(seed, verbose=False)
            pool = b_pool.sample(n_lab, random_state=seed)
            fit_df, val_df = train_test_split(pool, test_size=0.2, random_state=seed)
            runs = {
                "scratch_index": dict(strategy="index", src_encoder=None),
                "scratch_semantic": dict(strategy="semantic", src_encoder=None),
                "transfer_index": dict(strategy="index", src_encoder=enc_idx),
                "transfer_semantic": dict(strategy="semantic", src_encoder=enc_sem),
            }
            for arm, kw in runs.items():
                m = fit_regressor(
                    fit_df,
                    val_df,
                    cfg_b,
                    B_DESCR,
                    tag=f"{arm}_{frac}_{seed}",
                    epochs=ft_epochs,
                    patience=ft_pat,
                    **kw,
                )
                res[frac][arm].append(rmse(m, b_test, cfg_b, tscale_b))
            res[frac]["xgb"].append(xgb_rmse(fit_df, b_test, b_cols, tscale_b, seed))
        row = " | ".join(
            f"{arm.split('_')[0][:3]}_{arm.split('_')[-1][:3]} {np.mean(res[frac][arm]):.3f}"
            for arm in arms
        )
        print(f"  frac {frac:>5.2f} (n={n_lab:>5}): {row}", flush=True)

    # --- Report + pre-registered verdict ---
    print(
        "\n=== Plant-B test RMSE (NOx units, mean +/- std over seeds) ===", flush=True
    )
    header = f"{'frac':>6} " + " ".join(f"{arm:>18}" for arm in arms)
    print(header, flush=True)
    for frac in a.fractions:
        cells = " ".join(
            f"{np.mean(res[frac][arm]):>9.3f}+-{np.std(res[frac][arm]):<6.3f}"
            for arm in arms
        )
        print(f"{frac:>6.2f} {cells}", flush=True)

    low = [f for f in a.fractions if f <= 0.10]
    smallest = min(a.fractions)

    def mean(frac, arm):
        return float(np.mean(res[frac][arm]))

    a_ok = any(mean(f, "transfer_semantic") < mean(f, "scratch_semantic") for f in low)
    b_ok = any(mean(f, "transfer_semantic") < mean(f, "transfer_index") for f in low)
    c_ok = mean(smallest, "transfer_semantic") < mean(smallest, "xgb")
    print("\n=== PRE-REGISTERED VERDICT (low-label fractions <= 0.10) ===", flush=True)
    print(
        f"  (a) transfer_semantic < scratch_semantic : {'PASS' if a_ok else 'FAIL'}",
        flush=True,
    )
    print(
        f"  (b) transfer_semantic < transfer_index   : {'PASS' if b_ok else 'FAIL'}",
        flush=True,
    )
    print(
        f"  (c) transfer_semantic < xgb @ frac={smallest:.2f}: {'PASS' if c_ok else 'FAIL'}",
        flush=True,
    )
    verdict = (
        "VALIDATED: semantic column transfer earns its keep"
        if (a_ok and b_ok)
        else "KILL: names carry no transferable signal on this pair"
    )
    print(f"\n  {'PASS' if (a_ok and b_ok) else 'MISS'} -> {verdict}", flush=True)


if __name__ == "__main__":
    main()
