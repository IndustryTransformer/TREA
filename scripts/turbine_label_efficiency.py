"""Turbine NOx label-efficiency: XGBoost baseline (the bar the deep model must beat).

UCI Gas Turbine CO/NOx dataset (data/nox/gt_20xx.csv), single turbine, 5 years.
Task: predict NOX (regression). Honest split is TEMPORAL: train 2011-2013, test
2014-2015 (no random shuffling — temporally adjacent rows would leak).

This establishes XGBoost test-RMSE as a function of how many labeled training rows
are available. The thesis (per docs/LESSONS.md §3): a pretrained->finetuned model
should beat this at low label counts (≤10%). This script is the baseline half.

Run:
    uv run python scripts/turbine_label_efficiency.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

NOX_DIR = Path(__file__).resolve().parent.parent / "data" / "nox"
TARGET = "NOX"
# 9 operational variables; exclude CO (co-emission, not a real predictor at inference).
FEATURES = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP"]
TRAIN_YEARS = [2011, 2012, 2013]
TEST_YEARS = [2014, 2015]
FRACTIONS = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
SEEDS = [0, 1, 2, 3, 4]


def load():
    frames = []
    for f in sorted(NOX_DIR.glob("gt_*.csv")):
        year = int(f.stem.split("_")[1])
        d = pd.read_csv(f)
        d["year"] = year
        frames.append(d)
    if not frames:
        sys.exit(f"No gt_*.csv found in {NOX_DIR}")
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["temporal", "random"], default="temporal",
                    help="temporal = honest (train early years, test late); "
                         "random = leaky for time series (same-distribution shuffle).")
    a = ap.parse_args()

    df = load()
    if a.split == "temporal":
        tr = df[df.year.isin(TRAIN_YEARS)]
        te = df[df.year.isin(TEST_YEARS)]
    else:
        from sklearn.model_selection import train_test_split

        tr, te = train_test_split(df, test_size=0.4, random_state=42)
    print(f"SPLIT = {a.split}")

    # Standardize features on train only.
    mu, sd = tr[FEATURES].mean(), tr[FEATURES].std().replace(0, 1)
    Xtr_all = ((tr[FEATURES] - mu) / sd).to_numpy(np.float32)
    ytr_all = tr[TARGET].to_numpy(np.float32)
    Xte = ((te[FEATURES] - mu) / sd).to_numpy(np.float32)
    yte = te[TARGET].to_numpy(np.float32)

    print(f"Turbine NOx | train {len(ytr_all):,} rows ({TRAIN_YEARS}) | "
          f"test {len(yte):,} rows ({TEST_YEARS}) | {len(FEATURES)} features")
    print(f"NOX test range [{yte.min():.1f}, {yte.max():.1f}], std {yte.std():.2f}\n")
    print(f"{'frac':>6} {'n_labels':>9} {'XGB test-RMSE (mean±std)':>26}")

    results = {}
    for frac in FRACTIONS:
        rmses = []
        n_lab = max(20, int(round(len(ytr_all) * frac)))
        for seed in SEEDS:
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(ytr_all), size=n_lab, replace=False)
            model = XGBRegressor(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=seed,
            )
            model.fit(Xtr_all[idx], ytr_all[idx])
            rmse = float(np.sqrt(mean_squared_error(yte, model.predict(Xte))))
            rmses.append(rmse)
        results[frac] = (np.mean(rmses), np.std(rmses), n_lab)
        print(f"{frac:>6.2f} {n_lab:>9,} {np.mean(rmses):>14.3f} ± {np.std(rmses):.3f}")

    print("\nThis is the bar. The pretrained->finetuned model must beat the low-frac rows "
          "(≤10%) to satisfy the kill criterion (docs/LESSONS.md §3).")


if __name__ == "__main__":
    main()
