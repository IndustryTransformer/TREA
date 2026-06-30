"""3W leakage check: does well-grouping change the RF baseline?

3W missingness is structural per well (whole sensor columns absent depending on the
well), and the same well has multiple instance files. A file-level split lets a model
shortcut on "which well is this." This script quantifies that by training the SAME RF
on stat features under two grouped 5-fold splits:

  - file-grouped : files are atomic but a well's files may span train/test (the current
                   benchmark setup — "leaky" wrt wells)
  - well-grouped : a well's files never span train/test (honest)

The macro-F1 gap between them is the well-leakage effect. If file-grouped >> well-grouped,
every existing 3W number (including RF's 0.92) is optimistic.

Run (use --max-files-per-class for a fast pass):
    uv run python scripts/leakage_check.py --max-files-per-class 80
    uv run python scripts/leakage_check.py            # all real WELL files
"""

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.three_w import SENSOR_COLUMNS  # noqa: E402

RAW = Path("/home/kailukowiak/Work/3W/dataset")
WELL_RE = re.compile(r"(WELL-\d+)")


def window_stats(feat: np.ndarray, w: int) -> np.ndarray:
    """[T, C] -> [n_win, C*6] nan-aware stats per window (mean/std/min/max/median/nanrate)."""
    n = (feat.shape[0] // w) * w
    if n < w:
        return np.empty((0, feat.shape[1] * 6), np.float32)
    x = feat[:n].reshape(-1, w, feat.shape[1])  # [n_win, w, C]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # all-NaN slices
        mean = np.nanmean(x, axis=1)
        std = np.nanstd(x, axis=1)
        mn = np.nanmin(x, axis=1)
        mx = np.nanmax(x, axis=1)
        med = np.nanmedian(x, axis=1)
    nanrate = np.isnan(x).mean(axis=1)
    out = np.concatenate([mean, std, mn, mx, med, nanrate], axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def window_labels(cls: np.ndarray, w: int) -> np.ndarray:
    """Majority valid label per non-overlapping window; -1 if none valid."""
    n = (len(cls) // w) * w
    lab = cls[:n].reshape(-1, w)
    out = np.full(lab.shape[0], -1, np.int64)
    for i in range(lab.shape[0]):
        v = lab[i][lab[i] >= 0]
        if len(v):
            out[i] = int(np.median(v))
    return out


def load(max_per_class: int | None, window: int):
    X, y, file_id, well_id = [], [], [], []
    fid = 0
    for cls_dir in sorted(RAW.glob("[0-9]")):
        files = sorted(cls_dir.glob("WELL-*.parquet"))
        if max_per_class:
            files = files[:max_per_class]
        for f in files:
            well = WELL_RE.search(f.name).group(1)
            df = pl.read_parquet(f)
            cols = [c for c in SENSOR_COLUMNS if c in df.columns]
            arr = df.select(cols).to_numpy().astype(np.float32)
            # reorder/pad to full SENSOR_COLUMNS with NaN for missing sensors
            full = np.full((arr.shape[0], len(SENSOR_COLUMNS)), np.nan, np.float32)
            for j, c in enumerate(cols):
                full[:, SENSOR_COLUMNS.index(c)] = arr[:, j]
            cl = df["class"].to_numpy()
            cl = np.where(np.isnan(cl.astype(np.float64)), -1, cl).astype(np.int64)
            cl = np.where(cl >= 100, cl % 100, cl)
            feats = window_stats(full, window)
            labs = window_labels(cl, window)
            keep = labs >= 0
            if keep.sum() == 0:
                continue
            X.append(feats[keep]); y.append(labs[keep])
            file_id.append(np.full(keep.sum(), fid)); well_id.append([well] * int(keep.sum()))
            fid += 1
    X = np.concatenate(X); y = np.concatenate(y)
    file_id = np.concatenate(file_id)
    well_id = np.array([w for sub in well_id for w in sub])
    return X, y, file_id, well_id


def run_cv(X, y, groups, n_splits, seed, label):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    f1s, bals, missing = [], [], []
    for tr, te in sgkf.split(X, y, groups):
        clf = RandomForestClassifier(
            n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=seed
        )
        clf.fit(X[tr], y[tr])
        p = clf.predict(X[te])
        f1s.append(f1_score(y[te], p, average="macro", zero_division=0))
        bals.append(balanced_accuracy_score(y[te], p))
        missing.append(sorted(set(range(10)) - set(np.unique(y[te]))))
    print(f"  {label:14s} macro-F1 {np.mean(f1s):.4f} ± {np.std(f1s):.4f} | "
          f"bal-acc {np.mean(bals):.4f} | n_groups {len(set(groups))} | "
          f"classes-missing-from-test-folds {missing}")
    return np.mean(f1s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-files-per-class", type=int, default=None)
    ap.add_argument("--window", type=int, default=192)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    print(f"Loading 3W windows (window={a.window}, cap/class={a.max_files_per_class})...")
    X, y, file_id, well_id = load(a.max_files_per_class, a.window)
    print(f"  {len(y):,} windows | {len(set(file_id))} files | {len(set(well_id))} wells "
          f"| class dist {np.bincount(y, minlength=10).tolist()}\n")

    print("RF on stat features, StratifiedGroupKFold:")
    leaky = run_cv(X, y, file_id, a.n_splits, a.seed, "file-grouped")   # wells can span
    clean = run_cv(X, y, well_id, a.n_splits, a.seed, "well-grouped")   # honest
    print(f"\n==> well-leakage effect on RF: {leaky - clean:+.4f} macro-F1 "
          f"(file-grouped {leaky:.4f} -> well-grouped {clean:.4f})")


if __name__ == "__main__":
    main()
