"""Pretrained in-context tabular models vs trees on the local corpus.

The question (LESSONS §3e follow-up, 2026-09-15): the field claims pretrained ICL
models (TabPFNv2 / TabICLv2) beat tuned GBDTs. Does that hold on OUR corpus and
splits? If yes, TREA's ceiling was corpus starvation and the GAUGE-scale
pretraining bet is funded; if no, the benchmarks are the problem.

Reuses the exact split/standardization and tree arms from
scripts/tabular_vs_trees.py, so numbers are comparable to the from-scratch deep
run. Trees and ICL both get NaN mean-imputed (post-standardization 0).

  classification -> TabICL (classification-only model)
  regression     -> TabPFN regressor (needs TABPFN_TOKEN; skipped if missing)

Run:  uv run python scripts/icl_vs_trees.py --tasks classification > logs/icl_vs_trees.log 2>&1
"""

import argparse
import json
import os
import time
import warnings

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import f1_score, mean_squared_error

from scripts.tabular_vs_trees import CORPUS, load_corpus, prep, tree_score


warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent.parent / "logs" / "icl_vs_trees.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TRAIN = 20000  # in-context cap (never hit: corpus tables are <=20k rows, 70% train)


def icl_score(Xtr, ytr, Xte, yte, task, tscale, seed, n_classes):
    Xtr_i, Xte_i = np.nan_to_num(Xtr, nan=0.0), np.nan_to_num(Xte, nan=0.0)
    if len(ytr) > MAX_TRAIN:
        idx = np.random.RandomState(seed).choice(len(ytr), MAX_TRAIN, replace=False)
        Xtr_i, ytr = Xtr_i[idx], ytr[idx]
    t0 = time.perf_counter()
    if task == "classification":
        from tabicl import TabICLClassifier

        m = TabICLClassifier(device=DEVICE, random_state=seed)
        m.fit(Xtr_i, ytr)
        score = float(f1_score(yte, m.predict(Xte_i), average="macro"))
    else:
        from tabpfn import TabPFNRegressor

        m = TabPFNRegressor(device=DEVICE, random_state=seed)
        m.fit(Xtr_i, ytr)
        score = float(np.sqrt(mean_squared_error(yte, m.predict(Xte_i)))) * tscale
    return score, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["classification"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if "regression" in a.tasks and not os.environ.get("TABPFN_TOKEN"):
        print("TABPFN_TOKEN not set -> skipping regression (TabPFN)", flush=True)
        a.tasks = [t for t in a.tasks if t != "regression"]

    _, datasets = load_corpus(a.tasks)
    if a.limit:
        datasets = datasets[: a.limit]
    print(f"datasets={len(datasets)} tasks={a.tasks} seeds={a.seeds} device={DEVICE}", flush=True)
    results = []
    for i, m in enumerate(datasets):
        name, task, feat = m["name"], m["task"], m["feature_cols"]
        df = pd.read_parquet(CORPUS / f"{name}.parquet")
        row = {k: [] for k in ("icl", "xgb", "rf", "cat")}
        secs = []
        try:
            for seed in range(a.seeds):
                Xtr, ytr, Xte, yte, tscale, nc = prep(df, feat, task, seed)
                s, dt = icl_score(Xtr, ytr, Xte, yte, task, tscale, seed, nc)
                row["icl"].append(s)
                secs.append(dt)
                for k in ("xgb", "rf", "cat"):
                    row[k].append(tree_score(k, Xtr, ytr, Xte, yte, task, tscale, seed))
        except Exception as e:
            print(f"[{i:>2}] FAIL {name}: {type(e).__name__} {str(e)[:80]}", flush=True)
            continue
        higher = task == "classification"
        rec = {"name": name, "task": task, "n_rows": m["n_rows"], "n_features": len(feat)}
        for k, v in row.items():
            rec[k], rec[k + "_std"] = float(np.mean(v)), float(np.std(v))
        rec["icl_seconds"] = float(np.mean(secs))
        trees = [rec[k] for k in ("xgb", "rf", "cat")]
        bt = max(trees) if higher else min(trees)
        rec["best_tree"] = bt
        rec["win"] = (rec["icl"] > bt) if higher else (rec["icl"] < bt)
        rec["tie"] = (
            (rec["icl"] >= bt - rec["icl_std"]) if higher else (rec["icl"] <= bt + rec["icl_std"])
        )
        results.append(rec)
        metric = "F1" if higher else "RMSE"
        print(
            f"[{i:>2}] {name:34s} {task[:3]} {metric} icl {rec['icl']:.3f}±{rec['icl_std']:.3f} "
            f"| xgb {rec['xgb']:.3f} rf {rec['rf']:.3f} cat {rec['cat']:.3f} "
            f"| {'WIN' if rec['win'] else ('tie' if rec['tie'] else 'loss')} ({rec['icl_seconds']:.1f}s)",
            flush=True,
        )
        OUT.write_text(json.dumps(results, indent=2))

    print("\n=== AGGREGATE: pretrained ICL vs best tree ===", flush=True)
    for task in a.tasks:
        rows = [r for r in results if r["task"] == task]
        if not rows:
            continue
        n, nw, nt = len(rows), sum(r["win"] for r in rows), sum(r["tie"] for r in rows)
        gaps = [
            (r["icl"] - r["best_tree"]) if task == "classification" else (r["best_tree"] - r["icl"]) / max(r["best_tree"], 1e-9)
            for r in rows
        ]
        print(
            f"  {task:14s} n={n:>2} | icl strictly beats best tree on {nw}/{n} | "
            f"ties-or-beats {nt}/{n} | median edge {np.median(gaps):+.3f}",
            flush=True,
        )
    print(f"\nresults -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
