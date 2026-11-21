"""Release gate for trea.transfer: does column-name transfer beat every control?

This re-runs the schema-scaling test (docs/LESSONS.md §3b) through the shipped API,
``trea.transfer.SchemaTransferRegressor``, so the gate tests the code users get
rather than a script-local copy. It adds the arm §3b lacked: ``scratch_index``
(§3g found it beats ``scratch_semantic`` on 4/5 held-outs at 200 labels).

Setup (corpus from scripts/download_tabular_corpus.py):
  - Each dataset is split 70/30 train/test (fixed, random_state=42).
  - Source tables for pretraining = the train split of K datasets drawn from the
    pool that excludes the held-out dataset.
  - Few-shot = n_labels rows sampled from the held-out train split. Every deep arm
    standardizes on those labeled rows only (the API's behaviour; §3b used the full
    train split, which leaked target scale).
  - Metric = held-out test RMSE in original units, mean over fits.

Arms:
  transfer_semantic : pretrain on K sources (semantic embedder), fit on held-out
  transfer_index    : same with embedder="index" (unseen columns -> [UNK])
  scratch_semantic  : fit on held-out only, semantic embedder
  scratch_index     : fit on held-out only, index embedder
  xgb               : XGBoost on the same labels (reported, not gated)

Fits: transfer arms = source_seeds x fewshot_seeds (each pretrained model is
fine-tuned once per few-shot draw); scratch arms and xgb = fewshot_seeds.

== PRE-REGISTERED GATE (locked 2026-11-21, before any run; do not revise) ==
At K=4, n_labels=200, 3 source seeds, 3 few-shot seeds, on the 5 held-outs below:
PASS iff, on at least 4 of 5 held-outs, transfer_semantic has lower mean RMSE
than ALL THREE of transfer_index, scratch_semantic and scratch_index.
FAIL against scratch_index only -> the mechanism is real but has no user value at
200 labels: do not ship the class as a feature; record the result and stop.
Corpus = the §3b regression corpus (download_tabular_corpus.py entries only).
(Clarified 2026-11-21 after a crash, before any complete run: the shared manifest
had grown to 89 tables incl. classification; the pool was always meant to be §3b's.)

Run (trains many small models; progress off):
    uv run python scripts/schema_scaling.py > /tmp/schema_gate.log 2>&1
    uv run python scripts/schema_scaling.py --smoke
"""

import argparse
import copy
import json
import math
import os
import warnings

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from trea.transfer import SchemaTransferRegressor, Table  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "tabular_corpus"
RESULTS = ROOT / "logs" / "schema_transfer_gate.json"
TARGET = "__target__"
HELD_OUT = [
    "california_housing",
    "concrete_compressive_strength",
    "abalone",
    "airfoil_self_noise",
    "Bike_Sharing_Demand",
]
GATED = ("transfer_index", "scratch_semantic", "scratch_index")


def load_corpus():
    """Load the §3b corpus: the regression sets from download_tabular_corpus.py.

    data/tabular_corpus/manifest.json is shared with download_openml_suites.py,
    which appends OpenML suite tables (classification included) carrying a "task"
    key. Those were not in the §3b corpus, so they are excluded here; including
    them put class labels into the regression source pool.
    """
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    manifest = [m for m in manifest if "task" not in m]
    out = {}
    for m in manifest:
        df = pd.read_parquet(CORPUS / f"{m['name']}.parquet")
        feat = m["feature_cols"]
        df = df[feat + [TARGET]]
        if not pd.api.types.is_numeric_dtype(df[TARGET]):
            raise ValueError(f"{m['name']}: non-numeric target in regression corpus")
        tr, te = train_test_split(df, test_size=0.3, random_state=42)
        out[m["name"]] = {
            "train": tr.reset_index(drop=True),
            "test": te.reset_index(drop=True),
            "descr": {c: m["descriptions"][c] for c in feat},
        }
    return out


def table(d, df, name):
    return Table(df, TARGET, d["descr"], name=name)


def rmse(pred, y):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(y)) ** 2)))


def fewshot_rows(d, n_labels, seed):
    rng = np.random.RandomState(seed)
    n = min(n_labels, len(d["train"]))
    return d["train"].iloc[rng.choice(len(d["train"]), size=n, replace=False)]


def xgb_rmse(d, rows, seed):
    feat = list(d["descr"])
    m = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=seed,
    )
    m.fit(rows[feat], rows[TARGET])
    return rmse(m.predict(d["test"][feat]), d["test"][TARGET])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-labels", type=int, default=200)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--held-out", nargs="+", default=HELD_OUT)
    ap.add_argument("--source-seeds", type=int, default=3)
    ap.add_argument("--fewshot-seeds", type=int, default=3)
    ap.add_argument("--pretrain-epochs", type=int, default=40)
    ap.add_argument("--fit-epochs", type=int, default=80)
    a = ap.parse_args()
    if a.smoke:
        a.held_out, a.k = a.held_out[:1], 2
        a.source_seeds = a.fewshot_seeds = 1
        a.pretrain_epochs, a.fit_epochs = 1, 2

    corpus = load_corpus()
    held_list = [h for h in a.held_out if h in corpus]
    missing = sorted(set(a.held_out) - set(held_list))
    print(
        f"corpus={len(corpus)} held_out={held_list} missing={missing} K={a.k} "
        f"n_labels={a.n_labels} source_seeds={a.source_seeds} "
        f"fewshot_seeds={a.fewshot_seeds}",
        flush=True,
    )

    results = []
    for held in held_list:
        h = corpus[held]
        test_df, y_test = h["test"], h["test"][TARGET]
        pool = sorted(n for n in corpus if n != held)
        scores = {arm: [] for arm in ("xgb", *GATED, "transfer_semantic")}
        draws = [fewshot_rows(h, a.n_labels, f) for f in range(a.fewshot_seeds)]

        for f, rows in enumerate(draws):
            scores["xgb"].append(xgb_rmse(h, rows, f))
            for emb in ("semantic", "index"):
                m = SchemaTransferRegressor(embedder=emb, seed=f)
                m.fit(table(h, rows, held), epochs=a.fit_epochs, seed=f)
                scores[f"scratch_{emb}"].append(rmse(m.predict(test_df), y_test))

        for s in range(a.source_seeds):
            rng = np.random.RandomState(1000 * a.k + s)
            src = [pool[i] for i in rng.choice(len(pool), size=a.k, replace=False)]
            sources = [table(corpus[n], corpus[n]["train"], n) for n in src]
            for emb in ("semantic", "index"):
                base = SchemaTransferRegressor(embedder=emb, seed=s)
                base.pretrain(sources, epochs=a.pretrain_epochs)
                for f, rows in enumerate(draws):
                    m = copy.deepcopy(base)
                    m.fit(table(h, rows, held), epochs=a.fit_epochs, seed=f)
                    scores[f"transfer_{emb}"].append(rmse(m.predict(test_df), y_test))
            print(f"  [{held}] source seed {s}: {src}", flush=True)

        row = {
            "held_out": held,
            **{arm: float(np.mean(v)) for arm, v in scores.items()},
            **{f"{arm}_std": float(np.std(v)) for arm, v in scores.items()},
            "n_fits": {arm: len(v) for arm, v in scores.items()},
        }
        row["beats"] = {arm: row["transfer_semantic"] < row[arm] for arm in GATED}
        results.append(row)
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(results, indent=2))
        print(
            f"[{held}] "
            + " | ".join(
                f"{arm} {row[arm]:.3f}+-{row[arm + '_std']:.3f}" for arm in scores
            ),
            flush=True,
        )

    print("\n=== RELEASE GATE (held-out test RMSE, lower is better) ===", flush=True)
    header = ["held_out", "transfer_semantic", *GATED, "xgb", "pass"]
    print(" ".join(f"{c:>18}" for c in header), flush=True)
    n_pass = {arm: 0 for arm in GATED}
    for r in results:
        ok = all(r["beats"].values())
        for arm in GATED:
            n_pass[arm] += r["beats"][arm]
        cells = [r["held_out"][:18]] + [
            f"{r[c]:.3f}" for c in ("transfer_semantic", *GATED, "xgb")
        ]
        print(
            " ".join(f"{c:>18}" for c in cells) + f" {'yes' if ok else 'no':>18}",
            flush=True,
        )
    need = math.ceil(0.8 * len(results))
    n_all = sum(all(r["beats"].values()) for r in results)
    n_no_scratch_idx = sum(
        all(v for arm, v in r["beats"].items() if arm != "scratch_index")
        for r in results
    )
    print(
        "\n"
        + " | ".join(f"beats {arm}: {n_pass[arm]}/{len(results)}" for arm in GATED)
        + f" | beats all three: {n_all}/{len(results)} (need {need})",
        flush=True,
    )
    if a.smoke:
        verdict = "SMOKE (not a verdict)"
    elif n_all >= need:
        verdict = "PASS: ship trea.transfer as a feature"
    elif n_no_scratch_idx >= need:
        verdict = (
            "FAIL vs scratch_index only: mechanism real, no user value; do not ship"
        )
    else:
        verdict = "FAIL"
    print(f"VERDICT: {verdict}\nresults -> {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
