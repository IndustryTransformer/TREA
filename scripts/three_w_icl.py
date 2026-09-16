"""TabICL on 3W binary (well-disjoint) from cached hand-stat features.

Two stages, run as separate processes so the 645k-window store (which OOM-killed the
in-process attempt twice at ~48 GB RSS) is gone before TabICL runs:

  extract : build ThreeWDataset (same args as examples/benchmark_3w_binary.py
            --split-mode well), remap binary, hand-stat features -> npz
  icl     : TabICL on a random cap of train rows, chunked predict_proba on val,
            same binary_metrics as the RF gate. Reference RF (LESSONS §3d): AUC 0.958,
            macro-F1 0.820.

Run:
  PYTHONPATH=. uv run python scripts/three_w_icl.py extract
  PYTHONPATH=. uv run python scripts/three_w_icl.py icl --cap 20000 --seeds 42 43 44
"""

import argparse
import json
import resource
import sys
import time

from pathlib import Path

import numpy as np


sys.path.insert(0, ".")
NPZ = Path("logs/three_w_binary_handstats.npz")
OUT = Path("logs/three_w_icl.json")


def extract():
    from examples.benchmark_3w_binary import (
        extract_engineered_features,
        remap_to_binary,
    )
    from utils.three_w import ThreeWDataset

    kw = dict(
        toolkit_path="/home/kailukowiak/Work/3W",
        data_path="/home/kailukowiak/Work/3W/dataset",
        window_size=96,
        stride=96,
        split_mode="well",
        val_fraction=0.2,
        max_files=None,
        seed=42,
        augment=False,
    )
    tr = ThreeWDataset(split="train", **kw)
    va = ThreeWDataset(split="val", normalization_stats=tr.normalization_stats, **kw)
    remap_to_binary(tr)
    remap_to_binary(va)
    Xtr = extract_engineered_features(tr.windows)
    ytr = tr.labels.copy()
    del tr
    Xva = extract_engineered_features(va.windows)
    yva = va.labels.copy()
    del va
    NPZ.parent.mkdir(exist_ok=True)
    np.savez_compressed(NPZ, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva)
    print(f"saved {NPZ}: train {Xtr.shape} val {Xva.shape}", flush=True)


def icl(cap, seeds, chunk):
    import torch

    from examples.benchmark_3w_binary import binary_metrics
    from tabicl import TabICLClassifier

    d = np.load(NPZ)
    Xtr, ytr, Xva, yva = d["Xtr"], d["ytr"], d["Xva"], d["yva"]
    print(f"train {Xtr.shape} val {Xva.shape} cap={cap}", flush=True)
    runs = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(ytr), size=min(cap, len(ytr)), replace=False)
        clf = TabICLClassifier(device="cuda" if torch.cuda.is_available() else "cpu")
        t0 = time.perf_counter()
        clf.fit(Xtr[idx], ytr[idx])
        pos = list(clf.classes_).index(1)
        score = np.concatenate(
            [
                clf.predict_proba(Xva[i : i + chunk])[:, pos]
                for i in range(0, len(Xva), chunk)
            ]
        )
        secs = time.perf_counter() - t0
        m = binary_metrics((score >= 0.5).astype(np.int64), yva, score)
        m.update(seed=seed, cap=int(cap), seconds=secs)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        print(
            f"[tabicl cap={cap}] seed={seed} auc={m['auc']:.4f} "
            f"anomaly_f1={m['anomaly_f1']:.4f} normal_f1={m['normal_f1']:.4f} "
            f"macro_f1={m['macro_f1']:.4f} acc={m['accuracy']:.4f} "
            f"({secs:.0f}s, peak RSS {peak:.1f} GB)",
            flush=True,
        )
        runs.append(m)
        OUT.write_text(json.dumps(runs, indent=2))
    auc = np.mean([r["auc"] for r in runs])
    mf1 = np.mean([r["macro_f1"] for r in runs])
    print(
        f"\n=== 3W binary well-disjoint: TabICL(cap={cap}) AUC {auc:.4f} macro-F1 {mf1:.4f} "
        f"| RF ref AUC 0.9584 macro-F1 0.8202 -> "
        f"{'ICL WINS' if auc > 0.9584 else 'RF wins'} on AUC ===",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "icl"])
    ap.add_argument("--cap", type=int, default=20000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--chunk", type=int, default=5000)
    a = ap.parse_args()
    if a.stage == "extract":
        extract()
    else:
        icl(a.cap, a.seeds, a.chunk)


if __name__ == "__main__":
    main()
