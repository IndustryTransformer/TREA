"""Grow the tabular corpus from OpenML's CURATED benchmark suites -- the standard,
vetted "good examples" for tabular ML, so results are diverse and comparable:

  - OpenML-CTR23 (study 353): 35 curated regression datasets
  - OpenML-CC18  (study 99):  72 curated classification datasets

For each dataset: keep numeric features, cap rows/features, detect task, and record a
`meaningful` flag (fraction of non-generic column names) + `task`. The full set firms up
"single-row deep vs trees in error bands" (names irrelevant); the meaningful-named subset
feeds the semantic-column / schema-scaling work. Categorical FEATURES are dropped for now
(numeric-only corpus, consistent with the existing 15) but their count is recorded.

Writes parquet per dataset + merges into data/tabular_corpus/manifest.json (dedup by name).

Run:  uv run python scripts/download_openml_suites.py
"""

import json
import re
import socket
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import urllib.request

warnings.filterwarnings("ignore")
socket.setdefaulttimeout(180)  # bound each fetch so one hang can't stall the batch

OUT = Path(__file__).resolve().parent.parent / "data" / "tabular_corpus"
SUITES = {"regression": "353", "classification": "99"}
MAX_ROWS, MAX_FEAT, MIN_ROWS, MIN_FEAT = 20_000, 150, 300, 3
MAX_CLASSES, MIN_CLASS_COUNT = 50, 10
GENERIC = re.compile(r"^(v|x|f|col|att|attr|feature|var|pixel|p|dim)_?\d+$", re.IGNORECASE)


def suite_data_ids(study_id: str) -> list[str]:
    url = f"https://www.openml.org/api/v1/json/study/{study_id}"
    d = json.load(urllib.request.urlopen(url, timeout=30))["study"]
    return list(d.get("data", {}).get("data_id", []))


def clean(c: str) -> str:
    return re.sub(r"[._\-]+", " ", str(c)).strip().lower()


def meaningful_frac(cols) -> float:
    if not len(cols):
        return 0.0
    return sum(0 if GENERIC.match(str(c).strip()) else 1 for c in cols) / len(cols)


def process(did: str, task: str):
    from sklearn.datasets import fetch_openml

    d = fetch_openml(data_id=int(did), as_frame=True)
    name = (getattr(d, "details", {}) or {}).get("name") or f"oml{did}"
    name = re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_")[:48]
    X, y = d.data, d.target
    if X is None or y is None:
        return None

    num = X.apply(pd.to_numeric, errors="coerce")
    keep = [c for c in num.columns if num[c].notna().mean() > 0.5]
    n_cat_dropped = X.shape[1] - len(keep)
    num = num[keep]
    if not (MIN_FEAT <= num.shape[1] <= MAX_FEAT):
        return None

    df = num.copy()
    if task == "regression":
        yy = pd.to_numeric(y, errors="coerce")
        if yy.notna().sum() < MIN_ROWS or yy.nunique() < 10:
            return None
        df["__target__"] = yy.values
    else:
        yy = y.astype(str)
        vc = yy.value_counts()
        if not (2 <= len(vc) <= MAX_CLASSES) or vc.min() < MIN_CLASS_COUNT:
            return None
        df["__target__"] = yy.values

    df = df[df["__target__"].notna()].reset_index(drop=True)
    if len(df) < MIN_ROWS:
        return None
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=0).reset_index(drop=True)

    feat = [c for c in df.columns if c != "__target__"]
    rec = {
        "name": name, "source": f"openml:{did}", "task": task,
        "target": "__target__", "n_rows": len(df), "n_features": len(feat),
        "n_cat_dropped": int(n_cat_dropped),
        "meaningful": round(meaningful_frac(feat), 3),
        "feature_cols": feat, "descriptions": {c: clean(c) for c in feat},
    }
    if task == "classification":
        rec["n_classes"] = int(df["__target__"].nunique())
    return name, df, rec


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mpath = OUT / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else []
    have = {m["name"] for m in manifest}

    added = 0
    for task, study in SUITES.items():
        try:
            ids = suite_data_ids(study)
        except Exception as e:
            print(f"SUITE {task} ({study}) FAIL: {type(e).__name__} {str(e)[:60]}", flush=True)
            continue
        print(f"\n== {task}: {len(ids)} candidate datasets (OpenML study {study}) ==", flush=True)
        for did in ids:
            try:
                res = process(did, task)
            except Exception as e:
                print(f"  FAIL  id={did:>7} {type(e).__name__}: {str(e)[:50]}", flush=True)
                continue
            if res is None:
                print(f"  skip  id={did:>7} (filtered: task/size/features)", flush=True)
                continue
            name, df, rec = res
            if name in have:
                print(f"  dup   id={did:>7} {name} (already in corpus)", flush=True)
                continue
            df.to_parquet(OUT / f"{name}.parquet")
            manifest.append(rec)
            have.add(name)
            added += 1
            mf = rec["meaningful"]
            tag = "MEANINGFUL" if mf >= 0.6 else "generic   "
            print(f"  OK   id={did:>7} {name:36s} {rec['n_rows']:>6}x{rec['n_features']:<3} "
                  f"{tag} ({mf:.2f})", flush=True)

    mpath.write_text(json.dumps(manifest, indent=2))
    reg = [m for m in manifest if m.get("task", "regression") == "regression"]
    clf = [m for m in manifest if m.get("task") == "classification"]
    named = [m for m in manifest if m.get("meaningful", 1.0) >= 0.6]
    print(f"\n=== corpus now {len(manifest)} datasets (+{added} this run) ===", flush=True)
    print(f"  {len(reg)} regression | {len(clf)} classification | "
          f"{len(named)} with meaningful column names", flush=True)


if __name__ == "__main__":
    main()
