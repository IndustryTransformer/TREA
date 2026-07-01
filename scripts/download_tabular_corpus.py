"""Download a small, diverse corpus of tabular REGRESSION datasets with meaningful
column names -- the substrate for the schema-scaling / semantic-column-transfer test.

The semantic-column thesis needs *schema diversity*, not row volume: many datasets with
different, human-readable column vocabularies. This fetches a curated set (sklearn
bundled + OpenML by name), keeps only those with (a) a numeric regression target and
(b) non-generic column names (rejects V1/x2/attr3-style), caps rows for tractable
training, drops non-numeric feature columns for a clean numeric-only pilot, and writes
one parquet per dataset + a manifest.json.

Column "descriptions" for the semantic embedder = the cleaned column names (dots/
underscores -> spaces). Good enough: these names are already meaningful.

Run:  uv run python scripts/download_tabular_corpus.py
Out:  data/tabular_corpus/<name>.parquet  +  data/tabular_corpus/manifest.json
"""

import json
import re
import socket
import sys
import warnings

from pathlib import Path

import pandas as pd


warnings.filterwarnings("ignore")
socket.setdefaulttimeout(120)  # bound each network call so one hang can't eat the night

OUT = Path(__file__).resolve().parent.parent / "data" / "tabular_corpus"
MAX_ROWS = 20_000
MIN_ROWS = 300
GENERIC = re.compile(r"^(v|x|f|col|attr|feature|att|var)_?\d+$", re.IGNORECASE)

# (source, name/version, optional explicit target col). OpenML default target used if None.
CANDIDATES = [
    ("sklearn", "california_housing", None),
    ("sklearn", "diabetes", None),
    ("openml", "wine_quality", None),
    ("openml", "Bike_Sharing_Demand", None),
    ("openml", "abalone", None),
    ("openml", "autoMpg", None),
    ("openml", "auto_price", None),
    ("openml", "concrete_compressive_strength", None),
    ("openml", "energy-efficiency", None),
    ("openml", "airfoil_self_noise", None),
    ("openml", "forest_fires", None),
    ("openml", "house_sales", None),
    ("openml", "Moneyball", None),
    ("openml", "cps_85_wages", None),
    ("openml", "us_crime", None),
    ("openml", "student_performance_por", None),
    ("openml", "solar_flare", None),
    ("openml", "sensory", None),
    ("openml", "meta", None),
    ("openml", "pbc", None),
]


def clean_name(c: str) -> str:
    return re.sub(r"[._\-]+", " ", str(c)).strip().lower()


def meaningful(cols) -> bool:
    generic = sum(1 for c in cols if GENERIC.match(str(c).strip()))
    return generic / max(1, len(cols)) <= 0.4


def load(source, name):
    from sklearn.datasets import fetch_california_housing, fetch_openml, load_diabetes

    if source == "sklearn":
        if name == "california_housing":
            d = fetch_california_housing(as_frame=True)
            return d.data.copy(), d.target.copy(), "MedHouseVal"
        if name == "diabetes":
            d = load_diabetes(as_frame=True, scaled=False)
            return d.data.copy(), d.target.copy(), "disease_progression"
        raise ValueError(name)
    for kw in ({"version": 1}, {"version": "active"}):
        try:
            d = fetch_openml(name=name, as_frame=True, **kw)
            break
        except Exception:
            d = None
    if d is None:
        raise RuntimeError("fetch failed")
    y = d.target
    tname = getattr(y, "name", None) or "target"
    return d.data.copy(), pd.to_numeric(y, errors="coerce"), tname


def process(X: pd.DataFrame, y: pd.Series):
    # numeric target with real spread => regression
    y = pd.to_numeric(y, errors="coerce")
    if y.notna().sum() < MIN_ROWS or y.nunique() < 10:
        return None
    # keep numeric features only (clean numeric-only pilot)
    Xn = X.apply(pd.to_numeric, errors="coerce")
    keep = [c for c in Xn.columns if Xn[c].notna().mean() > 0.5]
    Xn = Xn[keep]
    if Xn.shape[1] < 3 or not meaningful(Xn.columns):
        return None
    df = Xn.copy()
    df["__target__"] = y.values
    df = df[df["__target__"].notna()].reset_index(drop=True)
    if len(df) < MIN_ROWS:
        return None
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=0).reset_index(drop=True)
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for source, name, _ in CANDIDATES:
        try:
            X, y, tname = load(source, name)
            df = process(X, y)
            if df is None:
                print(
                    f"  SKIP  {name:32s} (no regression target / generic names / too small)",
                    flush=True,
                )
                continue
            feat_cols = [c for c in df.columns if c != "__target__"]
            df.to_parquet(OUT / f"{name}.parquet")
            manifest.append(
                {
                    "name": name,
                    "source": source,
                    "target": tname,
                    "n_rows": len(df),
                    "n_features": len(feat_cols),
                    "feature_cols": feat_cols,
                    "descriptions": {c: clean_name(c) for c in feat_cols},
                }
            )
            print(
                f"  OK    {name:32s} {len(df):>6} rows x {len(feat_cols):>2} feats | "
                f"e.g. {[clean_name(c) for c in feat_cols[:4]]}",
                flush=True,
            )
        except Exception as e:
            print(f"  FAIL  {name:32s} {type(e).__name__}: {str(e)[:60]}", flush=True)

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nCorpus: {len(manifest)} datasets -> {OUT}", flush=True)
    if manifest:
        tot = sum(m["n_rows"] for m in manifest)
        print(
            f"  {tot:,} total rows | feature counts {sorted(m['n_features'] for m in manifest)}",
            flush=True,
        )
    if len(manifest) < 4:
        print(
            "WARNING: <4 datasets fetched; schema-scaling sweep needs more diversity.",
            flush=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
