"""Broaden the time-series benchmark: download a diverse set of UEA multivariate
datasets from the aeon archive into ``data/uea`` (the dir the axial sweep reads).

For each dataset: fetch + extract the ``.ts`` zip, read the header metadata, and record
whether it is compatible with the current parser (equal length, no missing, no
timestamps) plus dims / series length / #classes / train-test sizes. Writes
``data/uea/manifest.json`` so the sweep can pick the usable, appropriately-sized ones.

Run:  uv run python scripts/download_uea_archive.py
"""

import json
import re
import socket
import warnings
import zipfile
from pathlib import Path

import urllib.request

warnings.filterwarnings("ignore")
socket.setdefaulttimeout(300)

OUT = Path(__file__).resolve().parent.parent / "data" / "uea"
BASE = "https://www.timeseriesclassification.com/aeon-toolkit"

# Equal-length multivariate UEA datasets spanning domains/sizes (the giant high-dim ones
# -- MotorImagery, PEMS-SF, DuckDuckGeese, FaceDetection -- are left out for now).
DATASETS = [
    "ArticularyWordRecognition", "BasicMotions", "Epilepsy", "NATOPS", "RacketSports",
    "Cricket", "ERing", "EthanolConcentration", "FingerMovements",
    "HandMovementDirection", "Handwriting", "Heartbeat", "Libras", "LSST", "PenDigits",
    "SelfRegulationSCP1", "SelfRegulationSCP2", "StandWalkJump", "UWaveGestureLibrary",
    "AtrialFibrillation",
]


def download(dataset: str) -> Path:
    target = OUT / dataset
    target.mkdir(parents=True, exist_ok=True)
    zip_path = target / f"{dataset}.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(f"{BASE}/{dataset}.zip", zip_path)
    tr, te = target / f"{dataset}_TRAIN.ts", target / f"{dataset}_TEST.ts"
    if not (tr.exists() and te.exists()):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target)
    if not (tr.exists() and te.exists()):
        raise FileNotFoundError("missing TRAIN/TEST .ts after extract")
    return target


def read_meta(ts_path: Path) -> dict:
    meta, n_rows = {}, 0
    with open(ts_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s.lower().startswith("@data"):
                for d in f:
                    if d.strip():
                        n_rows += 1
                break
            m = re.match(r"@(\w+)\s+(.*)", s)
            if m:
                meta[m.group(1).lower()] = m.group(2).strip()
    meta["n_rows"] = n_rows
    return meta


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for ds in DATASETS:
        try:
            target = download(ds)
            tr = read_meta(target / f"{ds}_TRAIN.ts")
            te = read_meta(target / f"{ds}_TEST.ts")
            labels = tr.get("classlabel", "").split()
            n_classes = len(labels) - 1 if labels and labels[0].lower() == "true" else 0
            equal = tr.get("equallength", "").lower() == "true"
            missing = tr.get("missing", "false").lower() == "true"
            stamps = tr.get("timestamps", "false").lower() == "true"
            compatible = equal and not missing and not stamps
            rec = {
                "name": ds, "dimensions": int(tr.get("dimensions", 0) or 0),
                "series_length": int(tr.get("serieslength", 0) or 0),
                "n_classes": n_classes, "n_train": tr["n_rows"], "n_test": te["n_rows"],
                "equal_length": equal, "missing": missing,
                "parser_compatible": compatible,
            }
            manifest.append(rec)
            flag = "OK  " if compatible else "INCOMPAT"
            print(f"  {flag} {ds:28s} dim={rec['dimensions']:>3} len={rec['series_length']:>4} "
                  f"cls={n_classes:>2} train/test={tr['n_rows']}/{te['n_rows']}"
                  f"{'' if compatible else '  (unequal/missing/timestamps)'}", flush=True)
        except Exception as e:
            print(f"  FAIL {ds:28s} {type(e).__name__}: {str(e)[:55]}", flush=True)

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    ok = [m for m in manifest if m["parser_compatible"]]
    print(f"\n=== {len(manifest)} UEA datasets fetched, {len(ok)} parser-compatible -> {OUT} ===", flush=True)
    print(f"  usable now: {sorted(m['name'] for m in ok)}", flush=True)


if __name__ == "__main__":
    main()
