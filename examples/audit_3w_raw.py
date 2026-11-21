"""Audit raw sensor scales using only the frozen training-file manifest."""

import argparse
import json

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from utils.three_w import SENSOR_COLUMNS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    if manifest["split"] != "train":
        raise ValueError("Raw audit is restricted to training files")
    root = Path(manifest["data_path"])
    samples = {c: [] for c in SENSOR_COLUMNS}
    counts = {c: Counter() for c in SENSOR_COLUMNS}
    extremes = {c: Counter() for c in SENSOR_COLUMNS}
    for i, file in enumerate(manifest["files"]):
        frame = pd.read_parquet(root / file)
        for col in SENSOR_COLUMNS:
            if col not in frame:
                continue
            values = frame[col].to_numpy(dtype=np.float64)
            counts[col]["total"] += len(values)
            counts[col]["missing_or_nonfinite"] += int((~np.isfinite(values)).sum())
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            counts[col]["above_legacy_clip"] += int((np.abs(values) > 1e6).sum())
            counts[col]["above_1e12"] += int((np.abs(values) > 1e12).sum())
            extreme = values[np.abs(values) > 1e12]
            if len(extreme):
                unique, n = np.unique(extreme, return_counts=True)
                extremes[col].update(
                    dict(zip(unique.tolist(), n.tolist(), strict=True))
                )
            samples[col].append(values[:: max(1, len(values) // 512)][:512])
        if (i + 1) % 200 == 0:
            print(f"Audited {i + 1} training files", flush=True)
    result = {}
    for col in SENSOR_COLUMNS:
        x = np.concatenate(samples[col]) if samples[col] else np.array([])
        plausible = x[np.abs(x) <= 1e12]
        result[col] = {
            "counts": dict(counts[col]),
            "sample_quantiles": np.quantile(
                x, [0, 0.001, 0.01, 0.5, 0.99, 0.999, 1]
            ).tolist()
            if len(x)
            else [],
            "excluding_extreme_quantiles": np.quantile(
                plausible, [0, 0.01, 0.5, 0.99, 1]
            ).tolist()
            if len(plausible)
            else [],
            "common_extremes": extremes[col].most_common(5),
        }
    Path(args.output).write_text(json.dumps(result, indent=2))
    for col, values in result.items():
        print(col, json.dumps(values), flush=True)


if __name__ == "__main__":
    main()
