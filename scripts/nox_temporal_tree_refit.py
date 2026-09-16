"""Secondary tree baseline: fit all available labels, including neural validation.

Same frozen XGBoost preset, no tuning on future labels. This follows the old
label-budget convention; the primary experiment instead matches fitting rows.
"""

import argparse
import json

from pathlib import Path

import numpy as np

from xgboost import XGBRegressor

from scripts.nox_encoding_ablation import ROOT
from scripts.nox_reference_models import save_json
from scripts.nox_temporal_pretraining import FRACTIONS, SEEDS


def run(output):
    arrays = np.load(output / "data.npz")
    splits = np.load(output / "splits.npz")
    x, y, future = arrays["x"], arrays["y"], arrays["future"]
    destination = output / "tree_all_labels"
    destination.mkdir(exist_ok=False)
    save_json(
        destination / "protocol.json",
        dict(
            role="Additional diagnostic: fit all available labels, as in old protocol",
            change="Fit reserved neural validation labels as well as fitting labels",
            selection="No tuning; retain frozen 800-tree/depth-6/lr-.03 preset",
            fractions=FRACTIONS,
            seeds=SEEDS,
        ),
    )
    summary = {}
    for fraction in FRACTIONS:
        fit = splits[f"fit_{fraction}"]
        labeled = np.concatenate([fit, splits[f"val_{fraction}"]])
        mu, sd = y[fit].mean(), y[fit].std(ddof=1)
        predictions, scores = [], []
        for seed in SEEDS:
            model = XGBRegressor(
                n_estimators=800,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=2,
                random_state=seed,
            )
            model.fit(x[labeled], (y[labeled] - mu) / sd)
            prediction = model.predict(x[future]) * sd + mu
            score = float(np.sqrt(np.mean((y[future] - prediction) ** 2)))
            np.save(destination / f"prediction_{fraction}_{seed}.npy", prediction)
            predictions.append(prediction)
            scores.append(score)
        ensemble = np.mean(predictions, axis=0)
        np.save(destination / f"ensemble_{fraction}.npy", ensemble)
        summary[str(fraction)] = dict(
            fit_rows=len(labeled),
            mean=float(np.mean(scores)),
            std=float(np.std(scores, ddof=1)),
            individual=scores,
            ensemble=float(np.sqrt(np.mean((y[future] - ensemble) ** 2))),
        )
    save_json(destination / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "logs/nox_temporal_pretraining"
    )
    run(parser.parse_args().output)
