"""Freeze four outer group folds within current training groups, without signals.

The current internal validation remains fixed for selection. Both consumed test
partitions are excluded. This is a development-population robustness study with
historical fitting exposure, not a newly acquired external confirmation cohort.
"""

import argparse
import hashlib
import json

from pathlib import Path

import numpy as np

from examples.diagnose_3w_calibration import calibration_folds


def make_folds(parent, splits=4, seed=20260920):
    original = parent["partitions"]
    groups = np.array(sorted(original["train"]))
    assignment = calibration_folds(groups, splits, seed)
    excluded = sorted(original["test"] + original["excluded"])
    folds = []
    for fold in range(splits):
        partitions = {
            "train": groups[assignment != fold].tolist(),
            "val": sorted(original["val"]),
            "test": groups[assignment == fold].tolist(),
            "excluded": excluded,
        }
        folds.append(
            {
                "protocol": "3w_training_population_grouped_robustness_v1",
                "outer_fold": fold,
                "outer_seed": seed,
                "partitions": partitions,
                "file_inventory_sha256": parent["file_inventory_sha256"],
                "historical_exposure": (
                    "Outer groups previously used for fitting and raw-value audit. "
                    "All current outer-fold models and preprocessing are refitted "
                    "without those groups. Fixed validation is used for selection. "
                    "Previously consumed test groups are excluded entirely."
                ),
                "counts": {
                    s: {
                        "groups": len(ids),
                        "physical_wells": sum(g.startswith("WELL-") for g in ids),
                    }
                    for s, ids in partitions.items()
                },
            }
        )
    return folds


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--parent", default="docs/results/3w_range_confirmation_protocol.json"
    )
    p.add_argument("--output-dir", default="docs/results/3w_grouped_robustness_v1")
    args = p.parse_args()
    root = Path(args.output_dir)
    if root.exists():
        raise FileExistsError("Robustness membership/plan already frozen")
    parent = json.loads(Path(args.parent).read_text())
    folds = make_folds(parent)
    root.mkdir(parents=True)
    for i, fold in enumerate(folds):
        (root / f"fold_{i}.json").write_text(json.dumps(fold, indent=2))
    plan = {
        "parent_sha256": hashlib.sha256(Path(args.parent).read_bytes()).hexdigest(),
        "scope": __doc__,
        "seeds_per_recipe": [42, 43, 44],
        "neural_candidates": [
            "single_row_window",
            "single_row_causal",
            "tabm_ple_causal",
        ],
        "tree_candidates": ["xgb_window", "xgb_causal"],
        "shared_recipe": {
            "group_loss_power": 0.5,
            "value_policy": "preserve_range",
            "train_cap": 200000,
            "width": 128,
            "layers": 2,
            "dropout": 0,
            "lr": 3e-4,
            "epochs": 30,
            "patience": 6,
            "causal_alpha": 0.1,
        },
        "selection": "Within each outer fold select neural and tree recipes by highest "
        "three-fit ensemble AUC on fixed internal validation; no outer selection.",
        "threshold": "Exact macro-F1 optimum on internal validation only.",
        "unblinding": (
            "Fit all candidates/all folds, save selection and hashes, "
            "then evaluate outer groups. No outer-driven refitting "
            "or threshold tuning."
        ),
        "reporting": [
            "Each outer-fold AUC and macro-F1",
            "Mean and spread across outer folds",
            "Pooled out-of-fold per-window metrics",
            "Single-fit and explicit three-fit ensembles separately",
            "Physical-only and simulation-only results",
            "Paired whole-group uncertainty and historical exposure",
        ],
        "fold_counts": [f["counts"] for f in folds],
    }
    (root / "plan.json").write_text(json.dumps(plan, indent=2))
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
