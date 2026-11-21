"""Post-hoc audit of frozen predictions; no model fitting or threshold tuning."""

import json

from pathlib import Path

import numpy as np

from sklearn.metrics import f1_score, roc_auc_score

from examples.evaluate_3w_robustness import verify_selection
from examples.freeze_3w_robustness import file_digest
from examples.summarize_3w_grouped import json_safe


def pair_decomposition(records, family):
    """Decompose pooled AUC into within/cross-fold positive-negative pairs."""
    positives = sum(int(r["labels"].sum()) for r in records)
    negatives = sum(int((r["labels"] == 0).sum()) for r in records)
    cells = []
    for i, positive in enumerate(records):
        p = positive[f"{family}_ensemble_score"][positive["labels"] == 1]
        for j, negative in enumerate(records):
            n = negative[f"{family}_ensemble_score"][negative["labels"] == 0]
            weight = len(p) * len(n) / (positives * negatives)
            auc = roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(n))], np.r_[p, n])
            cells.append(
                {
                    "positive_fold": i,
                    "negative_fold": j,
                    "auc": auc,
                    "pair_weight": weight,
                    "weighted_contribution": weight * auc,
                }
            )
    y = np.concatenate([r["labels"] for r in records])
    score = np.concatenate([r[f"{family}_ensemble_score"] for r in records])
    reconstructed = sum(c["weighted_contribution"] for c in cells)
    np.testing.assert_allclose(reconstructed, roc_auc_score(y, score), atol=1e-12)
    return {
        "pooled_auc": reconstructed,
        "cells": cells,
        "within_fold_contribution": sum(
            c["weighted_contribution"]
            for c in cells
            if c["positive_fold"] == c["negative_fold"]
        ),
        "cross_fold_contribution": sum(
            c["weighted_contribution"]
            for c in cells
            if c["positive_fold"] != c["negative_fold"]
        ),
    }


def mean_fold_interval(records, repeats=500):
    """Post-hoc sensitivity: equal-fold AUC difference, stratified group resampling."""
    prepared = []
    for r in records:
        groups, inverse = np.unique(r["groups"], return_inverse=True)
        physical = np.char.startswith(groups, "WELL-")
        prepared.append(
            (r, inverse, [np.flatnonzero(physical), np.flatnonzero(~physical)])
        )
    rng = np.random.default_rng(20260921)
    differences = []
    for _ in range(repeats):
        fold_differences = []
        for r, inverse, strata in prepared:
            counts = np.zeros(inverse.max() + 1, dtype=int)
            for members in strata:
                if len(members):
                    counts += np.bincount(
                        rng.choice(members, len(members)), minlength=len(counts)
                    )
            weights = counts[inverse]
            if any(weights[r["labels"] == k].sum() == 0 for k in [0, 1]):
                break
            fold_differences.append(
                roc_auc_score(
                    r["labels"], r["neural_ensemble_score"], sample_weight=weights
                )
                - roc_auc_score(
                    r["labels"], r["tree_ensemble_score"], sample_weight=weights
                )
            )
        if len(fold_differences) == len(records):
            differences.append(float(np.mean(fold_differences)))
    return {
        "post_hoc": True,
        "valid_replicates": len(differences),
        "mean_fold_auc_delta_95_interval": np.quantile(
            differences, [0.025, 0.975]
        ).tolist(),
        "conditional_on_fitted_models": True,
        "strata": "fold and physical/simulation source; whole groups resampled",
    }


def main():
    root = Path("logs/benchmark_3w_binary/robustness_outer_v1")
    out = root / "posthoc_audit.json"
    if out.exists():
        raise FileExistsError("Post-hoc audit already exists")
    selection_path = Path("docs/results/3w_robustness_selection.json")
    selection = json.loads(selection_path.read_text())
    verify_selection(selection)
    results = json.loads((root / "results.json").read_text())
    records, well_rows, cohort_rows = [], [], []
    hashes = {
        str(p): file_digest(p)
        for p in [
            selection_path,
            root / "results.json",
            root / "preflight.json",
            Path(__file__),
        ]
    }
    for fold in range(4):
        path = root / f"fold_{fold}_predictions.npz"
        hashes[str(path)] = file_digest(path)
        with np.load(path) as stored:
            r = {k: np.array(stored[k]) for k in stored.files}
        records.append(r)
        manifest = json.loads(
            Path(f"docs/results/3w_grouped_robustness_v1/fold_{fold}.json").read_text()
        )
        actual = set(r["groups"])
        assert actual == set(manifest["partitions"]["test"])
        assert not actual.intersection(
            set().union(
                *[set(manifest["partitions"][s]) for s in ["train", "val", "excluded"]]
            )
        )
        cohort_rows.append(
            {
                "fold": fold,
                "groups": len(actual),
                "windows": len(r["labels"]),
                "normal": int((r["labels"] == 0).sum()),
                "fault": int(r["labels"].sum()),
            }
        )
        for family in ["neural", "tree"]:
            np.testing.assert_array_equal(
                r[f"{family}_ensemble_score"],
                np.mean([r[f"{family}_{seed}_score"] for seed in [42, 43, 44]], axis=0),
            )
            frozen = selection["folds"][fold][f"selected_{family}"]["validation"]
            np.testing.assert_array_equal(
                r[f"{family}_ensemble_prediction"],
                (
                    r[f"{family}_ensemble_score"] >= frozen["threshold_from_validation"]
                ).astype(int),
            )
            for fit in ["42", "43", "44", "ensemble"]:
                row = next(
                    item
                    for item in results["fold_results"]
                    if item["fold"] == fold
                    and item["family"] == family
                    and item["fit"] == fit
                )
                np.testing.assert_allclose(
                    roc_auc_score(r["labels"], r[f"{family}_{fit}_score"]),
                    row["outer"]["auc"],
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    f1_score(
                        r["labels"],
                        r[f"{family}_{fit}_prediction"],
                        labels=[0, 1],
                        average="macro",
                    ),
                    row["outer"]["macro_f1"],
                    atol=1e-12,
                )
            for group in sorted(g for g in actual if g.startswith("WELL-")):
                mask = r["groups"] == group
                y = r["labels"][mask]
                score = r[f"{family}_ensemble_score"][mask]
                prediction = r[f"{family}_ensemble_prediction"][mask]
                well_rows.append(
                    {
                        "fold": fold,
                        "group": group,
                        "family": family,
                        "windows": int(mask.sum()),
                        "normal": int((y == 0).sum()),
                        "fault": int(y.sum()),
                        "auc": roc_auc_score(y, score)
                        if len(np.unique(y)) == 2
                        else None,
                        "macro_f1": f1_score(
                            y, prediction, labels=[0, 1], average="macro"
                        ),
                        "false_positives": int(((y == 0) & (prediction == 1)).sum()),
                        "false_negatives": int(((y == 1) & (prediction == 0)).sum()),
                    }
                )
    decomposed = {
        family: pair_decomposition(records, family) for family in ["neural", "tree"]
    }
    result = {
        "post_hoc_diagnostic_only": True,
        "no_fitting_or_threshold_changes": True,
        "all_32_fit_metrics_recomputed": True,
        "group_coverage_verified": True,
        "ensemble_averaging_and_frozen_thresholds_verified": True,
        "cohorts": cohort_rows,
        "physical_wells": well_rows,
        "auc_pair_decomposition": decomposed,
        "equal_fold_sensitivity": mean_fold_interval(records),
        "source_sha256": hashes,
    }
    out.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False))
    print(
        json.dumps(
            {
                "decomposition": {
                    k: {a: b for a, b in v.items() if a != "cells"}
                    for k, v in decomposed.items()
                },
                "uncertainty": result["equal_fold_sensitivity"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
