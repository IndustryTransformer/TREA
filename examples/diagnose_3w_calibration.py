"""Development-only grouped threshold/calibration diagnostics.

The base checkpoints used these validation groups for early stopping. Cross-fitting
calibration reduces threshold fitting optimism, but does not make these scores an
independent model test. This script never loads confirmation arrays or results.
"""

import argparse
import json

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from examples.benchmark_3w_grouped import group_loss_weights, load_data
from examples.summarize_3w_grouped import json_safe


POLICIES = [
    "global_macro",
    "global_balanced",
    "global_robust",
    "group_macro",
    "source_macro",
    "source_robust",
    "source_platt_macro",
]


def threshold_for(y, score, objective="macro", weights=None):
    """Exact score-boundary search; robust objective fixes priors before fitting."""
    y, score = np.asarray(y), np.asarray(score)
    if len(y) == 0 or not np.isfinite(score).all():
        raise ValueError("Nonempty finite scores required")
    weights = np.ones(len(y)) if weights is None else np.asarray(weights)
    order = np.argsort(-score, kind="stable")
    y, score, weights = y[order], score[order], weights[order]
    ends = np.r_[np.flatnonzero(score[:-1] != score[1:]), len(score) - 1]
    tp = np.r_[0, np.cumsum(weights * y)[ends]]
    fp = np.r_[0, np.cumsum(weights * (1 - y))[ends]]
    positive, negative = (weights * y).sum(), (weights * (1 - y)).sum()
    if min(positive, negative) <= 0:
        raise ValueError("Both classes required for threshold fitting")
    tpr, fpr = tp / positive, fp / negative
    if objective == "balanced":
        utility = (tpr + 1 - fpr) / 2
    else:
        if objective == "macro":
            priors = [positive / (positive + negative)]
        elif objective == "robust":
            priors = [0.1, 0.3, 0.5, 0.7, 0.9]
        else:
            raise ValueError("Unknown threshold objective")
        utilities = []
        for prior in priors:
            a, b = prior * tpr, (1 - prior) * fpr
            c, d = prior * (1 - tpr), (1 - prior) * (1 - fpr)
            utilities.append(
                a / np.maximum(2 * a + b + c, 1e-15)
                + d / np.maximum(2 * d + b + c, 1e-15)
            )
        utility = np.min(utilities, axis=0)
    thresholds = np.r_[np.nextafter(score[0], np.inf), score[ends]]
    return float(thresholds[np.argmax(utility)])


def calibration_folds(groups, splits=4, seed=42):
    """Source-stratified group assignment; no labels or predictions used."""
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    assignment = {}
    for real in [False, True]:
        source = unique[np.char.startswith(unique, "WELL-") == real]
        if len(source) < splits:
            raise ValueError("Too few groups per source for requested folds")
        for i, group in enumerate(rng.permutation(source)):
            assignment[group] = i % splits
    return np.array([assignment[group] for group in groups])


def calibration_features(score, groups):
    probability = np.clip(score, 1e-6, 1 - 1e-6)
    logit = np.log(probability / (1 - probability))
    real = np.char.startswith(np.asarray(groups), "WELL-").astype(float)
    return np.column_stack([logit, real, logit * real])


def apply_policy(policy, y, score, groups, target_score, target_groups):
    """Fit using calibration labels only; target labels are not an input."""
    objective = "robust" if policy.endswith("robust") else "macro"
    if policy == "global_balanced":
        objective = "balanced"
    if policy == "source_platt_macro":
        model = LogisticRegression(C=1.0, max_iter=200, random_state=42)
        model.fit(
            calibration_features(score, groups),
            y,
            sample_weight=group_loss_weights(groups, 1.0),
        )
        fitted = model.predict_proba(calibration_features(score, groups))[:, 1]
        target = model.predict_proba(calibration_features(target_score, target_groups))[
            :, 1
        ]
        threshold = threshold_for(y, fitted)
        return (
            target >= threshold,
            target,
            {
                "threshold": threshold,
                "coefficients": model.coef_.tolist(),
                "intercept": model.intercept_.tolist(),
            },
        )
    if policy in {"source_macro", "source_robust"}:
        prediction = np.zeros(len(target_score), dtype=bool)
        parameters = {}
        for real in [False, True]:
            fitting = np.char.startswith(groups, "WELL-") == real
            target = np.char.startswith(target_groups, "WELL-") == real
            # Fallback uses calibration data only if a source lacks a class.
            if len(np.unique(y[fitting])) < 2:
                fitting = np.ones(len(y), dtype=bool)
            threshold = threshold_for(y[fitting], score[fitting], objective)
            prediction[target] = target_score[target] >= threshold
            parameters["physical" if real else "simulation"] = threshold
        return prediction, target_score, parameters
    weights = group_loss_weights(groups, 1.0) if policy == "group_macro" else None
    threshold = threshold_for(y, score, objective, weights)
    return target_score >= threshold, target_score, {"threshold": threshold}


def classification_metrics(y, prediction, score):
    negative, positive = y == 0, y == 1
    return {
        "auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
        "macro_f1": float(f1_score(y, prediction, average="macro", labels=[0, 1])),
        "false_positive_rate": float(prediction[negative].mean())
        if negative.any()
        else None,
        "false_negative_rate": float((~prediction[positive]).mean())
        if positive.any()
        else None,
        "windows": len(y),
        "fault_fraction": float(y.mean()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir", default="logs/benchmark_3w_binary/calibration_development"
    )
    args = p.parse_args()
    out = Path(args.output_dir)
    if (out / "results.json").exists():
        raise FileExistsError("Development diagnostic already exists")
    out.mkdir(parents=True, exist_ok=True)
    selection = json.loads(Path("docs/results/3w_range_selection.json").read_text())
    first = Path(next(iter(selection["selected"]["single_row"]["sources"].values())))
    cfg = SimpleNamespace(**json.loads((first / "config.json").read_text()))
    cfg.evaluate_test = False
    data, _ = load_data(cfg)
    val = data["val"]
    y, groups = np.asarray(val["labels"]), np.asarray(val["group_ids"])
    folds = calibration_folds(groups)
    physical = np.char.startswith(groups, "WELL-")
    rows, per_group = [], {}
    for label in ["single_row", "tabm_ple", "xgb"]:
        chosen = selection["selected"][label]
        model = chosen["recipe"]["model"]
        score = np.mean(
            [
                np.load(Path(directory) / f"{model}_{seed}_predictions.npz")["val"]
                for seed, directory in chosen["sources"].items()
            ],
            axis=0,
        )
        for policy in POLICIES:
            predictions = np.zeros(len(y), dtype=bool)
            probabilities = np.zeros(len(y))
            fold_parameters = []
            for fold in range(4):
                fit, target = folds != fold, folds == fold
                prediction, probability, parameters = apply_policy(
                    policy,
                    y[fit],
                    score[fit],
                    groups[fit],
                    score[target],
                    groups[target],
                )
                predictions[target], probabilities[target] = prediction, probability
                fold_parameters.append(
                    {
                        "fold": fold,
                        "parameters": parameters,
                        "metrics": classification_metrics(
                            y[target], prediction, probability
                        ),
                    }
                )
            result = {
                "model": label,
                "policy": policy,
                "pooled": classification_metrics(y, predictions, probabilities),
                "physical": classification_metrics(
                    y[physical], predictions[physical], probabilities[physical]
                ),
                "simulation": classification_metrics(
                    y[~physical], predictions[~physical], probabilities[~physical]
                ),
                "folds": fold_parameters,
            }
            rows.append(result)
            np.savez_compressed(
                out / f"{label}_{policy}.npz",
                prediction=predictions,
                probability=probabilities,
                folds=folds,
            )
            if policy == "global_macro":
                per_group[label] = {
                    g: classification_metrics(
                        y[groups == g],
                        predictions[groups == g],
                        probabilities[groups == g],
                    )
                    for g in np.unique(groups[physical])
                }
            print(label, policy, json.dumps(result["pooled"]), flush=True)
    result = {
        "scope": __doc__,
        "policies_fixed_before_reading_diagnostic_results": POLICIES,
        "fold_seed": 42,
        "fold_groups": {
            str(i): np.unique(groups[folds == i]).tolist() for i in range(4)
        },
        "results": rows,
        "physical_groups_global_macro": per_group,
    }
    (out / "results.json").write_text(
        json.dumps(json_safe(result), indent=2, allow_nan=False)
    )


if __name__ == "__main__":
    main()
