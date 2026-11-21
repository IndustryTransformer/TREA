"""Verify the fixed regularization contrast against its matched development fits."""

import json

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sklearn.metrics import roc_auc_score

from examples.benchmark_3w_grouped import choose_threshold, load_data, metrics
from examples.benchmark_3w_initial_reference import validate_args
from examples.collect_3w_reference_screen import summarize
from examples.freeze_3w_robustness import file_digest
from examples.summarize_3w_grouped import json_safe


def verify_contrast(baseline, candidate):
    ignored = {"models", "output_dir", "weight_decay"}
    for key in set(baseline) | set(candidate):
        if key not in ignored and baseline.get(key) != candidate.get(key):
            raise ValueError(f"Unexpected recipe change: {key}")
    if baseline["weight_decay"] != 0.0003 or candidate["weight_decay"] != 0.03:
        raise ValueError("Wrong weight-decay contrast")


def main():
    root = Path("logs/benchmark_3w_binary")
    new = root / "causal_regularization_v1"
    config = json.loads((new / "config.json").read_text())
    args = SimpleNamespace(**config)
    validate_args(args)
    rows = json.loads((new / "results.json").read_text())
    for model in ["single_row", "tabm_ple"]:
        summarize(rows, model)  # Require completion before any result collection.
    hashes = json.loads((new / "source_hashes.json").read_text())
    for path, digest in hashes.items():
        if file_digest(path) != digest:
            raise ValueError(f"Source changed: {path}")
    data, _ = load_data(args)
    y, groups = data["val"]["labels"], data["val"]["group_ids"]
    physical = np.char.startswith(groups, "WELL-")
    result = {
        "scope": "Repeated internal validation only, not independent confirmation.",
        "plan": json.loads((new / "plan.json").read_text()),
        "checks": "Recipes, sampled rows, probability ensembles and metrics verified.",
        "runs": {},
        "comparisons": {},
        "artifact_sha256": hashes,
        "limitations": [
            "Validation selects both checkpoints and F1 thresholds.",
            "Per-well diagnostics below are exploratory development analysis.",
            "Neither consumed tests nor frozen outer folds are evaluated here.",
            "Seed population standard deviations do not measure new-well uncertainty.",
        ],
    }
    for model, baseline_dir in [
        ("single_row", root / "causal_context_trea"),
        ("tabm_ple", root / "causal_context_trees_tabm"),
    ]:
        baseline = json.loads((baseline_dir / "config.json").read_text())
        verify_contrast(baseline, config)
        for seed in [42, 43, 44]:
            np.testing.assert_array_equal(
                np.load(baseline_dir / f"train_indices_{seed}.npy"),
                np.load(new / f"train_indices_{seed}.npy"),
            )
        for arm, directory in [("baseline", baseline_dir), ("stronger_decay", new)]:
            fits = json.loads((directory / "results.json").read_text())
            summary = summarize(fits, model)
            scores = [
                np.load(directory / f"{model}_{seed}_predictions.npz")["val"]
                for seed in [42, 43, 44]
            ]
            ensemble = np.load(directory / f"{model}_ensemble_predictions.npz")["val"]
            np.testing.assert_array_equal(np.mean(scores, axis=0), ensemble)
            for seed, score in list(zip([42, 43, 44], scores, strict=True)) + [
                (None, ensemble)
            ]:
                name = model if seed is not None else model + "_ensemble"
                row = next(
                    r for r in fits if r["model"] == name and r.get("seed") == seed
                )
                threshold = choose_threshold(y, score)
                measured = metrics(y, score, threshold, data["val"]["file_ids"])
                for key in ["auc", "macro_f1", "threshold_from_validation"]:
                    np.testing.assert_allclose(
                        measured[key], row["validation"][key], rtol=0, atol=1e-10
                    )
            threshold = choose_threshold(y, ensemble)
            summary["source_ensemble"] = {
                source: metrics(
                    y[mask], ensemble[mask], threshold, data["val"]["file_ids"][mask]
                )
                for source, mask in [("physical", physical), ("simulation", ~physical)]
            }
            summary["physical_wells"] = []
            for group in np.unique(groups[physical]):
                mask = groups == group
                summary["physical_wells"].append(
                    {
                        "group": str(group),
                        "windows": int(mask.sum()),
                        "auc": float(roc_auc_score(y[mask], ensemble[mask]))
                        if len(np.unique(y[mask])) == 2
                        else None,
                    }
                )
            result["runs"][model + "/" + arm] = summary
            for filename in (
                ["config.json", "results.json"]
                + [f"{model}_{seed}_predictions.npz" for seed in [42, 43, 44]]
                + [f"{model}_ensemble_predictions.npz"]
            ):
                path = directory / filename
                result["artifact_sha256"][str(path)] = file_digest(path)
        old, current = [
            result["runs"][model + "/" + arm] for arm in ["baseline", "stronger_decay"]
        ]
        result["comparisons"][model] = {
            "ensemble_auc_delta": current["ensemble"]["auc"] - old["ensemble"]["auc"],
            "ensemble_macro_f1_delta": current["ensemble"]["macro_f1"]
            - old["ensemble"]["macro_f1"],
            "single_fit_auc_mean_delta": current["single_fit_auc"]["mean"]
            - old["single_fit_auc"]["mean"],
        }
    output = Path("docs/results/3w_regularization_screen.json")
    output.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False))
    for name, summary in result["runs"].items():
        print(
            name,
            summary["single_fit_auc"],
            summary["ensemble"]["auc"],
            summary["ensemble"]["macro_f1"],
        )
    print("contrasts", json.dumps(result["comparisons"]))


if __name__ == "__main__":
    main()
