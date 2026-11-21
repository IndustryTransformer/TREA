"""Verify and package matched development reference-feature screens, never tests."""

import json

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from examples.benchmark_3w_grouped import choose_threshold, load_data, metrics
from examples.benchmark_3w_initial_reference import validate_args
from examples.freeze_3w_robustness import file_digest
from examples.summarize_3w_grouped import json_safe


def summarize(rows, model):
    fits = sorted([r for r in rows if r["model"] == model], key=lambda r: r["seed"])
    if [r["seed"] for r in fits] != [42, 43, 44]:
        raise ValueError("All three distinct seeds are required")
    ensembles = [r for r in rows if r["model"] == model + "_ensemble"]
    if len(ensembles) != 1:
        raise ValueError("Exactly one explicit probability ensemble is required")
    result = {"model": model, "seeds": [42, 43, 44]}
    for metric in ["auc", "macro_f1"]:
        values = [r["validation"][metric] for r in fits]
        result["single_fit_" + metric] = {
            "values": values,
            "mean": float(np.mean(values)),
            "population_std": float(np.std(values)),
        }
    result["ensemble"] = ensembles[0]["validation"]
    result["source_ensemble"] = {
        k: ensembles[0][k] for k in ["physical", "simulation"] if k in ensembles[0]
    }
    return result


def main():
    root = Path("logs/benchmark_3w_binary")
    reference = root / "initial_reference_v1"
    config = json.loads((reference / "config.json").read_text())
    args = SimpleNamespace(**config)
    validate_args(args)
    data, _ = load_data(args)  # validate_args rejects evaluate_test before this call.
    y = data["val"]["labels"]
    recipe_keys = [
        "group_manifest",
        "value_policy",
        "feature_family",
        "relative_stats",
        "group_loss_power",
        "train_cap",
        "preprocess",
        "width",
        "layers",
        "lr",
        "weight_decay",
        "epochs",
        "patience",
        "row_dropout",
        "seeds",
    ]
    runs = {}
    for name, models in [
        ("causal_context_trea", ["single_row"]),
        ("causal_context_trees_tabm", ["xgb", "tabm_ple"]),
        ("initial_reference_v1", ["xgb", "tabm_ple", "single_row"]),
        ("combined_reference_fast_v1", ["xgb", "tabm_ple"]),
    ]:
        directory = root / name
        current = json.loads((directory / "config.json").read_text())
        for key in recipe_keys:
            if current.get(key) != config.get(key):
                raise ValueError(f"Unmatched recipe {name}: {key}")
        source_file = directory / "source_hashes.json"
        if source_file.exists():
            for path, digest in json.loads(source_file.read_text()).items():
                if file_digest(path) != digest:
                    raise ValueError(f"Source changed: {path}")
        for seed in [42, 43, 44]:
            np.testing.assert_array_equal(
                np.load(reference / f"train_indices_{seed}.npy"),
                np.load(directory / f"train_indices_{seed}.npy"),
            )
        rows = json.loads((directory / "results.json").read_text())
        summaries = []
        for model in models:
            summary = summarize(rows, model)
            scores = [
                np.load(directory / f"{model}_{s}_predictions.npz")["val"]
                for s in [42, 43, 44]
            ]
            ensemble = np.load(directory / f"{model}_ensemble_predictions.npz")["val"]
            np.testing.assert_array_equal(np.mean(scores, axis=0), ensemble)
            for seed, score in list(zip([42, 43, 44], scores, strict=True)) + [
                (None, ensemble)
            ]:
                expected_model = model if seed is not None else model + "_ensemble"
                row = next(
                    r
                    for r in rows
                    if r["model"] == expected_model and r.get("seed") == seed
                )
                measured = metrics(
                    y, score, choose_threshold(y, score), data["val"]["file_ids"]
                )
                for key in ["auc", "macro_f1", "threshold_from_validation"]:
                    np.testing.assert_allclose(
                        measured[key], row["validation"][key], atol=1e-10, rtol=0
                    )
            summaries.append(summary)
        runs[name] = {
            "config": current,
            "summaries": summaries,
            "results_sha256": file_digest(directory / "results.json"),
        }
    result = {
        "scope": (
            "Development-only repeated internal validation. "
            "Not confirmation or a new independent test."
        ),
        "checks": (
            "Same recipe and sample indices; all single-fit metrics and "
            "probability ensembles recomputed; recorded source hashes verified."
        ),
        "limitations": [
            "All arms have longer context than isolated 96-reading windows.",
            "Checkpoints and F1 thresholds both selected on internal validation.",
            "Repeated selection on eight physical validation wells risks overfitting.",
            "Consumed held-out results remain frozen; no target-attainment claim.",
        ],
        "runs": runs,
    }
    output = Path("docs/results/3w_reference_screen.json")
    output.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False))
    for name, run in runs.items():
        for summary in run["summaries"]:
            print(
                name,
                summary["model"],
                summary["single_fit_auc"],
                summary["ensemble"]["auc"],
                summary["ensemble"]["macro_f1"],
            )


if __name__ == "__main__":
    main()
