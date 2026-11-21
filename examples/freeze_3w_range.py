"""Freeze validation-selected recipes and artifact hashes before confirmation.

Run once without --evaluate to save selection, then with --evaluate to verify
the frozen files and evaluate. No confirmation signals are read during selection.
"""

import argparse
import hashlib
import json

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from examples.benchmark_3w_grouped import choose_threshold, load_data, metrics
from examples.evaluate_3w_reserved import evaluate, validate_evaluation_plan
from examples.summarize_3w_grouped import json_safe


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection", default="docs/results/3w_range_selection.json")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--output-dir", default="logs/benchmark_3w_binary/range_reserved")
    args = p.parse_args()
    path = Path(args.selection)
    if args.evaluate:
        selection = json.loads(path.read_text())
        for file, expected in selection["artifact_sha256"].items():
            if digest(file) != expected:
                raise ValueError(f"Frozen artifact changed: {file}")
        evaluate(
            selection["runs"], selection["models"], args.output_dir, selection["blends"]
        )
        return
    if path.exists():
        raise FileExistsError("Selection is already frozen")
    summary = json.loads(Path("docs/results/3w_range_development.json").read_text())
    candidates = [e for e in summary["experiments"] if e["seeds"] == [42, 43, 44]]

    def best(model, condition=lambda recipe: True):
        choices = [
            e
            for e in candidates
            if e["recipe"]["model"] == model and condition(e["recipe"])
        ]
        return max(choices, key=lambda e: e["ensemble"]["auc"])

    selected = {
        "single_row": best("single_row"),
        "tabm_ple": best("tabm_ple"),
        "xgb": best("xgb"),
        "xgb_standard": best(
            "xgb", lambda r: r["group_loss_power"] == 0 and not r["relative_stats"]
        ),
        "xgb_naive": best("xgb_naive"),
        "rf": best("rf"),
    }
    runs, models, hashes = [], set(), {}
    for label, experiment in selected.items():
        model = experiment["recipe"]["model"]
        models.add(model)
        for directory in dict.fromkeys(experiment["sources"].values()):
            runs.append(f"{directory}::{model}={label}")
        for seed, directory in experiment["sources"].items():
            run = Path(directory)
            extension = "joblib" if model in {"xgb", "xgb_naive", "rf"} else "pt"
            files = [
                run / f"{model}_{seed}.{extension}",
                run / f"{model}_{seed}_predictions.npz",
                run / f"train_indices_{seed}.npy",
                run / "config.json",
                run / "protocol.json",
            ]
            preprocess = run / f"{model}_{seed}_preprocess.joblib"
            if preprocess.exists():
                files.append(preprocess)
            for file in files:
                hashes[str(file)] = digest(file)
    cfg = SimpleNamespace(
        **summary["runs"][
            Path(next(iter(selected["single_row"]["sources"].values()))).name
        ]["config"]
    )
    cfg.evaluate_test = False
    data, _ = load_data(cfg)
    val = data["val"]
    recipe_scores = {}
    for label in ["single_row", "tabm_ple"]:
        recipe_scores[label] = np.mean(
            [
                np.load(Path(directory) / f"{label}_{seed}_predictions.npz")["val"]
                for seed, directory in selected[label]["sources"].items()
            ],
            axis=0,
        )
    blend_score = np.mean(list(recipe_scores.values()), axis=0)
    threshold = choose_threshold(val["labels"], blend_score)
    blend_validation = metrics(val["labels"], blend_score, threshold, val["file_ids"])
    # One predetermined equal mixture, not a weight search.
    use_blend = blend_validation["auc"] > max(
        selected[n]["ensemble"]["auc"] for n in recipe_scores
    )
    blends = ["neural_blend=single_row,tabm_ple"] if use_blend else []
    validate_evaluation_plan(runs, sorted(models), blends)
    hashes[cfg.group_manifest] = digest(cfg.group_manifest)
    for file in [
        "examples/benchmark_3w_grouped.py",
        "examples/evaluate_3w_reserved.py",
        "examples/freeze_3w_range.py",
        "utils/three_w.py",
        "trea/models/triple_attention.py",
        "trea/models/single_row.py",
    ]:
        hashes[file] = digest(file)
    payload = {
        "selection_rule": (
            "Highest three-fit internal-validation ensemble AUC per model family"
        ),
        "selected": selected,
        "runs": runs,
        "models": sorted(models),
        "blends": blends,
        "equal_neural_blend_validation": blend_validation,
        "primary_neural": "neural_blend"
        if use_blend
        else max(recipe_scores, key=lambda n: selected[n]["ensemble"]["auc"]),
        "artifact_sha256": hashes,
        "chronology": (
            "Saved before current-stage confirmation signals or "
            "results are read. Historical exposure is disclosed "
            "in the group protocol manifest."
        ),
    }
    path.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False))
    print(
        json.dumps(
            {
                "primary": payload["primary_neural"],
                "blends": blends,
                "blend_validation": blend_validation,
                "selected": {k: v["ensemble"] for k, v in selected.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
