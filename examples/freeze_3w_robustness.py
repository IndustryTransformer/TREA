"""Freeze all-fold selection before any outer-fold evaluation."""

import hashlib
import json

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from examples.benchmark_3w_grouped import choose_threshold, load_data, metrics
from examples.summarize_3w_grouped import json_safe


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def candidate_rows(rows, model):
    fits = sorted(
        [r for r in rows if r["model"] == model and r.get("seed") is not None],
        key=lambda r: r["seed"],
    )
    if [r["seed"] for r in fits] != [42, 43, 44]:
        raise ValueError(f"Incomplete or duplicated seed fits: {model}")
    ensembles = [r for r in rows if r["model"] == model + "_ensemble"]
    if len(ensembles) != 1 or ensembles[0]["seeds"] != [42, 43, 44]:
        raise ValueError(f"Missing explicit three-fit ensemble: {model}")
    return fits, ensembles[0]


def verify_history_gate(fold_root):
    path = fold_root / "history_equivalence.json"
    gate = json.loads(path.read_text())
    if not gate.get("passed"):
        raise ValueError("Label-independent history equivalence not verified")
    if gate["config_sha256"] != file_digest(fold_root / "window/config.json"):
        raise ValueError("History gate belongs to different fitting inputs")
    if gate["reconstruction_source_sha256"] != file_digest(
        "examples/three_w_label_blind_history.py"
    ):
        raise ValueError("Raw reconstruction changed since verification")
    if [r["split"] for r in gate["reports"]] != ["train", "val"]:
        raise ValueError("History verification must cover train and validation")
    if not all(
        r["annotation_blind_history"]
        and r["cache_history_equivalence_required"]
        and r["scored_windows"] > 0
        for r in gate["reports"]
    ):
        raise ValueError("Incomplete raw-history verification")
    return path


def main():
    root = Path("logs/benchmark_3w_binary/robustness_v1")
    output = Path("docs/results/3w_robustness_selection.json")
    if output.exists():
        raise FileExistsError("Robustness selection is already frozen")
    plan_path = Path("docs/results/3w_grouped_robustness_v1/plan.json")
    plan = json.loads(plan_path.read_text())
    recipe = plan["shared_recipe"]
    hashes = {str(plan_path): file_digest(plan_path)}
    selected_folds = []
    for fold in range(4):
        fold_root = root / f"fold_{fold}"
        state = json.loads((fold_root / "training_state.json").read_text())
        if state["state"] != "fits_complete_outer_unread":
            raise ValueError(f"Fold {fold} fits are not complete")
        history_gate = verify_history_gate(fold_root)
        hashes[str(history_gate)] = file_digest(history_gate)
        for file, expected in state["source_sha256"].items():
            if file_digest(file) != expected:
                raise ValueError(f"Training source changed: {file}")
            hashes[file] = expected
        candidates = []
        reference_protocol = None
        expected_indices = {}
        for view, models in [
            ("window", ["xgb", "single_row"]),
            ("causal", ["xgb", "single_row", "tabm_ple"]),
        ]:
            directory = fold_root / view
            config = json.loads((directory / "config.json").read_text())
            if config["evaluate_test"]:
                raise ValueError("Outer data was enabled during fitting")
            if Path(config["group_manifest"]).name != f"fold_{fold}.json":
                raise ValueError("Wrong outer group manifest")
            for key in [
                "group_loss_power",
                "value_policy",
                "train_cap",
                "width",
                "layers",
                "lr",
                "epochs",
                "patience",
            ]:
                if config[key] != recipe[key]:
                    raise ValueError(f"Recipe deviated from plan: {key}")
            if config.get("row_dropout", 0) != recipe["dropout"]:
                raise ValueError("Dropout deviated from plan")
            if (
                view == "causal"
                and config["context_information"]["alpha"] != recipe["causal_alpha"]
            ):
                raise ValueError("Causal decay deviated from plan")
            args = SimpleNamespace(**config)
            args.evaluate_test = False
            data, protocol = load_data(args)
            if reference_protocol is not None and protocol != reference_protocol:
                raise ValueError("Different datasets across candidate recipes")
            reference_protocol = protocol
            dataset_config = {
                key: protocol["train"][key]
                for key in [
                    "seed",
                    "holdout",
                    "internal_val",
                    "window",
                    "data_path",
                    "version",
                    "value_policy",
                    "group_manifest_sha256",
                ]
            }
            cache_key = hashlib.sha256(
                json.dumps(dataset_config, sort_keys=True).encode()
            ).hexdigest()[:16]
            if (Path(args.cache_dir) / cache_key / "test.json").exists():
                raise ValueError(
                    "Outer arrays were materialized before selection freeze"
                )
            for seed in [42, 43, 44]:
                idx_path = directory / f"train_indices_{seed}.npy"
                indices = np.load(idx_path)
                if seed not in expected_indices:
                    expected_indices[seed] = np.random.default_rng(seed).choice(
                        len(data["train"]["labels"]),
                        min(recipe["train_cap"], len(data["train"]["labels"])),
                        replace=False,
                    )
                np.testing.assert_array_equal(indices, expected_indices[seed])
                hashes[str(idx_path)] = file_digest(idx_path)
            rows = json.loads((directory / "results.json").read_text())
            for model in models:
                fits, ensemble = candidate_rows(rows, model)
                paths = [
                    directory / f"{model}_{seed}_predictions.npz"
                    for seed in [42, 43, 44]
                ]
                predictions = [np.load(path)["val"] for path in paths]
                score = np.mean(predictions, axis=0)
                threshold = choose_threshold(data["val"]["labels"], score)
                measured = metrics(
                    data["val"]["labels"], score, threshold, data["val"]["file_ids"]
                )
                reported = ensemble.get(
                    "validation", ensemble.get("splits", {}).get("val")
                )
                for metric in ["auc", "macro_f1"]:
                    np.testing.assert_allclose(
                        measured[metric], reported[metric], atol=1e-10, rtol=0
                    )
                for seed in [42, 43, 44]:
                    extension = "joblib" if model == "xgb" else "pt"
                    paths.append(directory / f"{model}_{seed}.{extension}")
                    if model != "xgb":
                        paths.append(directory / f"{model}_{seed}_preprocess.joblib")
                for path in paths:
                    hashes[str(path)] = file_digest(path)
                candidates.append(
                    {
                        "candidate": f"{model}_{view}",
                        "model": model,
                        "view": view,
                        "directory": str(directory),
                        "validation": measured,
                        "fit_seeds": [r["seed"] for r in fits],
                    }
                )
            for file in ["config.json", "protocol.json", "results.json"]:
                path = directory / file
                hashes[str(path)] = file_digest(path)
        selected_folds.append(
            {
                "fold": fold,
                "candidates": candidates,
                "selected_neural": max(
                    [c for c in candidates if c["model"] != "xgb"],
                    key=lambda c: c["validation"]["auc"],
                ),
                "selected_tree": max(
                    [c for c in candidates if c["model"] == "xgb"],
                    key=lambda c: c["validation"]["auc"],
                ),
            }
        )
    hashes[__file__] = file_digest(__file__)
    hashes["examples/three_w_label_blind_history.py"] = file_digest(
        "examples/three_w_label_blind_history.py"
    )
    hashes["examples/evaluate_3w_robustness.py"] = file_digest(
        "examples/evaluate_3w_robustness.py"
    )
    result = {
        "scope": plan["scope"],
        "plan": plan,
        "folds": selected_folds,
        "artifact_sha256": hashes,
        "outer_groups_unread_at_freeze": True,
    }
    output.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False))
    print(
        json.dumps(
            [
                {k: f[k] for k in ["fold", "selected_neural", "selected_tree"]}
                for f in selected_folds
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
