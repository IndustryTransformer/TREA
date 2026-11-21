"""Evaluate frozen validation-selected 3W fits once on the reserved grouped test.

Example: python -m examples.evaluate_3w_reserved --runs RUN1 RUN2 --output-dir OUT
All model selection must be complete before running this script.
"""

import argparse
import hashlib
import json

from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import torch

from sklearn.metrics import roc_auc_score

from examples.benchmark_3w_grouped import (
    build_model,
    choose_threshold,
    extract_naive_features,
    load_data,
    metrics,
    predict,
    summary_features,
)


def grouped_bootstrap(y, a, b, groups, repeats=500):
    """Paired cluster bootstrap: resample groups, never independent windows."""
    _, inv = np.unique(groups, return_inverse=True)
    n = inv.max() + 1
    rng = np.random.default_rng(20260917)
    differences = []
    for _ in range(repeats):
        counts = np.bincount(rng.integers(n, size=n), minlength=n)
        weights = counts[inv]
        if min(weights[y == 0].sum(), weights[y == 1].sum()) == 0:
            continue
        differences.append(
            roc_auc_score(y, a, sample_weight=weights)
            - roc_auc_score(y, b, sample_weight=weights)
        )
    return {
        "delta_auc": float(roc_auc_score(y, a) - roc_auc_score(y, b)),
        "cluster_bootstrap_95_interval": np.quantile(
            differences, [0.025, 0.975]
        ).tolist(),
        "valid_replicates": len(differences),
        "groups": int(n),
    }


def validate_evaluation_plan(run_paths, model_names, blends=()):
    """Reject incomplete/ambiguous plans before opening any reserved signals."""
    found, reference = {}, None
    for spec in run_paths:
        directory, _, restriction = spec.partition("::")
        model, _, alias = restriction.partition("=")
        path = Path(directory)
        protocol = json.loads((path / "protocol.json").read_text())
        signature = {s: protocol[s] for s in ["train", "val"]}
        if reference is not None and signature != reference:
            raise ValueError("Cannot compare fits from different cached protocols")
        reference = signature
        config = json.loads((path / "config.json").read_text())
        if config.get("group_manifest"):
            digest = hashlib.sha256(
                json.dumps(
                    json.loads(Path(config["group_manifest"]).read_text()),
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            if digest != protocol["train"].get("group_manifest_sha256"):
                raise ValueError("Frozen manifest changed after fitting")
        for row in json.loads((path / "results.json").read_text()):
            name, seed = row["model"], row.get("seed")
            if seed is None or name not in model_names or (model and model != name):
                continue
            label = alias or name
            if seed in found.setdefault(label, set()):
                raise ValueError(f"Duplicate fit {label}/{seed}")
            found[label].add(seed)
    if not found or any(seeds != {42, 43, 44} for seeds in found.values()):
        raise ValueError("Each selected recipe requires seeds 42,43,44")
    for spec in blends:
        label, _, components = spec.partition("=")
        members = components.split(",")
        if (
            not label
            or label in found
            or len(set(members)) < 2
            or len(members) != len(set(members))
            or not set(members) <= set(found)
        ):
            raise ValueError("Invalid frozen blend plan")


def evaluate(run_paths, model_names, out, blends=()):
    validate_evaluation_plan(run_paths, model_names, blends)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise FileExistsError("Reserved evaluation already exists")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = None
    reference = None
    outputs, costs, results = {}, {}, []
    for spec in run_paths:
        directory, _, restricted_model = spec.partition("::")
        restricted_model, _, alias = restricted_model.partition("=")
        if alias and not restricted_model:
            raise ValueError("An output alias requires an explicit model restriction")
        path = Path(directory)
        args = SimpleNamespace(**json.loads((path / "config.json").read_text()))
        args.evaluate_test = True
        torch.set_num_threads(args.threads)
        signature = (
            args.split_seed,
            args.holdout_fraction,
            args.internal_val_fraction,
            args.data_path,
            getattr(args, "value_policy", "legacy_clip"),
            hashlib.sha256(
                json.dumps(
                    json.loads(Path(args.group_manifest).read_text()), sort_keys=True
                ).encode()
            ).hexdigest()
            if getattr(args, "group_manifest", None)
            else None,
        )
        if reference is not None and signature != reference:
            raise ValueError("Cannot compare fits from different protocols")
        reference = signature
        if data is None:
            data, manifest = load_data(args)
            (out / "protocol.json").write_text(json.dumps(manifest, indent=2))
        for row in json.loads((path / "results.json").read_text()):
            name, seed = row["model"], row.get("seed")
            if name not in model_names or seed is None:
                continue
            if restricted_model and name != restricted_model:
                continue
            output_name = alias or name
            if any(seed == existing[0] for existing in outputs.get(output_name, [])):
                raise ValueError(f"Duplicate fit {output_name}/{seed}")
            temporal = name in {"temporal", "direct_stat", "stats_only"}
            classical = name in {"xgb", "xgb_naive", "rf"}
            xs = {}
            for split, d in data.items():
                if temporal:
                    xs[split] = d["windows"]
                elif name == "xgb_naive":
                    xs[split] = extract_naive_features(d["windows"])
                else:
                    xs[split] = summary_features(
                        d,
                        manifest["train"]["normalization"],
                        args.feature_family,
                        getattr(args, "relative_stats", False),
                    )
            indices = np.load(path / f"train_indices_{seed}.npy")
            xs["train"] = xs["train"][indices]
            if not classical and not temporal and args.preprocess != "raw":
                scaler = joblib.load(path / f"{name}_{seed}_preprocess.joblib")
                xs = {s: scaler.transform(x).astype(np.float32) for s, x in xs.items()}
            if classical:
                model = joblib.load(path / f"{name}_{seed}.joblib")
                val = model.predict_proba(xs["val"])[:, 1]
                score = model.predict_proba(xs["test"])[:, 1]
            else:
                model = build_model(name, xs["train"], args).to(device)
                state = torch.load(
                    path / f"{name}_{seed}.pt", map_location="cpu", weights_only=False
                )
                model.load_state_dict(state["state_dict"], strict=True)
                val = predict(model, xs["val"], args.batch_size, device)
                score = predict(model, xs["test"], args.batch_size, device)
            saved = np.load(path / f"{name}_{seed}_predictions.npz")["val"]
            np.testing.assert_allclose(val, saved, atol=1e-6, rtol=1e-6)
            threshold = choose_threshold(data["val"]["labels"], saved)
            detail = metrics(
                data["test"]["labels"], score, threshold, data["test"]["file_ids"]
            )
            result = {
                "model": output_name,
                "architecture": name,
                "seed": seed,
                "source": str(path),
                "test": detail,
            }
            results.append(result)
            outputs.setdefault(output_name, []).append((seed, saved, score))
            costs.setdefault(output_name, []).append(row["train_seconds"])
            np.savez_compressed(
                out / f"{output_name}_{seed}_predictions.npz", val=saved, test=score
            )
            (out / "results.json").write_text(json.dumps(results, indent=2))
            print(json.dumps(result), flush=True)
            del model, xs
            if device.type == "cuda":
                torch.cuda.empty_cache()
    ensemble_scores, ensemble_validation = {}, {}
    for name, items in outputs.items():
        if sorted(i[0] for i in items) != [42, 43, 44]:
            raise ValueError(f"Missing seeds for {name}")
        val = np.mean([i[1] for i in items], axis=0)
        score = np.mean([i[2] for i in items], axis=0)
        threshold = choose_threshold(data["val"]["labels"], val)
        detail = metrics(
            data["test"]["labels"], score, threshold, data["test"]["file_ids"]
        )
        result = {
            "model": name + "_ensemble",
            "seeds": [42, 43, 44],
            "train_seconds": sum(costs[name]),
            "test": detail,
        }
        # Separately report identifiable wells versus unlinked simulation groups.
        real = np.char.startswith(np.asarray(data["test"]["group_ids"]), "WELL-")
        for label, mask in [("physical_wells", real), ("simulation_instances", ~real)]:
            if mask.any():
                result[label] = metrics(
                    data["test"]["labels"][mask],
                    score[mask],
                    threshold,
                    data["test"]["file_ids"][mask],
                )
        results.append(result)
        ensemble_scores[name] = score
        ensemble_validation[name] = val
        np.savez_compressed(
            out / f"{name}_ensemble_predictions.npz", val=val, test=score
        )
        print(json.dumps(result), flush=True)
    for spec in blends:
        name, _, components = spec.partition("=")
        members = components.split(",")
        if not name or name in ensemble_scores or len(set(members)) < 2:
            raise ValueError("Blend needs a unique name and distinct fitted recipes")
        score = np.mean([ensemble_scores[n] for n in members], axis=0)
        val = np.mean([ensemble_validation[n] for n in members], axis=0)
        threshold = choose_threshold(data["val"]["labels"], val)
        result = {
            "model": name,
            "components": members,
            "independent_fits": 3 * len(members),
            "train_seconds": sum(sum(costs[n]) for n in members),
            "test": metrics(
                data["test"]["labels"], score, threshold, data["test"]["file_ids"]
            ),
        }
        for label, mask in [("physical_wells", real), ("simulation_instances", ~real)]:
            if mask.any():
                result[label] = metrics(
                    data["test"]["labels"][mask],
                    score[mask],
                    threshold,
                    data["test"]["file_ids"][mask],
                )
        results.append(result)
        ensemble_scores[name] = score
        np.savez_compressed(out / f"{name}_predictions.npz", val=val, test=score)
        print(json.dumps(result), flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    intervals = {}
    if "xgb" in ensemble_scores:
        for name, score in ensemble_scores.items():
            if name in {"xgb", "xgb_naive", "rf"}:
                continue
            intervals[name] = grouped_bootstrap(
                data["test"]["labels"],
                score,
                ensemble_scores["xgb"],
                data["test"]["group_ids"],
            )
    (out / "uncertainty.json").write_text(json.dumps(intervals, indent=2))
    print(json.dumps(intervals), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--models", default="xgb,xgb_naive,rf,tabm,tabm_ple")
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--blend",
        action="append",
        default=[],
        help="Frozen equal-probability mixture, e.g. neural=single_row,tabm_ple",
    )
    args = p.parse_args()
    evaluate(args.runs, args.models.split(","), args.output_dir, args.blend)
