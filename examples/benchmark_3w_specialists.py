"""Development-only physical/simulation specialist ablation on exact summaries.

Each seed uses the same sampled rows as the joint model, partitioned between two
experts. Preprocessing, class balancing and early stopping are source-specific;
this is a training-recipe comparison, not an isolated negative-transfer estimate.
"""

import copy
import json
import time

from pathlib import Path

import joblib
import numpy as np
import torch

from sklearn.preprocessing import QuantileTransformer, StandardScaler
from xgboost import XGBClassifier

from examples.benchmark_3w_grouped import (
    choose_threshold,
    group_loss_weights,
    load_data,
    metrics,
    parse_args,
    summary_features,
    train_neural,
)


def source_masks(groups):
    physical = np.char.startswith(np.asarray(groups), "WELL-")
    return {"physical": physical, "simulation": ~physical}


def run(args):
    if args.evaluate_test:
        raise ValueError(
            "Specialist development must not load consumed confirmation data"
        )
    models = args.models.split(",")
    if not set(models) <= {"xgb", "mlp", "single_row", "tabm", "tabm_ple"}:
        raise ValueError("Specialist ablation supports exact-summary models only")
    out = Path(args.output_dir)
    if (out / "results.json").exists():
        raise FileExistsError("Specialist result directory already exists")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("high")
    data, protocol = load_data(args)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    features = {
        s: summary_features(
            d,
            protocol["train"]["normalization"],
            args.feature_family,
            args.relative_stats,
        )
        for s, d in data.items()
    }
    seeds = [int(s) for s in args.seeds.split(",")]
    labels = {s: np.asarray(d["labels"]) for s, d in data.items()}
    rows = []
    for model in models:
        seed_scores, seed_costs = [], []
        for seed in seeds:
            indices = np.random.default_rng(seed).choice(
                len(labels["train"]),
                min(args.train_cap or len(labels["train"]), len(labels["train"])),
                replace=False,
            )
            np.save(out / f"train_indices_{seed}.npy", indices)
            groups = np.asarray(data["train"]["group_ids"])[indices]
            train_masks = source_masks(groups)
            val_masks = source_masks(data["val"]["group_ids"])
            score = np.zeros(len(labels["val"]), dtype=np.float32)
            costs, details = [], {}
            for source in ["physical", "simulation"]:
                train_indices = indices[train_masks[source]]
                val_mask = val_masks[source]
                xs = {
                    "train": features["train"][train_indices],
                    "val": features["val"][val_mask],
                }
                ys = {
                    "train": labels["train"][train_indices],
                    "val": labels["val"][val_mask],
                }
                if min(np.bincount(ys["train"], minlength=2)) == 0:
                    raise ValueError("Each specialist requires both training classes")
                weights = group_loss_weights(
                    groups[train_masks[source]], args.group_loss_power
                )
                expert_out = out / source
                expert_out.mkdir(exist_ok=True)
                np.save(expert_out / f"train_indices_{seed}.npy", train_indices)
                if model != "xgb" and args.preprocess != "raw":
                    scaler = (
                        StandardScaler()
                        if args.preprocess == "standard"
                        else QuantileTransformer(
                            n_quantiles=min(1000, len(xs["train"])),
                            output_distribution="normal",
                            subsample=100000,
                            random_state=seed,
                        )
                    )
                    scaler.fit(xs["train"])
                    xs = {
                        s: scaler.transform(x).astype(np.float32) for s, x in xs.items()
                    }
                    joblib.dump(
                        scaler, expert_out / f"{model}_{seed}_preprocess.joblib"
                    )
                if model == "xgb":
                    negative, positive = np.bincount(ys["train"], minlength=2)
                    estimator = XGBClassifier(
                        n_estimators=500,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        tree_method="hist",
                        n_jobs=args.threads,
                        scale_pos_weight=negative / positive,
                        random_state=seed,
                    )
                    start = time.perf_counter()
                    estimator.fit(xs["train"], ys["train"], sample_weight=weights)
                    seconds = time.perf_counter() - start
                    prediction = estimator.predict_proba(xs["val"])[:, 1]
                    joblib.dump(estimator, expert_out / f"{model}_{seed}.joblib")
                    history, parameters = [], 0
                else:
                    expert_args = copy.copy(args)
                    expert_args.output_dir = str(expert_out)
                    expert_args.source_expert = source
                    predictions, seconds, parameters, history = train_neural(
                        model, xs, ys, seed, expert_args, expert_out, weights
                    )
                    prediction = predictions["val"]
                score[val_mask] = prediction
                costs.append(seconds)
                details[source] = {
                    "train_n": len(ys["train"]),
                    "train_groups": sorted(set(groups[train_masks[source]])),
                    "train_class_counts": np.bincount(
                        ys["train"], minlength=2
                    ).tolist(),
                    "train_seconds": seconds,
                    "parameters": parameters,
                    "history": history,
                    "validation_auc": metrics(
                        ys["val"], prediction, 0.5, data["val"]["file_ids"][val_mask]
                    )["auc"],
                }
            threshold = choose_threshold(labels["val"], score)
            row = {
                "model": model,
                "seed": seed,
                "experts_per_fit": 2,
                "train_seconds": sum(costs),
                "experts": details,
                "validation": metrics(
                    labels["val"], score, threshold, data["val"]["file_ids"]
                ),
            }
            rows.append(row)
            seed_scores.append(score)
            seed_costs.append(sum(costs))
            np.savez_compressed(out / f"{model}_{seed}_predictions.npz", val=score)
            (out / "results.json").write_text(json.dumps(rows, indent=2))
            print(model, seed, json.dumps(row["validation"]), flush=True)
        if len(seeds) > 1:
            score = np.mean(seed_scores, axis=0)
            threshold = choose_threshold(labels["val"], score)
            row = {
                "model": model + "_ensemble",
                "seeds": seeds,
                "trained_experts": 2 * len(seeds),
                "train_seconds": sum(seed_costs),
                "validation": metrics(
                    labels["val"], score, threshold, data["val"]["file_ids"]
                ),
            }
            for source, mask in source_masks(data["val"]["group_ids"]).items():
                row[source] = metrics(
                    labels["val"][mask],
                    score[mask],
                    threshold,
                    data["val"]["file_ids"][mask],
                )
            rows.append(row)
            np.savez_compressed(out / f"{model}_ensemble_predictions.npz", val=score)
            print(model, "ensemble", json.dumps(row["validation"]), flush=True)
        (out / "results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    run(parse_args())
