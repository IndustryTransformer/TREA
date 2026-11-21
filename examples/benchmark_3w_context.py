"""Development ablation: exact window summaries plus strictly past-only history.

This is a longer-context setting, not the original isolated 96-step-window task.
Every model receives identical causal history; consumed tests remain untouched.
"""

import hashlib
import json
import time

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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
from examples.three_w_causal_features import causal_history_features


def audit_chronology(protocol):
    checked = []
    for split in ["train", "val"]:
        manifest = protocol[split]
        root = Path(manifest["data_path"])
        for file in manifest["files"]:
            index = pd.read_parquet(root / file, columns=[]).index
            if (
                not isinstance(index, pd.DatetimeIndex)
                or index.hasnans
                or not index.is_monotonic_increasing
            ):
                raise ValueError(f"Cannot establish chronological order: {file}")
            checked.append(str(root / file))
    return {
        "files": len(checked),
        "all_datetime_monotonic": True,
        "inventory_sha256": hashlib.sha256("\n".join(checked).encode()).hexdigest(),
    }


def run(args):
    if args.evaluate_test:
        raise ValueError("Context development must not load consumed confirmation data")
    names = args.models.split(",")
    if not set(names) <= {"xgb", "mlp", "single_row", "tabm", "tabm_ple"}:
        raise ValueError("Unsupported context model")
    out = Path(args.output_dir)
    if (out / "results.json").exists():
        raise FileExistsError("Context experiment already exists")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("high")
    data, protocol = load_data(args)
    chronology = audit_chronology(protocol)
    args.context_information = {
        "alpha": 0.1,
        "causal": True,
        "blocks": [
            "mean_minus_first_past",
            "mean_minus_ewma_past",
            "std_over_ewma_past",
            "past_reference_available",
        ],
        "past_scope": "same file, earlier cached windows only",
        "chronology": chronology,
    }
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    features = {}
    for split, d in data.items():
        base = summary_features(
            d,
            protocol["train"]["normalization"],
            args.feature_family,
            args.relative_stats,
        )
        history = causal_history_features(d["features"], d["file_ids"], alpha=0.1)
        features[split] = np.concatenate([base, history], axis=1)
    labels = {s: np.asarray(d["labels"]) for s, d in data.items()}
    seeds = [int(s) for s in args.seeds.split(",")]
    results = []
    for name in names:
        seed_scores, costs = [], []
        for seed in seeds:
            indices = np.random.default_rng(seed).choice(
                len(labels["train"]),
                min(args.train_cap or len(labels["train"]), len(labels["train"])),
                replace=False,
            )
            np.save(out / f"train_indices_{seed}.npy", indices)
            xs = {"train": features["train"][indices], "val": features["val"]}
            ys = {"train": labels["train"][indices], "val": labels["val"]}
            weights = group_loss_weights(
                data["train"]["group_ids"][indices], args.group_loss_power
            )
            if name != "xgb" and args.preprocess != "raw":
                scaler = (
                    StandardScaler()
                    if args.preprocess == "standard"
                    else QuantileTransformer(
                        n_quantiles=1000,
                        output_distribution="normal",
                        subsample=100000,
                        random_state=seed,
                    )
                )
                scaler.fit(xs["train"])
                xs = {s: scaler.transform(x).astype(np.float32) for s, x in xs.items()}
                joblib.dump(scaler, out / f"{name}_{seed}_preprocess.joblib")
            if name == "xgb":
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
                score = estimator.predict_proba(xs["val"])[:, 1]
                joblib.dump(estimator, out / f"{name}_{seed}.joblib")
                history, parameters = [], 0
            else:
                prediction, seconds, parameters, history = train_neural(
                    name, xs, ys, seed, args, out, weights
                )
                score = prediction["val"]
            threshold = choose_threshold(ys["val"], score)
            row = {
                "model": name,
                "seed": seed,
                "train_seconds": seconds,
                "parameters": parameters,
                "history": history,
                "validation": metrics(
                    ys["val"], score, threshold, data["val"]["file_ids"]
                ),
            }
            results.append(row)
            seed_scores.append(score)
            costs.append(seconds)
            np.savez_compressed(out / f"{name}_{seed}_predictions.npz", val=score)
            (out / "results.json").write_text(json.dumps(results, indent=2))
            print(name, seed, json.dumps(row["validation"]), flush=True)
        if len(seeds) > 1:
            score = np.mean(seed_scores, axis=0)
            threshold = choose_threshold(labels["val"], score)
            row = {
                "model": name + "_ensemble",
                "seeds": seeds,
                "train_seconds": sum(costs),
                "validation": metrics(
                    labels["val"], score, threshold, data["val"]["file_ids"]
                ),
            }
            physical = np.char.startswith(np.asarray(data["val"]["group_ids"]), "WELL-")
            for source, mask in [("physical", physical), ("simulation", ~physical)]:
                row[source] = metrics(
                    labels["val"][mask],
                    score[mask],
                    threshold,
                    data["val"]["file_ids"][mask],
                )
            results.append(row)
            np.savez_compressed(out / f"{name}_ensemble_predictions.npz", val=score)
            print(name, "ensemble", json.dumps(row["validation"]), flush=True)
        (out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    run(parse_args())
