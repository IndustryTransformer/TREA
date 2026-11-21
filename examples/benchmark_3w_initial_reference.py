"""Development-only matched neural/tree screen with raw initial-sensor references."""

import hashlib
import json
import time

from pathlib import Path

import joblib
import numpy as np
import torch

from sklearn.preprocessing import QuantileTransformer
from xgboost import XGBClassifier

from examples.benchmark_3w_grouped import (
    choose_threshold,
    group_loss_weights,
    load_data,
    metrics,
    parse_args,
    train_neural,
)
from examples.summarize_3w_grouped import json_safe
from examples.three_w_initial_reference import reconstruct_development
from examples.three_w_label_blind_history import file_inventory


def validate_args(args):
    if args.evaluate_test:
        raise ValueError("Consumed evaluation groups must not be loaded")
    if (
        args.value_policy != "preserve_range"
        or args.feature_family != "all"
        or args.relative_stats
        or args.preprocess != "quantile"
        or not args.group_manifest
    ):
        raise ValueError(
            "Initial-reference screen requires frozen groups, preserved range, "
            "exact eight-stat base and quantile preprocessing"
        )
    if not set(args.models.split(",")) <= {"xgb", "single_row", "tabm_ple"}:
        raise ValueError("Unsupported initial-reference model")


def run(args):
    validate_args(args)
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError("Initial-reference experiment already exists")
    out.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("high")
    args.reference_information = {
        "prefix_limit": 37,
        "strictly_past": True,
        "annotation_blind": True,
        "initial_period_assumed_healthy": False,
        "replaces_old_four_history_blocks": True,
        "blocks": [
            "signed_log_mean_offset",
            "log_std_ratio",
            "log_range_ratio",
            "initial_reference_available",
        ],
        "reference_location": "median of first min(37,j) earlier observed window means",
        "reference_scale": "median window std + 1.4826*MAD of window means; floor1e-6",
        "scope": "Fixed train/internal-validation development; no held-out scoring.",
    }
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    sources = [
        __file__,
        "examples/three_w_initial_reference.py",
        "examples/three_w_label_blind_history.py",
        "examples/benchmark_3w_grouped.py",
        "utils/three_w.py",
        "trea/models/single_row.py",
        args.group_manifest,
    ]
    (out / "source_hashes.json").write_text(
        json.dumps(
            {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
            indent=2,
        )
    )
    data, protocol = load_data(args)
    inventory = file_inventory(args)
    features, reports = {}, []
    for split in ["train", "val"]:
        features[split], report = reconstruct_development(
            data[split], protocol[split], inventory
        )
        reports.append(report)
        np.save(out / f"{split}_features.npy", features[split])
        print("raw_reference_audit", json.dumps(report), flush=True)
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    (out / "input_audit.json").write_text(json.dumps(reports, indent=2))
    if args.prepare_only:
        return
    y = {s: np.asarray(d["labels"]) for s, d in data.items()}
    physical = np.char.startswith(data["val"]["group_ids"], "WELL-")

    def detail(score):
        threshold = choose_threshold(y["val"], score)
        result = {
            "validation": metrics(y["val"], score, threshold, data["val"]["file_ids"])
        }
        for source, mask in [("physical", physical), ("simulation", ~physical)]:
            result[source] = metrics(
                y["val"][mask], score[mask], threshold, data["val"]["file_ids"][mask]
            )
        return result

    seeds = [int(s) for s in args.seeds.split(",")]
    rows = []
    for name in args.models.split(","):
        scores, costs = [], []
        for seed in seeds:
            indices = np.random.default_rng(seed).choice(
                len(y["train"]), min(args.train_cap, len(y["train"])), replace=False
            )
            np.save(out / f"train_indices_{seed}.npy", indices)
            xs = {"train": features["train"][indices], "val": features["val"]}
            ys = {"train": y["train"][indices], "val": y["val"]}
            weights = group_loss_weights(
                data["train"]["group_ids"][indices], args.group_loss_power
            )
            if name == "xgb":
                negative, positive = np.bincount(ys["train"], minlength=2)
                model = XGBClassifier(
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
                model.fit(xs["train"], ys["train"], sample_weight=weights)
                seconds = time.perf_counter() - start
                score = model.predict_proba(xs["val"])[:, 1]
                joblib.dump(model, out / f"{name}_{seed}.joblib")
                parameters, history = 0, []
            else:
                scaler = QuantileTransformer(
                    n_quantiles=1000,
                    output_distribution="normal",
                    subsample=100000,
                    random_state=seed,
                )
                scaler.fit(xs["train"])
                xs = {s: scaler.transform(x).astype(np.float32) for s, x in xs.items()}
                joblib.dump(scaler, out / f"{name}_{seed}_preprocess.joblib")
                prediction, seconds, parameters, history = train_neural(
                    name, xs, ys, seed, args, out, weights
                )
                score = prediction["val"]
            scores.append(score)
            costs.append(seconds)
            row = {
                "model": name,
                "seed": seed,
                "train_seconds": seconds,
                "parameters": parameters,
                "history": history,
                **detail(score),
            }
            rows.append(row)
            np.savez_compressed(out / f"{name}_{seed}_predictions.npz", val=score)
            (out / "results.json").write_text(
                json.dumps(json_safe(rows), indent=2, allow_nan=False)
            )
            print(
                name,
                seed,
                json.dumps(json_safe(detail(score)), allow_nan=False),
                flush=True,
            )
        if len(seeds) > 1:
            score = np.mean(scores, axis=0)
            row = {
                "model": name + "_ensemble",
                "seeds": seeds,
                "train_seconds": sum(costs),
                **detail(score),
            }
            rows.append(row)
            np.savez_compressed(out / f"{name}_ensemble_predictions.npz", val=score)
            (out / "results.json").write_text(
                json.dumps(json_safe(rows), indent=2, allow_nan=False)
            )
            print(
                name,
                "ensemble",
                json.dumps(json_safe(detail(score)), allow_nan=False),
                flush=True,
            )


if __name__ == "__main__":
    run(parse_args())
