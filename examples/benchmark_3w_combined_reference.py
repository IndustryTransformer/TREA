"""Development-only additive control: retain recent history plus initial reference."""

import argparse
import json
import time

from pathlib import Path
from types import SimpleNamespace

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
    train_neural,
)
from examples.benchmark_3w_initial_reference import validate_args
from examples.freeze_3w_robustness import file_digest
from examples.summarize_3w_grouped import json_safe
from examples.three_w_label_blind_history import file_inventory, reconstruct_split


def combine_features(initial, causal):
    if initial.shape != causal.shape or initial.ndim != 2 or initial.shape[1] % 12:
        raise ValueError("Expected aligned twelve-block inputs")
    c = initial.shape[1] // 12
    np.testing.assert_array_equal(initial[:, : 8 * c], causal[:, : 8 * c])
    return np.concatenate([initial, causal[:, 8 * c :]], axis=1)


def run(reference_run, output_dir, models):
    reference = Path(reference_run)
    args = SimpleNamespace(**json.loads((reference / "config.json").read_text()))
    args.models = models
    validate_args(args)  # Must reject outer loading before any dataset access.
    for path, expected in json.loads(
        (reference / "source_hashes.json").read_text()
    ).items():
        if file_digest(path) != expected:
            raise ValueError(f"Reference-generating source changed: {path}")
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError("Combined-reference screen already exists")
    out.mkdir(parents=True)
    args.output_dir = str(out)
    args.reference_information.update(
        replaces_old_four_history_blocks=False,
        includes_recent_history=True,
        total_summary_blocks=16,
        input_reference_run=str(reference),
        screening_rule="Matched development screen; no held-out evaluation.",
    )
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    sources = [
        Path(__file__),
        Path("examples/benchmark_3w_initial_reference.py"),
        Path("examples/benchmark_3w_grouped.py"),
        Path("examples/three_w_label_blind_history.py"),
        Path("examples/three_w_causal_features.py"),
        Path("trea/models/single_row.py"),
        reference / "config.json",
        reference / "input_audit.json",
        reference / "train_features.npy",
        reference / "val_features.npy",
    ]
    (out / "source_hashes.json").write_text(
        json.dumps({str(p): file_digest(p) for p in sources}, indent=2)
    )
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("high")
    data, protocol = load_data(args)
    inventory = file_inventory(args)
    xs_all, reports = {}, []
    for split in ["train", "val"]:
        # Verify old cached-history semantics against raw annotation-blind inputs.
        causal, report = reconstruct_split(data[split], protocol[split], inventory)
        initial = np.load(reference / f"{split}_features.npy", mmap_mode="r")
        xs_all[split] = combine_features(initial, causal)
        reports.append(report)
    (out / "input_audit.json").write_text(json.dumps(reports, indent=2))
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    labels = {s: np.asarray(d["labels"]) for s, d in data.items()}
    physical = np.char.startswith(data["val"]["group_ids"], "WELL-")

    def detail(score):
        threshold = choose_threshold(labels["val"], score)
        measured = {
            "validation": metrics(
                labels["val"], score, threshold, data["val"]["file_ids"]
            )
        }
        for source, mask in [("physical", physical), ("simulation", ~physical)]:
            measured[source] = metrics(
                labels["val"][mask],
                score[mask],
                threshold,
                data["val"]["file_ids"][mask],
            )
        return measured

    seeds = [int(s) for s in args.seeds.split(",")]
    rows = []
    for name in models.split(","):
        predictions, costs = [], []
        for seed in seeds:
            indices = np.load(reference / f"train_indices_{seed}.npy")
            expected = np.random.default_rng(seed).choice(
                len(labels["train"]),
                min(args.train_cap, len(labels["train"])),
                replace=False,
            )
            np.testing.assert_array_equal(indices, expected)
            np.save(out / f"train_indices_{seed}.npy", indices)
            xs = {"train": xs_all["train"][indices], "val": xs_all["val"]}
            ys = {"train": labels["train"][indices], "val": labels["val"]}
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
                predicted, seconds, parameters, history = train_neural(
                    name, xs, ys, seed, args, out, weights
                )
                score = predicted["val"]
            predictions.append(score)
            costs.append(seconds)
            rows.append(
                {
                    "model": name,
                    "seed": seed,
                    "train_seconds": seconds,
                    "parameters": parameters,
                    "history": history,
                    **detail(score),
                }
            )
            np.savez_compressed(out / f"{name}_{seed}_predictions.npz", val=score)
            (out / "results.json").write_text(
                json.dumps(json_safe(rows), indent=2, allow_nan=False)
            )
            print(name, seed, json.dumps(rows[-1]["validation"]), flush=True)
        score = np.mean(predictions, axis=0)
        rows.append(
            {
                "model": name + "_ensemble",
                "seeds": seeds,
                "train_seconds": sum(costs),
                **detail(score),
            }
        )
        np.savez_compressed(out / f"{name}_ensemble_predictions.npz", val=score)
        (out / "results.json").write_text(
            json.dumps(json_safe(rows), indent=2, allow_nan=False)
        )
        print(name, "ensemble", json.dumps(rows[-1]["validation"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", default="xgb,tabm_ple")
    options = parser.parse_args()
    run(options.reference_run, options.output_dir, options.models)
