"""Summarize range-repair development fits; never load confirmation signals."""

import argparse
import hashlib
import json

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from examples.benchmark_3w_grouped import choose_threshold, load_data, metrics
from examples.summarize_3w_grouped import json_safe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/results/3w_range_development.json")
    args = parser.parse_args()
    root = Path("logs/benchmark_3w_binary")
    buckets, runs, data, manifests = {}, {}, None, None
    defaults = {
        "preprocess": "standard",
        "feature_family": "all",
        "balance": "sqrt",
        "group_loss_power": 0.0,
        "relative_stats": False,
        "shuffle_time": False,
        "width": 128,
        "layers": 2,
        "row_dropout": 0.0,
        "lr": 3e-4,
        "weight_decay": 3e-4,
        "train_cap": 200000,
        "epochs": 30,
        "patience": 6,
        "tabm_k": 16,
        "batch_size": 512,
    }
    reference = None
    for directory in sorted(root.glob("range_confirmation*")):
        if not (directory / "results.json").exists():
            continue
        config = json.loads((directory / "config.json").read_text())
        protocol = json.loads((directory / "protocol.json").read_text())
        signature = {
            s: {
                k: protocol[s].get(k)
                for k in [
                    "seed",
                    "holdout",
                    "internal_val",
                    "value_policy",
                    "group_manifest_sha256",
                    "data_path",
                    "groups",
                    "normalization",
                ]
            }
            for s in ["train", "val"]
        }
        if reference is not None and signature != reference:
            raise ValueError("Mismatched development protocols")
        reference = signature
        if data is None:
            cfg = SimpleNamespace(**config)
            cfg.evaluate_test = False
            data, manifests = load_data(cfg)
        rows = json.loads((directory / "results.json").read_text())
        runs[directory.name] = {"config": config, "results": rows}
        for row in rows:
            if row.get("seed") is None:
                continue
            recipe = {k: config.get(k, v) for k, v in defaults.items()}
            recipe["model"] = row["model"]
            # These switches do not affect tree fitting or temporal inputs.
            if row["model"] in {"xgb", "xgb_naive", "rf"}:
                recipe["preprocess"] = "raw"
            key = json.dumps(recipe, sort_keys=True)
            bucket = buckets.setdefault(key, {"recipe": recipe, "fits": {}})
            seed = row["seed"]
            if seed in bucket["fits"]:
                raise ValueError(f"Duplicate fit: {key}/{seed}")
            bucket["fits"][seed] = (directory, row)
    summary = []
    for key, bucket in buckets.items():
        fits = bucket["fits"]
        seeds = sorted(fits)
        row = {
            "id": hashlib.sha256(key.encode()).hexdigest()[:12],
            "recipe": bucket["recipe"],
            "seeds": seeds,
            "sources": {s: str(fits[s][0]) for s in seeds},
            "single_fit_auc_mean": float(
                np.mean([fits[s][1]["splits"]["val"]["auc"] for s in seeds])
            ),
            "single_fit_auc_std": float(
                np.std([fits[s][1]["splits"]["val"]["auc"] for s in seeds])
            ),
            "single_fit_macro_f1_mean": float(
                np.mean([fits[s][1]["splits"]["val"]["macro_f1"] for s in seeds])
            ),
        }
        if seeds == [42, 43, 44]:
            model = bucket["recipe"]["model"]
            score = np.mean(
                [
                    np.load(fits[s][0] / f"{model}_{s}_predictions.npz")["val"]
                    for s in seeds
                ],
                axis=0,
            )
            val = data["val"]
            threshold = choose_threshold(val["labels"], score)
            row["ensemble"] = metrics(val["labels"], score, threshold, val["file_ids"])
            physical = np.char.startswith(np.asarray(val["group_ids"]), "WELL-")
            for label, mask in [("physical", physical), ("simulation", ~physical)]:
                row[label] = metrics(
                    val["labels"][mask], score[mask], threshold, val["file_ids"][mask]
                )
        summary.append(row)
    summary.sort(
        key=lambda row: row.get("ensemble", {}).get("auc", row["single_fit_auc_mean"]),
        reverse=True,
    )
    result = {
        "scope": "Internal validation only; no confirmation selection by test",
        "protocol": manifests,
        "experiments": summary,
        "runs": runs,
    }
    Path(args.output).write_text(
        json.dumps(json_safe(result), indent=2, allow_nan=False)
    )
    for row in summary:
        r = row["recipe"]
        print(
            row["id"],
            r["model"],
            r["feature_family"],
            "group",
            r["group_loss_power"],
            "relative",
            r["relative_stats"],
            "seeds",
            row["seeds"],
            "mean",
            round(row["single_fit_auc_mean"], 5),
            "ensemble",
            round(row.get("ensemble", {}).get("auc", 0), 5),
            flush=True,
        )


if __name__ == "__main__":
    main()
