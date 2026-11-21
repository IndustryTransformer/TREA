"""Stage 2 of docs/3W_ENCODER_HISTORY_PREREG.md: outer-fold scoring.

Scores the fitted ``history`` and ``none`` arms of each outer fold on that fold's
held-out groups and compares them with the frozen causal TREA and causal XGBoost
outer predictions in ``logs/benchmark_3w_binary/robustness_outer_v1``. Thresholds
come from internal validation only. Every checkpoint must reproduce its saved
validation scores before the outer rows of any fold are loaded. The consumed test
partitions stay excluded by the fold manifests.
"""

from __future__ import annotations

import argparse
import json

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import QuantileTransformer

from examples.benchmark_3w_grouped import build_model, choose_threshold, load_data
from examples.benchmark_3w_history_attention import arm_indices, predict
from examples.evaluate_3w_robustness import paired_intervals
from examples.three_w_history_attention import (
    EncoderHistoryClassifier,
    EncoderHistoryInputs,
)


SEEDS = (42, 43, 44)
ARMS = ("history", "none")


def load_fits(directory, device, evaluate_test):
    """Rebuild every fit of one fold and return its scores on val (and outer)."""
    args = SimpleNamespace(**json.loads((directory / "config.json").read_text()))
    args.evaluate_test = evaluate_test
    torch.set_num_threads(args.threads)
    data, _ = load_data(args)
    splits = ["train", "val"] + (["test"] if evaluate_test else [])
    features = {s: np.asarray(data[s]["features"], dtype=np.float32) for s in splits}
    scores = {arm: {} for arm in ARMS}
    for seed in SEEDS:
        train_rows = np.load(directory / f"train_indices_{seed}.npy")
        scaler = QuantileTransformer(
            n_quantiles=min(1000, len(train_rows)),
            output_distribution="normal",
            subsample=100000,
            random_state=seed,
        ).fit(features["train"][train_rows])
        standardized = {
            s: scaler.transform(features[s]).astype(np.float32) for s in splits
        }
        for arm in ARMS:
            indices = arm_indices(arm, {s: data[s] for s in splits}, args, seed)
            base = build_model("single_row", standardized["train"][train_rows], args)
            model = EncoderHistoryClassifier(
                base.model.model, args.width, slots=indices["train"].shape[1] - 1
            )
            checkpoint = torch.load(directory / f"{arm}_{seed}.pt", map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.to(device)
            saved = np.load(directory / f"{arm}_{seed}_predictions.npz")["val"]
            result = {"val": saved}
            for split in splits[1:]:
                inputs = EncoderHistoryInputs(
                    standardized[split], indices[split], device
                )
                result[split] = predict(
                    model, inputs, len(features[split]), args.eval_batch_size, device
                )
            gap = float(np.abs(result["val"] - saved).max())
            if gap > 5e-3:
                raise RuntimeError(f"{directory.name}/{arm}/{seed}: val gap {gap:.4f}")
            result["val"], result["val_gap"] = saved, gap
            scores[arm][seed] = result
            del model
    return data, scores


def frozen_decisions(labels_val, fits):
    """Ensemble outer score and its decision at the internal-validation threshold."""
    val = np.mean([f["val"] for f in fits.values()], axis=0)
    threshold = choose_threshold(labels_val, val)
    outer = np.mean([f["test"] for f in fits.values()], axis=0)
    return outer, (outer >= threshold).astype(int), threshold


def within_file_auc(y, score, files, mask):
    total, rows = 0.0, 0
    for file in np.unique(files[mask]):
        m = mask & (files == file)
        if 0 < y[m].mean() < 1:
            total += m.sum() * roc_auc_score(y[m], score[m])
            rows += int(m.sum())
    return (total / rows if rows else None), rows


def summarise(y, score, prediction):
    return {
        "auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
        "macro_f1": float(f1_score(y, prediction, labels=[0, 1], average="macro")),
        "windows": len(y),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--fits", default="logs/benchmark_3w_binary/encoder_history_outer_v1"
    )
    p.add_argument("--frozen", default="logs/benchmark_3w_binary/robustness_outer_v1")
    p.add_argument("--validation-only", action="store_true")
    args = p.parse_args(argv)
    root = Path(args.fits)
    out = root / "outer"
    if not args.validation_only and (out / "results.json").exists():
        raise FileExistsError("Outer folds already scored for this experiment")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Pass 1: every fold must reproduce validation before any outer row is read.
    for fold in range(4):
        _, scores = load_fits(root / f"fold_{fold}", device, evaluate_test=False)
        gaps = [scores[a][s]["val_gap"] for a in ARMS for s in SEEDS]
        print(
            f"fold {fold}: validation reproduced, max gap {max(gaps):.2e}", flush=True
        )
    if args.validation_only:
        return
    out.mkdir(parents=True)
    models = ("history", "none", "trea_causal", "xgb_causal")
    records = []
    for fold in range(4):
        data, scores = load_fits(root / f"fold_{fold}", device, evaluate_test=True)
        frozen = np.load(Path(args.frozen) / f"fold_{fold}_predictions.npz")
        y = np.asarray(data["test"]["labels"])
        files = np.asarray(data["test"]["file_ids"])
        groups = np.asarray(data["test"]["group_ids"]).astype(str)
        if not (
            np.array_equal(y, frozen["labels"])
            and np.array_equal(files, frozen["file_ids"])
            and np.array_equal(groups, frozen["groups"].astype(str))
        ):
            raise RuntimeError("Outer rows do not align with the frozen predictions")
        record = {"fold": fold, "labels": y, "groups": groups, "files": files}
        y_val = np.asarray(data["val"]["labels"])
        for arm in ARMS:
            score, prediction, threshold = frozen_decisions(y_val, scores[arm])
            record[arm] = {"score": score, "prediction": prediction}
            record[arm]["threshold"] = threshold
            record[arm]["single"] = {
                seed: float(roc_auc_score(y, fit["test"]))
                for seed, fit in scores[arm].items()
            }
        for name, key in [("trea_causal", "neural"), ("xgb_causal", "tree")]:
            record[name] = {
                "score": frozen[f"{key}_ensemble_score"],
                "prediction": frozen[f"{key}_ensemble_prediction"],
            }
        np.savez_compressed(
            out / f"fold_{fold}_predictions.npz",
            labels=y,
            groups=groups,
            file_ids=files,
            **{f"{m}_score": record[m]["score"] for m in models},
            **{f"{m}_prediction": record[m]["prediction"] for m in models},
        )
        records.append(record)
    y = np.concatenate([r["labels"] for r in records])
    groups = np.concatenate([r["groups"] for r in records])
    files = np.concatenate([r["files"] + 10**6 * r["fold"] for r in records])
    folds = np.concatenate([np.full(len(r["labels"]), r["fold"]) for r in records])
    physical = np.char.startswith(groups, "WELL-")
    results = {"models": {}}
    pooled = {}
    for m in models:
        score = np.concatenate([r[m]["score"] for r in records])
        prediction = np.concatenate([r[m]["prediction"] for r in records])
        pooled[m] = (score, prediction)
        fold_rows = [
            summarise(r["labels"], r[m]["score"], r[m]["prediction"]) for r in records
        ]
        within, rows = within_file_auc(y, score, files, physical)
        results["models"][m] = {
            "fold_auc": [f["auc"] for f in fold_rows],
            "fold_macro_f1": [f["macro_f1"] for f in fold_rows],
            "mean_fold_auc": float(np.mean([f["auc"] for f in fold_rows])),
            "mean_fold_macro_f1": float(np.mean([f["macro_f1"] for f in fold_rows])),
            "pooled": summarise(y, score, prediction),
            "physical": summarise(y[physical], score[physical], prediction[physical]),
            "simulation": summarise(
                y[~physical], score[~physical], prediction[~physical]
            ),
            "within_file_auc_physical_mixed": within,
            "within_file_rows": rows,
        }
        if m in ARMS:
            results["models"][m]["single_fit_fold_auc"] = [
                r[m]["single"] for r in records
            ]
            results["models"][m]["thresholds"] = [r[m]["threshold"] for r in records]
    wells = {}
    for well in np.unique(groups[physical]):
        mask = groups == well
        wells[well] = {
            "windows": int(mask.sum()),
            "fault_rate": float(y[mask].mean()),
            "fold": int(folds[mask][0]),
            **{
                m: {
                    "auc": float(roc_auc_score(y[mask], pooled[m][0][mask]))
                    if 0 < y[mask].mean() < 1
                    else None,
                    "accuracy": float((pooled[m][1][mask] == y[mask]).mean()),
                }
                for m in models
            },
        }
    results["physical_wells"] = wells
    results["paired_group_bootstrap"] = {}
    for reference in ["xgb_causal", "trea_causal", "none"]:
        interval = paired_intervals(
            y,
            pooled["history"][0],
            pooled[reference][0],
            pooled["history"][1],
            pooled[reference][1],
            groups,
            folds,
        )
        mean_fold = []
        rng = np.random.default_rng(20260920)
        unique, inv = np.unique(groups, return_inverse=True)
        for _ in range(500):
            delta = []
            for fold in range(4):
                for_fold = folds == fold
                counts = np.zeros(len(unique), dtype=int)
                for source in [True, False]:
                    members = np.unique(inv[for_fold & (physical == source)])
                    counts += np.bincount(
                        rng.choice(members, len(members)), minlength=len(unique)
                    )
                w = counts[inv][for_fold]
                yf = y[for_fold]
                if any(w[yf == c].sum() == 0 for c in [0, 1]):
                    break
                delta.append(
                    roc_auc_score(yf, pooled["history"][0][for_fold], sample_weight=w)
                    - roc_auc_score(yf, pooled[reference][0][for_fold], sample_weight=w)
                )
            if len(delta) == 4:
                mean_fold.append(np.mean(delta))
        interval["mean_fold_auc_delta_95_interval"] = np.quantile(
            mean_fold, [0.025, 0.975]
        ).tolist()
        results["paired_group_bootstrap"][f"history_minus_{reference}"] = interval
    (out / "results.json").write_text(json.dumps(results, indent=2))
    for m in models:
        r = results["models"][m]
        print(
            f"OUTER {m:12s} fold_auc "
            + " ".join(f"{a:.4f}" for a in r["fold_auc"])
            + f" mean {r['mean_fold_auc']:.4f} | fold_f1 "
            + " ".join(f"{a:.4f}" for a in r["fold_macro_f1"])
            + f" | pooled auc {r['pooled']['auc']:.4f} f1 {r['pooled']['macro_f1']:.4f}"
            + f" | phys auc {r['physical']['auc']:.4f}"
            + f" f1 {r['physical']['macro_f1']:.4f}"
            + f" | sim auc {r['simulation']['auc']:.4f}"
            + f" | within-file {r['within_file_auc_physical_mixed']:.4f}",
            flush=True,
        )
    for name, interval in results["paired_group_bootstrap"].items():
        print("BOOT", name, json.dumps(interval), flush=True)
    for well, row in wells.items():
        print(
            f"WELL {well} fold {row['fold']} n={row['windows']:6d} "
            f"fault={row['fault_rate']:.3f} "
            + " ".join(
                f"{m}={row[m]['auc'] and round(row[m]['auc'], 3)}"
                f"/{row[m]['accuracy']:.3f}"
                for m in models
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
