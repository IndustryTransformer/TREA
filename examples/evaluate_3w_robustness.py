"""One-shot outer scoring of the frozen, internally selected comparison.

No fitting, threshold optimization or candidate selection uses outer labels.
All selected checkpoints must reproduce validation predictions before any outer
arrays are loaded. Causal inputs are reconstructed from sensors before labels.
"""

import json

from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import torch

from sklearn.metrics import f1_score, roc_auc_score

from examples.benchmark_3w_grouped import (
    build_model,
    choose_threshold,
    load_data,
    metrics,
    predict,
    summary_features,
)
from examples.freeze_3w_robustness import file_digest
from examples.summarize_3w_grouped import json_safe
from examples.three_w_label_blind_history import file_inventory, reconstruct_split


def verify_selection(selection):
    if not selection.get("outer_groups_unread_at_freeze"):
        raise ValueError("Selection was not frozen before outer access")
    if [f["fold"] for f in selection["folds"]] != list(range(4)):
        raise ValueError("Expected all four outer folds")
    for path, expected in selection["artifact_sha256"].items():
        if file_digest(path) != expected:
            raise ValueError(f"Frozen artifact changed: {path}")
    evaluator = "examples/evaluate_3w_robustness.py"
    if evaluator not in selection["artifact_sha256"]:
        raise ValueError("Evaluator must be frozen before outer access")
    for fold in selection["folds"]:
        for family in ["neural", "tree"]:
            candidate = fold[f"selected_{family}"]
            if candidate["fit_seeds"] != [42, 43, 44]:
                raise ValueError("Incomplete selected ensemble")
            if candidate not in fold["candidates"]:
                raise ValueError("Selection is not a frozen candidate")
            if (candidate["model"] == "xgb") != (family == "tree"):
                raise ValueError("Selected candidate has wrong family")


def feature_views(data, protocol, args, splits, causal):
    if args.feature_family != "all" or getattr(args, "relative_stats", False):
        raise ValueError("This frozen evaluator requires exact eight-stat base inputs")
    result, reports = {}, []
    inventory = file_inventory(args) if causal else None
    for split in splits:
        result[split] = {
            "window": summary_features(data[split], protocol["train"]["normalization"])
        }
        if causal:
            features, report = reconstruct_split(
                data[split],
                protocol[split],
                inventory,
                compare_cached_history=split != "test",
            )
            result[split]["causal"] = features
            reports.append(report)
    return result, reports


def fit_predict(fit, values, device):
    if fit["scaler"] is not None:
        values = fit["scaler"].transform(values).astype(np.float32)
    if fit["name"] == "xgb":
        return fit["model"].predict_proba(values)[:, 1]
    model = fit["model"].to(device)
    result = predict(model, values, fit["batch_size"], device)
    model.cpu()
    return result


def prepare_fold(fold, device):
    selected = {k: fold[f"selected_{k}"] for k in ["neural", "tree"]}
    directory = Path(selected["neural"]["directory"])
    args = SimpleNamespace(**json.loads((directory / "config.json").read_text()))
    args.evaluate_test = False
    torch.set_num_threads(args.threads)
    data, protocol = load_data(args)
    features, reports = feature_views(
        data,
        protocol,
        args,
        ["train", "val"],
        any(c["view"] == "causal" for c in selected.values()),
    )
    prepared = {
        "fold": fold["fold"],
        "args": args,
        "families": {},
        "validation_reconstruction": reports,
    }
    for family, candidate in selected.items():
        path = Path(candidate["directory"])
        config = SimpleNamespace(**json.loads((path / "config.json").read_text()))
        fits, saved_scores = [], []
        name, view = candidate["model"], candidate["view"]
        for seed in [42, 43, 44]:
            scaler = None
            if name == "xgb":
                model = joblib.load(path / f"{name}_{seed}.joblib")
            else:
                indices = np.load(path / f"train_indices_{seed}.npy")
                training = features["train"][view][indices]
                if config.preprocess != "raw":
                    scaler = joblib.load(path / f"{name}_{seed}_preprocess.joblib")
                    training = scaler.transform(training).astype(np.float32)
                model = build_model(name, training, config)
                checkpoint = torch.load(
                    path / f"{name}_{seed}.pt", map_location="cpu", weights_only=False
                )
                model.load_state_dict(checkpoint["state_dict"], strict=True)
            saved = np.load(path / f"{name}_{seed}_predictions.npz")["val"]
            fit = {
                "seed": seed,
                "name": name,
                "model": model,
                "scaler": scaler,
                "batch_size": config.batch_size,
                "threshold": choose_threshold(data["val"]["labels"], saved),
            }
            reproduced = fit_predict(fit, features["val"][view], device)
            np.testing.assert_allclose(reproduced, saved, atol=1e-6, rtol=1e-6)
            fits.append(fit)
            saved_scores.append(saved)
        prepared["families"][family] = {
            "candidate": candidate,
            "fits": fits,
            "ensemble_threshold": choose_threshold(
                data["val"]["labels"], np.mean(saved_scores, axis=0)
            ),
        }
    return prepared


def pooled_metrics(y, score, prediction):
    """Use already fixed fold-specific decisions, never a pooled refit threshold."""
    return {
        "auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
        "macro_f1": float(f1_score(y, prediction, labels=[0, 1], average="macro")),
        "windows": len(y),
    }


def paired_intervals(y, a, b, ap, bp, groups, folds, repeats=500):
    """Paired whole-group resampling within each outer-fold/source stratum.

    Conditional on these fitted models and partition; not a retraining interval.
    """
    unique, inv = np.unique(groups, return_inverse=True)
    strata = []
    for fold in np.unique(folds):
        for physical in [False, True]:
            mask = (folds == fold) & (np.char.startswith(groups, "WELL-") == physical)
            members = np.unique(inv[mask])
            if len(members):
                strata.append(members)
    rng = np.random.default_rng(20260920)
    deltas = []
    for _ in range(repeats):
        counts = np.zeros(len(unique), dtype=int)
        for members in strata:
            counts += np.bincount(
                rng.choice(members, len(members)), minlength=len(unique)
            )
        weights = counts[inv]
        if any(weights[y == label].sum() == 0 for label in [0, 1]):
            continue
        deltas.append(
            [
                roc_auc_score(y, a, sample_weight=weights)
                - roc_auc_score(y, b, sample_weight=weights),
                f1_score(y, ap, average="macro", labels=[0, 1], sample_weight=weights)
                - f1_score(
                    y, bp, average="macro", labels=[0, 1], sample_weight=weights
                ),
            ]
        )
    return {
        "groups": len(unique),
        "valid_replicates": len(deltas),
        "auc_delta_95_interval": np.quantile(deltas, [0.025, 0.975], axis=0)[
            :, 0
        ].tolist(),
        "macro_f1_delta_95_interval": np.quantile(deltas, [0.025, 0.975], axis=0)[
            :, 1
        ].tolist(),
        "conditional_on_fitted_models": True,
        "stratification": "outer fold and physical/simulation source",
    }


def aggregate(records):
    y = np.concatenate([r["labels"] for r in records])
    groups = np.concatenate([r["groups"] for r in records])
    folds = np.concatenate([np.full(len(r["labels"]), r["fold"]) for r in records])
    seen = set()
    for record in records:
        current = set(record["groups"])
        if seen & current:
            raise ValueError("A group occurs in multiple outer folds")
        seen |= current
    result, pooled = {}, {}
    for family in ["neural", "tree"]:
        result[family], pooled[family] = {}, {}
        for fit in ["42", "43", "44", "ensemble"]:
            score = np.concatenate([r[family][fit]["score"] for r in records])
            prediction = np.concatenate([r[family][fit]["prediction"] for r in records])
            pooled[family][fit] = (score, prediction)
            detail = pooled_metrics(y, score, prediction)
            aucs = [
                roc_auc_score(r["labels"], r[family][fit]["score"]) for r in records
            ]
            f1s = [
                f1_score(
                    r["labels"],
                    r[family][fit]["prediction"],
                    labels=[0, 1],
                    average="macro",
                )
                for r in records
            ]
            detail.update(
                fold_auc=aucs,
                mean_fold_auc=float(np.mean(aucs)),
                std_fold_auc=float(np.std(aucs)),
                fold_macro_f1=f1s,
                mean_fold_macro_f1=float(np.mean(f1s)),
                std_fold_macro_f1=float(np.std(f1s)),
            )
            for source, mask in [
                ("physical", np.char.startswith(groups, "WELL-")),
                ("simulation", ~np.char.startswith(groups, "WELL-")),
            ]:
                detail[source] = pooled_metrics(y[mask], score[mask], prediction[mask])
            result[family][fit] = detail
        result[family]["single_fit_summary"] = {
            "seeds": [42, 43, 44],
            "standard_deviation_definition": "population spread over fitting seeds",
            **{
                metric: {
                    "mean": float(
                        np.mean([result[family][s][metric] for s in ["42", "43", "44"]])
                    ),
                    "std": float(
                        np.std([result[family][s][metric] for s in ["42", "43", "44"]])
                    ),
                }
                for metric in ["auc", "macro_f1", "mean_fold_auc", "mean_fold_macro_f1"]
            },
        }
    a, ap = pooled["neural"]["ensemble"]
    b, bp = pooled["tree"]["ensemble"]
    result["paired_ensemble"] = paired_intervals(y, a, b, ap, bp, groups, folds)
    result["paired_ensemble"].update(
        delta_auc=result["neural"]["ensemble"]["auc"]
        - result["tree"]["ensemble"]["auc"],
        delta_macro_f1=result["neural"]["ensemble"]["macro_f1"]
        - result["tree"]["ensemble"]["macro_f1"],
    )
    return result


def main():
    selection_path = Path("docs/results/3w_robustness_selection.json")
    selection = json.loads(selection_path.read_text())
    verify_selection(selection)
    out = Path("logs/benchmark_3w_binary/robustness_outer_v1")
    if out.exists():
        raise FileExistsError("Outer evaluation already started; do not overwrite it")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Complete ALL-fold validation reproduction before the first outer read.
    prepared = []
    for fold in selection["folds"]:
        prepared.append(prepare_fold(fold, device))
        print(
            f"Fold {fold['fold']}: all selected validation probabilities reproduced",
            flush=True,
        )
    verify_selection(selection)
    out.mkdir()
    (out / "preflight.json").write_text(
        json.dumps(
            {
                "selection_sha256": file_digest(selection_path),
                "all_selected_validation_predictions_reproduced": True,
                "reconstruction": [p["validation_reconstruction"] for p in prepared],
            },
            indent=2,
        )
    )
    records, rows = [], []
    for item in prepared:
        args = item["args"]
        args.evaluate_test = True
        data, protocol = load_data(args)
        features, reports = feature_views(
            data,
            protocol,
            args,
            ["test"],
            any(c["candidate"]["view"] == "causal" for c in item["families"].values()),
        )
        d = data["test"]
        record = {
            "fold": item["fold"],
            "labels": np.asarray(d["labels"]),
            "groups": np.asarray(d["group_ids"]),
        }
        arrays = {
            "labels": record["labels"],
            "groups": record["groups"],
            "file_ids": d["file_ids"],
        }
        for family, selected in item["families"].items():
            candidate = selected["candidate"]
            scores, record[family] = [], {}
            for fit in selected["fits"]:
                score = fit_predict(fit, features["test"][candidate["view"]], device)
                scores.append(score)
            scores.append(np.mean(scores, axis=0))
            thresholds = [f["threshold"] for f in selected["fits"]] + [
                selected["ensemble_threshold"]
            ]
            for fit, score, threshold in zip(
                ["42", "43", "44", "ensemble"], scores, thresholds, strict=True
            ):
                prediction = (score >= threshold).astype(int)
                record[family][fit] = {"score": score, "prediction": prediction}
                arrays[f"{family}_{fit}_score"] = score
                arrays[f"{family}_{fit}_prediction"] = prediction
                row = {
                    "fold": item["fold"],
                    "family": family,
                    "fit": fit,
                    "candidate": candidate["candidate"],
                    "outer": metrics(d["labels"], score, threshold, d["file_ids"]),
                }
                rows.append(row)
                physical = np.char.startswith(record["groups"], "WELL-")
                for source, mask in [("physical", physical), ("simulation", ~physical)]:
                    if mask.any():
                        row[source] = metrics(
                            d["labels"][mask],
                            score[mask],
                            threshold,
                            d["file_ids"][mask],
                        )
                print(json.dumps(json_safe(row), allow_nan=False), flush=True)
        np.savez_compressed(out / f"fold_{item['fold']}_predictions.npz", **arrays)
        (out / f"fold_{item['fold']}_protocol.json").write_text(
            json.dumps(
                {"protocol": protocol["test"], "raw_history_reconstruction": reports},
                indent=2,
            )
        )
        records.append(record)
        (out / "fold_results.json").write_text(
            json.dumps(json_safe(rows), indent=2, allow_nan=False)
        )
    result = {
        "scope": selection["scope"],
        "fold_results": rows,
        "summary": aggregate(records),
        "limitations": [
            "Outer groups were used in historical development; "
            "this is not pristine external validation.",
            "Neural/tree recipes can differ by fold; "
            "results evaluate the frozen selection pipelines.",
            "Causal candidates use longer sensor context "
            "than isolated-window candidates.",
            "Pooled AUC mixes separately calibrated fold models; "
            "also report mean fold AUC.",
            "Simulation instances have no verified shared latent-generator identity.",
            "Bootstrap conditions on the fitted models; "
            "overlapping training folds add uncertainty.",
        ],
    }
    (out / "results.json").write_text(
        json.dumps(json_safe(result), indent=2, allow_nan=False)
    )


if __name__ == "__main__":
    main()
