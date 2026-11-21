"""Read-only audit of fitted development losses; never load consumed test groups."""

import json

from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import f1_score, roc_auc_score

from examples.benchmark_3w_grouped import (
    build_model,
    choose_threshold,
    group_loss_weights,
    load_data,
)
from examples.benchmark_3w_initial_reference import validate_args
from examples.freeze_3w_robustness import file_digest
from examples.summarize_3w_grouped import json_safe
from examples.three_w_label_blind_history import file_inventory, reconstruct_split


def row_outputs(logits, labels):
    """Retain TabM's mean member CE, not CE of its mean probability."""
    if logits.ndim == 3:
        targets = labels[:, None].expand(-1, logits.shape[1])
        loss = (
            F.cross_entropy(
                logits.float().reshape(-1, 2), targets.reshape(-1), reduction="none"
            )
            .reshape(targets.shape)
            .mean(1)
        )
        scores = logits.float().softmax(-1).mean(1)[:, 1]
    else:
        loss = F.cross_entropy(logits.float(), labels, reduction="none")
        scores = logits.float().softmax(-1)[:, 1]
    return scores, loss


@torch.no_grad()
def predict_with_loss(model, x, y, batch_size, device):
    model.eval()
    scores, losses = [], []
    for start in range(0, len(y), batch_size):
        xb = torch.as_tensor(np.array(x[start : start + batch_size]), device=device)
        yb = torch.as_tensor(
            y[start : start + batch_size], dtype=torch.long, device=device
        )
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = model(xb)
        score, loss = row_outputs(logits, yb)
        scores.append(score.cpu().numpy())
        losses.append(loss.cpu().numpy())
    return np.concatenate(scores), np.concatenate(losses)


def summarize_groups(y, score, ce, groups, weights, threshold):
    y, score, ce, groups, weights = map(np.asarray, (y, score, ce, groups, weights))
    if not (y.shape == score.shape == ce.shape == groups.shape == weights.shape):
        raise ValueError("Aligned per-row arrays required")
    if not np.all(np.isfinite(ce)) or not np.all(weights > 0):
        raise ValueError("Finite losses and positive weights required")
    total_mass = weights.sum()
    total_loss = (weights * ce).sum()

    def measure(mask):
        target, predicted = y[mask], score[mask]
        return {
            "windows": int(mask.sum()),
            "fault_fraction": float(target.mean()),
            "auc": float(roc_auc_score(target, predicted))
            if len(np.unique(target)) == 2
            else None,
            "error_at_05": float(np.mean((predicted >= 0.5) != target)),
            "macro_f1_frozen_validation_threshold": float(
                f1_score(target, predicted >= threshold, labels=[0, 1], average="macro")
            ),
            "mean_member_ce": float(ce[mask].mean()),
            "weighted_mean_member_ce": float(
                np.average(ce[mask], weights=weights[mask])
            ),
            "weight_mass_fraction": float(weights[mask].sum() / total_mass),
            "weighted_loss_fraction": float(
                (weights[mask] * ce[mask]).sum() / total_loss
            )
            if total_loss > 0
            else 0.0,
        }

    physical = np.char.startswith(groups, "WELL-")
    rows = [{"group": str(g), **measure(groups == g)} for g in np.unique(groups)]
    return {
        "overall": measure(np.ones(len(y), dtype=bool)),
        "sources": {
            name: measure(mask)
            for name, mask in [("physical", physical), ("simulation", ~physical)]
            if mask.any()
        },
        "groups": sorted(rows, key=lambda r: r["mean_member_ce"], reverse=True),
    }


def main():
    root = Path("logs/benchmark_3w_binary")
    out = root / "training_group_audit_v1"
    if out.exists():
        raise FileExistsError("Do not overwrite an existing training audit")
    config_path = root / "causal_context_trea/config.json"
    args = SimpleNamespace(**json.loads(config_path.read_text()))
    validate_args(args)  # evaluate_test must be false before any dataset access.
    frozen = json.loads(Path("docs/results/3w_robustness_selection.json").read_text())
    for path, digest in frozen["artifact_sha256"].items():
        if file_digest(path) != digest:
            raise ValueError(f"Frozen source/artifact changed: {path}")
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, protocol = load_data(args)
    inventory = file_inventory(args)
    features, audits = {}, []
    for split in ["train", "val"]:
        features[split], audit = reconstruct_split(
            data[split], protocol[split], inventory
        )
        audits.append(audit)
    out.mkdir()
    result = {
        "scope": "Read-only current train/internal-validation checkpoint audit.",
        "limitations": [
            "Training means in-sample, not held-out generalization.",
            "Losses use eval-mode selected checkpoints, not last epoch training mode.",
            "TabM reports mean member CE; classification uses mean member probability.",
            "Weighted loss is full-sample ratio, not average minibatch loss ratios.",
            "Validation checkpoint and threshold selection reused these labels.",
        ],
        "reconstruction": audits,
        "fits": [],
        "artifact_sha256": {
            str(path): file_digest(path)
            for path in [
                Path(__file__),
                Path("examples/benchmark_3w_grouped.py"),
                Path("examples/three_w_label_blind_history.py"),
                Path("docs/results/3w_robustness_selection.json"),
                Path(args.group_manifest),
            ]
        },
    }
    for name, directory in [
        ("single_row", root / "causal_context_trea"),
        ("tabm_ple", root / "causal_context_trees_tabm"),
    ]:
        config_file = directory / "config.json"
        config = SimpleNamespace(**json.loads(config_file.read_text()))
        validate_args(config)
        for key in [
            "group_manifest",
            "value_policy",
            "feature_family",
            "relative_stats",
        ]:
            if getattr(config, key) != getattr(args, key):
                raise ValueError(f"Mismatched feature configuration: {key}")
        for seed in [42, 43, 44]:
            idx_file = directory / f"train_indices_{seed}.npy"
            scaler_file = directory / f"{name}_{seed}_preprocess.joblib"
            checkpoint_file = directory / f"{name}_{seed}.pt"
            score_file = directory / f"{name}_{seed}_predictions.npz"
            for file in [
                config_file,
                idx_file,
                scaler_file,
                checkpoint_file,
                score_file,
            ]:
                result["artifact_sha256"][str(file)] = file_digest(file)
            indices = np.load(idx_file)
            scaler = joblib.load(scaler_file)
            xs = {
                "train": scaler.transform(features["train"][indices]).astype(
                    np.float32
                ),
                "val": scaler.transform(features["val"]).astype(np.float32),
            }
            ys = {
                "train": data["train"]["labels"][indices],
                "val": data["val"]["labels"],
            }
            gs = {
                "train": data["train"]["group_ids"][indices],
                "val": data["val"]["group_ids"],
            }
            model = build_model(name, xs["train"], config).to(device)
            checkpoint = torch.load(
                checkpoint_file, map_location="cpu", weights_only=False
            )
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            saved = np.load(score_file)["val"]
            score, ce = predict_with_loss(
                model, xs["val"], ys["val"], config.batch_size, device
            )
            np.testing.assert_allclose(score, saved, rtol=1e-6, atol=1e-6)
            threshold = choose_threshold(ys["val"], saved)
            val = summarize_groups(
                ys["val"], score, ce, gs["val"], np.ones(len(score)), threshold
            )
            score, ce = predict_with_loss(
                model, xs["train"], ys["train"], config.batch_size, device
            )
            counts = np.bincount(ys["train"], minlength=2).astype(float)
            exponent = {"none": 0, "sqrt": 0.5, "full": 1}[config.balance]
            class_weights = counts**-exponent
            row_weights = group_loss_weights(gs["train"], config.group_loss_power)
            weights = class_weights[ys["train"]] * (
                1 if row_weights is None else row_weights
            )
            train = summarize_groups(
                ys["train"], score, ce, gs["train"], weights, threshold
            )
            result["fits"].append(
                {"model": name, "seed": seed, "train": train, "val": val}
            )
            np.savez_compressed(
                out / f"{name}_{seed}_training.npz", indices=indices, score=score, ce=ce
            )
            (out / "results.json").write_text(
                json.dumps(json_safe(result), indent=2, allow_nan=False)
            )
            print(
                name,
                seed,
                "train",
                json.dumps(train["sources"]),
                "val",
                json.dumps(val["sources"]),
                flush=True,
            )
            del model, checkpoint, xs


if __name__ == "__main__":
    main()
