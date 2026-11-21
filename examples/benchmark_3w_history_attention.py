"""Stage 1 of docs/3W_HISTORY_ATTENTION_PREREG.md: internal validation only.

Arms: ``history`` (own file's earlier windows), ``shuffled`` (another file's
context, same source), ``none`` (no context). The shared recipe mirrors the existing
causal-context arms in ``examples/benchmark_3w_context.py``. Consumed test
partitions and the outer robustness folds are never loaded here.
"""

from __future__ import annotations

import argparse
import json
import time

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.preprocessing import QuantileTransformer
from torch import nn

from examples.benchmark_3w_context import audit_chronology
from examples.benchmark_3w_grouped import (
    build_model,
    choose_threshold,
    group_loss_weights,
    load_data,
    metrics,
)
from examples.three_w_history_attention import (
    EncoderHistoryClassifier,
    EncoderHistoryInputs,
    HistoryAttentionClassifier,
    HistoryInputs,
    difference_scale,
    history_indices,
    offset_history_indices,
    shuffled_history_indices,
)
from trea.training.single_row import adamw_parameter_groups


ARMS = ("history", "shuffled", "none")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--history", type=int, default=32)
    p.add_argument(
        "--model",
        choices=["tokens", "encoder"],
        default="tokens",
        help="tokens: 3W_HISTORY_ATTENTION_PREREG; encoder: 3W_ENCODER_HISTORY_PREREG",
    )
    p.add_argument("--offsets", default="32,16,8,4,2,1", help="Encoder model only.")
    p.add_argument("--row-dropout", type=float, default=0.0)
    p.add_argument("--split-seed", type=int, default=20260917)
    p.add_argument("--holdout-fraction", type=float, default=0.2)
    p.add_argument("--internal-val-fraction", type=float, default=0.25)
    p.add_argument("--toolkit-path", default="/home/kailukowiak/Work/3W")
    p.add_argument("--data-path", default="/home/kailukowiak/Work/3W/dataset")
    p.add_argument("--cache-dir", default="logs/benchmark_3w_binary/grouped_cache")
    p.add_argument(
        "--group-manifest", default="docs/results/3w_range_confirmation_protocol.json"
    )
    p.add_argument("--value-policy", default="preserve_range")
    p.add_argument("--train-cap", type=int, default=200000)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--eval-batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=3e-4)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--balance", choices=["none", "sqrt", "full"], default="sqrt")
    p.add_argument("--group-loss-power", type=float, default=0.5)
    p.add_argument("--smoke-rows", type=int, default=0, help="Tests only.")
    args = p.parse_args(argv)
    args.evaluate_test = False
    return args


def arm_indices(arm, data, args, seed):
    """Context row ids per split for one arm (and seed, for the shuffled control)."""
    result = {}
    for offset, (split, d) in enumerate(data.items()):
        files = np.asarray(d["file_ids"])
        if arm == "none":
            result[split] = history_indices(files, history=0)
            continue
        if args.model == "encoder":
            offsets = tuple(int(o) for o in args.offsets.split(","))
            indices = offset_history_indices(files, offsets=offsets)
        else:
            indices = history_indices(files, history=args.history)
        if arm == "shuffled":
            physical = np.char.startswith(np.asarray(d["group_ids"]), "WELL-")
            indices = shuffled_history_indices(
                indices, files, physical, seed=1000 * seed + offset
            )
        result[split] = indices
    return result


@torch.no_grad()
def predict(model, inputs, n, batch_size, device):
    model.eval()
    scores = []
    # CUDA attention fails once (rows in batch) x (context length) reaches 65536.
    batch_size = min(batch_size, 65535 // inputs.attention_fanout)
    for start in range(0, n, batch_size):
        rows = torch.arange(start, min(start + batch_size, n), device=device)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = model(*inputs(rows))
        scores.append(logits.float().softmax(-1)[:, 1].cpu().numpy())
    return np.concatenate(scores)


def train_arm(arm, seed, features, labels, data, train_rows, args, out):
    """Fit one arm/seed with the shared recipe; return validation scores and costs."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = QuantileTransformer(
        n_quantiles=min(1000, len(train_rows)),
        output_distribution="normal",
        subsample=100000,
        random_state=seed,
    ).fit(features["train"][train_rows])
    indices = arm_indices(arm, data, args, seed)
    standardized = {
        split: scaler.transform(features[split]).astype(np.float32)
        for split in features
    }
    if args.model == "encoder":
        inputs = {
            split: EncoderHistoryInputs(standardized[split], indices[split], device)
            for split in features
        }
        # Exactly the grouped harness's single-row model; context is added on top.
        base = build_model("single_row", standardized["train"][train_rows], args)
        model = EncoderHistoryClassifier(
            base.model.model,
            d_model=args.width,
            slots=indices["train"].shape[1] - 1,
        ).to(device)
    else:
        scale = difference_scale(
            features["train"].reshape(len(features["train"]), 8, -1).transpose(0, 2, 1),
            train_rows,
            data["train"]["file_ids"],
        )
        inputs = {
            split: HistoryInputs(
                standardized[split], features[split], indices[split], scale, device
            )
            for split in features
        }
        model = HistoryAttentionClassifier(
            n_sensors=features["train"].shape[1] // 8,
            length=indices["train"].shape[1],
            d_model=args.width,
            n_head=4,
            layers=args.layers,
        ).to(device)
    opt = torch.optim.AdamW(
        adamw_parameter_groups(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.95)
    )
    counts = np.bincount(labels["train"][train_rows], minlength=2)
    exponent = {"none": 0, "sqrt": 0.5, "full": 1}[args.balance]
    class_weights = counts.astype(float) ** -exponent
    class_weights /= class_weights.mean()
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    ytrain = torch.as_tensor(labels["train"][train_rows], device=device).long()
    row_weights = group_loss_weights(
        np.asarray(data["train"]["group_ids"])[train_rows], args.group_loss_power
    )
    row_weights = (
        torch.as_tensor(row_weights, device=device) if row_weights is not None else None
    )
    rows_device = torch.as_tensor(train_rows, device=device)
    rng = np.random.default_rng(seed)
    best, stale, history, state = -np.inf, 0, [], None
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        order = torch.as_tensor(rng.permutation(len(train_rows)), device=device)
        losses = []
        for j in range(0, len(order), args.batch_size):
            idx = order[j : j + args.batch_size]
            yb = ytrain[idx]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(*inputs["train"](rows_device[idx]))
            per_row = F.cross_entropy(
                logits.float(), yb, weight=class_weights, reduction="none"
            )
            wb = (
                row_weights[idx]
                if row_weights is not None
                else torch.ones_like(per_row)
            )
            loss = (per_row * wb).sum() / (class_weights[yb] * wb).sum()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss: {arm}/{seed}/{epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        score = predict(
            model, inputs["val"], len(labels["val"]), args.eval_batch_size, device
        )
        auc = metrics(labels["val"], score, 0.5, np.arange(len(score)))["auc"]
        history.append(
            {"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_auc": auc}
        )
        print(
            f"{arm}/{seed} epoch={epoch + 1} loss={np.mean(losses):.5f} "
            f"val_auc={auc:.5f} elapsed={time.perf_counter() - start:.0f}s",
            flush=True,
        )
        if auc > best + 1e-5:
            best, stale = auc, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(state)
    seconds = time.perf_counter() - start
    torch.save(
        {"state_dict": state, "args": vars(args), "history": history},
        out / f"{arm}_{seed}.pt",
    )
    score = predict(
        model, inputs["val"], len(labels["val"]), args.eval_batch_size, device
    )
    parameters = sum(p.numel() for p in model.parameters())
    del model, opt, inputs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return score, seconds, parameters, history


def run(args):
    arms = args.arms.split(",")
    if not set(arms) <= set(ARMS):
        raise ValueError("Unsupported arm")
    out = Path(args.output_dir)
    if (out / "results.json").exists():
        raise FileExistsError("History-attention experiment already exists")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("high")
    data, protocol = load_data(args)
    if set(data) != {"train", "val"}:
        raise RuntimeError("Only training and internal validation may be loaded")
    if args.smoke_rows:
        data = {
            s: {k: v[: args.smoke_rows] for k, v in d.items()} for s, d in data.items()
        }
    else:
        args.chronology = audit_chronology(protocol)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    features = {s: np.asarray(d["features"], dtype=np.float32) for s, d in data.items()}
    labels = {s: np.asarray(d["labels"]) for s, d in data.items()}
    physical = np.char.startswith(np.asarray(data["val"]["group_ids"]), "WELL-")
    val_files = np.asarray(data["val"]["file_ids"])
    seeds = [int(s) for s in args.seeds.split(",")]
    results = []
    for arm in arms:
        scores, costs = [], []
        for seed in seeds:
            n = len(labels["train"])
            train_rows = np.random.default_rng(seed).choice(
                n, min(args.train_cap or n, n), replace=False
            )
            np.save(out / f"train_indices_{seed}.npy", train_rows)
            score, seconds, parameters, history = train_arm(
                arm, seed, features, labels, data, train_rows, args, out
            )
            threshold = choose_threshold(labels["val"], score)
            row = {
                "model": arm,
                "seed": seed,
                "train_seconds": seconds,
                "parameters": parameters,
                "history": history,
                "validation": metrics(labels["val"], score, threshold, val_files),
            }
            results.append(row)
            scores.append(score)
            costs.append(seconds)
            np.savez_compressed(out / f"{arm}_{seed}_predictions.npz", val=score)
            (out / "results.json").write_text(json.dumps(results, indent=2))
            print("RESULT", arm, seed, json.dumps(row["validation"]), flush=True)
        if len(seeds) > 1:
            score = np.mean(scores, axis=0)
            threshold = choose_threshold(labels["val"], score)
            row = {
                "model": arm + "_ensemble",
                "seeds": seeds,
                "train_seconds": sum(costs),
                "validation": metrics(labels["val"], score, threshold, val_files),
            }
            for source, mask in [("physical", physical), ("simulation", ~physical)]:
                if mask.any() and len(np.unique(labels["val"][mask])) == 2:
                    row[source] = metrics(
                        labels["val"][mask], score[mask], threshold, val_files[mask]
                    )
            results.append(row)
            np.savez_compressed(out / f"{arm}_ensemble_predictions.npz", val=score)
            print("RESULT", arm, "ensemble", json.dumps(row["validation"]), flush=True)
        (out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    run(parse_args())
