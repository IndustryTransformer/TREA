"""History as a correction to a frozen single-row base (exploratory, 2026-11-11).

Follow-up to ``docs/3W_ENCODER_HISTORY_PREREG.md``. The base is the already-fitted
``none`` checkpoint of the same seed; its prediction is kept and a small module adds
a margin correction computed from the file's earlier windows. Internal validation
only: consumed test partitions and the outer robustness folds are never loaded.

Arms: ``diff`` (differences to the past only), ``full`` (also absolute embeddings),
``diff_shuffled`` (``diff`` with another file's context, same source).
"""

from __future__ import annotations

import argparse
import json
import time

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer
from torch import nn

from examples.benchmark_3w_grouped import (
    build_model,
    choose_threshold,
    group_loss_weights,
    load_data,
    metrics,
)
from examples.benchmark_3w_history_attention import arm_indices, parse_args as base_args
from examples.three_w_history_attention import (
    EncoderHistoryClassifier,
    HistoryCorrection,
)
from trea.training.single_row import adamw_parameter_groups


ARMS = ("diff", "full", "diff_shuffled")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base-dirs",
        default="logs/benchmark_3w_binary/encoder_history_v1,"
        "logs/benchmark_3w_binary/encoder_history_v1_seeds45_49",
        help="Directories searched for none_<seed>.pt and its saved predictions.",
    )
    p.add_argument("--correction-arms", default=",".join(ARMS))
    own, rest = p.parse_known_args(argv)
    args = base_args([*rest, "--model", "encoder"])
    args.base_dirs, args.correction_arms = own.base_dirs, own.correction_arms
    return args


@torch.no_grad()
def embed_base(model, x, device, batch=8192):
    """Pooled embedding and fault-minus-normal margin of the frozen base."""
    hs, margins = [], []
    for i in range(0, len(x), batch):
        xb = torch.as_tensor(x[i : i + batch], device=device)
        h = model.embed(xb).float()
        logits = model.base.head(h).float()
        hs.append(h)
        margins.append(logits[:, 1] - logits[:, 0])
    return torch.cat(hs), torch.cat(margins)


def load_base(seed, standardized, train_rows, args, device):
    base = build_model("single_row", standardized["train"][train_rows], args)
    model = EncoderHistoryClassifier(base.model.model, args.width, slots=0)
    for folder in args.base_dirs.split(","):
        path = Path(folder) / f"none_{seed}.pt"
        if path.exists():
            model.load_state_dict(torch.load(path, map_location="cpu")["state_dict"])
            saved = np.load(Path(folder) / f"none_{seed}_predictions.npz")["val"]
            return model.to(device).eval(), saved
    raise FileNotFoundError(f"No base checkpoint for seed {seed}")


class Split:
    """Frozen base embeddings of one split plus one arm's context row ids."""

    def __init__(self, h, margin, indices, device):
        self.h, self.margin = h, margin
        self.indices = torch.as_tensor(indices, device=device)

    def __call__(self, rows):
        ids = self.indices[rows]
        context = ids[:, :-1]
        query = ids[:, -1]
        return self.h[query], self.h[context.clamp_min(0)], context < 0, query


@torch.no_grad()
def predict(model, split, n, device, batch=8192):
    model.eval()
    out = []
    for i in range(0, n, batch):
        rows = torch.arange(i, min(i + batch, n), device=device)
        h, encoded, absent, query = split(rows)
        out.append(torch.sigmoid(split.margin[query] + model(h, encoded, absent)))
    return torch.cat(out).cpu().numpy()


def train_correction(arm, seed, embedded, data, labels, train_rows, args, device):
    torch.manual_seed(seed)
    args.model = "encoder"
    indices = arm_indices(
        "shuffled" if arm.endswith("shuffled") else "history", data, args, seed
    )
    splits = {s: Split(*embedded[s], indices[s], device) for s in ["train", "val"]}
    sampled = embedded["train"][0][torch.as_tensor(train_rows, device=device)]
    model = HistoryCorrection(
        args.width,
        slots=indices["train"].shape[1] - 1,
        mean=sampled.mean(0),
        scale=sampled.std(0).clamp_min(1e-6),
        absolute=arm == "full",
    ).to(device)
    opt = torch.optim.AdamW(
        adamw_parameter_groups(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.95)
    )
    y = labels["train"][train_rows]
    counts = np.bincount(y, minlength=2)
    exponent = {"none": 0, "sqrt": 0.5, "full": 1}[args.balance]
    class_weights = counts.astype(float) ** -exponent
    class_weights /= class_weights.mean()
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    ytrain = torch.as_tensor(y, device=device).long()
    row_weights = group_loss_weights(
        np.asarray(data["train"]["group_ids"])[train_rows], args.group_loss_power
    )
    row_weights = (
        torch.as_tensor(row_weights, device=device)
        if row_weights is not None
        else torch.ones(len(train_rows), device=device)
    )
    rows_device = torch.as_tensor(train_rows, device=device)
    rng = np.random.default_rng(seed)
    n_val = len(labels["val"])
    # Epoch 0 is the untouched base, so selection can never prefer a worse model.
    score = predict(model, splits["val"], n_val, device)
    best, stale = roc_auc_score(labels["val"], score), 0
    history = [{"epoch": 0, "val_auc": best}]
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        order = torch.as_tensor(rng.permutation(len(train_rows)), device=device)
        losses = []
        for j in range(0, len(order), args.batch_size):
            idx = order[j : j + args.batch_size]
            h, encoded, absent, query = splits["train"](rows_device[idx])
            margin = splits["train"].margin[query] + model(h, encoded, absent)
            yb = ytrain[idx]
            per_row = F.binary_cross_entropy_with_logits(
                margin, yb.float(), reduction="none"
            )
            wb = class_weights[yb] * row_weights[idx]
            loss = (per_row * wb).sum() / wb.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        score = predict(model, splits["val"], n_val, device)
        auc = roc_auc_score(labels["val"], score)
        history.append(
            {"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_auc": auc}
        )
        if auc > best + 1e-5:
            best, stale = auc, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(state)
    return (
        predict(model, splits["val"], n_val, device),
        time.perf_counter() - start,
        history,
        state,
    )


def run(args):
    arms = args.correction_arms.split(",")
    if not set(arms) <= set(ARMS):
        raise ValueError("Unsupported arm")
    out = Path(args.output_dir)
    if (out / "results.json").exists():
        raise FileExistsError("History-correction experiment already exists")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, _ = load_data(args)
    if set(data) != {"train", "val"}:
        raise RuntimeError("Only training and internal validation may be loaded")
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    features = {s: np.asarray(d["features"], dtype=np.float32) for s, d in data.items()}
    labels = {s: np.asarray(d["labels"]) for s, d in data.items()}
    val_files = np.asarray(data["val"]["file_ids"])
    seeds = [int(s) for s in args.seeds.split(",")]
    results, scores = [], {arm: [] for arm in arms}
    for seed in seeds:
        n = len(labels["train"])
        train_rows = np.random.default_rng(seed).choice(
            n, min(args.train_cap or n, n), replace=False
        )
        scaler = QuantileTransformer(
            n_quantiles=min(1000, len(train_rows)),
            output_distribution="normal",
            subsample=100000,
            random_state=seed,
        ).fit(features["train"][train_rows])
        standardized = {
            s: scaler.transform(features[s]).astype(np.float32) for s in features
        }
        base, saved = load_base(seed, standardized, train_rows, args, device)
        embedded = {s: embed_base(base, standardized[s], device) for s in features}
        rebuilt = torch.sigmoid(embedded["val"][1]).cpu().numpy()
        # Saved scores came from a bfloat16 forward pass, so compare ranking.
        gap = abs(
            roc_auc_score(labels["val"], rebuilt) - roc_auc_score(labels["val"], saved)
        )
        print(
            f"base {seed}: auc gap {gap:.5f}, mean abs score gap "
            f"{np.abs(rebuilt - saved).mean():.5f}",
            flush=True,
        )
        if gap > 0.002:
            raise RuntimeError(f"Base checkpoint does not reproduce (gap {gap:.4f})")
        del base
        for arm in arms:
            score, seconds, history, state = train_correction(
                arm, seed, embedded, data, labels, train_rows, args, device
            )
            threshold = choose_threshold(labels["val"], score)
            row = {
                "model": arm,
                "seed": seed,
                "train_seconds": seconds,
                "base_reproduction_gap": gap,
                "best_epoch": int(np.argmax([h["val_auc"] for h in history])),
                "history": history,
                "validation": metrics(labels["val"], score, threshold, val_files),
            }
            results.append(row)
            scores[arm].append(score)
            np.savez_compressed(out / f"{arm}_{seed}_predictions.npz", val=score)
            torch.save({"state_dict": state}, out / f"{arm}_{seed}.pt")
            (out / "results.json").write_text(json.dumps(results, indent=2))
            v = row["validation"]
            print(
                f"RESULT {arm} {seed} auc={v['auc']:.5f} macro_f1={v['macro_f1']:.5f} "
                f"base_auc={history[0]['val_auc']:.5f} best_epoch={row['best_epoch']} "
                f"seconds={seconds:.0f}",
                flush=True,
            )
        del embedded
        if device.type == "cuda":
            torch.cuda.empty_cache()
    for arm in arms:
        score = np.mean(scores[arm], axis=0)
        threshold = choose_threshold(labels["val"], score)
        v = metrics(labels["val"], score, threshold, val_files)
        results.append({"model": arm + "_ensemble", "seeds": seeds, "validation": v})
        np.savez_compressed(out / f"{arm}_ensemble_predictions.npz", val=score)
        print(
            f"RESULT {arm} ensemble auc={v['auc']:.5f} macro_f1={v['macro_f1']:.5f}",
            flush=True,
        )
    (out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    run(parse_args())
