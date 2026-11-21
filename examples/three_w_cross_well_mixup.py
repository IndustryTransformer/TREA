"""Training-only, source-restricted cross-group mixup for summary models."""

import time

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn

from examples.benchmark_3w_grouped import (
    build_model,
    metrics,
    predict,
    train_neural,
)
from trea.training.single_row import adamw_parameter_groups


class CrossGroupPartners:
    """Uniformly sample other-group rows within the anchor's source."""

    def __init__(self, groups):
        names, self.inverse, self.counts = np.unique(
            np.asarray(groups), return_inverse=True, return_counts=True
        )
        self.source = np.char.startswith(names, "WELL-")
        self.starts = np.zeros(len(names), dtype=np.int64)
        self.pools = {}
        for source in np.unique(self.source):
            members = np.flatnonzero(self.source == source)
            if len(members) < 2:
                raise ValueError("Each source needs at least two training groups")
            rows = np.flatnonzero(self.source[self.inverse] == source)
            self.pools[bool(source)] = rows[
                np.argsort(self.inverse[rows], kind="stable")
            ]
            self.starts[members] = np.r_[0, np.cumsum(self.counts[members])[:-1]]

    def sample(self, anchors, rng):
        group = self.inverse[anchors]
        partner = np.empty(len(anchors), dtype=np.int64)
        for source, pool in self.pools.items():
            selected = np.flatnonzero(self.source[group] == source)
            own = group[selected]
            rank = rng.integers(0, len(pool) - self.counts[own])
            rank += (rank >= self.starts[own]) * self.counts[own]
            partner[selected] = pool[rank]
        return partner


def mixed_loss(logits, ya, yb, wa, wb, class_weights, fraction):
    """Apply original class/group weighting to both interpolated targets."""
    if not 0 <= fraction <= 1:
        raise ValueError("Mixing fraction must lie in [0,1]")
    if logits.ndim == 3:
        members = logits.shape[1]
        expanded_a = ya[:, None].expand(-1, members).reshape(-1)
        expanded_b = yb[:, None].expand(-1, members).reshape(-1)
        flat = logits.float().reshape(-1, 2)
        ca = F.cross_entropy(flat, expanded_a, weight=class_weights, reduction="none")
        cb = F.cross_entropy(flat, expanded_b, weight=class_weights, reduction="none")
        ca, cb = ca.reshape(-1, members).mean(1), cb.reshape(-1, members).mean(1)
    else:
        ca = F.cross_entropy(logits.float(), ya, weight=class_weights, reduction="none")
        cb = F.cross_entropy(logits.float(), yb, weight=class_weights, reduction="none")
    numerator = (fraction * wa * ca + (1 - fraction) * wb * cb).sum()
    denominator = (
        fraction * wa * class_weights[ya] + (1 - fraction) * wb * class_weights[yb]
    ).sum()
    return numerator / denominator


def train_mixup(name, xs, ys, seed, args, out, sample_weights, groups):
    alpha = args.mixup_alpha
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("mixup_alpha must be finite and nonnegative")
    if set(xs) != {"train", "val"} or set(ys) != {"train", "val"}:
        raise ValueError("Cross-well screen must not receive held-out arrays")
    if alpha == 0:
        return train_neural(name, xs, ys, seed, args, out, sample_weights)
    partners = CrossGroupPartners(groups)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(name, xs["train"], args).to(device)
    opt = torch.optim.AdamW(
        adamw_parameter_groups(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.95)
    )
    counts = np.bincount(ys["train"], minlength=2)
    exponent = {"none": 0, "sqrt": 0.5, "full": 1}[args.balance]
    weights = counts.astype(float) ** -exponent
    weights /= weights.mean()
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    targets = torch.tensor(ys["train"], dtype=torch.long, device=device)
    row_weights = torch.tensor(
        np.ones(len(targets)) if sample_weights is None else sample_weights,
        dtype=torch.float32,
        device=device,
    )
    order_rng = np.random.default_rng(seed)
    mix_rng = np.random.default_rng(seed + 20260917)
    best, stale, history, state = -np.inf, 0, [], None
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        order = order_rng.permutation(len(targets))
        losses, fractions = [], []
        for j in range(0, len(order), args.batch_size):
            anchor = order[j : j + args.batch_size]
            other = partners.sample(anchor, mix_rng)
            fraction = float(mix_rng.beta(alpha, alpha))
            # Input interpolation is in train-fitted quantile coordinates.
            values = (
                fraction * xs["train"][anchor] + (1 - fraction) * xs["train"][other]
            )
            xb = torch.as_tensor(np.asarray(values, dtype=np.float32), device=device)
            ia, ib = (
                torch.as_tensor(anchor, device=device),
                torch.as_tensor(other, device=device),
            )
            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(xb)
                loss = mixed_loss(
                    logits,
                    targets[ia],
                    targets[ib],
                    row_weights[ia],
                    row_weights[ib],
                    weights,
                    fraction,
                )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite mixup loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            fractions.append(fraction)
        score = predict(model, xs["val"], args.batch_size, device)
        auc = metrics(ys["val"], score, 0.5, np.arange(len(score)))["auc"]
        history.append(
            {
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
                "val_auc": auc,
                "mean_mix_fraction": float(np.mean(fractions)),
            }
        )
        print(
            f"{name}/{seed} mixup epoch={epoch + 1} loss={np.mean(losses):.5f} "
            f"val_auc={auc:.5f}",
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
        out / f"{name}_{seed}.pt",
    )
    scores = {"val": predict(model, xs["val"], args.batch_size, device)}
    params = sum(p.numel() for p in model.parameters())
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scores, seconds, params, history
