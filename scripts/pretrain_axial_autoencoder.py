"""Masked autoencoding pretraining for axial time-series encoders.

This is a lightweight SSL harness for the UEA/HAR-style raw windows. It masks a
mixture of random cells, contiguous spans, and whole feature channels, then trains
the axial encoder to reconstruct the masked patch-level values from raw windows.

Usage:
    uv run python scripts/pretrain_axial_autoencoder.py --dataset BasicMotions \
        --epochs 3 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.benchmark_har_axial import seed_everything  # noqa: E402
from examples.benchmark_uea_axial import make_data  # noqa: E402
from trea.models.axial_attention import AxialEncoder  # noqa: E402


class AxialMaskedAutoencoder(nn.Module):
    def __init__(self, encoder: AxialEncoder):
        super().__init__()
        self.encoder = encoder
        self.reconstruct = nn.Linear(encoder.d_model, 1)

    def forward(self, values: torch.Tensor, missing: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(values, missing, pool=False)
        feature_tokens = tokens[:, :, : self.encoder.value_num_features]
        return self.reconstruct(feature_tokens).squeeze(-1)


def encoder_kwargs(args: argparse.Namespace, num_features: int, T: int) -> dict:
    common = dict(
        num_features=num_features,
        value_num_features=num_features,
        T=T,
        d_model=args.d_model,
        n_head=args.n_head,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_feature_id_embedding=True,
    )
    if args.arch == "axial":
        return common
    if args.arch == "convpatch_p8":
        return {
            **common,
            "time_patch_len": 8,
            "tokenizer": "conv_patch_mlp",
            "tokenizer_kernel_size": 5,
        }
    if args.arch == "axis_patches":
        return {
            **common,
            "time_patch_len": 8,
            "tokenizer": "conv_patch_mlp",
            "tokenizer_kernel_size": 5,
            "use_axis_patch_tokens": True,
            "feature_patch_size": 4,
        }
    if args.arch == "llmish":
        return {
            **common,
            "time_patch_len": 8,
            "tokenizer": "conv_patch_mlp",
            "tokenizer_kernel_size": 5,
            "use_moment_tokens": True,
            "num_global_tokens": 4,
            "pooling": "latent_mean",
            "mixer": "bilinear_feature",
        }
    raise ValueError(f"Unknown arch: {args.arch}")


def random_mask(
    x: torch.Tensor,
    cell_ratio: float,
    span_ratio: float,
    channel_ratio: float,
    span_len: int,
) -> torch.Tensor:
    observed = torch.isfinite(x)
    mask = (torch.rand_like(x) < cell_ratio) & observed
    B, Fdim, T = x.shape

    if span_ratio > 0 and span_len > 0:
        span_count = max(1, int(Fdim * span_ratio))
        for b in range(B):
            feats = torch.randperm(Fdim, device=x.device)[:span_count]
            for f in feats:
                start = int(torch.randint(0, max(1, T - span_len + 1), (1,)).item())
                mask[b, f, start : start + span_len] |= observed[
                    b, f, start : start + span_len
                ]

    if channel_ratio > 0:
        channel_mask = torch.rand(B, Fdim, device=x.device) < channel_ratio
        mask |= channel_mask.unsqueeze(-1) & observed

    return mask


def patch_targets(
    encoder: AxialEncoder,
    x: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.nan_to_num(x, nan=0.0)
    observed = torch.isfinite(x).float()
    target_patch, observed_patch = encoder._patch_view(values, 1.0 - observed)
    mask_patch, _missing_patch = encoder._patch_view(mask.float(), torch.zeros_like(x))
    denom = observed_patch.neg().add(1.0).sum(dim=-1).clamp(min=1.0)
    target = (target_patch * (1.0 - observed_patch)).sum(dim=-1) / denom
    masked = mask_patch.sum(dim=-1) > 0
    return target.permute(0, 2, 1), masked.permute(0, 2, 1)


def train_epoch(
    model: AxialMaskedAutoencoder,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    model.train()
    losses = []
    for (x,) in loader:
        x = x.to(device)
        mask = random_mask(
            x,
            args.cell_mask_ratio,
            args.span_mask_ratio,
            args.channel_mask_ratio,
            args.span_len,
        )
        values = torch.nan_to_num(x, nan=0.0)
        missing = torch.isnan(x).float()
        values = values.masked_fill(mask, 0.0)
        missing = torch.maximum(missing, mask.float())

        target, target_mask = patch_targets(model.encoder, x, mask)
        pred = model(values, missing)
        if not target_mask.any():
            continue
        loss = nn.functional.mse_loss(pred[target_mask], target[target_mask])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def eval_epoch(
    model: AxialMaskedAutoencoder,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for (x,) in loader:
        x = x.to(device)
        mask = random_mask(
            x,
            args.cell_mask_ratio,
            args.span_mask_ratio,
            args.channel_mask_ratio,
            args.span_len,
        )
        values = torch.nan_to_num(x, nan=0.0)
        missing = torch.maximum(torch.isnan(x).float(), mask.float())
        values = values.masked_fill(mask, 0.0)
        target, target_mask = patch_targets(model.encoder, x, mask)
        pred = model(values, missing)
        if target_mask.any():
            losses.append(
                float(nn.functional.mse_loss(pred[target_mask], target[target_mask]))
            )
    return float(np.mean(losses)) if losses else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="BasicMotions")
    parser.add_argument("--data-dir", type=str, default="data/uea")
    parser.add_argument(
        "--arch",
        choices=["axial", "convpatch_p8", "axis_patches", "llmish"],
        default="llmish",
    )
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--cell-mask-ratio", type=float, default=0.15)
    parser.add_argument("--span-mask-ratio", type=float, default=0.15)
    parser.add_argument("--channel-mask-ratio", type=float, default=0.05)
    parser.add_argument("--span-len", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output", type=str, default="checkpoints/axial_autoencoder.pt"
    )
    parser.add_argument("--log-json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")

    train, val, _test, _num_classes = make_data(args)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train[0])),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val[0])),
        batch_size=args.batch_size,
        shuffle=False,
    )
    encoder = AxialEncoder(**encoder_kwargs(args, train[0].shape[1], train[0].shape[2]))
    model = AxialMaskedAutoencoder(encoder).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    history = []
    best = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, opt, args, device)
        val_loss = eval_epoch(model, val_loader, args, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best:
            best = val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}",
            flush=True,
        )

    assert best_state is not None
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "args": vars(args),
            "model_state": best_state,
            "encoder_kwargs": encoder_kwargs(
                args, train[0].shape[1], train[0].shape[2]
            ),
            "history": history,
        },
        args.output,
    )
    payload = {"args": vars(args), "best_val_loss": best, "history": history}
    if args.log_json:
        Path(args.log_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.log_json).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
