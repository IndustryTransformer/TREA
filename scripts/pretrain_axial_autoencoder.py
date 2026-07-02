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
import math
import sys
import time

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.benchmark_har_axial import seed_everything  # noqa: E402
from examples.benchmark_uea_axial import (  # noqa: E402
    download_uea_dataset,
    make_data,
    parse_uea_ts,
    standardize_channels,
)
from trea.models.axial_attention import AxialEncoder  # noqa: E402


class AxialMaskedAutoencoder(nn.Module):
    def __init__(self, encoder: AxialEncoder, use_conv_stem: bool = False):
        super().__init__()
        self.encoder = encoder
        self.stem = (
            nn.Sequential(
                nn.Conv1d(
                    encoder.value_num_features,
                    encoder.value_num_features,
                    kernel_size=5,
                    padding=2,
                    groups=encoder.value_num_features,
                ),
                nn.GELU(),
                nn.Conv1d(
                    encoder.value_num_features,
                    encoder.value_num_features,
                    kernel_size=5,
                    padding=2,
                    groups=encoder.value_num_features,
                ),
            )
            if use_conv_stem
            else None
        )
        self.reconstruct = nn.Linear(encoder.d_model, 1)

    def forward(self, values: torch.Tensor, missing: torch.Tensor) -> torch.Tensor:
        if self.stem is not None:
            values = values + self.stem(values)
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
    if args.arch == "conv_axis_patches":
        return {
            **common,
            "time_patch_len": 8,
            "tokenizer": "patch_mlp",
            "use_axis_patch_tokens": True,
            "feature_patch_size": args.feature_patch_size,
        }
    if args.arch == "conv_axis_global":
        return {
            **common,
            "time_patch_len": 8,
            "tokenizer": "patch_mlp",
            "use_axis_patch_tokens": True,
            "feature_patch_size": args.feature_patch_size,
            "num_global_tokens": args.num_global_tokens,
            "pooling": "latent_mean",
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


def uses_conv_stem(arch: str) -> bool:
    return arch.startswith("conv_")


def resample_time(x: np.ndarray, target_length: int) -> np.ndarray:
    """Resample ``[N, F, T]`` windows to a shared length, preserving NaNs."""
    if x.shape[-1] == target_length:
        return x.astype(np.float32, copy=False)
    old = np.linspace(0.0, 1.0, x.shape[-1], dtype=np.float32)
    new = np.linspace(0.0, 1.0, target_length, dtype=np.float32)
    out = np.empty((x.shape[0], x.shape[1], target_length), dtype=np.float32)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            y = x[i, j].astype(np.float32, copy=False)
            good = np.isfinite(y)
            if good.sum() == 0:
                out[i, j] = np.nan
            elif good.sum() == 1:
                out[i, j] = y[good][0]
            else:
                out[i, j] = np.interp(new, old[good], y[good]).astype(np.float32)
    return out


def pad_features(x: np.ndarray, target_features: int) -> np.ndarray:
    if x.shape[1] > target_features:
        raise ValueError(
            f"source has {x.shape[1]} features, target_features={target_features}"
        )
    if x.shape[1] == target_features:
        return x.astype(np.float32, copy=False)
    out = np.full((x.shape[0], target_features, x.shape[2]), np.nan, dtype=np.float32)
    out[:, : x.shape[1]] = x
    return out


def prepare_windows(
    x: np.ndarray,
    target_length: int,
    target_features: int,
    max_windows: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    x = resample_time(x, target_length)
    x = pad_features(x, target_features)
    if max_windows is not None and len(x) > max_windows:
        idx = rng.choice(len(x), size=max_windows, replace=False)
        x = x[idx]
    return x.astype(np.float32, copy=False)


class SyntheticTimeSeriesDataset(Dataset):
    """On-the-fly simulated multivariate windows with local/global structure."""

    def __init__(
        self,
        n: int,
        num_features: int,
        T: int,
        seed: int,
        missing_rate: float = 0.02,
    ):
        self.n = n
        self.num_features = num_features
        self.T = T
        self.seed = seed
        self.missing_rate = missing_rate

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        gen = torch.Generator().manual_seed(self.seed + idx * 9973)
        t = torch.linspace(0.0, 1.0, self.T)
        latent_count = 4
        latents = []
        for k in range(latent_count):
            freq = 0.5 + 5.5 * torch.rand((), generator=gen)
            phase = 2.0 * math.pi * torch.rand((), generator=gen)
            amp = 0.2 + 1.3 * torch.rand((), generator=gen)
            trend = (2.0 * torch.rand((), generator=gen) - 1.0) * (t - 0.5)
            wave = amp * torch.sin(2.0 * math.pi * freq * t + phase)
            latents.append(wave + trend)
        z = torch.stack(latents)
        weights = torch.randn(self.num_features, latent_count, generator=gen)
        x = weights @ z
        x += 0.15 * torch.randn(self.num_features, self.T, generator=gen)

        # Inject local motifs and nonlinear channel interactions.
        for _ in range(int(torch.randint(1, 5, (1,), generator=gen).item())):
            f = int(torch.randint(0, self.num_features, (1,), generator=gen).item())
            start = int(
                torch.randint(0, max(1, self.T - 16), (1,), generator=gen).item()
            )
            width = int(
                torch.randint(
                    4, min(32, self.T - start) + 1, (1,), generator=gen
                ).item()
            )
            height = 4.0 * torch.rand((), generator=gen) - 2.0
            ramp = torch.linspace(0.0, 1.0, width)
            x[f, start : start + width] += height * ramp
        if self.num_features > 1:
            x[0] += 0.2 * x[1].tanh()
        if self.num_features > 3:
            x[2] += 0.1 * x[3].square().clamp(max=4.0)

        x = (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True).clamp_min(
            1e-4
        )
        if self.missing_rate > 0:
            missing = (
                torch.rand(self.num_features, self.T, generator=gen)
                < self.missing_rate
            )
            x = x.masked_fill(missing, float("nan"))
        return (x.float(),)


def load_uea_windows(args: argparse.Namespace) -> tuple[list[np.ndarray], list[str]]:
    rng = np.random.default_rng(args.seed)
    arrays = []
    names = []
    for name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        dataset_dir = download_uea_dataset(name, Path(args.data_dir))
        x_train, _labels, _classes = parse_uea_ts(dataset_dir / f"{name}_TRAIN.ts")
        if args.include_test:
            x_test, _test_labels, _test_classes = parse_uea_ts(
                dataset_dir / f"{name}_TEST.ts"
            )
            x_train, _dummy_val, x_test = standardize_channels(
                x_train, x_train[:1].copy(), x_test
            )
            x = np.concatenate([x_train, x_test], axis=0)
        else:
            x_train, _dummy_val, _dummy_test = standardize_channels(
                x_train, x_train[:1].copy(), x_train[:1].copy()
            )
            x = x_train
        arrays.append(
            prepare_windows(
                x,
                args.target_length,
                args.target_features,
                args.max_windows_per_dataset,
                rng,
            )
        )
        names.append(name)
    return arrays, names


def make_pretrain_loaders(
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader, dict]:
    if args.multi_dataset:
        arrays, names = load_uea_windows(args)
        x = np.concatenate(arrays, axis=0)
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(x))
        val_n = max(1, int(len(x) * args.val_fraction))
        val_idx = perm[:val_n]
        train_idx = perm[val_n:]
        train_sets: list[Dataset] = [
            TensorDataset(torch.from_numpy(x[train_idx])),
        ]
        val_sets: list[Dataset] = [
            TensorDataset(torch.from_numpy(x[val_idx])),
        ]
        if args.synthetic_windows > 0:
            train_sets.append(
                SyntheticTimeSeriesDataset(
                    args.synthetic_windows,
                    args.target_features,
                    args.target_length,
                    args.seed + 100_000,
                    args.synthetic_missing_rate,
                )
            )
            val_sets.append(
                SyntheticTimeSeriesDataset(
                    max(1, args.synthetic_windows // 20),
                    args.target_features,
                    args.target_length,
                    args.seed + 200_000,
                    args.synthetic_missing_rate,
                )
            )
        info = {
            "mode": "multi_dataset",
            "datasets": names,
            "uea_windows": int(len(x)),
            "train_uea_windows": int(len(train_idx)),
            "val_uea_windows": int(len(val_idx)),
            "synthetic_windows": int(args.synthetic_windows),
            "target_features": args.target_features,
            "target_length": args.target_length,
        }
        train_dataset: Dataset = ConcatDataset(train_sets)
        val_dataset: Dataset = ConcatDataset(val_sets)
    else:
        train, val, _test, _num_classes = make_data(args)
        train_dataset = TensorDataset(torch.from_numpy(train[0]))
        val_dataset = TensorDataset(torch.from_numpy(val[0]))
        info = {
            "mode": "single_dataset",
            "dataset": args.dataset,
            "target_features": int(train[0].shape[1]),
            "target_length": int(train[0].shape[2]),
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device != "cpu",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device != "cpu",
        persistent_workers=args.num_workers > 0,
    )
    return train_loader, val_loader, info


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


def throughput_probe(
    args: argparse.Namespace,
    model: AxialMaskedAutoencoder,
    device: torch.device,
) -> list[dict[str, float]]:
    results = []
    original_batch = args.batch_size
    for batch_size in args.batch_size_probe:
        x = torch.randn(
            batch_size,
            args.target_features,
            args.target_length,
            device=device,
        )
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
        torch.cuda.empty_cache() if device.type == "cuda" else None
        try:
            start = time.perf_counter()
            steps = args.probe_steps
            for _ in range(steps):
                mask = random_mask(
                    x,
                    args.cell_mask_ratio,
                    args.span_mask_ratio,
                    args.channel_mask_ratio,
                    args.span_len,
                )
                values = x.masked_fill(mask, 0.0)
                missing = mask.float()
                target, target_mask = patch_targets(model.encoder, x, mask)
                pred = model(values, missing)
                loss = nn.functional.mse_loss(pred[target_mask], target[target_mask])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            mem = (
                torch.cuda.max_memory_allocated(device) / 1024**3
                if device.type == "cuda"
                else 0.0
            )
            results.append(
                {
                    "batch_size": float(batch_size),
                    "steps_per_sec": float(steps / elapsed),
                    "samples_per_sec": float(batch_size * steps / elapsed),
                    "max_mem_gb": float(mem),
                    "ok": 1.0,
                }
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            results.append(
                {
                    "batch_size": float(batch_size),
                    "steps_per_sec": 0.0,
                    "samples_per_sec": 0.0,
                    "max_mem_gb": 0.0,
                    "ok": 0.0,
                }
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
    args.batch_size = original_batch
    return results


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
    parser.add_argument(
        "--datasets",
        type=str,
        default=(
            "ArticularyWordRecognition,AtrialFibrillation,BasicMotions,Cricket,"
            "ERing,Epilepsy,EthanolConcentration,FingerMovements,"
            "HandMovementDirection,Handwriting,Heartbeat,LSST,Libras,NATOPS,"
            "PenDigits,RacketSports,SelfRegulationSCP1,SelfRegulationSCP2,"
            "StandWalkJump,UWaveGestureLibrary"
        ),
    )
    parser.add_argument("--data-dir", type=str, default="data/uea")
    parser.add_argument("--multi-dataset", action="store_true")
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--target-length", type=int, default=128)
    parser.add_argument("--target-features", type=int, default=32)
    parser.add_argument("--max-windows-per-dataset", type=int, default=None)
    parser.add_argument("--synthetic-windows", type=int, default=0)
    parser.add_argument("--synthetic-missing-rate", type=float, default=0.02)
    parser.add_argument(
        "--arch",
        choices=[
            "axial",
            "convpatch_p8",
            "axis_patches",
            "conv_axis_patches",
            "conv_axis_global",
            "llmish",
        ],
        default="llmish",
    )
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--feature-patch-size", type=int, default=4)
    parser.add_argument("--num-global-tokens", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
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
    parser.add_argument("--batch-size-probe", nargs="*", type=int, default=[])
    parser.add_argument("--probe-steps", type=int, default=10)
    parser.add_argument("--probe-only", action="store_true")
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

    train_loader, val_loader, data_info = make_pretrain_loaders(args)
    num_features = data_info["target_features"]
    T = data_info["target_length"]
    encoder = AxialEncoder(**encoder_kwargs(args, num_features, T))
    model = AxialMaskedAutoencoder(encoder, use_conv_stem=uses_conv_stem(args.arch)).to(
        device
    )
    if args.batch_size_probe:
        probe = throughput_probe(args, model, device)
        print(json.dumps({"batch_size_probe": probe}, indent=2), flush=True)
        if args.probe_only:
            return
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
            "encoder_kwargs": encoder_kwargs(args, num_features, T),
            "use_conv_stem": uses_conv_stem(args.arch),
            "data_info": data_info,
            "history": history,
        },
        args.output,
    )
    payload = {
        "args": vars(args),
        "best_val_loss": best,
        "data_info": data_info,
        "history": history,
    }
    if args.log_json:
        Path(args.log_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.log_json).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
