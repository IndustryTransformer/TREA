"""Diagnostic synthetic benchmark for axial tabular time-series attention.

The task is intentionally built to expose the row-pooling bottleneck:

  class 0: feature 0 has waveform A, feature 1 has waveform B
  class 1: feature 0 has waveform B, feature 1 has waveform A

Every timestep has the same multiset of feature values in both classes, so a model
that embeds cells with shared weights and mean-pools features before temporal
attention cannot solve it. The axial model keeps feature tokens alive and adds
feature identity, so it should learn the assignment.

Usage:
    uv run python examples/benchmark_axial_synthetic.py --epochs 12
"""

from __future__ import annotations

import argparse
import random

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from trea.models.axial_attention import AxialTransformer


@dataclass
class Metrics:
    loss: float
    accuracy: float


class RowPooledTransformer(nn.Module):
    """Baseline that destroys feature identity before temporal attention."""

    def __init__(
        self,
        num_features: int,
        sequence_length: int,
        d_model: int,
        n_head: int,
        num_layers: int,
        num_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.value_proj = nn.Linear(2, d_model)
        self.time_pos = nn.Embedding(sequence_length, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        self.sequence_length = sequence_length
        self.num_features = num_features

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        # x_num: [B, F, T], NaNs intact.
        missing = torch.isnan(x_num).float()
        values = torch.nan_to_num(x_num, nan=0.0)
        cell = torch.stack([values, missing], dim=-1).permute(0, 2, 1, 3)
        tokens = self.value_proj(cell).mean(dim=2)

        steps = torch.arange(self.sequence_length, device=x_num.device)
        tokens = tokens + self.time_pos(steps).unsqueeze(0)
        encoded = self.encoder(tokens)
        return self.head(encoded.mean(dim=1))


class AxialClassifier(nn.Module):
    """Thin nn.Module wrapper around the axial backbone for a plain training loop."""

    def __init__(
        self,
        num_features: int,
        sequence_length: int,
        d_model: int,
        n_head: int,
        num_layers: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.model = AxialTransformer(
            C_num=num_features,
            C_cat=0,
            cat_cardinalities=[],
            T=sequence_length,
            d_model=d_model,
            task="classification",
            num_classes=2,
            n_head=n_head,
            num_layers=num_layers,
            dropout=dropout,
            use_feature_id_embedding=True,
        )

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        x_cat = torch.empty(
            x_num.size(0),
            0,
            x_num.size(2),
            dtype=torch.long,
            device=x_num.device,
        )
        return self.model(x_num, x_cat)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_feature_assignment_data(
    n_samples: int,
    num_features: int,
    sequence_length: int,
    noise: float,
    missing_prob: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return windows where only feature identity distinguishes the classes."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, sequence_length, dtype=np.float32)

    wave_a = np.sin(2.0 * np.pi * t).astype(np.float32)
    wave_b = np.sign(t - 0.5).astype(np.float32)
    distractor = 0.35 * np.cos(4.0 * np.pi * t).astype(np.float32)

    x = np.zeros((n_samples, num_features, sequence_length), dtype=np.float32)
    y = rng.integers(0, 2, size=n_samples, dtype=np.int64)

    for i, label in enumerate(y):
        amp = rng.uniform(0.85, 1.15)
        offset = rng.normal(0.0, 0.05)
        a = amp * wave_a + offset
        b = amp * wave_b - offset

        if label == 0:
            x[i, 0] = a
            x[i, 1] = b
        else:
            x[i, 0] = b
            x[i, 1] = a

        for f in range(2, num_features):
            phase = rng.uniform(0.0, 2.0 * np.pi)
            x[i, f] = distractor * np.sin(2.0 * np.pi * t + phase)

    x += rng.normal(0.0, noise, size=x.shape).astype(np.float32)

    if missing_prob > 0.0:
        mask = rng.random(size=x.shape) < missing_prob
        x[mask] = np.nan

    return torch.from_numpy(x), torch.from_numpy(y)


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    x, y = make_feature_assignment_data(
        n_samples=args.samples,
        num_features=args.features,
        sequence_length=args.sequence_length,
        noise=args.noise,
        missing_prob=args.missing_prob,
        seed=args.seed,
    )
    perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(args.seed))
    split = int((1.0 - args.val_fraction) * len(y))
    train_idx, val_idx = perm[:split], perm[split:]

    train = TensorDataset(x[train_idx], y[train_idx])
    val = TensorDataset(x[val_idx], y[val_idx])
    return (
        DataLoader(train, batch_size=args.batch_size, shuffle=True),
        DataLoader(val, batch_size=args.batch_size, shuffle=False),
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Metrics:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += float(loss.item()) * y.numel()
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += y.numel()
    return Metrics(loss=total_loss / total, accuracy=correct / total)


def train_one(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> Metrics:
    model.to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

        if epoch == 1 or epoch == args.epochs or epoch % args.print_every == 0:
            metrics = evaluate(model, val_loader, device)
            print(
                f"{name:<12} epoch={epoch:02d} "
                f"val_loss={metrics.loss:.4f} val_acc={metrics.accuracy:.3f}",
                flush=True,
            )

    return evaluate(model, val_loader, device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--features", type=int, default=6)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--noise", type=float, default=0.08)
    parser.add_argument("--missing-prob", type=float, default=0.03)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--print-every", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    train_loader, val_loader = make_loaders(args)
    common = dict(
        num_features=args.features,
        sequence_length=args.sequence_length,
        d_model=args.d_model,
        n_head=args.n_head,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    models: list[tuple[str, nn.Module]] = [
        ("row_pooled", RowPooledTransformer(**common)),
        ("axial", AxialClassifier(**common)),
    ]

    results: dict[str, Metrics] = {}
    for name, model in models:
        results[name] = train_one(name, model, train_loader, val_loader, args, device)

    print("\nFinal validation:")
    for name, metrics in results.items():
        print(f"  {name:<12} loss={metrics.loss:.4f} acc={metrics.accuracy:.3f}")


if __name__ == "__main__":
    main()
