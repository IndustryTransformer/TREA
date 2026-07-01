"""Axial vs row-pooled transformer on UCI HAR raw inertial signals.

HAR is a better quick time-series test than the NOx rows: each sample is a
9-channel, 128-step phone accelerometer/gyroscope window, and the official split
separates train/test subjects. We compare:

  xgb_stats    XGBoost on engineered window statistics
  row_pooled   feature-mean pooled temporal transformer
  axial        axial feature/time transformer over raw windows
  axial_stats  axial plus the same engineered statistics

Usage:
    uv run python examples/benchmark_har_axial.py --epochs 8 --device cpu
"""

from __future__ import annotations

import argparse
import json
import random
import sys

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.download_har_dataset import download_har_dataset, load_har_data
from trea.models.axial_attention import AxialEncoder, AxialTransformer


@dataclass
class Metrics:
    accuracy: float
    macro_f1: float


class RowPooledClassifier(nn.Module):
    """Temporal transformer after mean-pooling feature tokens per timestep."""

    def __init__(
        self,
        num_features: int,
        sequence_length: int,
        num_classes: int,
        d_model: int,
        n_head: int,
        num_layers: int,
        dropout: float,
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

    def forward(
        self, x_num: torch.Tensor, _x_stats: torch.Tensor | None = None
    ) -> torch.Tensor:
        missing = torch.isnan(x_num).float()
        values = torch.nan_to_num(x_num, nan=0.0)
        cell = torch.stack([values, missing], dim=-1).permute(0, 2, 1, 3)
        tokens = self.value_proj(cell).mean(dim=2)
        steps = torch.arange(self.sequence_length, device=x_num.device)
        encoded = self.encoder(tokens + self.time_pos(steps).unsqueeze(0))
        return self.head(encoded.mean(dim=1))


class AxialClassifier(nn.Module):
    """AxialTransformer wrapper for a plain PyTorch loop."""

    def __init__(
        self,
        num_features: int,
        sequence_length: int,
        num_classes: int,
        d_model: int,
        n_head: int,
        num_layers: int,
        dropout: float,
        time_patch_len: int,
    ) -> None:
        super().__init__()
        self.model = AxialTransformer(
            C_num=num_features,
            C_cat=0,
            cat_cardinalities=[],
            T=sequence_length,
            d_model=d_model,
            task="classification",
            num_classes=num_classes,
            n_head=n_head,
            num_layers=num_layers,
            dropout=dropout,
            use_feature_id_embedding=True,
            time_patch_len=time_patch_len,
        )

    def forward(
        self, x_num: torch.Tensor, _x_stats: torch.Tensor | None = None
    ) -> torch.Tensor:
        x_cat = torch.empty(
            x_num.size(0),
            0,
            x_num.size(2),
            dtype=torch.long,
            device=x_num.device,
        )
        return self.model(x_num, x_cat)


class AxialStatsClassifier(nn.Module):
    """Axial encoder plus an MLP over hand-engineered window statistics."""

    def __init__(
        self,
        num_features: int,
        sequence_length: int,
        num_classes: int,
        stats_dim: int,
        d_model: int,
        n_head: int,
        num_layers: int,
        dropout: float,
        time_patch_len: int,
    ) -> None:
        super().__init__()
        self.encoder = AxialEncoder(
            num_features=num_features,
            T=sequence_length,
            d_model=d_model,
            n_head=n_head,
            num_layers=num_layers,
            dropout=dropout,
            use_feature_id_embedding=True,
            time_patch_len=time_patch_len,
        )
        self.stats_mlp = nn.Sequential(
            nn.LayerNorm(stats_dim),
            nn.Linear(stats_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(
        self, x_num: torch.Tensor, x_stats: torch.Tensor | None
    ) -> torch.Tensor:
        if x_stats is None:
            raise ValueError("AxialStatsClassifier requires engineered stats")
        missing = torch.isnan(x_num).float()
        values = torch.nan_to_num(x_num, nan=0.0)
        axial = self.encoder(values, missing, pool=True)
        stats = self.stats_mlp(x_stats)
        return self.head(torch.cat([axial, stats], dim=1))


class ConvAxialClassifier(nn.Module):
    """Depthwise temporal conv stem, then axial feature/time attention."""

    def __init__(
        self,
        num_features: int,
        sequence_length: int,
        num_classes: int,
        d_model: int,
        n_head: int,
        num_layers: int,
        dropout: float,
        time_patch_len: int,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(
                num_features,
                num_features,
                kernel_size=5,
                padding=2,
                groups=num_features,
            ),
            nn.GELU(),
            nn.Conv1d(
                num_features,
                num_features,
                kernel_size=5,
                padding=2,
                groups=num_features,
            ),
        )
        self.encoder = AxialEncoder(
            num_features=num_features,
            T=sequence_length,
            d_model=d_model,
            n_head=n_head,
            num_layers=num_layers,
            dropout=dropout,
            use_feature_id_embedding=True,
            time_patch_len=time_patch_len,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(
        self, x_num: torch.Tensor, _x_stats: torch.Tensor | None = None
    ) -> torch.Tensor:
        missing = torch.isnan(x_num).float()
        values = torch.nan_to_num(x_num, nan=0.0)
        values = values + self.stem(values)
        axial = self.encoder(values, missing, pool=True)
        return self.head(axial)


class ConvAxialStatsClassifier(nn.Module):
    """Depthwise temporal conv stem + axial attention + engineered statistics."""

    def __init__(
        self,
        num_features: int,
        sequence_length: int,
        num_classes: int,
        stats_dim: int,
        d_model: int,
        n_head: int,
        num_layers: int,
        dropout: float,
        time_patch_len: int,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(
                num_features,
                num_features,
                kernel_size=5,
                padding=2,
                groups=num_features,
            ),
            nn.GELU(),
            nn.Conv1d(
                num_features,
                num_features,
                kernel_size=5,
                padding=2,
                groups=num_features,
            ),
        )
        self.encoder = AxialEncoder(
            num_features=num_features,
            T=sequence_length,
            d_model=d_model,
            n_head=n_head,
            num_layers=num_layers,
            dropout=dropout,
            use_feature_id_embedding=True,
            time_patch_len=time_patch_len,
        )
        self.stats_mlp = nn.Sequential(
            nn.LayerNorm(stats_dim),
            nn.Linear(stats_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(
        self, x_num: torch.Tensor, x_stats: torch.Tensor | None
    ) -> torch.Tensor:
        if x_stats is None:
            raise ValueError("ConvAxialStatsClassifier requires engineered stats")
        missing = torch.isnan(x_num).float()
        values = torch.nan_to_num(x_num, nan=0.0)
        values = values + self.stem(values)
        axial = self.encoder(values, missing, pool=True)
        stats = self.stats_mlp(x_stats)
        return self.head(torch.cat([axial, stats], dim=1))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def window_stats(x: np.ndarray) -> np.ndarray:
    """Window-level time-series features for tree baselines and stats ablations."""
    t = np.linspace(-1.0, 1.0, x.shape[-1], dtype=np.float32)
    denom = float(np.sum(t**2))
    diffs = np.diff(x, axis=2)
    stats = [
        np.nanmean(x, axis=2),
        np.nanstd(x, axis=2),
        np.nanmin(x, axis=2),
        np.nanmax(x, axis=2),
        x[:, :, 0],
        x[:, :, -1],
        x[:, :, -1] - x[:, :, 0],
        np.nansum(x * t.reshape(1, 1, -1), axis=2) / denom,
        np.nanmean(np.abs(diffs), axis=2),
    ]
    return np.concatenate(stats, axis=1).astype(np.float32)


def standardize_channels(
    x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(x_train, axis=(0, 2), keepdims=True)
    sd = np.nanstd(x_train, axis=(0, 2), keepdims=True)
    sd = np.where(sd == 0.0, 1.0, sd)
    return (
        ((x_train - mu) / sd).astype(np.float32),
        ((x_val - mu) / sd).astype(np.float32),
        ((x_test - mu) / sd).astype(np.float32),
    )


def standardize_stats(
    s_train: np.ndarray, s_val: np.ndarray, s_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(s_train, axis=0, keepdims=True)
    sd = np.nanstd(s_train, axis=0, keepdims=True)
    sd = np.where(sd == 0.0, 1.0, sd)
    return (
        ((s_train - mu) / sd).astype(np.float32),
        ((s_val - mu) / sd).astype(np.float32),
        ((s_test - mu) / sd).astype(np.float32),
    )


def cap_samples(
    x: np.ndarray,
    y: np.ndarray,
    max_samples: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples is None or len(y) <= max_samples:
        return x, y
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(y), size=max_samples, replace=False))
    return x[idx], y[idx]


def make_data(
    args: argparse.Namespace,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    int,
]:
    base = download_har_dataset(args.data_dir) if args.download else Path(args.data_dir)
    if base.name != "UCI HAR Dataset":
        base = base / "UCI HAR Dataset"
    x_train_all, y_train_all = load_har_data(base, train=True)
    x_test, y_test = load_har_data(base, train=False)

    x_train_all = x_train_all.numpy().astype(np.float32)
    y_train_all = y_train_all.numpy().astype(np.int64)
    x_test = x_test.numpy().astype(np.float32)
    y_test = y_test.numpy().astype(np.int64)

    perm = np.random.default_rng(args.seed).permutation(len(y_train_all))
    split = int((1.0 - args.val_fraction) * len(y_train_all))
    train_idx, val_idx = perm[:split], perm[split:]
    x_train, y_train = x_train_all[train_idx], y_train_all[train_idx]
    x_val, y_val = x_train_all[val_idx], y_train_all[val_idx]

    x_train, y_train = cap_samples(x_train, y_train, args.max_train_samples, args.seed)
    x_val, y_val = cap_samples(x_val, y_val, args.max_val_samples, args.seed + 1)
    x_test, y_test = cap_samples(x_test, y_test, args.max_test_samples, args.seed + 2)

    x_train, x_val, x_test = standardize_channels(x_train, x_val, x_test)
    s_train, s_val, s_test = standardize_stats(
        window_stats(x_train), window_stats(x_val), window_stats(x_test)
    )
    num_classes = int(max(y_train.max(), y_val.max(), y_test.max()) + 1)
    return (
        (x_train, y_train, s_train),
        (x_val, y_val, s_val),
        (x_test, y_test, s_test),
        num_classes,
    )


def make_loaders(
    train: tuple[np.ndarray, np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray, np.ndarray],
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    def loader(
        data: tuple[np.ndarray, np.ndarray, np.ndarray], shuffle: bool
    ) -> DataLoader:
        x, y, stats = data
        ds = TensorDataset(
            torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(stats)
        )
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    return loader(train, True), loader(val, False), loader(test, False)


def macro_f1(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    f1s = []
    for cls in range(num_classes):
        tp = int(((pred == cls) & (target == cls)).sum().item())
        fp = int(((pred == cls) & (target != cls)).sum().item())
        fn = int(((pred != cls) & (target == cls)).sum().item())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return float(sum(f1s) / len(f1s))


def evaluate(model: nn.Module, loader: DataLoader, num_classes: int, device) -> Metrics:
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for x, y, stats in loader:
            logits = model(x.to(device), stats.to(device))
            preds.append(logits.argmax(dim=1).cpu())
            targets.append(y)
    pred = torch.cat(preds)
    target = torch.cat(targets)
    return Metrics(
        accuracy=float((pred == target).float().mean().item()),
        macro_f1=macro_f1(pred, target, num_classes),
    )


def xgb_stats_baseline(
    train: tuple[np.ndarray, np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray, np.ndarray],
    num_classes: int,
    args: argparse.Namespace,
) -> Metrics:
    _x_train, y_train, s_train = train
    _x_test, y_test, s_test = test
    model = XGBClassifier(
        n_estimators=args.xgb_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_lr,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_jobs=args.xgb_jobs,
        random_state=args.seed,
    )
    model.fit(s_train, y_train)
    pred = torch.from_numpy(model.predict(s_test).astype(np.int64))
    target = torch.from_numpy(y_test)
    return Metrics(
        accuracy=float((pred == target).float().mean().item()),
        macro_f1=macro_f1(pred, target, num_classes),
    )


def train_one(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Metrics:
    model.to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()
    best_state = None
    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y, stats in train_loader:
            x = x.to(device)
            y = y.to(device)
            stats = stats.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x, stats), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

        val = evaluate(model, val_loader, num_classes, device)
        if val.macro_f1 > best_f1:
            best_f1 = val.macro_f1
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        if epoch == 1 or epoch == args.epochs or epoch % args.print_every == 0:
            print(
                f"{name:<12} epoch={epoch:02d} "
                f"val_acc={val.accuracy:.3f} val_f1={val.macro_f1:.3f}",
                flush=True,
            )

    assert best_state is not None
    model.load_state_dict(best_state)
    test = evaluate(model, test_loader, num_classes, device)
    print(
        f"{name:<12} test_acc={test.accuracy:.3f} test_f1={test.macro_f1:.3f}",
        flush=True,
    )
    return test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/har")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--time-patch-len", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--print-every", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--xgb-estimators", type=int, default=300)
    parser.add_argument("--xgb-max-depth", type=int, default=4)
    parser.add_argument("--xgb-lr", type=float, default=0.05)
    parser.add_argument("--xgb-jobs", type=int, default=-1)
    parser.add_argument(
        "--models",
        type=str,
        default="xgb_stats,row_pooled,axial,axial_stats",
        help=(
            "Comma-separated subset: xgb_stats,row_pooled,axial,axial_stats,"
            "conv_axial,conv_axial_stats"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    train, val, test, num_classes = make_data(args)
    train_loader, val_loader, test_loader = make_loaders(train, val, test, args)
    print(
        "HAR windows | "
        f"train={len(train_loader.dataset):,} val={len(val_loader.dataset):,} "
        f"test={len(test_loader.dataset):,} T={train[0].shape[2]} "
        f"F={train[0].shape[1]} classes={num_classes} device={device}",
        flush=True,
    )

    common = dict(
        num_features=train[0].shape[1],
        sequence_length=train[0].shape[2],
        num_classes=num_classes,
        d_model=args.d_model,
        n_head=args.n_head,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    axial_common = {**common, "time_patch_len": args.time_patch_len}
    model_builders: list[tuple[str, type[nn.Module], dict]] = [
        ("row_pooled", RowPooledClassifier, common),
        ("axial", AxialClassifier, axial_common),
        (
            "axial_stats",
            AxialStatsClassifier,
            {**axial_common, "stats_dim": train[2].shape[1]},
        ),
        ("conv_axial", ConvAxialClassifier, axial_common),
        (
            "conv_axial_stats",
            ConvAxialStatsClassifier,
            {**axial_common, "stats_dim": train[2].shape[1]},
        ),
    ]

    selected = {m.strip() for m in args.models.split(",") if m.strip()}
    results = {}
    if "xgb_stats" in selected:
        xgb = xgb_stats_baseline(train, test, num_classes, args)
        results["xgb_stats"] = asdict(xgb)
        print(
            f"{'xgb_stats':<12} test_acc={xgb.accuracy:.3f} "
            f"test_f1={xgb.macro_f1:.3f}",
            flush=True,
        )
    for name, model_cls, model_kwargs in model_builders:
        if name not in selected:
            continue
        seed_everything(args.seed * 1000 + 17)
        model = model_cls(**model_kwargs)
        seed_everything(args.seed * 1000 + 777)
        metrics = train_one(
            name,
            model,
            train_loader,
            val_loader,
            test_loader,
            num_classes,
            args,
            device,
        )
        results[name] = asdict(metrics)

    print("\nFinal test:")
    for name, metrics in results.items():
        print(
            f"  {name:<12} acc={metrics['accuracy']:.3f} "
            f"f1={metrics['macro_f1']:.3f}"
        )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        payload = {"args": vars(args), "results": results}
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
