"""Axial vs row-pooled transformer on the local gas-turbine NOx time series.

This is a small real-data follow-up to ``benchmark_axial_synthetic.py``. It uses the
local UCI gas-turbine CSVs in ``data/nox`` and an honest temporal split:

  train/val: 2011-2013
  test:      2014-2015

Each example is a multivariate sensor window; the target is NOX at the final
timestep of that window. Results are reported as RMSE in original NOX units.

Usage:
    uv run python examples/benchmark_nox_axial.py --epochs 6 --device cpu
"""

from __future__ import annotations

import argparse
import json
import random

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

from trea.models.axial_attention import AxialEncoder, AxialTransformer


FEATURES = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP"]
TARGET = "NOX"
TRAIN_YEARS = [2011, 2012, 2013]
TEST_YEARS = [2014, 2015]


@dataclass
class Metrics:
    rmse_scaled: float
    rmse_nox: float
    mae_nox: float


class RowPooledRegressor(nn.Module):
    """Temporal transformer after mean-pooling feature tokens per row."""

    def __init__(
        self,
        num_features: int,
        sequence_length: int,
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
            nn.Linear(d_model, 1),
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
        return self.head(encoded.mean(dim=1)).squeeze(-1)


class AxialRegressor(nn.Module):
    """Axial backbone wrapped for a plain PyTorch regression loop."""

    def __init__(
        self,
        num_features: int,
        sequence_length: int,
        d_model: int,
        n_head: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.model = AxialTransformer(
            C_num=num_features,
            C_cat=0,
            cat_cardinalities=[],
            T=sequence_length,
            d_model=d_model,
            task="regression",
            num_classes=None,
            n_head=n_head,
            num_layers=num_layers,
            dropout=dropout,
            use_feature_id_embedding=True,
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
        return self.model(x_num, x_cat).squeeze(-1)


class AxialStatsRegressor(nn.Module):
    """Axial encoder plus an MLP over hand-engineered window statistics."""

    def __init__(
        self,
        num_features: int,
        sequence_length: int,
        stats_dim: int,
        d_model: int,
        n_head: int,
        num_layers: int,
        dropout: float,
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
            nn.Linear(d_model, 1),
        )

    def forward(
        self, x_num: torch.Tensor, x_stats: torch.Tensor | None
    ) -> torch.Tensor:
        if x_stats is None:
            raise ValueError("AxialStatsRegressor requires engineered stats")
        missing = torch.isnan(x_num).float()
        values = torch.nan_to_num(x_num, nan=0.0)
        axial = self.encoder(values, missing, pool=True)
        stats = self.stats_mlp(x_stats)
        return self.head(torch.cat([axial, stats], dim=1)).squeeze(-1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_nox(data_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(data_dir.glob("gt_*.csv")):
        year = int(path.stem.split("_")[1])
        df = pd.read_csv(path)
        df["year"] = year
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No gt_*.csv files found in {data_dir}")
    return pd.concat(frames, ignore_index=True)


def make_windows(
    df: pd.DataFrame,
    sequence_length: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for _, year_df in df.groupby("year", sort=True):
        values = year_df[FEATURES].to_numpy(np.float32)
        target = year_df[TARGET].to_numpy(np.float32)
        for start in range(0, len(year_df) - sequence_length + 1, stride):
            end = start + sequence_length
            xs.append(values[start:end].T)
            ys.append(target[end - 1])
    return np.stack(xs), np.asarray(ys, dtype=np.float32)


def window_stats(x: np.ndarray) -> np.ndarray:
    """Window-level time-series features for tree baselines and the stats ablation."""
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


def cap_windows(
    x: np.ndarray,
    y: np.ndarray,
    max_windows: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_windows is None or len(y) <= max_windows:
        return x, y
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(y), size=max_windows, replace=False))
    return x[idx], y[idx]


def make_data(
    args: argparse.Namespace,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    float,
    float,
]:
    raw = load_nox(Path(args.data_dir))
    train_raw = raw[raw.year.isin(TRAIN_YEARS)].copy()
    test_raw = raw[raw.year.isin(TEST_YEARS)].copy()

    feature_mu = train_raw[FEATURES].mean()
    feature_sd = train_raw[FEATURES].std().replace(0, 1)
    target_mu = float(train_raw[TARGET].mean())
    target_sd = float(train_raw[TARGET].std() or 1.0)

    for df in (train_raw, test_raw):
        df[FEATURES] = (df[FEATURES] - feature_mu) / feature_sd
        df[TARGET] = (df[TARGET] - target_mu) / target_sd

    x_train_all, y_train_all = make_windows(
        train_raw, args.sequence_length, args.stride
    )
    x_test, y_test = make_windows(test_raw, args.sequence_length, args.stride)

    split = int((1.0 - args.val_fraction) * len(y_train_all))
    x_train, y_train = x_train_all[:split], y_train_all[:split]
    x_val, y_val = x_train_all[split:], y_train_all[split:]

    x_train, y_train = cap_windows(x_train, y_train, args.max_train_windows, args.seed)
    x_val, y_val = cap_windows(x_val, y_val, args.max_val_windows, args.seed + 1)
    x_test, y_test = cap_windows(x_test, y_test, args.max_test_windows, args.seed + 2)

    s_train = window_stats(x_train)
    s_val = window_stats(x_val)
    s_test = window_stats(x_test)
    stats_mu = np.nanmean(s_train, axis=0, keepdims=True)
    stats_sd = np.nanstd(s_train, axis=0, keepdims=True)
    stats_sd = np.where(stats_sd == 0.0, 1.0, stats_sd)
    s_train = (s_train - stats_mu) / stats_sd
    s_val = (s_val - stats_mu) / stats_sd
    s_test = (s_test - stats_mu) / stats_sd

    return (
        (x_train, y_train, s_train.astype(np.float32)),
        (x_val, y_val, s_val.astype(np.float32)),
        (x_test, y_test, s_test.astype(np.float32)),
        target_mu,
        target_sd,
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

def xgb_stats_baseline(
    train: tuple[np.ndarray, np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray, np.ndarray],
    target_mu: float,
    target_sd: float,
    args: argparse.Namespace,
) -> Metrics:
    _x_train, y_train, s_train = train
    _x_test, y_test, s_test = test
    model = XGBRegressor(
        n_estimators=args.xgb_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_lr,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=args.xgb_jobs,
        random_state=args.seed,
    )
    model.fit(s_train, y_train)
    pred = torch.from_numpy(model.predict(s_test).astype(np.float32))
    target = torch.from_numpy(y_test)
    err_scaled = pred - target
    err_nox = (pred * target_sd + target_mu) - (target * target_sd + target_mu)
    return Metrics(
        rmse_scaled=float(torch.sqrt(torch.mean(err_scaled**2)).item()),
        rmse_nox=float(torch.sqrt(torch.mean(err_nox**2)).item()),
        mae_nox=float(torch.mean(torch.abs(err_nox)).item()),
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    target_mu: float,
    target_sd: float,
    device: torch.device,
) -> Metrics:
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for x, y, stats in loader:
            x = x.to(device)
            stats = stats.to(device)
            pred = model(x, stats).cpu()
            preds.append(pred)
            targets.append(y)
    pred = torch.cat(preds)
    target = torch.cat(targets)
    err_scaled = pred - target
    pred_nox = pred * target_sd + target_mu
    target_nox = target * target_sd + target_mu
    err_nox = pred_nox - target_nox
    return Metrics(
        rmse_scaled=float(torch.sqrt(torch.mean(err_scaled**2)).item()),
        rmse_nox=float(torch.sqrt(torch.mean(err_nox**2)).item()),
        mae_nox=float(torch.mean(torch.abs(err_nox)).item()),
    )


def train_one(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    target_mu: float,
    target_sd: float,
    args: argparse.Namespace,
    device: torch.device,
) -> Metrics:
    model.to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = nn.MSELoss()
    best_state = None
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y, stats in train_loader:
            x = x.to(device)
            stats = stats.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x, stats), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

        val = evaluate(model, val_loader, target_mu, target_sd, device)
        if val.rmse_scaled < best_val:
            best_val = val.rmse_scaled
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        if epoch == 1 or epoch == args.epochs or epoch % args.print_every == 0:
            print(
                f"{name:<12} epoch={epoch:02d} "
                f"val_rmse={val.rmse_nox:.3f} val_mae={val.mae_nox:.3f}",
                flush=True,
            )

    assert best_state is not None
    model.load_state_dict(best_state)
    test = evaluate(model, test_loader, target_mu, target_sd, device)
    print(
        f"{name:<12} test_rmse={test.rmse_nox:.3f} test_mae={test.mae_nox:.3f}",
        flush=True,
    )
    return test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/nox")
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-train-windows", type=int, default=4000)
    parser.add_argument("--max-val-windows", type=int, default=1000)
    parser.add_argument("--max-test-windows", type=int, default=2000)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
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
    parser.add_argument("--xgb-estimators", type=int, default=400)
    parser.add_argument("--xgb-max-depth", type=int, default=4)
    parser.add_argument("--xgb-lr", type=float, default=0.05)
    parser.add_argument("--xgb-jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    train, val, test, target_mu, target_sd = make_data(args)
    train_loader, val_loader, test_loader = make_loaders(train, val, test, args)
    print(
        "NOx windows | "
        f"train={len(train_loader.dataset):,} val={len(val_loader.dataset):,} "
        f"test={len(test_loader.dataset):,} T={args.sequence_length} "
        f"F={len(FEATURES)} device={device}",
        flush=True,
    )

    common = dict(
        num_features=len(FEATURES),
        sequence_length=args.sequence_length,
        d_model=args.d_model,
        n_head=args.n_head,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model_builders: list[tuple[str, type[nn.Module], dict]] = [
        ("row_pooled", RowPooledRegressor, common),
        ("axial", AxialRegressor, common),
        (
            "axial_stats",
            AxialStatsRegressor,
            {**common, "stats_dim": train[2].shape[1]},
        ),
    ]

    results = {}
    xgb = xgb_stats_baseline(train, test, target_mu, target_sd, args)
    results["xgb_stats"] = asdict(xgb)
    print(
        f"{'xgb_stats':<12} test_rmse={xgb.rmse_nox:.3f} test_mae={xgb.mae_nox:.3f}",
        flush=True,
    )
    for name, model_cls, model_kwargs in model_builders:
        seed_everything(args.seed * 1000 + 17)
        model = model_cls(**model_kwargs)
        seed_everything(args.seed * 1000 + 777)
        metrics = train_one(
            name,
            model,
            train_loader,
            val_loader,
            test_loader,
            target_mu,
            target_sd,
            args,
            device,
        )
        results[name] = asdict(metrics)

    print("\nFinal test:")
    for name, metrics in results.items():
        print(
            f"  {name:<12} rmse={metrics['rmse_nox']:.3f} "
            f"mae={metrics['mae_nox']:.3f}"
        )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        payload = {"args": vars(args), "results": results}
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
