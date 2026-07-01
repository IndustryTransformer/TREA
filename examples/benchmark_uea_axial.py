"""Axial/raw-window benchmarks for UEA multivariate time-series datasets.

Downloads a dataset from the time-series classification archive, parses the UEA
``.ts`` files, and runs the same comparison as the HAR benchmark:

  xgb_stats         XGBoost on engineered window statistics
  row_pooled        feature-mean pooled temporal transformer
  axial             axial feature/time transformer over raw windows
  axial_stats       axial plus engineered stats
  conv_axial        depthwise temporal conv stem + axial
  conv_axial_stats  conv stem + axial + engineered stats

Usage:
    uv run python examples/benchmark_uea_axial.py --dataset BasicMotions --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch.nn as nn


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.benchmark_har_axial import (  # noqa: E402
    AxialClassifier,
    AxialStatsClassifier,
    ConvAxialClassifier,
    ConvAxialStatsClassifier,
    RowPooledClassifier,
    make_loaders,
    seed_everything,
    standardize_channels,
    standardize_stats,
    train_one,
    window_stats,
    xgb_stats_baseline,
)


def download_uea_dataset(dataset: str, data_dir: Path) -> Path:
    target = data_dir / dataset
    target.mkdir(parents=True, exist_ok=True)
    zip_path = target / f"{dataset}.zip"
    if not zip_path.exists():
        url = f"https://www.timeseriesclassification.com/aeon-toolkit/{dataset}.zip"
        print(f"Downloading {dataset} from {url}", flush=True)
        urllib.request.urlretrieve(url, zip_path)
    train_path = target / f"{dataset}_TRAIN.ts"
    test_path = target / f"{dataset}_TEST.ts"
    if not train_path.exists() or not test_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target)
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing {dataset}_TRAIN.ts / {dataset}_TEST.ts")
    return target


def parse_uea_ts(path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    """Parse equal-length, no-timestamp UEA ``.ts`` files."""
    class_order: list[str] = []
    dimensions = None
    data_started = False
    xs: list[list[list[float]]] = []
    labels: list[str] = []

    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if not data_started:
            if low.startswith("@dimensions"):
                dimensions = int(line.split()[-1])
            elif low.startswith("@classlabel"):
                parts = line.split()
                if len(parts) > 2 and parts[1].lower() == "true":
                    class_order = parts[2:]
            elif low.startswith("@data"):
                data_started = True
            continue

        parts = line.split(":")
        if dimensions is None:
            dimensions = len(parts) - 1
        if len(parts) != dimensions + 1:
            raise ValueError(f"Expected {dimensions} dimensions plus label in {path}")
        series = []
        for dim_values in parts[:dimensions]:
            vals = [
                float(v) if v != "?" else np.nan
                for v in dim_values.split(",")
                if v
            ]
            series.append(vals)
        xs.append(series)
        labels.append(parts[-1])

    if not xs:
        raise ValueError(f"No data rows parsed from {path}")
    x = np.asarray(xs, dtype=np.float32)
    if not class_order:
        class_order = sorted(set(labels))
    return x, labels, class_order


def label_indices(labels: list[str], class_order: list[str]) -> np.ndarray:
    mapping = {label: i for i, label in enumerate(class_order)}
    return np.asarray([mapping[label] for label in labels], dtype=np.int64)


def make_data(
    args: argparse.Namespace,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    int,
]:
    dataset_dir = download_uea_dataset(args.dataset, Path(args.data_dir))
    x_train_all, labels_train, class_order = parse_uea_ts(
        dataset_dir / f"{args.dataset}_TRAIN.ts"
    )
    x_test, labels_test, class_order_test = parse_uea_ts(
        dataset_dir / f"{args.dataset}_TEST.ts"
    )
    if set(class_order_test) != set(class_order):
        raise ValueError("Train/test class labels differ")
    y_train_all = label_indices(labels_train, class_order)
    y_test = label_indices(labels_test, class_order)

    perm = np.random.default_rng(args.seed).permutation(len(y_train_all))
    split = max(1, int((1.0 - args.val_fraction) * len(y_train_all)))
    train_idx, val_idx = perm[:split], perm[split:]
    if len(val_idx) == 0:
        train_idx, val_idx = perm[:-1], perm[-1:]

    x_train, y_train = x_train_all[train_idx], y_train_all[train_idx]
    x_val, y_val = x_train_all[val_idx], y_train_all[val_idx]
    x_train, x_val, x_test = standardize_channels(x_train, x_val, x_test)
    s_train, s_val, s_test = standardize_stats(
        window_stats(x_train), window_stats(x_val), window_stats(x_test)
    )
    return (
        (x_train, y_train, s_train),
        (x_val, y_val, s_val),
        (x_test, y_test, s_test),
        len(class_order),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="BasicMotions")
    parser.add_argument("--data-dir", type=str, default="data/uea")
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--time-patch-len", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--xgb-estimators", type=int, default=200)
    parser.add_argument("--xgb-max-depth", type=int, default=3)
    parser.add_argument("--xgb-lr", type=float, default=0.05)
    parser.add_argument("--xgb-jobs", type=int, default=-1)
    parser.add_argument(
        "--models",
        type=str,
        default="xgb_stats,row_pooled,axial,axial_stats,conv_axial,conv_axial_stats",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = "cuda" if args.device == "auto" else args.device

    train, val, test, num_classes = make_data(args)
    train_loader, val_loader, test_loader = make_loaders(train, val, test, args)
    print(
        f"{args.dataset} | train={len(train_loader.dataset)} "
        f"val={len(val_loader.dataset)} test={len(test_loader.dataset)} "
        f"T={train[0].shape[2]} F={train[0].shape[1]} "
        f"classes={num_classes} device={device}",
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
    builders: list[tuple[str, type[nn.Module], dict]] = [
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
            f"{'xgb_stats':<16} test_acc={xgb.accuracy:.3f} "
            f"test_f1={xgb.macro_f1:.3f}",
            flush=True,
        )
    for name, model_cls, kwargs in builders:
        if name not in selected:
            continue
        seed_everything(args.seed * 1000 + 17)
        model = model_cls(**kwargs)
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
            f"  {name:<16} acc={metrics['accuracy']:.3f} "
            f"f1={metrics['macro_f1']:.3f}"
        )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps({"args": vars(args), "results": results}, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
