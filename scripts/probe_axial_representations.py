"""Probe whether axial time-series representations expose simple window statistics.

This is a diagnostic, not a headline benchmark. It trains one supervised axial
variant on a UEA dataset, freezes the encoder, then fits cheap linear probes from
early/final representations to the engineered ``window_stats`` targets used by
the XGBoost stats baseline.

Usage:
    uv run python scripts/probe_axial_representations.py --dataset NATOPS --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, r2_score


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.benchmark_har_axial import (  # noqa: E402
    AxialClassifier,
    make_loaders,
    train_one,
    window_stats,
)
from examples.benchmark_uea_axial import make_data  # noqa: E402


def model_kwargs(name: str, common: dict) -> dict:
    variants = {
        "axial": common,
        "axial_patchmlp_p8": {
            **common,
            "time_patch_len": 8,
            "tokenizer": "patch_mlp",
        },
        "axial_convpatch_p8": {
            **common,
            "time_patch_len": 8,
            "tokenizer": "conv_patch_mlp",
            "tokenizer_kernel_size": 5,
        },
        "axial_moment_tokens": {
            **common,
            "time_patch_len": 4,
            "use_moment_tokens": True,
        },
        "axial_global_latents": {
            **common,
            "num_global_tokens": 4,
            "pooling": "latent_mean",
        },
        "axial_axis_patches": {
            **common,
            "time_patch_len": 8,
            "tokenizer": "conv_patch_mlp",
            "tokenizer_kernel_size": 5,
            "use_axis_patch_tokens": True,
            "feature_patch_size": 4,
        },
        "axial_bilinear_mixer": {**common, "mixer": "bilinear_feature"},
        "axial_llmish": {
            **common,
            "time_patch_len": 8,
            "tokenizer": "conv_patch_mlp",
            "tokenizer_kernel_size": 5,
            "use_moment_tokens": True,
            "num_global_tokens": 4,
            "pooling": "latent_mean",
            "mixer": "bilinear_feature",
        },
    }
    if name not in variants:
        raise ValueError(f"Unsupported probe model: {name}")
    return variants[name]


def encode_arrays(
    model: AxialClassifier,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    out: dict[str, list[np.ndarray]] = {"early_mean": [], "final_mean": []}
    encoder = model.model.encoder
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device)
            missing = torch.isnan(xb).float()
            values = torch.nan_to_num(xb, nan=0.0)
            early = encoder.build_tokens(values, missing)
            final = encoder(values, missing, pool=False)
            out["early_mean"].append(early.mean(dim=(1, 2)).cpu().numpy())
            out["final_mean"].append(final.mean(dim=(1, 2)).cpu().numpy())
            if encoder.num_global_tokens > 0:
                out.setdefault("final_latents", []).append(
                    final[:, :, -encoder.num_global_tokens :, :]
                    .mean(dim=(1, 2))
                    .cpu()
                    .numpy()
                )
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def probe_stats(
    z_train: np.ndarray,
    z_test: np.ndarray,
    stats_train: np.ndarray,
    stats_test: np.ndarray,
) -> float:
    probe = Ridge(alpha=1.0)
    probe.fit(z_train, stats_train)
    pred = probe.predict(z_test)
    return float(r2_score(stats_test, pred, multioutput="uniform_average"))


def probe_labels(
    z_train: np.ndarray,
    z_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(z_train, y_train)
    pred = clf.predict(z_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="BasicMotions")
    parser.add_argument("--data-dir", type=str, default="data/uea")
    parser.add_argument("--model", type=str, default="axial")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")

    train, val, test, num_classes = make_data(args)
    train_loader, val_loader, test_loader = make_loaders(train, val, test, args)
    common = dict(
        num_features=train[0].shape[1],
        sequence_length=train[0].shape[2],
        num_classes=num_classes,
        d_model=args.d_model,
        n_head=args.n_head,
        num_layers=args.num_layers,
        dropout=args.dropout,
        time_patch_len=1,
    )
    model = AxialClassifier(**model_kwargs(args.model, common)).to(device)
    supervised = train_one(
        args.model,
        model,
        train_loader,
        val_loader,
        test_loader,
        num_classes,
        args,
        device,
    )

    z_train = encode_arrays(model, train[0], args.batch_size, device)
    z_test = encode_arrays(model, test[0], args.batch_size, device)
    stats_train = window_stats(train[0])
    stats_test = window_stats(test[0])

    probes = {}
    for key in z_train:
        probes[key] = {
            "stats_r2": probe_stats(z_train[key], z_test[key], stats_train, stats_test),
            "label_probe": probe_labels(z_train[key], z_test[key], train[1], test[1]),
        }

    payload = {
        "args": vars(args),
        "supervised": asdict(supervised),
        "probes": probes,
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
