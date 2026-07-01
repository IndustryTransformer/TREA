"""Run a multi-dataset UEA axial benchmark sweep and aggregate results.

This wraps ``examples/benchmark_uea_axial.py`` so we can launch a reproducible
dataset battery without hand-running every dataset/seed pair.

Usage:
    uv run python scripts/sweep_uea_axial.py --device cuda --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys

from pathlib import Path
from statistics import mean, pstdev


# The 20 parser-compatible UEA multivariate datasets from data/uea (see
# scripts/download_uea_archive.py), ordered roughly cheap -> expensive so partial
# sweeps cover the small ones first. Per-(dataset, seed) checkpointing makes it
# resumable; subset with --datasets for a quick pass.
DEFAULT_DATASETS = [
    "RacketSports", "ERing", "Libras", "BasicMotions", "PenDigits",
    "NATOPS", "Epilepsy", "ArticularyWordRecognition", "Handwriting",
    "UWaveGestureLibrary", "FingerMovements", "HandMovementDirection", "LSST",
    "AtrialFibrillation", "Heartbeat", "Cricket", "SelfRegulationSCP1",
    "SelfRegulationSCP2", "EthanolConcentration", "StandWalkJump",
]
DEFAULT_MODELS = (
    "xgb_raw_flat,xgb_stats,axial,axial_stats,conv_axial,conv_axial_stats"
)


def run_one(args: argparse.Namespace, dataset: str, seed: int, output: Path) -> None:
    if output.exists() and not args.force:
        print(f"SKIP {dataset} seed={seed}: {output} exists", flush=True)
        return

    cmd = [
        sys.executable,
        "examples/benchmark_uea_axial.py",
        "--dataset",
        dataset,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--seed",
        str(seed),
        "--models",
        args.models,
        "--output",
        str(output),
        "--xgb-estimators",
        str(args.xgb_estimators),
        "--xgb-max-depth",
        str(args.xgb_max_depth),
        "--xgb-lr",
        str(args.xgb_lr),
    ]
    if args.time_patch_len != 1:
        cmd.extend(["--time-patch-len", str(args.time_patch_len)])

    print(f"RUN {dataset} seed={seed}", flush=True)
    subprocess.run(cmd, check=True)


def load_rows(paths: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(paths):
        payload = json.loads(path.read_text())
        dataset = payload["args"]["dataset"]
        seed = int(payload["args"]["seed"])
        for model, metrics in payload["results"].items():
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "model": model,
                    "accuracy": float(metrics["accuracy"]),
                    "macro_f1": float(metrics["macro_f1"]),
                    "path": str(path),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "seed", "model", "accuracy", "macro_f1", "path"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["dataset"]), str(row["model"])), []).append(row)

    out = []
    for (dataset, model), group in sorted(grouped.items()):
        acc = [float(r["accuracy"]) for r in group]
        f1 = [float(r["macro_f1"]) for r in group]
        out.append(
            {
                "dataset": dataset,
                "model": model,
                "n_seeds": len(group),
                "accuracy_mean": mean(acc),
                "accuracy_std": pstdev(acc) if len(acc) > 1 else 0.0,
                "macro_f1_mean": mean(f1),
                "macro_f1_std": pstdev(f1) if len(f1) > 1 else 0.0,
            }
        )
    return out


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "model",
        "n_seeds",
        "accuracy_mean",
        "accuracy_std",
        "macro_f1_mean",
        "macro_f1_std",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, object]]) -> None:
    for dataset in sorted({str(r["dataset"]) for r in rows}):
        print(f"\n[{dataset}]")
        for row in [r for r in rows if r["dataset"] == dataset]:
            print(
                f"  {str(row['model']):<16} "
                f"f1={float(row['macro_f1_mean']):.3f}"
                f"+/-{float(row['macro_f1_std']):.3f} "
                f"acc={float(row['accuracy_mean']):.3f}"
                f"+/-{float(row['accuracy_std']):.3f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cuda")
    parser.add_argument("--models", type=str, default=DEFAULT_MODELS)
    parser.add_argument("--time-patch-len", type=int, default=1)
    parser.add_argument("--xgb-estimators", type=int, default=200)
    parser.add_argument("--xgb-max-depth", type=int, default=3)
    parser.add_argument("--xgb-lr", type=float, default=0.05)
    parser.add_argument("--out-dir", type=str, default="logs/uea_sweep")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for dataset in args.datasets:
        for seed in args.seeds:
            output = out_dir / f"{dataset.lower()}_seed{seed}.json"
            outputs.append(output)
            run_one(args, dataset, seed, output)

    rows = load_rows([p for p in outputs if p.exists()])
    write_csv(out_dir / "runs.csv", rows)
    summary = summarize(rows)
    write_summary(out_dir / "summary.csv", summary)
    print_summary(summary)
    print(f"\nWrote {out_dir / 'runs.csv'} and {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
