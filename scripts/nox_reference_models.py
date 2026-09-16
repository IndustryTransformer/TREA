"""Bounded NOx comparison using official FT-Transformer and TabM packages.

Run: .venv/bin/python -m scripts.nox_reference_models
Four presets per family, selected on validation with seed 0; three fresh seeds
10/11/12 evaluate the selected preset. Tuning never evaluates held-out targets.
Same fit-only standardization and exact row splits as nox_encoding_ablation.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins
from rtdl_revisiting_models import FTTransformer
from tabm import TabM
from xgboost import XGBRegressor

from scripts.nox_encoding_ablation import FEATURES, ROOT, prepare, rmse
from trea.models.single_row import TabularRegressor
from trea.training.single_row import adamw_parameter_groups


PRESETS = {
    "ft": [
        dict(depth=3, width=192, attention_dropout=0.2, dropout=0.1, lr=1e-4),
        dict(depth=2, width=128, attention_dropout=0.1, dropout=0.05, lr=3e-4),
        dict(depth=2, width=128, attention_dropout=0.1, dropout=0.05, lr=1e-3),
        dict(depth=3, width=192, attention_dropout=0.1, dropout=0.05, lr=3e-4),
    ],
    "tabm": [
        dict(depth=3, width=512, dropout=0.1, lr=2e-3, embedding="none"),
        dict(depth=2, width=256, dropout=0.1, lr=2e-3, embedding="none"),
        dict(depth=2, width=512, dropout=0.1, lr=2e-3, embedding="ple"),
        dict(depth=2, width=256, dropout=0.0, lr=1e-3, embedding="ple"),
    ],
    "trea": [
        dict(depth=4, width=128, dropout=0.2, lr=8e-4),
        dict(depth=2, width=128, dropout=0.1, lr=8e-4),
        dict(depth=2, width=128, dropout=0.0, lr=3e-4),
        dict(depth=4, width=128, dropout=0.1, lr=3e-4),
    ],
    "xgb": [
        dict(n_estimators=400, max_depth=6, learning_rate=0.05),
        dict(n_estimators=800, max_depth=3, learning_rate=0.05),
        dict(n_estimators=800, max_depth=4, learning_rate=0.05),
        dict(n_estimators=800, max_depth=6, learning_rate=0.03),
    ],
}


def save_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def regression_loss(output, target):
    # TabM: optimize each member's error, NOT the error of the ensemble mean.
    return (output.reshape(len(target), -1) - target[:, None]).square().mean()


@torch.no_grad()
def predict(model, x):
    model.eval()
    return torch.cat(
        [model(b, None).reshape(len(b), -1).mean(1) for b in x.split(512)]
    ).numpy()


def make_model(family, preset, config, x_fit):
    p = preset
    if family == "ft":
        model = FTTransformer(
            n_cont_features=len(FEATURES),
            cat_cardinalities=[],
            d_out=1,
            n_blocks=p["depth"],
            d_block=p["width"],
            attention_n_heads=8,
            attention_dropout=p["attention_dropout"],
            ffn_d_hidden=None,
            ffn_d_hidden_multiplier=4 / 3,
            ffn_dropout=p["dropout"],
            residual_dropout=0.0,
        )
        optimizer = torch.optim.AdamW(
            model.make_parameter_groups(), lr=p["lr"], weight_decay=1e-5
        )
    elif family == "tabm":
        embeddings = None
        if p["embedding"] == "ple":
            embeddings = PiecewiseLinearEmbeddings(
                compute_bins(x_fit, n_bins=48), 16, activation=False, version="B"
            )
        model = TabM.make(
            n_num_features=len(FEATURES),
            d_out=1,
            n_blocks=p["depth"],
            d_block=p["width"],
            dropout=p["dropout"],
            k=32,
            num_embeddings=embeddings,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=3e-4)
    else:
        assert family == "trea"
        model = TabularRegressor(
            config,
            d_model=p["width"],
            n_heads=4,
            n_layers=p["depth"],
            dropout=p["dropout"],
            value_encoder="identity",
            fusion="product_sum",
            first_prenormalization=False,
        )
        optimizer = torch.optim.AdamW(
            adamw_parameter_groups(model, 3e-4), lr=p["lr"], betas=(0.9, 0.95)
        )
    return model, optimizer


def run_job(args):
    family, stage, candidate, seed = args.job
    candidate, seed = int(candidate), int(seed)
    stem = f"{stage}_{family}_{candidate}_{seed}"
    destination = args.output / f"{stem}.json"
    if destination.exists():
        raise FileExistsError(destination)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    data, config, scale, indices = prepare(2219, 42)
    # Workers validate their exact splits against the frozen manifest.
    expected = np.load(args.output / "splits.npz")
    assert all(np.array_equal(indices[k], expected[k]) for k in indices)
    p = PRESETS[family][candidate]
    record = dict(family=family, stage=stage, candidate=candidate, seed=seed, preset=p)
    started = time.monotonic()
    x, y = data["fit"]
    if family == "xgb":
        model = XGBRegressor(
            **p, subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=2
        )
        model.fit(x.numpy(), y.numpy())

        def infer(xx):
            return model.predict(xx.numpy())
    else:
        model, optimizer = make_model(family, p, config, x)
        record["parameters"] = sum(v.numel() for v in model.parameters())
        record["initial_state_sha256"] = hashlib.sha256(
            b"".join(t.detach().numpy().tobytes() for t in model.state_dict().values())
        ).hexdigest()
        # Reference AdamW recipes; retain TREA's established betas and warmup.
        scheduler = None
        if family == "trea":
            warmup = max(1, int(args.epochs * ((len(x) + 255) // 256) * 0.05))
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda step: min(1.0, (step + 1) / warmup)
            )
        generator = torch.Generator().manual_seed(seed + 10000)
        best, best_epoch, best_state = float("inf"), -1, None
        history = []
        for epoch in range(args.epochs):
            model.train()
            total = 0.0
            for idx in torch.randperm(len(x), generator=generator).split(256):
                optimizer.zero_grad(set_to_none=True)
                loss = regression_loss(model(x[idx], None), y[idx])
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss in {stem}")
                loss.backward()
                if family == "trea":
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                total += loss.item() * len(idx)
            val = rmse(data["val"][1], predict(model, data["val"][0]), scale)
            history.append(dict(epoch=epoch + 1, train_loss=total / len(x), val=val))
            if val < best:
                best, best_epoch = val, epoch
                best_state = copy.deepcopy(model.state_dict())
            if epoch == 0 or (epoch + 1) % 25 == 0:
                print(
                    f"{stem} epoch={epoch + 1} val={val:.4f} best={best:.4f}",
                    flush=True,
                )
            if epoch - best_epoch >= args.patience:
                break
        if best_state is None:
            raise RuntimeError(f"No finite validation checkpoint: {stem}")
        model.load_state_dict(best_state)
        record.update(best_epoch=best_epoch + 1, epochs_run=epoch + 1, history=history)

        def infer(xx):
            return predict(model, xx)

    # Held-out data is not even passed to inference until selection is locked.
    splits = ["fit", "val"] if stage == "tune" else list(data)
    predictions = {s: infer(data[s][0]) for s in splits}
    record["rmse"] = {s: rmse(data[s][1], predictions[s], scale) for s in splits}
    record["seconds"] = time.monotonic() - started
    np.savez_compressed(args.output / f"{stem}.predictions.npz", **predictions)
    save_json(destination, record)
    print(f"RESULT {stem} {record['rmse']} seconds={record['seconds']:.1f}", flush=True)


def launch(args, job):
    stem = "_".join(map(str, [job[1], job[0], job[2], job[3]]))
    cmd = [
        sys.executable,
        "-m",
        "scripts.nox_reference_models",
        "--job",
        *map(str, job),
        "--output",
        str(args.output),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
    ]
    env = dict(
        os.environ,
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        MPLCONFIGDIR="/tmp/trea-mpl",
        HF_HUB_OFFLINE="1",
    )
    with (args.output / f"{stem}.log").open("w") as log:
        subprocess.run(
            cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    print(f"DONE {stem}", flush=True)


def select(records):
    return min(records, key=lambda r: (r["rmse"]["val"], r["candidate"]))["candidate"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "logs/nox_reference_models"
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--job", nargs=4)
    args = parser.parse_args()
    if args.job:
        run_job(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    data, _, scale, indices = prepare(2219, 42)
    historical_path = ROOT / "logs/nox_first_norm/product_sum_skip.splits.npz"
    if historical_path.exists():
        historical = np.load(historical_path)
        assert all(np.array_equal(indices[k], historical[k]) for k in indices)
    np.savez_compressed(args.output / "splits.npz", **indices)
    np.savez_compressed(
        args.output / "targets.npz", **{k: y.numpy() for k, (_, y) in data.items()}
    )
    protocol = dict(
        presets=PRESETS,
        tuning_seed=0,
        evaluation_seeds=[10, 11, 12],
        counts={k: len(v) for k, v in indices.items()},
        features=FEATURES,
        target_scale=scale,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=256,
        workers=args.workers,
        threads_per_worker=2,
        preprocessing="Fit-only feature/target standardization; PLE bins fit-only",
        selection="Lowest validation RMSE among four seed-0 presets per family",
        primary=(
            "Historical holdout RMSE; within 2% of XGB is practical proximity, "
            "not equivalence"
        ),
        secondary="Future-year RMSE, separately; no pretraining or rolling retraining",
        scope=(
            "Exploratory on repeatedly inspected NOx; "
            "equal trial counts, not equal FLOPs"
        ),
        versions={
            p: importlib.metadata.version(p)
            for p in [
                "torch",
                "numpy",
                "scikit-learn",
                "xgboost",
                "tabm",
                "rtdl-revisiting-models",
                "rtdl-num-embeddings",
            ]
        },
    )
    save_json(args.output / "protocol.json", protocol)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(
            pool.map(
                lambda job: launch(args, job),
                [(f, "tune", c, 0) for f in PRESETS for c in range(4)],
            )
        )
    selected = {
        f: select(
            [
                json.loads((args.output / f"tune_{f}_{c}_0.json").read_text())
                for c in range(4)
            ]
        )
        for f in PRESETS
    }
    save_json(args.output / "selected.json", selected)
    print("SELECTED", selected, flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(
            pool.map(
                lambda job: launch(args, job),
                [(f, "eval", c, s) for f, c in selected.items() for s in [10, 11, 12]],
            )
        )
    summary = {}
    for family, candidate in selected.items():
        records = [
            json.loads(
                (args.output / f"eval_{family}_{candidate}_{s}.json").read_text()
            )
            for s in [10, 11, 12]
        ]
        summary[family] = dict(
            candidate=candidate,
            preset=PRESETS[family][candidate],
            rmse={
                split: dict(
                    mean=float(np.mean([r["rmse"][split] for r in records])),
                    std=float(np.std([r["rmse"][split] for r in records], ddof=1)),
                )
                for split in data
            },
            seconds=sum(r["seconds"] for r in records),
        )
        # Additional diagnostic: equal average of every evaluation seed, no selection.
        with_predictions = [
            np.load(args.output / f"eval_{family}_{candidate}_{s}.predictions.npz")
            for s in [10, 11, 12]
        ]
        ensemble = {
            split: np.mean([p[split] for p in with_predictions], axis=0)
            for split in data
        }
        np.savez_compressed(
            args.output / f"ensemble_{family}.predictions.npz", **ensemble
        )
        summary[family]["three_seed_ensemble_rmse"] = {
            split: rmse(data[split][1], ensemble[split], scale) for split in data
        }
    save_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
