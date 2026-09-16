"""Updated TREA temporal/label-efficiency diagnostic, not a rerun of the old gate.

Train 2011-2013 with late-source-period validation; test 2014-2015.
Fixed label draws, three independent supervised initializations, and equal-weight
ensembles. Source-only and transductive masked pretraining each use one shared
seed-42 encoder. No test-based checkpoint, mode, or seed selection.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import time

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from xgboost import XGBRegressor

from scripts.nox_encoding_ablation import FEATURES, ROOT
from scripts.nox_reference_models import predict, save_json
from trea.models.single_row import MaskedTabularEncoder, TabularRegressor
from trea.training.single_row import adamw_parameter_groups
from trea.utils.single_row_data import SingleRowConfig


FRACTIONS = [0.01, 0.02, 0.05, 0.10, 1.0]
SEEDS = [10, 11, 12]
ARCH = dict(
    d_model=128,
    n_heads=4,
    n_layers=2,
    dropout=0.0,
    fusion="product_sum",
    first_prenormalization=False,
)


def draw_indices(historical, fraction):
    """Nested label draws; validation is always later than supervised fit rows."""
    cut = int(len(historical) * 0.9)
    fit_pool, val_pool = historical[:cut], historical[cut:]
    rng = np.random.RandomState(42)
    fit_order, val_order = rng.permutation(fit_pool), rng.permutation(val_pool)
    if fraction == 1.0:
        return fit_order, val_order
    n_lab = round(len(historical) * fraction)
    n_val = max(20, round(n_lab * 0.2))
    return fit_order[: n_lab - n_val], val_order[:n_val]


def target_stats(y_fit):
    return float(y_fit.mean()), float(y_fit.std(ddof=1))


def masked_mse(prediction, target, mask):
    return (prediction[mask] - target[mask]).square().mean()


def setup(args):
    args.output.mkdir(parents=True, exist_ok=False)
    frames = []
    for path in sorted((ROOT / "data/nox").glob("gt_*.csv")):
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower()
        df["year"] = int(path.stem.split("_")[1])
        frames.append(df)
    frame = pd.concat(frames, ignore_index=True)
    historical = frame.index[frame.year <= 2013].to_numpy()
    future = frame.index[frame.year >= 2014].to_numpy()
    fit_pool = historical[: int(len(historical) * 0.9)]
    # Unlabeled source features are available to all arms at every label budget.
    mu, sd = frame.loc[fit_pool, FEATURES].mean(), frame.loc[fit_pool, FEATURES].std()
    x = ((frame[FEATURES] - mu) / sd.replace(0, 1)).to_numpy(np.float32)
    y = frame.nox.to_numpy(np.float32)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    np.savez_compressed(
        args.output / "data.npz",
        x=x,
        y=y,
        historical=historical,
        future=future,
        feature_mean=mu,
        feature_std=sd,
    )
    splits = {"future": future, "historical": historical}
    counts = {}
    for fraction in FRACTIONS:
        fit, val = draw_indices(historical, fraction)
        key = str(fraction)
        splits[f"fit_{key}"], splits[f"val_{key}"] = fit, val
        assert fit.max() < val.min() < future.min()
        counts[key] = dict(fit=len(fit), val=len(val), total_labels=len(fit) + len(val))
    np.savez_compressed(args.output / "splits.npz", **splits)
    protocol = dict(
        features=FEATURES,
        architecture=ARCH,
        fractions=FRACTIONS,
        counts=counts,
        historical_rows=len(historical),
        test_rows=len(future),
        init_seeds=SEEDS,
        subset_seed=42,
        pretraining_seed=42,
        epochs=args.epochs,
        patience=args.patience,
        ssl_epochs=args.ssl_epochs,
        batch=256,
        ssl_batch=512,
        temporal_validation=(
            "Last 10% of historical rows; supplied within-year row order"
        ),
        standardization=(
            "Features: early 90% unlabeled source pool; target: labeled fit only"
        ),
        tree="Frozen prior reference preset: 800 trees, depth 6, lr .03, sample .8/.8",
        supervised="Frozen prior TREA preset: 2 layers, dropout 0, lr .0003",
        pretraining=(
            "30% masked numeric cells, lr .0008; "
            "select best fixed-mask SSL validation MSE"
        ),
        low_label_modes=(
            "At <=5%, select frozen probe lr .001 vs finetune lr .00008 "
            "on temporal validation"
        ),
        other_modes="Above 5%, finetune lr .0003",
        arms=["xgb", "scratch", "pre", "pre_target"],
        pre="Unlabeled 2011-2013 features only",
        pre_target="Unlabeled 2011-2015 features: transductive, never future labels",
        primary=(
            "Future RMSE, original NOx units; single fits and equal three-init averages"
        ),
        fairness=(
            "All models fit the identical labeled fit rows; "
            "validation labels select neural checkpoints/modes"
        ),
        scope=(
            "Exploratory updated-model diagnostic; one fixed draw per fraction, "
            "one SSL seed per arm; not the original gate"
        ),
    )
    save_json(args.output / "protocol.json", protocol)


def optimizer_for(model, lr, steps):
    opt = torch.optim.AdamW(
        adamw_parameter_groups(model, 3e-4), lr=lr, betas=(0.9, 0.95)
    )
    warmup = max(1, int(steps * 0.05))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: min(1.0, (step + 1) / warmup)
    )
    return opt, scheduler


def pretrain(args, arm, x, historical, future, config):
    indices = historical if arm == "pre" else np.concatenate([historical, future])
    order = np.random.RandomState(42).permutation(indices)
    cut = int(len(order) * 0.9)
    train_idx, val_idx = order[:cut], order[cut:]
    xx, xv = x[train_idx], x[val_idx]
    generator = torch.Generator().manual_seed(4242)
    val_mask = torch.rand(xv.shape, generator=generator) < 0.3
    masked_val = xv.masked_fill(val_mask, float("-inf"))
    model = MaskedTabularEncoder(config, **ARCH)
    opt, sched = optimizer_for(model, 8e-4, args.ssl_epochs * math.ceil(len(xx) / 512))
    best, best_epoch, best_state = float("inf"), -1, None
    history = []
    started = time.monotonic()
    for epoch in range(args.ssl_epochs):
        model.train()
        total, cells = 0.0, 0
        for batch_idx in torch.randperm(len(xx), generator=generator).split(512):
            batch = xx[batch_idx]
            mask = torch.rand(batch.shape, generator=generator) < 0.3
            opt.zero_grad(set_to_none=True)
            output = model(batch.masked_fill(mask, float("-inf")), None).numeric
            loss = masked_mse(output, batch, mask)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite SSL loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * int(mask.sum())
            cells += int(mask.sum())
        model.eval()
        with torch.no_grad():
            predictions = torch.cat(
                [model(b, None).numeric for b in masked_val.split(512)]
            )
            score = float(masked_mse(predictions, xv, val_mask))
        history.append(
            dict(epoch=epoch + 1, train_masked_mse=total / cells, val_masked_mse=score)
        )
        if score < best:
            best, best_epoch = score, epoch + 1
            best_state = copy.deepcopy(model.tabular_encoder.state_dict())
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"SSL {arm} epoch={epoch + 1} val={score:.5f} best={best:.5f}",
                flush=True,
            )
    torch.save(best_state, args.output / f"encoder_{arm}.pt")
    np.savez_compressed(
        args.output / f"ssl_{arm}.splits.npz",
        train=train_idx,
        val=val_idx,
        val_mask=val_mask.numpy(),
    )
    save_json(
        args.output / f"ssl_{arm}.json",
        dict(
            arm=arm,
            seed=42,
            train_rows=len(train_idx),
            val_rows=len(val_idx),
            best_epoch=best_epoch,
            best_masked_mse=best,
            history=history,
            seconds=time.monotonic() - started,
        ),
    )


def fit_neural(args, config, x, y, fit, val, seed, encoder, mode):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TabularRegressor(config, **ARCH)
    if encoder is not None:
        model.tabular_encoder.load_state_dict(encoder, strict=True)
    if mode == "probe":
        for p in model.tabular_encoder.parameters():
            p.requires_grad_(False)
    lr = 1e-3 if mode == "probe" else 8e-5 if mode == "lowft" else 3e-4
    opt, sched = optimizer_for(model, lr, args.epochs * math.ceil(len(fit) / 256))
    generator = torch.Generator().manual_seed(seed + 10000)
    best, best_epoch, best_state = float("inf"), -1, None
    history = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for ix in torch.randperm(len(fit), generator=generator).split(256):
            idx = fit[ix]
            opt.zero_grad(set_to_none=True)
            loss = (model(x[idx], None).flatten() - y[idx]).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite supervised loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        score = float(np.sqrt(np.mean((predict(model, x[val]) - y[val].numpy()) ** 2)))
        history.append(
            dict(epoch=epoch + 1, train_mse=total / len(fit), val_rmse_scaled=score)
        )
        if score < best:
            best, best_epoch, best_state = (
                score,
                epoch + 1,
                copy.deepcopy(model.state_dict()),
            )
        if epoch == 0 or (epoch + 1) % 50 == 0:
            print(
                f"{args.job} mode={mode} epoch={epoch + 1} "
                f"val={score:.5f} best={best:.5f}",
                flush=True,
            )
        if epoch + 1 - best_epoch >= args.patience:
            break
    model.load_state_dict(best_state)
    return model, dict(
        mode=mode,
        lr=lr,
        best_epoch=best_epoch,
        epochs_run=epoch + 1,
        best_val_rmse_scaled=best,
        history=history,
    )


def run_job(args):
    arm, fraction, seed = args.job
    fraction, seed = float(fraction), int(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    arrays = np.load(args.output / "data.npz")
    x = torch.from_numpy(arrays["x"])
    historical, future = arrays["historical"], arrays["future"]
    config = SingleRowConfig.generate(
        pd.DataFrame(
            x[historical[: int(len(historical) * 0.9)]].numpy(), columns=FEATURES
        )
    )
    if arm.startswith("ssl_"):
        pretrain(args, arm.removeprefix("ssl_"), x, historical, future, config)
        return
    fit, val = draw_indices(historical, fraction)
    saved = np.load(args.output / "splits.npz")
    assert np.array_equal(fit, saved[f"fit_{fraction}"])
    assert np.array_equal(val, saved[f"val_{fraction}"])
    raw_y = arrays["y"]
    mu, sd = target_stats(raw_y[fit])
    y = torch.from_numpy((raw_y - mu) / sd)
    started = time.monotonic()
    record = dict(arm=arm, fraction=fraction, seed=seed, target_mean=mu, target_std=sd)
    if arm == "xgb":
        model = XGBRegressor(
            n_estimators=800,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=2,
            random_state=seed,
        )
        model.fit(x[fit].numpy(), y[fit].numpy())

        def infer(xx):
            return model.predict(xx.numpy())
    else:
        encoder = (
            None
            if arm == "scratch"
            else torch.load(
                args.output / f"encoder_{arm}.pt", map_location="cpu", weights_only=True
            )
        )
        modes = (
            ["probe", "lowft"] if encoder is not None and fraction <= 0.05 else ["ft"]
        )
        selected, best = None, float("inf")
        records = []
        for mode in modes:
            fitted, rec = fit_neural(args, config, x, y, fit, val, seed, encoder, mode)
            records.append(rec)
            if rec["best_val_rmse_scaled"] < best:
                selected, best = fitted, rec["best_val_rmse_scaled"]
                record["selected_mode"] = mode
        model = selected
        record["modes"] = records

        def infer(xx):
            return predict(model, xx)

    indices = {"fit": fit, "val": val, "future": future}
    # Test inference only after the validation-selected model/mode is frozen.
    predictions = {s: infer(x[idx]) * sd + mu for s, idx in indices.items()}
    record["rmse"] = {
        s: float(np.sqrt(np.mean((raw_y[idx] - predictions[s]) ** 2)))
        for s, idx in indices.items()
    }
    record["constant_mean_future_rmse"] = float(
        np.sqrt(np.mean((raw_y[future] - mu) ** 2))
    )
    record["seconds"] = time.monotonic() - started
    stem = f"{arm}_{fraction}_{seed}"
    np.savez_compressed(args.output / f"{stem}.predictions.npz", **predictions)
    save_json(args.output / f"{stem}.json", record)
    print(
        f"RESULT {stem} {record['rmse']} mode={record.get('selected_mode')}", flush=True
    )


def launch(args, job):
    stem = "_".join(map(str, job))
    cmd = [
        sys.executable,
        "-m",
        "scripts.nox_temporal_pretraining",
        "--output",
        str(args.output),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--ssl-epochs",
        str(args.ssl_epochs),
        "--job",
        *map(str, job),
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
    print("DONE", stem, flush=True)


def summarize(output):
    arrays = np.load(output / "data.npz")
    target = arrays["y"][arrays["future"]]
    summary = {}
    for fraction in FRACTIONS:
        results = {}
        for arm in ["xgb", "scratch", "pre", "pre_target"]:
            paths = [output / f"{arm}_{fraction}_{seed}.json" for seed in SEEDS]
            if not all(p.exists() for p in paths):
                continue
            records = [json.loads(p.read_text()) for p in paths]
            ps = [np.load(p.with_suffix(".predictions.npz"))["future"] for p in paths]
            ensemble = np.mean(ps, axis=0)
            np.save(output / f"ensemble_{arm}_{fraction}.npy", ensemble)
            scores = [r["rmse"]["future"] for r in records]
            results[arm] = dict(
                mean=float(np.mean(scores)),
                std=float(np.std(scores, ddof=1)),
                ensemble=float(np.sqrt(np.mean((target - ensemble) ** 2))),
                selected_modes=[r.get("selected_mode") for r in records],
                constant_mean=records[0]["constant_mean_future_rmse"],
            )
        summary[str(fraction)] = results
    save_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "logs/nox_temporal_pretraining"
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--ssl-epochs", type=int, default=60)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--job", nargs=3)
    args = parser.parse_args()
    if args.job:
        run_job(args)
        return
    setup(args)
    paths = [
        Path(__file__),
        ROOT / "trea/models/single_row.py",
        ROOT / "uv.lock",
        *sorted((ROOT / "data/nox").glob("gt_*.csv")),
    ]
    save_json(
        args.output / "provenance.json",
        {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
    )
    phases = [
        [(a, f, s) for f in FRACTIONS for a in ["xgb", "scratch"] for s in SEEDS],
        [("ssl_pre", 1.0, 42), ("ssl_pre_target", 1.0, 42)],
        [(a, f, s) for f in FRACTIONS for a in ["pre", "pre_target"] for s in SEEDS],
    ]
    for phase in phases:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(lambda job: launch(args, job), phase))
        print(json.dumps(summarize(args.output), indent=2), flush=True)


if __name__ == "__main__":
    main()
