"""Replicate the real-NOx masked-pretraining result across label draws and SSL seeds.

LESSONS.md §3h: source-only masked pretraining improved the 1/2/5% ensembles
on one label draw with one SSL seed. This runner repeats that diagnostic with
the recipe frozen, on fresh label draws and independent SSL seeds, and applies
the pre-registered verdict in ``docs/NOX_SSL_REPLICATION.md``.

Everything about the recipe is imported from ``scripts.nox_temporal_pretraining``:
architecture, supervised and SSL optimizers, adaptation modes, early stopping,
and the XGBoost preset. Only three things vary here:

- the label draw (which historical rows are labeled, and which late rows validate),
- the SSL seed (encoder init, masks, and batch order; the SSL corpus split is fixed),
- the supervised init seed (10/11/12, as before).

Draw 42 with SSL seed 42 is the original cell. It runs as a reproduction check and
is excluded from the verdict, because its result is already known.

Run (long; see the run plan in the doc):

    uv run python -m scripts.nox_ssl_replication --output logs/nox_ssl_replication \
        > logs/nox_ssl_replication.log 2>&1 &

Re-running with the same ``--output`` resumes: finished jobs are skipped, and the
saved protocol must match exactly.
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

import scripts.nox_temporal_pretraining as base

from scripts.nox_encoding_ablation import FEATURES, ROOT
from scripts.nox_reference_models import predict, save_json
from trea.models.single_row import MaskedTabularEncoder
from trea.utils.single_row_data import SingleRowConfig


DRAW_SEEDS = [101, 102, 103, 104, 105]
SSL_SEEDS = [42, 43, 44]
INIT_SEEDS = base.SEEDS
FRACTIONS = [0.01, 0.02, 0.05, 0.10]
LOW = [0.01, 0.02, 0.05]
REPRO_DRAW, REPRO_SSL = 42, 42
FROZEN = dict(epochs=300, patience=30, ssl_epochs=60)
SSL_ARMS = ["pre", "pre_target", "random"]
# Ensemble RMSEs of the original cell (docs/results/nox_temporal_pretraining.json).
ORIGINAL = {
    "0.01": dict(xgb=13.102, scratch=12.453, pre=11.895, pre_target=11.270),
    "0.02": dict(xgb=12.254, scratch=12.014, pre=11.692, pre_target=11.516),
    "0.05": dict(xgb=12.775, scratch=11.903, pre=11.492, pre_target=11.682),
    "0.1": dict(xgb=13.009, scratch=12.224, pre=12.290, pre_target=12.583),
}
ORIGINAL_RANDOM = {"0.01": 12.968, "0.02": 13.760, "0.05": 12.861}


def stem(arm, fraction, draw, ssl, init):
    return f"{arm}_f{fraction}_d{draw}_s{ssl}_i{init}"


def plan_jobs(draws, ssl_seeds, fractions, repro=True):
    """All fits as (arm, fraction, draw, ssl_seed, init_seed); ssl_seed 0 = none."""
    cells = [(d, ssl_seeds) for d in draws]
    if repro:
        cells.append((REPRO_DRAW, [REPRO_SSL]))
    supervised, pretrained = [], []
    for draw, seeds in cells:
        for f in fractions:
            for i in INIT_SEEDS:
                supervised += [
                    (a, f, draw, 0, i) for a in ["xgb", "xgb_all", "scratch"]
                ]
                for s in seeds:
                    pretrained += [(a, f, draw, s, i) for a in ["pre", "pre_target"]]
                    if f in LOW:
                        # Untrained encoder: needs no SSL fit, so it runs early.
                        supervised.append(("random", f, draw, s, i))
    ssl = sorted(
        {
            (a, 1.0, 0, s, 0)
            for a in ["ssl_pre", "ssl_pre_target"]
            for _, seeds in cells
            for s in seeds
        }
    )
    return [supervised, ssl, pretrained]


def protocol_for(args):
    return dict(
        purpose="Replicate LESSONS §3h source-only SSL low-label benefit",
        recipe_source="scripts/nox_temporal_pretraining.py (imported, unchanged)",
        architecture=base.ARCH,
        xgb_preset=base.XGB_PRESET,
        epochs=args.epochs,
        patience=args.patience,
        ssl_epochs=args.ssl_epochs,
        frozen_recipe=all(getattr(args, k) == v for k, v in FROZEN.items()),
        fractions=args.fractions,
        low_fractions=LOW,
        draw_seeds=args.draws,
        ssl_seeds=args.ssl_seeds,
        init_seeds=INIT_SEEDS,
        reproduction_cell=dict(draw=REPRO_DRAW, ssl=REPRO_SSL) if args.repro else None,
        ssl_seed_controls="encoder init, train/val masks, batch order",
        ssl_split="fixed seed-42 corpus split for every SSL seed",
        arms=dict(
            xgb="Frozen preset, fits the same rows as the neural models",
            xgb_all="Frozen preset, fits fit + validation labels (old convention)",
            scratch="Supervised only",
            pre="SSL on unlabeled 2011-2013 features",
            pre_target="SSL on unlabeled 2011-2015 features (transductive)",
            random="Untrained encoder with the SSL seed's init; same adaptation",
        ),
        adaptation=(
            "<=5%: frozen probe lr .001 vs finetune lr .00008, picked on temporal "
            "validation; 10%: finetune lr .0003"
        ),
        verdict="docs/NOX_SSL_REPLICATION.md, pre-registered before any fit",
    )


def setup(args):
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = protocol_for(args)
    path = args.output / "protocol.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved != json.loads(json.dumps(protocol)):
            raise SystemExit(f"{path} differs from this invocation; refusing to resume")
        return
    historical, future = base.write_data(args.output)
    splits = {"future": future, "historical": historical}
    draws = args.draws + ([REPRO_DRAW] if args.repro else [])
    for draw in draws:
        for fraction in args.fractions:
            fit, val = base.draw_indices(historical, fraction, draw)
            assert fit.max() < val.min() < future.min()
            splits[f"fit_{fraction}_d{draw}"], splits[f"val_{fraction}_d{draw}"] = (
                fit,
                val,
            )
    np.savez_compressed(args.output / "splits.npz", **splits)
    paths = [
        Path(__file__),
        Path(base.__file__),
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
    save_json(path, protocol)


def random_encoder(config, ssl_seed):
    torch.manual_seed(ssl_seed)
    return copy.deepcopy(
        MaskedTabularEncoder(config, **base.ARCH).tabular_encoder.state_dict()
    )


def run_job(args):
    arm, fraction, draw, ssl, init = args.job
    fraction, draw, ssl, init = float(fraction), int(draw), int(ssl), int(init)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(init)
    np.random.seed(init)
    arrays = np.load(args.output / "data.npz")
    x = torch.from_numpy(arrays["x"])
    historical, future = arrays["historical"], arrays["future"]
    config = SingleRowConfig.generate(
        pd.DataFrame(
            x[historical[: int(len(historical) * 0.9)]].numpy(), columns=FEATURES
        )
    )
    if arm.startswith("ssl_"):
        name = arm.removeprefix("ssl_")
        base.pretrain(
            args, name, x, historical, future, config, seed=ssl, stem=f"{name}_s{ssl}"
        )
        return
    fit, val = base.draw_indices(historical, fraction, draw)
    saved = np.load(args.output / "splits.npz")
    assert np.array_equal(fit, saved[f"fit_{fraction}_d{draw}"])
    assert np.array_equal(val, saved[f"val_{fraction}_d{draw}"])
    raw_y = arrays["y"]
    mu, sd = base.target_stats(raw_y[fit])
    y = torch.from_numpy((raw_y - mu) / sd)
    started = time.monotonic()
    record = dict(
        arm=arm,
        fraction=fraction,
        draw=draw,
        ssl_seed=ssl,
        seed=init,
        target_mean=mu,
        target_std=sd,
    )
    if arm in ("xgb", "xgb_all"):
        rows = fit if arm == "xgb" else np.concatenate([fit, val])
        model = XGBRegressor(
            **{**base.XGB_PRESET, "n_jobs": args.threads}, random_state=init
        )
        model.fit(x[rows].numpy(), y[rows].numpy())
        record["fit_rows"] = len(rows)

        def infer(xx):
            return model.predict(xx.numpy())
    else:
        if arm == "scratch":
            encoder = None
        elif arm == "random":
            encoder = random_encoder(config, ssl)
        else:
            encoder = torch.load(
                args.output / f"encoder_{arm}_s{ssl}.pt",
                map_location="cpu",
                weights_only=True,
            )
        modes = (
            ["probe", "lowft"] if encoder is not None and fraction <= 0.05 else ["ft"]
        )
        selected, best, records = None, float("inf"), []
        for mode in modes:
            fitted, rec = base.fit_neural(
                args, config, x, y, fit, val, init, encoder, mode
            )
            records.append(rec)
            if rec["best_val_rmse_scaled"] < best:
                selected, best = fitted, rec["best_val_rmse_scaled"]
                record["selected_mode"] = mode
        record["modes"] = records

        def infer(xx):
            return predict(selected, xx)

    indices = {"fit": fit, "val": val, "future": future}
    # Test inference happens only after the validation-selected model is frozen.
    predictions = {s: infer(x[idx]) * sd + mu for s, idx in indices.items()}
    record["rmse"] = {
        s: float(np.sqrt(np.mean((raw_y[idx] - predictions[s]) ** 2)))
        for s, idx in indices.items()
    }
    record["constant_mean_future_rmse"] = float(
        np.sqrt(np.mean((raw_y[future] - mu) ** 2))
    )
    record["seconds"] = time.monotonic() - started
    name = stem(arm, fraction, draw, ssl, init)
    np.savez_compressed(args.output / f"{name}.predictions.npz", **predictions)
    save_json(args.output / f"{name}.json", record)
    print(
        f"RESULT {name} future={record['rmse']['future']:.4f} "
        f"mode={record.get('selected_mode')}",
        flush=True,
    )


def done_marker(args, job):
    arm, fraction, draw, ssl, init = job
    if arm.startswith("ssl_"):
        return args.output / f"ssl_{arm.removeprefix('ssl_')}_s{ssl}.json"
    return args.output / f"{stem(*job)}.json"


def launch(args, job):
    if done_marker(args, job).exists():
        return
    name = stem(*job)
    cmd = [
        sys.executable,
        "-m",
        "scripts.nox_ssl_replication",
        "--output",
        str(args.output),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--ssl-epochs",
        str(args.ssl_epochs),
        "--threads",
        str(args.threads),
        "--job",
        *map(str, job),
    ]
    threads = str(args.threads)
    env = dict(
        os.environ,
        OMP_NUM_THREADS=threads,
        MKL_NUM_THREADS=threads,
        MPLCONFIGDIR="/tmp/trea-mpl",
        HF_HUB_OFFLINE="1",
    )
    with (args.output / f"{name}.log").open("w") as log:
        subprocess.run(
            cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    print("DONE", name, flush=True)


# ---------------------------------------------------------------- analysis


def collect(output):
    """Ensemble and single-fit future RMSE per cell (arm, fraction, draw, ssl)."""
    arrays = np.load(output / "data.npz")
    target = arrays["y"][arrays["future"]]
    groups = {}
    for path in sorted(output.glob("*_f*_d*_s*_i*.json")):
        r = json.loads(path.read_text())
        key = (r["arm"], str(r["fraction"]), r["draw"], r["ssl_seed"])
        groups.setdefault(key, []).append(path)
    cells = {}
    for key, paths in groups.items():
        if len(paths) != len(INIT_SEEDS):
            continue
        records = [json.loads(p.read_text()) for p in paths]
        preds = [np.load(p.with_suffix(".predictions.npz"))["future"] for p in paths]
        single = [r["rmse"]["future"] for r in records]
        cells[key] = dict(
            ensemble=float(np.sqrt(np.mean((target - np.mean(preds, 0)) ** 2))),
            single_mean=float(np.mean(single)),
            single_sd=float(np.std(single, ddof=1)),
            modes=[r.get("selected_mode") for r in records],
        )
    return cells


def bootstrap_ci(values, n=10000, seed=0):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n, len(values)))].mean(1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def fraction_test(ens, arm, fraction, draws, ssl_seeds, min_draws, control=True):
    """Pre-registered test for one SSL arm at one label budget.

    ``ens`` maps (arm, fraction, draw, ssl) -> ensemble RMSE; ssl=0 for scratch.
    Passes iff (A) mean paired delta vs scratch < 0 over all draw x SSL cells,
    (B) the per-draw mean delta is < 0 in at least ``min_draws`` draws, and
    (C) mean delta vs the seed-matched random-encoder control < 0.
    ``control=False`` skips (C); the random control only runs at <=5% labels.
    Returns None when any required cell is missing.
    """
    try:
        deltas = np.array(
            [
                [
                    ens[(arm, fraction, d, s)] - ens[("scratch", fraction, d, 0)]
                    for s in ssl_seeds
                ]
                for d in draws
            ]
        )
        rand = np.array(
            [
                [
                    ens[(arm, fraction, d, s)] - ens[("random", fraction, d, s)]
                    for s in ssl_seeds
                ]
                for d in draws
            ]
            if control
            else np.nan
        )
        scratch = np.array([ens[("scratch", fraction, d, 0)] for d in draws])
    except KeyError:
        return None
    per_draw = deltas.mean(1)
    a = bool(deltas.mean() < 0)
    b = int((per_draw < 0).sum())
    c = bool(rand.mean() < 0) if control else None
    return dict(
        mean_delta_vs_scratch=float(deltas.mean()),
        relative_change_pct=float(100 * (deltas / scratch[:, None]).mean()),
        draw_ci95=bootstrap_ci(per_draw),
        per_draw_delta=per_draw.tolist(),
        draws_improved=b,
        sd_between_draws=float(per_draw.std(ddof=1)) if len(draws) > 1 else None,
        sd_within_draw_over_ssl=(
            float(deltas.std(1, ddof=1).mean()) if len(ssl_seeds) > 1 else None
        ),
        mean_delta_vs_random=float(rand.mean()) if control else None,
        criteria=dict(
            A_mean_beats_scratch=a, B_draws_improved=b >= min_draws, C_beats_random=c
        ),
        passes=bool(a and b >= min_draws and c is not False),
    )


def verdict(ens, draws, ssl_seeds, arm="pre"):
    min_draws = math.ceil(0.8 * len(draws))  # 4 of 5
    tests = {
        str(f): fraction_test(ens, arm, str(f), draws, ssl_seeds, min_draws)
        for f in LOW
    }
    if any(t is None for t in tests.values()):
        return dict(arm=arm, status="INCOMPLETE", tests=tests)
    passed = sum(t["passes"] for t in tests.values())
    status = (
        "REPLICATED"
        if passed == len(LOW)
        else ("PARTIAL" if passed == len(LOW) - 1 else "NOT REPLICATED")
    )
    return dict(
        arm=arm,
        status=status,
        fractions_passed=passed,
        min_draws=min_draws,
        tests=tests,
    )


def tree_comparison(ens, draws, ssl_seeds, fractions):
    out = {}
    for f in map(str, fractions):
        row = {}
        for arm in ["xgb", "xgb_all", "scratch", "pre", "pre_target", "random"]:
            seeds = [0] if arm in ("xgb", "xgb_all", "scratch") else ssl_seeds
            vals = [ens.get((arm, f, d, s)) for d in draws for s in seeds]
            if vals and None not in vals:
                row[arm] = dict(
                    mean=float(np.mean(vals)),
                    sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                )
        out[f] = row
    return out


def reproduction(ens):
    out = {}
    for f, ref in ORIGINAL.items():
        for arm, value in {
            **ref,
            **({"random": ORIGINAL_RANDOM[f]} if f in ORIGINAL_RANDOM else {}),
        }.items():
            ssl = REPRO_SSL if arm in SSL_ARMS else 0
            got = ens.get((arm, f, REPRO_DRAW, ssl))
            if got is not None:
                out[f"{arm}@{f}"] = dict(
                    original=value, rerun=round(got, 3), diff=round(got - value, 3)
                )
    return out


def summarize(args):
    cells = collect(args.output)
    ens = {k: v["ensemble"] for k, v in cells.items()}
    draws, seeds = args.draws, args.ssl_seeds
    summary = dict(
        admissible=protocol_for(args)["frozen_recipe"],
        primary=verdict(ens, draws, seeds, "pre"),
        secondary_transductive=verdict(ens, draws, seeds, "pre_target"),
        secondary_ten_percent={
            arm: fraction_test(ens, arm, "0.1", draws, seeds, 0, control=False)
            for arm in ["pre", "pre_target"]
        }
        if 0.1 in args.fractions
        else None,
        means_over_draws=tree_comparison(ens, draws, seeds, args.fractions),
        reproduction_check=reproduction(ens) if args.repro else None,
        cells={"|".join(map(str, k)): v for k, v in sorted(cells.items())},
    )
    save_json(args.output / "summary.json", summary)
    if not summary["admissible"]:
        print("NOT ADMISSIBLE: recipe differs from the frozen one (smoke run?)")
    for key in ["primary", "secondary_transductive"]:
        v = summary[key]
        print(f"VERDICT {key} arm={v['arm']}: {v['status']}")
        for f, t in v["tests"].items():
            if t:
                print(
                    f"  frac={f} delta={t['mean_delta_vs_scratch']:+.3f} "
                    f"({t['relative_change_pct']:+.2f}%) draws_improved="
                    f"{t['draws_improved']}/{len(draws)} vs_random="
                    f"{t['mean_delta_vs_random']:+.3f} pass={t['passes']}"
                )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "logs/nox_ssl_replication"
    )
    parser.add_argument("--epochs", type=int, default=FROZEN["epochs"])
    parser.add_argument("--patience", type=int, default=FROZEN["patience"])
    parser.add_argument("--ssl-epochs", type=int, default=FROZEN["ssl_epochs"])
    parser.add_argument("--draws", type=int, nargs="+", default=DRAW_SEEDS)
    parser.add_argument("--ssl-seeds", type=int, nargs="+", default=SSL_SEEDS)
    parser.add_argument("--fractions", type=float, nargs="+", default=FRACTIONS)
    parser.add_argument(
        "--no-repro",
        dest="repro",
        action="store_false",
        help="Skip the original draw-42/SSL-42 reproduction cell",
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2, help="Threads per job")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print job counts only")
    parser.add_argument("--job", nargs=5)
    args = parser.parse_args()
    if args.job:
        run_job(args)
        return
    phases = plan_jobs(args.draws, args.ssl_seeds, args.fractions, args.repro)
    if args.dry_run:
        for name, jobs in zip(
            ["supervised+random", "ssl", "pretrained"], phases, strict=True
        ):
            print(name, len(jobs))
        return
    if not args.summarize_only:
        setup(args)
        for jobs in phases:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(lambda job: launch(args, job), jobs))
    summarize(args)


if __name__ == "__main__":
    main()
