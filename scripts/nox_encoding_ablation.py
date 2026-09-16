"""Paired, supervised NOx diagnostic; does NOT revise the industrial kill gate.

Run: .venv/bin/python -m scripts.nox_encoding_ablation

Only intervention: first-block (Q,K,V) = (identity,value,value) versus
(value,value,value). For this numeric-only experiment, value already includes
column identity. This changes BOTH the query and residual, not just the skip.
Basis/fusion and optimizer flags default to the original diagnostic settings.
Use --arms identity_query to hold the first block fixed for basis ablations.
Historical experiment defaults remain explicit despite the new production defaults;
a parameter-free pre-hook implements the query intervention. Three paired
initializations, one fixed data draw. Primary endpoint is historical random-holdout
RMSE; future years measure drift separately.
Positive diagnostic: >=2% lower mean primary RMSE and improvement in >=2/3 pairs.
This is an exploratory, single-dataset criterion, not statistical equivalence.
"""

import argparse
import copy
import hashlib
import json
import math
import time

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from trea.models.single_row import TabularRegressor
from trea.training.single_row import adamw_parameter_groups
from trea.utils.single_row_data import SingleRowConfig


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["at", "ap", "ah", "afdp", "gtep", "tit", "tat", "tey", "cdp"]


def value_query(_module, args):
    """Preserve identity-plus-value tokens in query and residual paths."""
    _, k, v = args
    return k, k, v


def bypass_norm(_module, inputs, _output):
    """Experiment-only bypass; retains parameters and paired initial states."""
    return inputs[0]


def prepare(n_fit, split_seed):
    frames = []
    for path in sorted((ROOT / "data/nox").glob("gt_*.csv")):
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower()
        df["year"] = int(path.stem.split("_")[1])
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    historical = df.index[df.year <= 2013].to_numpy()
    pool, iid = train_test_split(historical, test_size=0.2, random_state=split_seed)
    fit, remaining = train_test_split(pool, train_size=n_fit, random_state=split_seed)
    val, _ = train_test_split(
        remaining, train_size=max(256, n_fit // 4), random_state=split_seed + 1
    )
    indices = {
        "fit": fit,
        "val": val,
        "iid": iid,
        "future": df.index[df.year >= 2014].to_numpy(),
    }
    assert sum(len(v) for v in indices.values()) == len(
        set(np.concatenate(list(indices.values())))
    )
    mu, sd = df.loc[fit, FEATURES].mean(), df.loc[fit, FEATURES].std().replace(0, 1)
    ymu, ysd = float(df.loc[fit, "nox"].mean()), float(df.loc[fit, "nox"].std())
    data = {}
    for name, idx in indices.items():
        x = ((df.loc[idx, FEATURES] - mu) / sd).to_numpy(np.float32)
        y = ((df.loc[idx, "nox"] - ymu) / ysd).to_numpy(np.float32)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        data[name] = (torch.from_numpy(x), torch.from_numpy(y))
    config = SingleRowConfig.generate(
        pd.DataFrame(data["fit"][0].numpy(), columns=FEATURES)
    )
    return data, config, ysd, indices


@torch.no_grad()
def predict(model, x):
    model.eval()
    return torch.cat([model(b, None).flatten().cpu() for b in x.split(1024)]).numpy()


def rmse(y, p, scale):
    return float(np.sqrt(np.mean((np.asarray(y) - p) ** 2)) * scale)


def fit_model(args, config, data, scale, seed, arm):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TabularRegressor(
        config,
        d_model=128,
        n_heads=4,
        n_layers=4,
        dropout=args.dropout,
        value_encoder=args.value_encoder,
        fusion=args.fusion,
        # Preserve this historical experiment's explicit keep/skip control.
        first_prenormalization=not args.skip_first_attention_norm,
    )
    initial_hash = hashlib.sha256(
        b"".join(t.detach().numpy().tobytes() for t in model.state_dict().values())
    ).hexdigest()
    if arm == "value_query":
        model.tabular_encoder.layers[0].register_forward_pre_hook(value_query)
    model.to(args.device)
    optimizer = torch.optim.AdamW(
        adamw_parameter_groups(model, 3e-4), lr=args.lr, betas=(0.9, 0.95)
    )
    n_steps = args.epochs * math.ceil(len(data["fit"][0]) / 512)
    warmup = max(1, int(n_steps * 0.05))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup)
    )
    generator = torch.Generator().manual_seed(seed + 10000)
    x, y = (t.to(args.device) for t in data["fit"])
    xv, yv = data["val"]
    xv = xv.to(args.device)
    best, best_epoch, best_state = float("inf"), -1, None
    history = []
    started = time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for idx in torch.randperm(len(x), generator=generator).split(512):
            idx = idx.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x[idx], None).flatten() - y[idx]).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss: {arm}, seed={seed}, epoch={epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(idx)
        val = rmse(yv, predict(model, xv), scale)
        history.append(
            {
                "epoch": epoch + 1,
                "train_dropout_rmse": math.sqrt(total / len(x)) * scale,
                "val_rmse": val,
            }
        )
        if val < best:
            best, best_epoch = val, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"{arm} seed={seed} epoch={epoch + 1} val={val:.4f} best={best:.4f} "
                f"seconds={time.monotonic() - started:.1f}",
                flush=True,
            )
        if epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(best_state)
    scores, predictions = {}, {}
    for name, (xx, yy) in data.items():
        predictions[name] = predict(model, xx.to(args.device))
        scores[name] = rmse(yy, predictions[name], scale)
    return {
        "arm": arm,
        "seed": seed,
        "initial_state_sha256": initial_hash,
        "best_epoch": best_epoch + 1,
        "epochs_run": epoch + 1,
        "seconds": time.monotonic() - started,
        "rmse": scores,
        "history": history,
    }, predictions


def summarize(records):
    summary = {}
    for arm in ("identity_query", "value_query", "xgb"):
        runs = [r for r in records if r["arm"] == arm]
        if runs:
            summary[arm] = {
                split: {
                    "mean": float(np.mean([r["rmse"][split] for r in runs])),
                    "std": float(np.std([r["rmse"][split] for r in runs], ddof=1))
                    if len(runs) > 1
                    else None,
                }
                for split in ("fit", "val", "iid", "future")
            }
    a = {r["seed"]: r for r in records if r["arm"] == "identity_query"}
    b = {r["seed"]: r for r in records if r["arm"] == "value_query"}
    paired = sorted(a.keys() & b.keys())
    if paired:
        assert all(
            a[s]["initial_state_sha256"] == b[s]["initial_state_sha256"] for s in paired
        )
        base = np.mean([a[s]["rmse"]["iid"] for s in paired])
        delta = [b[s]["rmse"]["iid"] - a[s]["rmse"]["iid"] for s in paired]
        summary["paired_iid"] = {
            "seeds": paired,
            "value_minus_identity": delta,
            "relative_improvement": float(-np.mean(delta) / base),
            "improved_pairs": sum(d < 0 for d in delta),
            "positive_diagnostic": bool(
                len(paired) >= 3
                and -np.mean(delta) / base >= 0.02
                and sum(d < 0 for d in delta) / len(paired) >= 2 / 3
            ),
        }
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", nargs="+", type=int, default=[10, 11, 12])
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--n-fit", type=int, default=2219)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument(
        "--value-encoder",
        choices=["identity", "ple", "periodic", "dual"],
        default="identity",
    )
    p.add_argument(
        "--fusion",
        choices=["additive", "hadamard", "product", "product_sum"],
        default="additive",
    )
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument(
        "--skip-first-attention-norm",
        action="store_true",
        help="Bypass only the first block's attention-input LayerNorm.",
    )
    p.add_argument(
        "--arms",
        nargs="+",
        choices=["identity_query", "value_query"],
        default=["identity_query", "value_query"],
    )
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--output", type=Path, default=ROOT / "logs/nox_encoding_ablation.json"
    )
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    data, config, scale, indices = prepare(args.n_fit, args.split_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": __doc__,
        "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "features": FEATURES,
        "counts": {k: len(v) for k, v in indices.items()},
        "target_scale": scale,
        "torch_version": torch.__version__,
        "records": [],
    }
    np.savez_compressed(args.output.with_suffix(".splits.npz"), **indices)
    predictions = {}

    def save():
        report["summary"] = summarize(report["records"])
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(args.output)
        np.savez_compressed(args.output.with_suffix(".predictions.npz"), **predictions)

    save()  # Persist the protocol before training or viewing any test result.
    print(json.dumps({"counts": report["counts"], "args": report["args"]}), flush=True)
    for seed in args.seeds:
        tree = XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=args.threads,
            random_state=seed,
        )
        tree.fit(data["fit"][0].numpy(), data["fit"][1].numpy())
        scores = {}
        for name, (xx, yy) in data.items():
            pred = tree.predict(xx.numpy())
            scores[name] = rmse(yy, pred, scale)
            predictions[f"xgb_{seed}_{name}"] = pred
        report["records"].append({"arm": "xgb", "seed": seed, "rmse": scores})
        save()
        for arm in args.arms:
            rec, pred = fit_model(args, config, data, scale, seed, arm)
            report["records"].append(rec)
            predictions.update({f"{arm}_{seed}_{name}": v for name, v in pred.items()})
            save()
            print(f"RESULT {arm} seed={seed}: {rec['rmse']}", flush=True)
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
