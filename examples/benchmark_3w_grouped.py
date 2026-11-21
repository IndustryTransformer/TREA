"""Controlled 3W experiments with internal grouped validation and reserved test.

The legacy benchmark remains available. This entrypoint never reads test signals
unless --evaluate-test is explicit. All fitting and decisions use train/val only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from torch import nn
from xgboost import XGBClassifier

from examples.benchmark_3w_binary import (
    binary_metrics,
    extract_engineered_features,
    extract_naive_features,
    remap_to_binary,
)
from trea.models import TriplePatchTransformer
from trea.training.single_row import SingleRowClassifier, adamw_parameter_groups
from trea.utils.single_row_data import SingleRowConfig
from utils.three_w import ThreeWDataset


FAMILIES = {
    "all": list(range(8)),
    "level": [0, 2, 3],
    "variation": [1, 7],
    "missing": [4],
    "direction": [5, 6],
    "no_direction": [0, 1, 2, 3, 4, 7],
    "no_missing": [0, 1, 2, 3, 5, 6, 7],
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="xgb,mlp,tabm,single_row")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--split-seed", type=int, default=20260917)
    p.add_argument("--holdout-fraction", type=float, default=0.2)
    p.add_argument("--internal-val-fraction", type=float, default=0.25)
    p.add_argument("--toolkit-path", default="/home/kailukowiak/Work/3W")
    p.add_argument("--data-path", default="/home/kailukowiak/Work/3W/dataset")
    p.add_argument("--cache-dir", default="logs/benchmark_3w_binary/grouped_cache")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--group-manifest", default=None)
    p.add_argument(
        "--value-policy",
        choices=["legacy_clip", "preserve_range"],
        default="legacy_clip",
    )
    p.add_argument("--evaluate-test", action="store_true")
    p.add_argument("--train-cap", type=int, default=200000)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=3e-4)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--row-dropout", type=float, default=0.0)
    p.add_argument(
        "--preprocess", choices=["standard", "quantile", "raw"], default="standard"
    )
    p.add_argument("--feature-family", choices=list(FAMILIES), default="all")
    p.add_argument("--relative-stats", action="store_true")
    p.add_argument("--shuffle-time", action="store_true")
    p.add_argument("--tabm-k", type=int, default=16)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--balance", choices=["none", "sqrt", "full"], default="sqrt")
    p.add_argument(
        "--group-loss-power", type=float, choices=[0.0, 0.5, 1.0], default=0.0
    )
    return p.parse_args()


def load_data(args):
    config = {
        "seed": args.split_seed,
        "holdout": args.holdout_fraction,
        "internal_val": args.internal_val_fraction,
        "window": 96,
        "data_path": args.data_path,
        "version": 1,
    }
    value_policy = getattr(args, "value_policy", "legacy_clip")
    group_manifest = None
    if getattr(args, "group_manifest", None):
        group_manifest = json.loads(Path(args.group_manifest).read_text())
        config["group_manifest_sha256"] = hashlib.sha256(
            json.dumps(group_manifest, sort_keys=True).encode()
        ).hexdigest()
    if value_policy != "legacy_clip":
        config["value_policy"] = value_policy
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    root = Path(args.cache_dir) / key
    root.mkdir(parents=True, exist_ok=True)
    splits = ["train", "val"] + (["test"] if args.evaluate_test else [])
    data = {}
    stats = None
    manifests = {}
    for split in splits:
        manifest_path = root / f"{split}.json"
        if not manifest_path.exists():
            ds = ThreeWDataset(
                toolkit_path=args.toolkit_path,
                data_path=args.data_path,
                window_size=96,
                stride=96,
                split=split,
                split_mode="well",
                val_fraction=args.holdout_fraction,
                internal_val_fraction=args.internal_val_fraction,
                normalization_stats=stats,
                seed=args.split_seed,
                value_policy=value_policy,
                group_manifest=group_manifest,
            )
            remap_to_binary(ds)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                features = extract_engineered_features(ds.windows)
            arrays = {
                "windows": ds.windows,
                "features": features,
                "labels": ds.labels,
                "file_ids": ds.file_ids,
                "group_ids": ds.group_ids,
            }
            for name, array in arrays.items():
                np.save(root / f"{split}_{name}.npy", array)
            manifest = {
                **config,
                "split": split,
                "n": len(ds),
                "files": ds.selected_files,
                "groups": ds.selected_groups,
                "normalization": {
                    k: v.tolist() for k, v in ds.normalization_stats.items()
                },
                "class_counts": ds.class_counts.tolist(),
            }
            manifest_path.write_text(json.dumps(manifest, indent=2))
            del ds, features, arrays
        manifest = json.loads(manifest_path.read_text())
        manifests[split] = manifest
        if split == "train":
            stats = {k: np.asarray(v) for k, v in manifest["normalization"].items()}
        data[split] = {
            name: np.load(root / f"{split}_{name}.npy", mmap_mode="r")
            for name in ["windows", "features", "labels", "file_ids", "group_ids"]
        }
    for i, a in enumerate(splits):
        for b in splits[i + 1 :]:
            if set(manifests[a]["groups"]) & set(manifests[b]["groups"]):
                raise RuntimeError(f"Group leakage: {a}/{b}")
    return data, manifests


def choose_threshold(y, scores):
    """Exact macro-F1 optimum among distinct validation scores, with >= ties."""
    order = np.argsort(-scores, kind="stable")
    s, y = np.asarray(scores)[order], np.asarray(y)[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s) - 1]
    tp = np.r_[0, np.cumsum(y)[ends]]
    fp = np.r_[0, (ends + 1) - np.cumsum(y)[ends]]
    fn, tn = y.sum() - tp, (len(y) - y.sum()) - fp
    f1 = tp / np.maximum(2 * tp + fp + fn, 1) + tn / np.maximum(2 * tn + fp + fn, 1)
    thresholds = np.r_[np.nextafter(s[0], np.inf), s[ends]]
    return float(thresholds[np.argmax(f1)])


def metrics(y, scores, threshold, file_ids):
    result = binary_metrics((scores >= threshold).astype(int), y, scores)
    result["macro_f1_at_05"] = binary_metrics((scores >= 0.5).astype(int), y, scores)[
        "macro_f1"
    ]
    result["threshold_from_validation"] = threshold
    # Instances may contain normal and fault windows. This secondary target means
    # majority-fault instance, not event detection; report the definition explicitly.
    _, inv = np.unique(file_ids, return_inverse=True)
    counts = np.bincount(inv)
    iy = (np.bincount(inv, weights=y) / counts >= 0.5).astype(int)
    ip = np.bincount(inv, weights=scores) / counts
    result["instance_majority"] = binary_metrics((ip >= threshold).astype(int), iy, ip)
    return result


class RowForward(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.model(x, None)


class TemporalForward(nn.Module):
    def __init__(self, model, shuffle=False):
        super().__init__()
        self.model = model
        self.shuffle = shuffle

    def forward(self, x):
        if self.shuffle:
            # Same timestamp permutation across sensors preserves simultaneous
            # observations; fresh order each training batch, fixed at evaluation.
            gen = (
                torch.Generator(device=x.device).manual_seed(1701)
                if not self.training
                else None
            )
            order = torch.rand(
                len(x), x.shape[-1], device=x.device, generator=gen
            ).argsort(dim=-1)
            x = x.gather(2, order[:, None, :].expand_as(x))
        cat = torch.empty(len(x), 0, x.shape[-1], dtype=torch.long, device=x.device)
        return self.model(x, cat)


def build_model(name, x, args):
    if name in {"single_row", "single_row_ple"}:
        config = SingleRowConfig.generate(
            pd.DataFrame(x, columns=[f"feature_{i}" for i in range(x.shape[1])])
        )
        return RowForward(
            SingleRowClassifier(
                config,
                num_classes=2,
                d_model=args.width,
                n_heads=4,
                n_layers=args.layers,
                dropout=getattr(args, "row_dropout", 0.0),
                logit_adjustment_tau=0.0,
                value_encoder="ple" if name == "single_row_ple" else "identity",
                fusion="hadamard" if name == "single_row_ple" else "product_sum",
            )
        )
    if name == "mlp":
        return nn.Sequential(
            nn.Linear(x.shape[1], args.width),
            nn.ReLU(),
            nn.Linear(args.width, args.width),
            nn.ReLU(),
            nn.Linear(args.width, 2),
        )
    if name in {"tabm", "tabm_ple"}:
        from tabm import TabM

        embeddings = None
        if name == "tabm_ple":
            from rtdl_num_embeddings import PiecewiseLinearEmbeddings

            # Quantile knots use training rows only. Constant columns receive a
            # harmless interval so the embedding library has two distinct knots.
            bins = []
            for col in x.T:
                knots = np.unique(np.quantile(col, np.linspace(0, 1, 33)))
                if len(knots) == 1:
                    knots = np.array([knots[0] - 1, knots[0] + 1])
                bins.append(torch.tensor(knots, dtype=torch.float32))
            embeddings = PiecewiseLinearEmbeddings(
                bins, d_embedding=16, activation=False, version="B"
            )

        return TabM.make(
            n_num_features=x.shape[1],
            cat_cardinalities=[],
            d_out=2,
            n_blocks=args.layers,
            d_block=args.width,
            dropout=0.0,
            k=args.tabm_k,
            num_embeddings=embeddings,
        )
    if name in {"temporal", "direct_stat", "stats_only"}:
        model = TriplePatchTransformer(
            C_num=x.shape[1],
            C_cat=0,
            cat_cardinalities=[],
            T=x.shape[2],
            d_model=args.width,
            num_classes=2,
            n_head=4,
            num_layers=args.layers,
            dropout=0.0,
            use_stat_tokens=True,
            use_pre_patch_feature_attention=False,
            direct_stat_pooling=name == "direct_stat",
            stats_only=name == "stats_only",
            stat_sensor_identity=name != "temporal",
            stat_include_delta=name != "temporal",
            stat_exact_engineered=name != "temporal",
        )
        return TemporalForward(model, args.shuffle_time)
    raise ValueError(f"Unknown neural model {name}")


@torch.no_grad()
def predict(model, x, batch_size, device):
    model.eval()
    scores = []
    for start in range(0, len(x), batch_size):
        batch = torch.as_tensor(np.array(x[start : start + batch_size]), device=device)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = model(batch)
        probs = logits.float().softmax(-1)
        if probs.ndim == 3:
            probs = probs.mean(1)
        scores.append(probs[:, 1].cpu().numpy())
    return np.concatenate(scores)


def group_loss_weights(groups, power=0.0):
    """Downweight long groups while preserving physical/simulation total mass."""
    if power == 0:
        return None
    if power not in {0.5, 1.0}:
        raise ValueError("Unsupported group loss power")
    groups = np.asarray(groups)
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = counts[inverse].astype(np.float64) ** -power
    real = np.char.startswith(groups, "WELL-")
    for mask in [real, ~real]:
        if mask.any():
            weights[mask] /= weights[mask].mean()
    return weights.astype(np.float32)


def summary_features(data, normalization, family="all", relative=False):
    """Exact summaries, optionally augmented with scale-relative shape measures.

    Recover physical means using training-only normalization. Ratios expose
    relative dispersion and change without discarding the absolute summaries.
    """
    c = data["windows"].shape[1]
    f = np.asarray(data["features"])
    cols = np.concatenate([np.arange(i * c, (i + 1) * c) for i in FAMILIES[family]])
    selected = np.asarray(f[:, cols], dtype=np.float32)
    if not relative:
        return selected
    mean, std, lo, hi, missing, delta, trend, _ = np.split(f, 8, axis=1)
    location = np.asarray(normalization["mean"])
    scale = np.asarray(normalization["std"])
    raw_mean = mean * scale + location
    raw_std = std * scale
    floor = 1e-6 * (np.abs(location) + scale)
    relative_std = raw_std / (np.abs(raw_mean) + floor)
    denominator = std + 1e-6
    extras = [
        relative_std,
        (hi - lo) / denominator,
        delta / denominator,
        trend / denominator,
    ]
    extras = [np.where(missing >= 1, 0, a) for a in extras]
    return np.concatenate([selected, *extras], axis=1).astype(np.float32)


def train_neural(name, xs, ys, seed, args, out, sample_weights=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(name, xs["train"], args).to(device)
    opt = torch.optim.AdamW(
        adamw_parameter_groups(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.95)
    )
    counts = np.bincount(ys["train"], minlength=2)
    exponent = {"none": 0, "sqrt": 0.5, "full": 1}[args.balance]
    weights = counts.astype(float) ** -exponent
    weights /= weights.mean()
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    ytrain = torch.tensor(ys["train"], dtype=torch.long, device=device)
    row_weights = (
        torch.as_tensor(sample_weights, device=device)
        if sample_weights is not None
        else None
    )
    rng = np.random.default_rng(seed)
    best, stale, history, state = -np.inf, 0, [], None
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(ytrain))
        losses = []
        for j in range(0, len(order), args.batch_size):
            idx = order[j : j + args.batch_size]
            xb = torch.as_tensor(np.array(xs["train"][idx]), device=device)
            yb = ytrain[torch.as_tensor(idx, device=device)]
            wb = row_weights[idx] if row_weights is not None else None
            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(xb)
                if logits.ndim == 3:
                    if wb is not None:
                        wb = wb[:, None].expand(-1, logits.shape[1]).reshape(-1)
                    yb = yb[:, None].expand(-1, logits.shape[1]).reshape(-1)
                    logits = logits.reshape(-1, 2)
                if wb is None:
                    loss = F.cross_entropy(logits.float(), yb, weight=weights)
                else:
                    loss = (
                        F.cross_entropy(
                            logits.float(), yb, weight=weights, reduction="none"
                        )
                        * wb
                    ).sum()
                    loss = loss / (weights[yb] * wb).sum()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss: {name}/{seed}/{epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        score = predict(model, xs["val"], args.batch_size, device)
        val = metrics(ys["val"], score, 0.5, np.arange(len(score)))
        auc = val["auc"]
        history.append(
            {"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_auc": auc}
        )
        print(
            f"{name}/{seed} epoch={epoch + 1} "
            f"loss={np.mean(losses):.5f} val_auc={auc:.5f}",
            flush=True,
        )
        if auc > best + 1e-5:
            best, stale = auc, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(state)
    seconds = time.perf_counter() - start
    torch.save(
        {"state_dict": state, "args": vars(args), "history": history},
        out / f"{name}_{seed}.pt",
    )
    scores = {
        split: predict(model, x, args.batch_size, device)
        for split, x in xs.items()
        if split != "train"
    }
    params = sum(p.numel() for p in model.parameters())
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scores, seconds, params, history


def run(args):
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("high")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed experiment: {out}")
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    sources = [
        Path(__file__),
        Path("utils/three_w.py"),
        Path("trea/models/triple_attention.py"),
    ]
    (out / "source_hashes.json").write_text(
        json.dumps(
            {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            indent=2,
        )
    )
    data, manifests = load_data(args)
    (out / "protocol.json").write_text(json.dumps(manifests, indent=2))
    if args.prepare_only:
        return
    results = []
    models = args.models.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    for name in models:
        outputs, cost, parameter_counts = [], [], []
        for seed in seeds:
            temporal = name in {"temporal", "direct_stat", "stats_only"}
            xs = {}
            for split, d in data.items():
                if temporal:
                    xs[split] = d["windows"]
                elif name == "xgb_naive":
                    xs[split] = extract_naive_features(d["windows"])
                else:
                    xs[split] = summary_features(
                        d,
                        manifests["train"]["normalization"],
                        args.feature_family,
                        getattr(args, "relative_stats", False),
                    )
            ys = {split: np.asarray(d["labels"]) for split, d in data.items()}
            # Identical cap and sampled row IDs for every model at each seed.
            idx = np.random.default_rng(seed).choice(
                len(ys["train"]),
                min(args.train_cap or len(ys["train"]), len(ys["train"])),
                replace=False,
            )
            xs["train"], ys["train"] = xs["train"][idx], ys["train"][idx]
            np.save(out / f"train_indices_{seed}.npy", idx)
            sample_weights = group_loss_weights(
                data["train"]["group_ids"][idx], getattr(args, "group_loss_power", 0.0)
            )
            classical = name in {"xgb", "xgb_naive", "rf"}
            if not classical and not temporal and args.preprocess != "raw":
                scaler = (
                    StandardScaler()
                    if args.preprocess == "standard"
                    else QuantileTransformer(
                        n_quantiles=1000,
                        output_distribution="normal",
                        subsample=100000,
                        random_state=seed,
                    )
                )
                scaler.fit(xs["train"])
                xs = {
                    split: scaler.transform(x).astype(np.float32)
                    for split, x in xs.items()
                }
                import joblib

                joblib.dump(scaler, out / f"{name}_{seed}_preprocess.joblib")
            history = []
            if classical:
                negatives, positives = np.bincount(ys["train"], minlength=2)
                if name == "rf":
                    model = RandomForestClassifier(
                        n_estimators=500,
                        n_jobs=args.threads,
                        random_state=seed,
                        class_weight="balanced_subsample",
                    )
                else:
                    model = XGBClassifier(
                        n_estimators=500,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        tree_method="hist",
                        n_jobs=args.threads,
                        scale_pos_weight=negatives / max(positives, 1),
                        random_state=seed,
                    )
                start = time.perf_counter()
                model.fit(xs["train"], ys["train"], sample_weight=sample_weights)
                seconds = time.perf_counter() - start
                scores = {
                    split: model.predict_proba(x)[:, 1]
                    for split, x in xs.items()
                    if split != "train"
                }
                params = 0
                import joblib

                joblib.dump(model, out / f"{name}_{seed}.joblib")
            else:
                scores, seconds, params, history = train_neural(
                    name, xs, ys, seed, args, out, sample_weights
                )
            threshold = choose_threshold(ys["val"], scores["val"])
            row = {
                "model": name,
                "seed": seed,
                "train_seconds": seconds,
                "num_params": params,
                "history": history,
                "splits": {
                    s: metrics(ys[s], p, threshold, data[s]["file_ids"])
                    for s, p in scores.items()
                },
            }
            results.append(row)
            outputs.append(scores)
            cost.append(seconds)
            parameter_counts.append(params)
            np.savez_compressed(out / f"{name}_{seed}_predictions.npz", **scores)
            (out / "results.json").write_text(json.dumps(results, indent=2))
            print(
                json.dumps({k: v for k, v in row.items() if k != "history"}), flush=True
            )
            del xs
        if len(outputs) > 1:
            scores = {s: np.mean([o[s] for o in outputs], axis=0) for s in outputs[0]}
            threshold = choose_threshold(ys["val"], scores["val"])
            row = {
                "model": name + "_ensemble",
                "seeds": seeds,
                "train_seconds": sum(cost),
                "num_params": sum(parameter_counts),
                "splits": {
                    s: metrics(ys[s], p, threshold, data[s]["file_ids"])
                    for s, p in scores.items()
                },
            }
            results.append(row)
            print(json.dumps(row), flush=True)
        (out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    run(parse_args())
