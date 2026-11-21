"""Collect exact grouped 3W results and correctly combine split seed runs."""

import hashlib
import importlib.metadata
import json
import math
import subprocess

from pathlib import Path

import numpy as np

from examples.benchmark_3w_grouped import choose_threshold, metrics
from examples.evaluate_3w_reserved import grouped_bootstrap


def json_safe(value):
    """Use JSON null for undefined metrics (e.g. one-class subset AUC)."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    root = Path("logs/benchmark_3w_binary")
    cache = root / "grouped_cache/fe77c39eec332d22"
    y = np.load(cache / "val_labels.npy")
    files = np.load(cache / "val_file_ids.npy")
    specs = {
        "xgb_engineered": ("xgb", ["grouped_baselines"]),
        "xgb_naive": ("xgb_naive", ["grouped_baselines"]),
        "rf": ("rf", ["grouped_baselines"]),
        "temporal": (
            "temporal",
            ["grouped_temporal_probe", "grouped_temporal_confirm"],
        ),
        "temporal_shuffled": ("temporal", ["grouped_shuffle"]),
        "direct_stat": (
            "direct_stat",
            ["grouped_temporal_probe", "grouped_temporal_confirm"],
        ),
        "stats_only": (
            "stats_only",
            ["grouped_temporal_probe", "grouped_temporal_confirm"],
        ),
        "single_row_variation": (
            "single_row",
            ["grouped_variation_probe", "grouped_single_row_confirm"],
        ),
        "tabm_quantile": ("tabm", ["grouped_tabm_confirm"]),
        "tabm_ple_quantile": ("tabm_ple", ["grouped_tabm_confirm"]),
        "tabm_ple_full_balance": ("tabm_ple", ["grouped_balanced_confirm"]),
    }
    for family in [
        "level",
        "variation",
        "missing",
        "direction",
        "no_direction",
        "no_missing",
    ]:
        specs[f"xgb_{family}"] = ("xgb", [f"grouped_ablation_{family}"])
    summary = {}
    for label, (model, directories) in specs.items():
        rows, scores = {}, {}
        for directory in directories:
            for row in json.loads((root / directory / "results.json").read_text()):
                if row["model"] != model or "seed" not in row:
                    continue
                seed = row["seed"]
                if seed in rows:
                    raise ValueError(f"Duplicate seed {label}/{seed}")
                rows[seed] = row
                scores[seed] = np.load(
                    root / directory / f"{model}_{seed}_predictions.npz"
                )["val"]
        if sorted(rows) != [42, 43, 44]:
            raise ValueError(f"Missing seeds: {label}")
        mean_scores = np.mean([scores[s] for s in sorted(scores)], axis=0)
        threshold = choose_threshold(y, mean_scores)
        entry = {
            "runs": directories,
            "single_fits": rows,
            "ensemble": metrics(y, mean_scores, threshold, files),
        }
        for metric in ["auc", "macro_f1", "macro_f1_at_05"]:
            values = [rows[s]["splits"]["val"][metric] for s in sorted(rows)]
            entry[f"single_fit_{metric}"] = {
                "mean": float(np.mean(values)),
                "std_population": float(np.std(values)),
            }
        summary[label] = entry
    raw = {}
    for p in sorted(root.glob("grouped_*/results.json")):
        if "reserved" not in p.parent.name:
            raw[p.parent.name] = {
                "config": json.loads((p.parent / "config.json").read_text()),
                "results": json.loads(p.read_text()),
            }
    test_dir = root / "grouped_reserved_test"
    reserved = json.loads((test_dir / "results.json").read_text())
    variation = json.loads(
        (root / "grouped_reserved_variation/results.json").read_text()
    )
    uncertainty = json.loads((test_dir / "uncertainty.json").read_text())
    ytest = np.load(cache / "test_labels.npy")
    groups = np.load(cache / "test_group_ids.npy")
    xgb_variation = np.load(
        root / "grouped_reserved_variation/xgb_ensemble_predictions.npz"
    )["test"]
    for model in ["single_row", "tabm_ple"]:
        score = np.load(test_dir / f"{model}_ensemble_predictions.npz")["test"]
        uncertainty[f"{model}_vs_xgb_variation"] = grouped_bootstrap(
            ytest, score, xgb_variation, groups
        )
    source_files = [
        "examples/benchmark_3w_grouped.py",
        "examples/evaluate_3w_reserved.py",
        "examples/benchmark_3w_binary.py",
        "utils/three_w.py",
        "trea/models/triple_attention.py",
        "trea/models/single_row.py",
    ]
    sources = {
        f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in source_files
    }
    versions = {
        p: importlib.metadata.version(p)
        for p in [
            "torch",
            "numpy",
            "scikit-learn",
            "xgboost",
            "tabm",
            "rtdl-num-embeddings",
        ]
    }
    payload = {
        "protocol": json.loads((test_dir / "protocol.json").read_text()),
        "selection": json.loads(
            Path("docs/results/3w_grouped_selection.json").read_text()
        ),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "uncommitted_source_sha256": sources,
        "versions": versions,
        "validation_summary": summary,
        "reserved_test": reserved,
        "reserved_xgb_variation": variation,
        "paired_group_uncertainty": uncertainty,
        "all_development_experiments": raw,
    }
    out = Path("docs/results/3w_grouped_neural_results.json")
    out.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False))
    print(f"Saved {out}")
    for name, result in summary.items():
        print(name, result["single_fit_auc"], result["ensemble"]["auc"])
    print(json.dumps(uncertainty, indent=2))


if __name__ == "__main__":
    main()
