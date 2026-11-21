"""Package development-only calibration, source and causal-context ablations."""

import json

from pathlib import Path

import numpy as np

from examples.summarize_3w_grouped import json_safe


def main():
    root = Path("logs/benchmark_3w_binary")
    selection = json.loads(Path("docs/results/3w_range_selection.json").read_text())
    baseline = json.loads(Path("docs/results/3w_range_development.json").read_text())
    summaries, runs = {}, {}
    for name in [
        "source_specialists_trea",
        "source_specialists_trees_tabm",
        "causal_context_trees_tabm",
        "causal_context_trea",
    ]:
        directory = root / name
        if not (directory / "results.json").exists():
            continue
        rows = json.loads((directory / "results.json").read_text())
        config = json.loads((directory / "config.json").read_text())
        protocol = json.loads((directory / "protocol.json").read_text())
        runs[name] = {"config": config, "protocol": protocol, "results": rows}
        for seed in [42, 43, 44]:
            sampled = directory / f"train_indices_{seed}.npy"
            if sampled.exists():
                original = Path(
                    selection["selected"]["single_row"]["sources"][str(seed)]
                )
                np.testing.assert_array_equal(
                    np.load(sampled), np.load(original / f"train_indices_{seed}.npy")
                )
        for model in sorted({r["model"] for r in rows if r.get("seed") is not None}):
            fits = [
                r for r in rows if r["model"] == model and r.get("seed") is not None
            ]
            ensemble = next(
                (r for r in rows if r["model"] == model + "_ensemble"), None
            )
            summaries[f"{name}/{model}"] = {
                "seeds_completed": [r["seed"] for r in fits],
                "single_fit_auc_mean": float(
                    np.mean([r["validation"]["auc"] for r in fits])
                ),
                "single_fit_auc_std": float(
                    np.std([r["validation"]["auc"] for r in fits])
                ),
                "single_fit_macro_f1_mean": float(
                    np.mean([r["validation"]["macro_f1"] for r in fits])
                ),
                "ensemble": ensemble,
            }
    result = {
        "scope": "Development diagnostics only. Consumed tests not reevaluated.",
        "calibration": json.loads(
            (root / "calibration_development/results.json").read_text()
        ),
        "summaries": summaries,
        "runs": runs,
        "matched_window_baselines": [
            e
            for e in baseline["experiments"]
            if e["seeds"] == [42, 43, 44]
            and e["recipe"]["group_loss_power"] == 0.5
            and e["recipe"]["feature_family"] == "all"
            and not e["recipe"]["relative_stats"]
            and e["recipe"].get("row_dropout", 0) == 0
        ],
        "sampled_window_indices_match_baseline": True,
        "next_study": json.loads(
            Path("docs/results/3w_grouped_robustness_v1/plan.json").read_text()
        ),
    }
    Path("docs/results/3w_development_followup.json").write_text(
        json.dumps(json_safe(result), indent=2, allow_nan=False)
    )
    for name, summary in summaries.items():
        ensemble = summary["ensemble"]
        print(
            name,
            summary["seeds_completed"],
            summary["single_fit_auc_mean"],
            ensemble["validation"] if ensemble else "ensemble pending",
            flush=True,
        )


if __name__ == "__main__":
    main()
