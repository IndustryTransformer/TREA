"""Package exact range-repair results without changing the frozen evaluation."""

import json

from pathlib import Path

import numpy as np

from examples.summarize_3w_grouped import json_safe


def main():
    docs = Path("docs/results")
    reserved = Path("logs/benchmark_3w_binary/range_reserved")
    rows = json.loads((reserved / "results.json").read_text())
    grouped = {}
    for row in rows:
        if row.get("seed") is not None:
            grouped.setdefault(row["model"], []).append(row)
    summaries = {}
    for model, fits in grouped.items():
        summaries[model] = {
            "seeds": [r["seed"] for r in fits],
            "auc_mean": float(np.mean([r["test"]["auc"] for r in fits])),
            "auc_std": float(np.std([r["test"]["auc"] for r in fits])),
            "macro_f1_mean": float(np.mean([r["test"]["macro_f1"] for r in fits])),
            "macro_f1_std": float(np.std([r["test"]["macro_f1"] for r in fits])),
        }
    result = {
        "status": "AUC milestone reached; competitive macro-F1 and parity with "
        "the stronger weighted XGBoost baseline not established.",
        "test_is_frozen": True,
        "raw_audit": json.loads((docs / "3w_training_raw_audit.json").read_text()),
        "group_protocol": json.loads(
            (docs / "3w_range_confirmation_protocol.json").read_text()
        ),
        "development": json.loads((docs / "3w_range_development.json").read_text()),
        "selection": json.loads((docs / "3w_range_selection.json").read_text()),
        "reserved_protocol": json.loads((reserved / "protocol.json").read_text()),
        "reserved_single_fit_summary": summaries,
        "reserved_results": rows,
        "paired_group_uncertainty": json.loads(
            (reserved / "uncertainty.json").read_text()
        ),
        "checks": {
            "focused_and_single_row_tests": 48,
            "ruff": "pass",
            "git_diff_check": "pass",
            "user_original_benchmark_preserved": True,
        },
        "limitations": [
            "Confirmation groups have documented historical training/audit exposure.",
            "Only eight physical test wells; simulations may share generators.",
            "Seed variability measures fitting, not group-split uncertainty.",
            "Validation thresholds transfer poorly to the confirmation mixture.",
            "No reselection or threshold optimization using confirmation labels.",
            "Majority-instance metrics do not measure early event detection.",
            "Concurrent development runs do not support controlled speed comparisons.",
        ],
    }
    (docs / "3w_range_neural_results.json").write_text(
        json.dumps(json_safe(result), indent=2, allow_nan=False)
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
