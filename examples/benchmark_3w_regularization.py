"""One fixed development contrast: stronger AdamW weight decay, unchanged inputs."""

import json

from pathlib import Path
from types import SimpleNamespace

from examples.benchmark_3w_context import run
from examples.benchmark_3w_initial_reference import validate_args
from examples.freeze_3w_robustness import file_digest


def contrast_config(original, output_dir):
    args = SimpleNamespace(**original)
    validate_args(args)
    if args.weight_decay != 0.0003 or args.seeds != "42,43,44":
        raise ValueError("Unexpected baseline recipe")
    args.weight_decay = 0.03
    args.models = "single_row,tabm_ple"
    args.output_dir = str(output_dir)
    return args


def main():
    base = Path("logs/benchmark_3w_binary/causal_context_trea/config.json")
    out = Path("logs/benchmark_3w_binary/causal_regularization_v1")
    args = contrast_config(json.loads(base.read_text()), out)
    if out.exists():
        raise FileExistsError("Do not overwrite the regularization contrast")
    audit_path = Path("logs/benchmark_3w_binary/training_group_audit_v1/results.json")
    audit = json.loads(audit_path.read_text())
    if {(f["model"], f["seed"]) for f in audit["fits"]} != {
        (name, seed) for name in ["single_row", "tabm_ple"] for seed in [42, 43, 44]
    }:
        raise ValueError("Complete all-seed diagnostic required before this contrast")
    for path, digest in audit["artifact_sha256"].items():
        if file_digest(path) != digest:
            raise ValueError(f"Audited source/checkpoint changed: {path}")
    out.mkdir()
    sources = [
        Path(__file__),
        base,
        audit_path,
        Path(args.group_manifest),
        Path("examples/benchmark_3w_context.py"),
        Path("examples/benchmark_3w_grouped.py"),
        Path("examples/three_w_causal_features.py"),
        Path("trea/models/single_row.py"),
        Path("utils/three_w.py"),
    ]
    plan = {
        "scope": "Development only; no consumed held-out evaluation.",
        "hypothesis": (
            "Physical training AUC is high and physical data already dominates "
            "weighted loss. Test regularization, not more group emphasis."
        ),
        "contrast": "AdamW decay 0.0003 -> 0.03; no other recipe change.",
        "selection": "Existing internal-validation AUC early stopping unchanged.",
        "reporting": "All seeds 42/43/44 and explicit probability ensembles.",
        "expansion": "No automatic decay grid, outer refit, or success claim.",
        "reference": "https://arxiv.org/abs/1911.08731",
    }
    (out / "plan.json").write_text(json.dumps(plan, indent=2))
    (out / "source_hashes.json").write_text(
        json.dumps({str(path): file_digest(path) for path in sources}, indent=2)
    )
    run(args)


if __name__ == "__main__":
    main()
