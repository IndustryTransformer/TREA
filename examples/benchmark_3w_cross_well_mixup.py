"""Fixed development-only cross-well augmentation screen; frozen tests untouched."""

import json

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from examples import benchmark_3w_context as context
from examples.benchmark_3w_grouped import load_data
from examples.benchmark_3w_initial_reference import validate_args
from examples.freeze_3w_robustness import file_digest
from examples.three_w_cross_well_mixup import train_mixup


def main():
    baseline = Path("logs/benchmark_3w_binary/causal_context_trea/config.json")
    args = SimpleNamespace(**json.loads(baseline.read_text()))
    validate_args(args)
    args.models = "tabm_ple,single_row"
    args.mixup_alpha = 0.2
    args.output_dir = "logs/benchmark_3w_binary/cross_well_mixup_v1"
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError("Never overwrite an augmentation experiment")
    frozen = json.loads(Path("docs/results/3w_robustness_selection.json").read_text())
    for path, digest in frozen["artifact_sha256"].items():
        if file_digest(path) != digest:
            raise ValueError(f"Frozen artifact changed: {path}")
    out.mkdir()
    plan = {
        "scope": "Development only; no new independent confirmation claim.",
        "intervention": "Cross-group same-source mixup, fixed Beta(0.2,0.2).",
        "pairing": "Other-group TRAINING rows; never cross physical/simulation.",
        "features": "Same causal summaries; train-fitted quantile coordinates.",
        "targets": "Interpolated labels with original class/group loss weights.",
        "unchanged": "Architecture, LR, decay, sampling cap, seeds, early stopping.",
        "caution": "Mixed summaries need not describe a realizable physical signal.",
        "reference": "https://arxiv.org/abs/1710.09412",
    }
    (out / "plan.json").write_text(json.dumps(plan, indent=2))
    sources = [
        Path(__file__),
        baseline,
        Path(args.group_manifest),
        Path("examples/three_w_cross_well_mixup.py"),
        Path("examples/benchmark_3w_grouped.py"),
        Path("examples/benchmark_3w_context.py"),
        Path("examples/three_w_causal_features.py"),
        Path("trea/models/single_row.py"),
        Path("trea/training/single_row.py"),
        Path("utils/three_w.py"),
    ]
    (out / "source_hashes.json").write_text(
        json.dumps({str(path): file_digest(path) for path in sources}, indent=2)
    )
    data, _ = load_data(args)
    groups, labels = data["train"]["group_ids"], data["train"]["labels"]
    del data
    original = context.train_neural

    def augmented(name, xs, ys, seed, config, directory, sample_weights):
        indices = np.load(directory / f"train_indices_{seed}.npy")
        np.testing.assert_array_equal(labels[indices], ys["train"])
        return train_mixup(
            name, xs, ys, seed, config, directory, sample_weights, groups[indices]
        )

    # Explicit process-local injection; the existing source stays immutable.
    context.train_neural = augmented
    try:
        context.run(args)
    finally:
        context.train_neural = original


if __name__ == "__main__":
    main()
