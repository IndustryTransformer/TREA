"""Bounded numeric-basis rerun: NOx, diabetes, balance_scale; three seeds.

Run: .venv/bin/python -m scripts.basic_numeric_experiments
Writes a locked protocol and separate per-job outputs. No test-based tuning.
"""

import json
import os
import subprocess
import sys

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/basic_numeric_experiments"
VARIANTS = [
    ("identity", "additive"),
    ("ple", "additive"),
    ("periodic", "additive"),
    ("dual", "additive"),
    ("identity", "hadamard"),
]


def run_job(job):
    dataset, basis, fusion = job
    name = f"{dataset}_{basis}_{fusion}"
    destination = OUT / f"{name}.json"
    if dataset == "nox":
        command = [
            sys.executable,
            "-m",
            "scripts.nox_encoding_ablation",
            "--arms",
            "identity_query",
            "--value-encoder",
            basis,
            "--fusion",
            fusion,
            "--threads",
            "2",
            "--output",
            str(destination),
        ]
        with (OUT / f"{name}.log").open("w") as log:
            subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True
            )
    else:
        # Import in the worker after setting thread limits in the parent.
        import torch

        from scripts.tabular_vs_trees import _score_dataset, load_corpus

        torch.set_num_threads(2)
        _, manifest = load_corpus(["classification", "regression"])
        entry = next(m for m in manifest if m["name"] == dataset)
        cfg = dict(
            seeds=3,
            epochs=100,
            patience=16,
            head="mse",
            bin_spacing="uniform",
            symlog=False,
            value_encoder=basis,
            fusion=fusion,
            label_smoothing=0.0,
            logit_adjustment_tau=1.0,
            warmup_frac=0.05,
            lr=8e-4,
            dropout=0.2,
            n_layers=3,  # Preserve the historical numeric-basis experiment.
            first_prenormalization=True,
            ensemble_k=0,
            tree_jobs=2,
            classification_margin=0.01,
            regression_margin=0.02,
        )
        _, result = _score_dataset((0, entry, cfg))
        destination.write_text(json.dumps(result, indent=2))
        if "__fail__" in result or "__skip__" in result:
            raise RuntimeError(result)
    return name


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    for key, value in {
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "MPLCONFIGDIR": "/tmp/trea-mpl",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }.items():
        os.environ[key] = value
    jobs = [
        (dataset, basis, fusion)
        for basis, fusion in VARIANTS
        for dataset in ("nox", "diabetes", "balance_scale")
    ]
    (OUT / "protocol.json").write_text(
        json.dumps(
            {
                "jobs": jobs,
                "workers": 4,
                "nox": "Same fixed split and seeds 10/11/12 as nox_encoding_ablation; "
                "100 epochs, patience 12, original identity-query first block.",
                "static": "shared-fit-v2; seeds 0/1/2; 100 epochs, patience 16",
                "primary_comparisons": (
                    "Each variant versus identity/additive within dataset"
                ),
                "interpretation": (
                    "Exploratory previously inspected datasets; no tuning. "
                    "NOx meaningful gain threshold 2% historical RMSE; "
                    "static margins 2% RMSE / 0.01 macro-F1. "
                    "Future NOx is a separate drift check. No pretraining."
                ),
            },
            indent=2,
        )
    )
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(run_job, job): job for job in jobs}
        failures = []
        for future in as_completed(futures):
            try:
                print("DONE", future.result(), flush=True)
            except Exception as exc:
                failures.append({"job": futures[future], "error": str(exc)})
                print("FAILED", failures[-1], flush=True)
        (OUT / "failures.json").write_text(json.dumps(failures, indent=2))
        if failures:
            raise RuntimeError(failures)


if __name__ == "__main__":
    main()
