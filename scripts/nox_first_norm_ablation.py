"""Run the fixed 3-encoding x 2-normalization NOx experiment, three seeds each."""

import json
import os
import subprocess
import sys

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/nox_first_norm"


def run(fusion, skip):
    name = f"{fusion}_{'skip' if skip else 'keep'}"
    command = [
        sys.executable,
        "-m",
        "scripts.nox_encoding_ablation",
        "--fusion",
        fusion,
        "--arms",
        "identity_query",
        "--threads",
        "2",
        "--output",
        str(OUT / f"{name}.json"),
    ]
    if skip:
        command.append("--skip-first-attention-norm")
    env = dict(
        os.environ,
        MPLCONFIGDIR="/tmp/trea-mpl",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    with (OUT / f"{name}.log").open("w") as log:
        subprocess.run(
            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    return name


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    jobs = [
        (f, skip)
        for f in ("additive", "product", "product_sum")
        for skip in (False, True)
    ]
    (OUT / "protocol.json").write_text(
        json.dumps(
            {
                "jobs": jobs,
                "seeds": [10, 11, 12],
                "workers": 3,
                "threads_per_job": 2,
                "split_seed": 42,
                "fit_rows": 2219,
                "epochs": 100,
                "patience": 12,
                "change": "Bypass first-block norm1 on Q/K/V only; retain norm2, "
                "per-head Q/K normalization, identity-query residual, later norms.",
                "primary": "Historical random-holdout RMSE; compare skip versus keep "
                "within each encoding, >=2% mean improvement and >=2/3 seeds.",
                "secondary": (
                    "Future-year frozen-model drift check, not a continuation gate."
                ),
                "scope": (
                    "Exploratory on previously inspected NOx; no tuning or pretraining."
                ),
            },
            indent=2,
        )
    )
    failures = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(run, *job): job for job in jobs}
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
