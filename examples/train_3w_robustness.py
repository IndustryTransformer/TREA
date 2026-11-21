"""Execute predeclared robustness training folds; never load outer-fold signals."""

import argparse
import hashlib
import json
import os
import subprocess
import sys

from pathlib import Path


def run(folds):
    protocol_root = Path("docs/results/3w_grouped_robustness_v1")
    plan = json.loads((protocol_root / "plan.json").read_text())
    recipe = plan["shared_recipe"]
    root = Path("logs/benchmark_3w_binary/robustness_v1")
    root.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        if fold not in range(4):
            raise ValueError("Unknown frozen fold")
        directory = root / f"fold_{fold}"
        # No restart from a lock/state file: callers must inspect live processes
        # and completed artifacts explicitly before deciding on recovery.
        directory.mkdir(exist_ok=False)
        sources = [
            "examples/benchmark_3w_grouped.py",
            "examples/benchmark_3w_context.py",
            "examples/three_w_causal_features.py",
            "utils/three_w.py",
            "trea/models/single_row.py",
            str(protocol_root / "plan.json"),
            str(protocol_root / f"fold_{fold}.json"),
        ]
        state = {
            "fold": fold,
            "worker_pid": os.getpid(),
            "state": "training",
            "source_sha256": {
                f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in sources
            },
            "jobs": [],
        }
        status = directory / "training_state.json"
        status.write_text(json.dumps(state, indent=2))
        for view, module, models in [
            ("window", "examples.benchmark_3w_grouped", "xgb,single_row"),
            ("causal", "examples.benchmark_3w_context", "xgb,single_row,tabm_ple"),
        ]:
            command = [
                sys.executable,
                "-u",
                "-m",
                module,
                "--models",
                models,
                "--seeds",
                "42,43,44",
                "--group-manifest",
                str(protocol_root / f"fold_{fold}.json"),
                "--value-policy",
                recipe["value_policy"],
                "--preprocess",
                "quantile",
                "--group-loss-power",
                str(recipe["group_loss_power"]),
                "--train-cap",
                str(recipe["train_cap"]),
                "--width",
                str(recipe["width"]),
                "--layers",
                str(recipe["layers"]),
                "--lr",
                str(recipe["lr"]),
                "--epochs",
                str(recipe["epochs"]),
                "--patience",
                str(recipe["patience"]),
                "--row-dropout",
                "0",
                "--output-dir",
                str(directory / view),
            ]
            job = {"view": view, "command": command, "state": "running"}
            state["jobs"].append(job)
            with (directory / f"{view}.log").open("w") as log:
                process = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT
                )
                job["pid"] = process.pid
                status.write_text(json.dumps(state, indent=2))
                print(f"Fold {fold} {view}: PID {process.pid}", flush=True)
                returncode = process.wait()
            job["returncode"] = returncode
            job["state"] = "complete" if returncode == 0 else "failed"
            state["state"] = "training" if returncode == 0 else "failed"
            status.write_text(json.dumps(state, indent=2))
            if returncode:
                raise RuntimeError(f"Fold {fold}/{view} failed; inspect its log")
        state["state"] = "fits_complete_outer_unread"
        status.write_text(json.dumps(state, indent=2))
        print(f"Fold {fold}: all fits complete, outer groups unread", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", required=True)
    args = parser.parse_args()
    run([int(value) for value in args.folds.split(",")])
