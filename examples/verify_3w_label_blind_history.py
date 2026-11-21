"""Verify training/validation history equivalence; never load outer groups."""

import argparse
import hashlib
import json

from pathlib import Path
from types import SimpleNamespace

from examples.benchmark_3w_grouped import load_data
from examples.three_w_label_blind_history import file_inventory, reconstruct_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", default="0,1,2,3")
    args = parser.parse_args()
    root = Path("logs/benchmark_3w_binary/robustness_v1")
    for fold in [int(i) for i in args.folds.split(",")]:
        config_path = root / f"fold_{fold}/window/config.json"
        config = SimpleNamespace(**json.loads(config_path.read_text()))
        config.evaluate_test = False
        data, manifests = load_data(config)
        inventory = file_inventory(config)
        reports = []
        for split in ["train", "val"]:
            features, report = reconstruct_split(
                data[split], manifests[split], inventory
            )
            reports.append(report)
            del features
            print(fold, json.dumps(report), flush=True)
        result = {
            "fold": fold,
            "passed": True,
            "reports": reports,
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "reconstruction_source_sha256": hashlib.sha256(
                Path("examples/three_w_label_blind_history.py").read_bytes()
            ).hexdigest(),
            "policy": "Fixed burn-in 37 physical/0 simulation windows; annotations "
            "select scored rows only after history construction.",
        }
        (root / f"fold_{fold}/history_equivalence.json").write_text(
            json.dumps(result, indent=2)
        )


if __name__ == "__main__":
    main()
