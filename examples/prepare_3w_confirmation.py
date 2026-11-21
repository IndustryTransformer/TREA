"""Freeze a physical-well-enriched confirmation partition without reading signals.

The earlier test is excluded. Confirmation groups come from the former training
partition: eight physical wells and 20% of simulation groups, using a fixed seed.
The historical use of these groups for fitting must remain disclosed.
"""

import argparse
import hashlib
import json

from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-cache", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError("Refusing to replace frozen group membership")
    root = Path(args.original_cache)
    manifests = {
        s: json.loads((root / f"{s}.json").read_text())
        for s in ["train", "val", "test"]
    }
    source = sorted(manifests["train"]["groups"])
    wells = [g for g in source if g.startswith("WELL-")]
    sims = [g for g in source if not g.startswith("WELL-")]
    rng = np.random.default_rng(20260918)
    reserved = sorted(
        rng.choice(wells, size=8, replace=False).tolist()
        + rng.choice(sims, size=round(len(sims) * 0.2), replace=False).tolist()
    )
    partitions = {
        "train": sorted(set(source) - set(reserved)),
        "val": manifests["val"]["groups"],
        "test": reserved,
        "excluded": manifests["test"]["groups"],
    }
    inventory = sorted(f for m in manifests.values() for f in m["files"])
    payload = {
        "protocol": "3w_range_confirmation_v1",
        "selection_seed": 20260918,
        "partitions": partitions,
        "file_inventory_sha256": hashlib.sha256(
            "\n".join(inventory).encode()
        ).hexdigest(),
        "counts": {
            s: {
                "groups": len(gs),
                "physical_wells": sum(g.startswith("WELL-") for g in gs),
            }
            for s, gs in partitions.items()
        },
        "historical_exposure": (
            "Confirmation groups were historically used for fitting and a "
            "training-only raw-scale audit. Four preliminary preserve_range "
            "fits completed before interruption, but their results were not "
            "read until after membership was frozen. Membership uses identifiers "
            "only. All current-stage models must be freshly initialized and "
            "exclude these groups from fitting and normalization. This is not "
            "an entirely untouched external dataset."
        ),
        "selection_uses": (
            "Group identifiers only; no signals, labels, predictions, "
            "or metrics used to select membership."
        ),
        "value_policy": (
            "preserve_range: abs(value)>1e12 or nonfinite becomes NaN; "
            "otherwise retain range"
        ),
        "freeze_rule": (
            "Do not change membership after inspecting results. Refit all "
            "normalization, feature selection, and models using only this "
            "training partition."
        ),
        "success": (
            "Neural AUC >=0.9474 with competitive macro-F1 and a contemporaneous "
            "XGB comparison; report independent fits and three-fit probability "
            "ensemble separately."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["counts"], indent=2))
    print(
        "Confirmation physical wells:", [g for g in reserved if g.startswith("WELL-")]
    )


if __name__ == "__main__":
    main()
