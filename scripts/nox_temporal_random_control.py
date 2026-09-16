"""Additional low-label control: identical adaptation modes, untrained SSL encoder."""

import argparse
import copy
import json

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.nox_encoding_ablation import FEATURES, ROOT
from scripts.nox_reference_models import save_json
from scripts.nox_temporal_pretraining import ARCH, SEEDS, launch
from trea.models.single_row import MaskedTabularEncoder
from trea.utils.single_row_data import SingleRowConfig


def run(args):
    manifest = args.output / "random_encoder_control.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    save_json(
        manifest,
        dict(
            scope="Additional diagnostic after initial pretrained low-label results",
            fractions=[0.01, 0.02, 0.05],
            init_seeds=SEEDS,
            encoder_seed=42,
            control="Untrained SSL encoder; identical head seeds and adaptation modes",
            modes="Frozen probe lr .001 vs finetune lr .00008; select on validation",
        ),
    )
    arrays = np.load(args.output / "data.npz")
    historical = arrays["historical"]
    x = arrays["x"][historical[: int(len(historical) * 0.9)]]
    config = SingleRowConfig.generate(pd.DataFrame(x, columns=FEATURES))
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = MaskedTabularEncoder(config, **ARCH)
    torch.save(
        copy.deepcopy(model.tabular_encoder.state_dict()),
        args.output / "encoder_random_encoder.pt",
    )
    jobs = [("random_encoder", f, s) for f in [0.01, 0.02, 0.05] for s in SEEDS]
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda job: launch(args, job), jobs))
    target = arrays["y"][arrays["future"]]
    summary = {}
    for fraction in [0.01, 0.02, 0.05]:
        paths = [args.output / f"random_encoder_{fraction}_{s}.json" for s in SEEDS]
        records = [json.loads(p.read_text()) for p in paths]
        predictions = [
            np.load(p.with_suffix(".predictions.npz"))["future"] for p in paths
        ]
        ensemble = np.mean(predictions, axis=0)
        np.save(args.output / f"ensemble_random_encoder_{fraction}.npy", ensemble)
        scores = [r["rmse"]["future"] for r in records]
        summary[str(fraction)] = dict(
            mean=float(np.mean(scores)),
            std=float(np.std(scores, ddof=1)),
            ensemble=float(np.sqrt(np.mean((target - ensemble) ** 2))),
            selected_modes=[r["selected_mode"] for r in records],
        )
    save_json(args.output / "random_encoder_summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "logs/nox_temporal_pretraining"
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--ssl-epochs", type=int, default=60)
    run(parser.parse_args())
