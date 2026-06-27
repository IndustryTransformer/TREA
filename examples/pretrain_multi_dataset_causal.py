"""Causal multi-dataset pretraining example for TREA-C.

This script demonstrates pretraining across datasets with different numbers of
numeric/categorical columns. Batches are padded to a unified schema and trained
with a causal next-row objective.

Usage:
    uv run python examples/pretrain_multi_dataset_causal.py
"""

import argparse
import sys


sys.path.insert(0, ".")

import pytorch_lightning as pl
import torch

from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset

from trea.models import MultiDatasetModel
from utils.multi_dataset_pretrain import DatasetSource, MultiDatasetPretrainDataModule
from utils.three_w import ThreeWDataset


class ToyVariableDataset(Dataset):
    """Synthetic dataset with variable schema and cross-feature interactions."""

    def __init__(
        self,
        num_samples: int,
        seq_len: int,
        n_numeric: int,
        n_categorical: int,
        cat_cardinalities: list[int] | None = None,
        missing_ratio: float = 0.05,
        seed: int = 42,
    ):
        gen = torch.Generator().manual_seed(seed)
        self.seq_len = seq_len
        self.n_numeric = n_numeric
        self.n_categorical = n_categorical
        self.cat_cardinalities = cat_cardinalities or [8] * n_categorical

        # Build multi-factor latent state so causal pretraining must model structure.
        z = torch.zeros(num_samples, seq_len)
        z2 = torch.zeros(num_samples, seq_len)
        eps = torch.randn(num_samples, seq_len, generator=gen) * 0.25
        eps2 = torch.randn(num_samples, seq_len, generator=gen) * 0.20
        for t in range(1, seq_len):
            z[:, t] = 0.92 * z[:, t - 1] + eps[:, t]
            z2[:, t] = 0.85 * z2[:, t - 1] + 0.25 * z[:, t - 1] + eps2[:, t]

        time = torch.linspace(0, 2 * torch.pi, seq_len)
        phase = torch.rand(num_samples, 1, generator=gen) * 2 * torch.pi
        freq = 1.0 + torch.rand(num_samples, 1, generator=gen) * 2.0
        seasonal = torch.sin(freq * time.unsqueeze(0) + phase) + 0.35 * torch.cos(
            2.0 * freq * time.unsqueeze(0) + 0.5 * phase
        )

        x_num = torch.empty(num_samples, n_numeric, seq_len)
        base_channels = []
        for i in range(n_numeric):
            w_latent = torch.randn(1, generator=gen).item() * 0.8 + 0.8
            w_seasonal = torch.randn(1, generator=gen).item() * 0.4
            lag = (i % 4) + 1
            lagged = torch.roll(z, shifts=lag, dims=1)
            lagged_cross = torch.roll(z2, shifts=((i + 1) % 5) + 1, dims=1)
            noise = torch.randn(num_samples, seq_len, generator=gen) * 0.08
            base = (
                w_latent * z
                + 0.4 * z2
                + 0.35 * lagged
                + 0.25 * lagged_cross
                + w_seasonal * seasonal
                + 0.1 * torch.tanh(z + 0.5 * z2)
                + noise
            )
            x_num[:, i, :] = base
            base_channels.append(base)

        # Explicit cross-column and lagged cross-column interactions.
        if n_numeric >= 3:
            for i in range(n_numeric):
                j = (i + 1) % n_numeric
                k = (i + 2) % n_numeric
                lag_i = torch.roll(base_channels[j], shifts=(i % 3) + 1, dims=1)
                lag_j = torch.roll(base_channels[k], shifts=((i + 1) % 4) + 1, dims=1)
                interaction = 0.22 * lag_i * torch.tanh(lag_j)
                additive = 0.18 * (base_channels[j] - 0.6 * lag_j)
                x_num[:, i, :] = x_num[:, i, :] + interaction + additive

        nan_mask = (
            torch.rand(num_samples, n_numeric, seq_len, generator=gen) < missing_ratio
        )
        x_num[nan_mask] = float("nan")
        self.x_num = x_num

        if n_categorical > 0:
            cat_channels = []
            for i, card in enumerate(self.cat_cardinalities):
                ref_a = x_num[:, i % n_numeric, :]
                ref_b = x_num[:, (i + 2) % n_numeric, :]
                ref_a = torch.nan_to_num(ref_a, nan=0.0)
                ref_b = torch.nan_to_num(ref_b, nan=0.0)
                cat_signal = (
                    0.5 * z
                    + 0.4 * z2
                    + 0.35 * seasonal
                    + 0.25 * torch.roll(ref_a * ref_b, shifts=(i % 5) + 1, dims=1)
                    + torch.randn(num_samples, seq_len, generator=gen) * 0.05
                )
                cat_signal = (cat_signal - cat_signal.mean(dim=1, keepdim=True)) / (
                    cat_signal.std(dim=1, keepdim=True) + 1e-6
                )
                scaled = ((cat_signal + 3.0) / 6.0) * (card - 1)
                cat_channels.append(torch.clamp(scaled, 0, card - 1).long())
            self.x_cat = torch.stack(cat_channels, dim=1)
        else:
            self.x_cat = None

    def __len__(self) -> int:
        return self.x_num.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = {
            "x_num": self.x_num[idx],
            # Unlabeled pretraining by default.
            "y": torch.tensor(-100, dtype=torch.long),
        }
        if self.x_cat is not None:
            sample["x_cat"] = self.x_cat[idx]
        return sample

    def get_feature_info(self) -> dict:
        return {
            "n_numeric": self.n_numeric,
            "n_categorical": self.n_categorical,
            "cat_cardinalities": self.cat_cardinalities,
            "sequence_length": self.seq_len,
            "num_classes": 2,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Causal multi-dataset pretraining with synthetic + optional 3W data."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/pretrain_multi_dataset_causal",
    )

    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--lambda-causal", type=float, default=1.0)
    parser.add_argument("--lambda-masked", type=float, default=0.3)
    parser.add_argument("--mask-ratio", type=float, default=0.15)

    parser.add_argument(
        "--synthetic-scale",
        type=float,
        default=1.0,
        help="Scale factor on synthetic sample counts.",
    )
    parser.add_argument(
        "--missing-ratio",
        type=float,
        default=0.05,
        help="NaN ratio for synthetic numeric features.",
    )

    parser.add_argument(
        "--include-three-w",
        action="store_true",
        help="Include 3W windows as a pretraining source.",
    )
    parser.add_argument(
        "--three-w-toolkit-path",
        type=str,
        default="/home/kailukowiak/Work/3W",
    )
    parser.add_argument(
        "--three-w-data-path",
        type=str,
        default="/home/kailukowiak/Work/3W/dataset",
    )
    parser.add_argument(
        "--three-w-max-files",
        type=int,
        default=400,
        help="Cap loaded 3W files for faster pretraining iteration.",
    )
    parser.add_argument(
        "--three-w-val-fraction",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--three-w-window-size",
        type=int,
        default=96,
    )
    parser.add_argument(
        "--three-w-stride",
        type=int,
        default=96,
    )

    return parser.parse_args()


def _scaled_count(base: int, scale: float) -> int:
    return max(64, int(base * scale))


def main():
    args = parse_args()
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")

    seq_len = args.sequence_length
    train_sources = [
        DatasetSource.from_dataset(
            name="industrial_a",
            dataset=ToyVariableDataset(
                num_samples=_scaled_count(2500, args.synthetic_scale),
                seq_len=seq_len,
                n_numeric=10,
                n_categorical=0,
                missing_ratio=args.missing_ratio,
                seed=1,
            ),
            num_classes=2,
        ),
        DatasetSource.from_dataset(
            name="building_b",
            dataset=ToyVariableDataset(
                num_samples=_scaled_count(2200, args.synthetic_scale),
                seq_len=seq_len,
                n_numeric=7,
                n_categorical=2,
                cat_cardinalities=[5, 12],
                missing_ratio=args.missing_ratio,
                seed=2,
            ),
            num_classes=3,
        ),
        DatasetSource.from_dataset(
            name="process_c",
            dataset=ToyVariableDataset(
                num_samples=_scaled_count(2000, args.synthetic_scale),
                seq_len=seq_len,
                n_numeric=14,
                n_categorical=1,
                cat_cardinalities=[20],
                missing_ratio=args.missing_ratio,
                seed=3,
            ),
            num_classes=4,
        ),
    ]

    if args.include_three_w:
        three_w_train = ThreeWDataset(
            toolkit_path=args.three_w_toolkit_path,
            data_path=args.three_w_data_path,
            window_size=args.three_w_window_size,
            stride=args.three_w_stride,
            split="train",
            val_fraction=args.three_w_val_fraction,
            max_files=args.three_w_max_files,
            seed=args.seed,
            augment=False,
        )
        train_sources.append(
            DatasetSource.from_dataset(
                name="3w",
                dataset=three_w_train,
                num_classes=10,
            )
        )
        print(f"Added 3W pretraining source with {len(three_w_train)} windows")

    print("Training sources:")
    for source in train_sources:
        print(
            f"  - {source.name}: {len(source.dataset)} samples "
            f"({source.n_numeric} num, {source.n_categorical} cat)"
        )

    dm = MultiDatasetPretrainDataModule(
        train_sources=train_sources,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sequence_length=seq_len,
    )

    schema = dm.model_schema_kwargs()
    print("Schema for pretraining:")
    print(f"  max_numeric_features: {schema['max_numeric_features']}")
    print(f"  max_categorical_features: {schema['max_categorical_features']}")
    print(f"  categorical_cardinalities: {schema['categorical_cardinalities']}")

    model = MultiDatasetModel(
        **schema,
        num_classes=16,  # Not used when lambda_supervised=0
        mode="pretrain",
        task="classification",
        patch_len=args.patch_len,
        stride=args.stride,
        d_model=args.d_model,
        n_head=args.n_head,
        num_layers=args.num_layers,
        max_sequence_length=seq_len,
        lr=args.lr,
        column_embedding_strategy="none",
        ssl_objectives={
            "causal_next_row": True,
            "masked_patch": True,
            "temporal_order": False,
            "contrastive": False,
        },
        mask_ratio=args.mask_ratio,
        lambda_causal=args.lambda_causal,
        lambda_masked=args.lambda_masked,
        lambda_temporal=0.0,
        lambda_contrastive=0.0,
        lambda_supervised=0.0,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        gradient_clip_val=1.0,
        log_every_n_steps=50,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        enable_progress_bar=True,
        callbacks=[
            ModelCheckpoint(
                dirpath=args.checkpoint_dir,
                filename="pretrain-{epoch:02d}",
                save_last=True,
                save_top_k=-1,
                every_n_epochs=1,
            )
        ],
    )

    print(f"Saving checkpoints to: {args.checkpoint_dir}")
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
