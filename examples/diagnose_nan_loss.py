"""Diagnose why we're getting NaN loss with 100% accuracy."""

import polars as pol
import torch

from train_patchtstnan_ultra_fast import W3SimpleDataset, extract_dataset_metadata

from treac.models import PatchTSTNan


def diagnose_nan_loss():
    print("Diagnosing NaN loss issue...")

    # Get metadata
    metadata = extract_dataset_metadata()

    # Create a small test dataset
    df_lazy = pol.scan_parquet("/home/ubuntu/TREA-C/data/W3/train.parquet")
    wells = df_lazy.select("well_name").unique().collect()["well_name"].to_list()

    test_dataset = W3SimpleDataset(
        parquet_path="/home/ubuntu/TREA-C/data/W3/train.parquet",
        well_names=wells[:2],  # Just 2 wells
        numeric_cols=metadata["numeric_cols"],
        state_to_label=metadata["state_to_label"],
        sequence_length=512,
        max_total_rows=1000,  # Small dataset for debugging
    )

    print(f"Test dataset size: {len(test_dataset)}")

    # Create model with same config as training
    model = PatchTSTNan(
        c_in=metadata["C_num"],
        seq_len=512,
        num_classes=metadata["unique_states"],
        patch_len=16,
        stride=8,
        d_model=64,
        n_head=4,
        num_layers=2,
        lr=1e-3,
        task="classification",
    )

    model.eval()

    # Test a few samples
    for i in range(min(5, len(test_dataset))):
        sample = test_dataset[i]
        x_num = sample["x_num"].unsqueeze(0)  # Add batch dimension
        x_cat = sample["x_cat"].unsqueeze(0)
        y = sample["y"].unsqueeze(0)

        print(f"\nSample {i}:")
        print(f"  Input shape: {x_num.shape}")
        print(f"  Input contains NaN: {torch.isnan(x_num).any()}")
        print(f"  Input contains Inf: {torch.isinf(x_num).any()}")
        print(f"  Input range: [{x_num.min():.3f}, {x_num.max():.3f}]")
        print(f"  True label: {y.item()}")

        with torch.no_grad():
            try:
                output = model(x_num, x_cat)
                print(f"  Output shape: {output.shape}")
                print(f"  Output contains NaN: {torch.isnan(output).any()}")
                print(f"  Output contains Inf: {torch.isinf(output).any()}")
                print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
                print(f"  Output raw values: {output.flatten()[:5]}")  # First 5 logits

                # Test loss calculation
                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(output, y)
                print(f"  Loss: {loss.item()}")
                print(f"  Loss is NaN: {torch.isnan(loss)}")

                # Test prediction
                pred = torch.argmax(output, dim=1)
                print(f"  Predicted: {pred.item()}")
                print(f"  Correct: {pred.item() == y.item()}")

                # Check if outputs are too extreme
                if torch.max(output) > 100:
                    print("  WARNING: Very large output values detected!")

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback

                traceback.print_exc()


if __name__ == "__main__":
    diagnose_nan_loss()
