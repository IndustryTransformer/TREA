"""Diagnose W3 training issues: slow speed and NaN loss."""

import time

import polars as pol
import torch

from treac.models import PatchTSTNan


def analyze_data_loading():
    """Analyze data loading performance and characteristics."""
    print("=" * 60)
    print("DIAGNOSING DATA LOADING")
    print("=" * 60)

    print("Step 1: Loading parquet file...")
    start_time = time.time()

    # Load much smaller subset for testing
    df_lazy = pol.scan_parquet("/home/ubuntu/TREA-C/data/W3/train.parquet")
    df = df_lazy.head(100_000).collect()  # Only 100K rows for testing

    load_time = time.time() - start_time
    print(f"✅ Loaded {len(df):,} rows in {load_time:.2f}s")

    print("\nStep 2: Analyzing data structure...")
    print(f"Columns: {df.columns}")
    print(f"Shape: {df.shape}")

    # Check data quality
    print(f"Wells: {df['well_name'].n_unique()}")
    print(f"States: {df['state'].value_counts()}")

    # Check for obvious data quality issues
    numeric_cols = [col for col in df.columns if col not in ["state", "well_name"]]
    print(f"Numeric columns: {len(numeric_cols)}")

    # Check NaN percentages
    print("\nStep 3: Checking data quality...")
    for col in numeric_cols[:5]:  # Check first 5 columns
        nan_pct = df[col].null_count() / len(df) * 100
        print(f"  {col}: {nan_pct:.1f}% NaN")

    # Check value ranges
    print("\nValue ranges (first 5 columns):")
    for col in numeric_cols[:5]:
        non_null = df[col].drop_nulls()
        if len(non_null) > 0:
            print(f"  {col}: [{non_null.min():.3f}, {non_null.max():.3f}]")

    return df, numeric_cols


def test_sequence_creation(df, numeric_cols):
    """Test sequence creation performance."""
    print("\n" + "=" * 60)
    print("DIAGNOSING SEQUENCE CREATION")
    print("=" * 60)

    from train_patchtstnan import W3Dataset

    print("Creating dataset with small sequence length...")
    start_time = time.time()

    # Test with very small sequence for speed
    dataset = W3Dataset(df, sequence_length=32, prediction_horizon=1)

    creation_time = time.time() - start_time
    print(f"✅ Created dataset with {len(dataset)} sequences in {creation_time:.2f}s")

    # Test getting a few samples
    print("\nTesting sample retrieval...")
    start_time = time.time()

    samples = []
    for i in range(min(10, len(dataset))):
        sample = dataset[i]
        samples.append(sample)

    sample_time = time.time() - start_time
    print(f"✅ Retrieved 10 samples in {sample_time:.3f}s")

    # Analyze sample characteristics
    if samples:
        sample = samples[0]
        print("\nSample analysis:")
        print(f"  x_num shape: {sample['x_num'].shape}")
        print(f"  x_cat shape: {sample['x_cat'].shape}")
        print(f"  y value: {sample['y']}")
        print(
            f"  x_num range: [{sample['x_num'].min():.3f}, {sample['x_num'].max():.3f}]"
        )
        print(f"  x_num contains NaN: {torch.isnan(sample['x_num']).any()}")
        nan_pct = torch.isnan(sample["x_num"]).float().mean() * 100
        print(f"  x_num NaN percentage: {nan_pct:.1f}%")

        # Check for suspicious patterns
        if torch.isnan(sample["x_num"]).all():
            print("⚠️  WARNING: Sample contains only NaN values!")

        if sample["y"] == 1 and torch.isnan(sample["x_num"]).float().mean() == 0:
            print("⚠️  WARNING: Perfect correlation between non-NaN data and labels!")

    return dataset, samples


def test_model_forward_pass(samples, numeric_cols):
    """Test model forward pass for NaN issues."""
    print("\n" + "=" * 60)
    print("DIAGNOSING MODEL FORWARD PASS")
    print("=" * 60)

    if not samples:
        print("❌ No samples to test")
        return

    # Create small batch
    batch_size = 4
    batch = {
        "x_num": torch.stack(
            [samples[i % len(samples)]["x_num"] for i in range(batch_size)]
        ),
        "x_cat": torch.stack(
            [samples[i % len(samples)]["x_cat"] for i in range(batch_size)]
        ),
        "y": torch.stack([samples[i % len(samples)]["y"] for i in range(batch_size)]),
    }

    print("Test batch shapes:")
    print(f"  x_num: {batch['x_num'].shape}")
    print(f"  x_cat: {batch['x_cat'].shape}")
    print(f"  y: {batch['y'].shape}")
    print(f"  y values: {batch['y']}")

    # Create model
    model = PatchTSTNan(
        C_num=len(numeric_cols),
        C_cat=0,
        T=32,  # Small sequence for testing
        d_model=64,  # Smaller model for testing
        patch_size=16,  # 32/16 = 2 patches
        num_classes=3,
        n_heads=4,
        n_layers=2,
        task="classification",
    )

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {params:,}")

    # Test forward pass
    print("\nTesting forward pass...")
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        output = model(batch["x_num"], batch["x_cat"])
        forward_time = time.time() - start_time

        print(f"✅ Forward pass completed in {forward_time:.3f}s")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
        print(f"  Output contains NaN: {torch.isnan(output).any()}")
        print(f"  Output contains Inf: {torch.isinf(output).any()}")

        # Test loss calculation
        loss_fn = torch.nn.CrossEntropyLoss()
        try:
            loss = loss_fn(output, batch["y"])
            print(f"  Loss: {loss.item():.4f}")
            print(f"  Loss is NaN: {torch.isnan(loss)}")

            # Calculate accuracy
            preds = torch.argmax(output, dim=1)
            acc = (preds == batch["y"]).float().mean()
            print(f"  Accuracy: {acc.item():.3f}")

            # Check for suspicious patterns
            if acc.item() == 1.0 and len(torch.unique(batch["y"])) > 1:
                print(
                    "⚠️  WARNING: Perfect accuracy on diverse labels - "
                    "possible data leakage!"
                )

            if torch.isnan(loss):
                print("⚠️  WARNING: NaN loss detected!")
                print(f"    Logits: {output}")
                print(f"    Labels: {batch['y']}")

        except Exception as e:
            print(f"❌ Loss calculation failed: {e}")


def estimate_training_time(dataset_size, batch_size):
    """Estimate total training time."""
    print("\n" + "=" * 60)
    print("TRAINING TIME ESTIMATION")
    print("=" * 60)

    num_batches = dataset_size // batch_size
    print(f"Dataset size: {dataset_size:,}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches per epoch: {num_batches:,}")

    # Time estimates
    time_per_batch_estimates = {
        "Current (slow)": 5 * 60,  # 5 minutes per batch
        "Optimized": 1.0,  # 1 second per batch
        "Target": 0.1,  # 0.1 second per batch
    }

    print("\nTime estimates per epoch:")
    for scenario, time_per_batch in time_per_batch_estimates.items():
        total_time = num_batches * time_per_batch
        hours = total_time / 3600
        print(f"  {scenario}: {hours:.1f} hours ({total_time / 60:.1f} minutes)")


def main():
    print("W3 Training Issues Diagnostic")
    print("=" * 60)

    # Step 1: Analyze data loading
    df, numeric_cols = analyze_data_loading()

    # Step 2: Test sequence creation
    dataset, samples = test_sequence_creation(df, numeric_cols)

    # Step 3: Test model forward pass
    test_model_forward_pass(samples, numeric_cols)

    # Step 4: Estimate training time
    estimate_training_time(len(dataset), 1024)

    # Final recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    print("🔧 Speed improvements:")
    print("  1. Use much smaller dataset (100K-500K rows) for initial testing")
    print("  2. Increase num_workers in DataLoader (try 4-8)")
    print("  3. Use shorter sequences (32 instead of 64)")
    print("  4. Pre-process data to avoid groupby operations during training")

    print("\n🐛 Debug NaN loss:")
    print("  1. Check for data leakage (state info in features)")
    print("  2. Normalize/standardize features")
    print("  3. Add gradient clipping (already done)")
    print("  4. Check label distribution and class imbalance")

    print("\n⚡ Quick test setup:")
    print("  - Use train_patchtstnan.py with 100K rows")
    print("  - batch_size=64, sequence_length=32, patch_size=16")
    print("  - Should complete ~100 batches in <10 minutes")


if __name__ == "__main__":
    main()
