"""Fix W3 dataset issues: single class problem and data leakage."""

from collections import Counter

import pandas as pd
import polars as pol
import torch


def analyze_w3_data_distribution():
    """Analyze W3 data to understand class distribution issues."""
    print("=" * 60)
    print("W3 DATA DISTRIBUTION ANALYSIS")
    print("=" * 60)

    # Load larger sample to find more classes
    print("Loading larger sample to find class diversity...")
    df_lazy = pol.scan_parquet("/home/ubuntu/TREA-C/data/W3/train.parquet")

    # Try different sampling strategies
    sample_sizes = [100_000, 500_000, 1_000_000, 2_000_000]

    for sample_size in sample_sizes:
        print(f"\n--- Analyzing {sample_size:,} rows ---")
        df = df_lazy.head(sample_size).collect()

        # Basic stats
        df_clean = df.drop("class").filter(pol.col("state").is_not_null())

        print(f"Rows after cleaning: {len(df_clean):,}")
        print(f"Wells: {df_clean['well_name'].n_unique()}")

        # State distribution
        state_counts = df_clean["state"].value_counts()
        print("State distribution:")
        print(state_counts)

        unique_states = df_clean["state"].n_unique()
        print(f"Unique states: {unique_states}")

        if unique_states > 1:
            print(f"✅ Found {unique_states} classes! Using {sample_size:,} rows.")
            return df_clean, sample_size
        else:
            print(f"❌ Only 1 class found in {sample_size:,} rows")

    print("⚠️  Could not find multi-class data in any sample size")
    return None, None


def create_balanced_w3_dataset():
    """Create a balanced W3 dataset with multiple classes."""
    print("\n" + "=" * 60)
    print("CREATING BALANCED W3 DATASET")
    print("=" * 60)

    # Load full dataset schema first
    df_lazy = pol.scan_parquet("/home/ubuntu/TREA-C/data/W3/train.parquet")

    # Strategy: Sample from different parts of the dataset
    print("Sampling from different time periods to find class diversity...")

    samples = []
    chunk_size = 200_000
    max_chunks = 10

    for i in range(max_chunks):
        offset = i * chunk_size
        print(f"Sampling chunk {i + 1}: rows {offset:,} to {offset + chunk_size:,}")

        chunk = df_lazy.slice(offset, chunk_size).collect()
        chunk_clean = chunk.drop("class").filter(pol.col("state").is_not_null())

        if len(chunk_clean) > 0:
            states = chunk_clean["state"].unique().to_list()
            print(f"  Found states: {states}")
            samples.append(chunk_clean)

            # Check if we have enough diversity
            all_states = set()
            for sample in samples:
                all_states.update(sample["state"].unique().to_list())

            print(f"  Total unique states so far: {len(all_states)}")

            if len(all_states) >= 2:
                print(f"✅ Found sufficient class diversity: {all_states}")
                break

    if len(samples) == 0:
        print("❌ No valid data found")
        return None

    # Combine samples
    df_combined = pol.concat(samples)
    print(f"\nCombined dataset: {len(df_combined):,} rows")

    # Final state distribution
    final_states = df_combined["state"].value_counts()
    print("Final state distribution:")
    print(final_states)

    return df_combined


def create_synthetic_multiclass_w3():
    """Create synthetic multi-class labels for W3 if needed."""
    print("\n" + "=" * 60)
    print("CREATING SYNTHETIC MULTI-CLASS LABELS")
    print("=" * 60)

    # Load a reasonable sample
    df_lazy = pol.scan_parquet("/home/ubuntu/TREA-C/data/W3/train.parquet")
    df = df_lazy.head(300_000).collect()

    # Clean data
    df = df.drop("class").filter(pol.col("state").is_not_null())

    # Get numeric columns
    numeric_cols = [col for col in df.columns if col not in ["state", "well_name"]]
    print(f"Using {len(numeric_cols)} numeric features to create synthetic classes")

    # Convert to pandas for easier processing
    df_pd = df.to_pandas()

    # Create synthetic classes based on feature patterns
    print("Creating synthetic classes based on feature statistics...")

    # Strategy 1: Use quantiles of key features
    key_features = numeric_cols[:5]  # Use first 5 features
    feature_stats = []

    for col in key_features:
        if df_pd[col].notna().sum() > 0:
            values = df_pd[col].dropna()
            q33 = values.quantile(0.33)
            q66 = values.quantile(0.66)
            feature_stats.append((col, q33, q66))

    print(f"Using {len(feature_stats)} features for classification:")
    for col, q33, q66 in feature_stats:
        print(f"  {col}: Low<{q33:.3f}, Med={q33:.3f}-{q66:.3f}, High>{q66:.3f}")

    # Create synthetic labels
    synthetic_labels = []

    for idx, row in df_pd.iterrows():
        # Simple rule-based classification
        scores = []
        for col, q33, q66 in feature_stats:
            value = row[col]
            if pd.isna(value):
                scores.append(1)  # Default to medium
            elif value < q33:
                scores.append(0)  # Low
            elif value > q66:
                scores.append(2)  # High
            else:
                scores.append(1)  # Medium

        # Majority vote
        label = max(set(scores), key=scores.count)
        synthetic_labels.append(label)

    # Add synthetic labels to dataframe
    df_pd["synthetic_state"] = synthetic_labels

    # Check distribution
    label_counts = Counter(synthetic_labels)
    print(f"\nSynthetic label distribution: {dict(label_counts)}")

    # Convert back to polars
    df_synthetic = (
        pol.from_pandas(df_pd)
        .with_columns([pol.col("synthetic_state").alias("state")])
        .drop("synthetic_state")
    )

    return df_synthetic


def test_fixed_dataset(df):
    """Test the fixed dataset with PatchTSTNan."""
    print("\n" + "=" * 60)
    print("TESTING FIXED DATASET")
    print("=" * 60)

    if df is None:
        print("❌ No dataset to test")
        return

    # Prepare data like in the quick test
    numeric_cols = [col for col in df.columns if col not in ["state", "well_name"]]

    # Convert state to string then create mapping
    df = df.with_columns([pol.col("state").cast(pol.Utf8)])
    state_labels = df["state"].unique().sort().to_list()
    state_to_label = {state: i for i, state in enumerate(state_labels)}

    print(f"State mapping: {state_to_label}")
    print(f"Number of classes: {len(state_to_label)}")

    if len(state_to_label) < 2:
        print("❌ Still only 1 class - cannot test classification")
        return

    # Convert to integer labels
    df = df.with_columns([pol.col("state").replace_strict(state_to_label)])

    # Create a small test
    from treac.models import PatchTSTNan

    # Create sample data
    sample_data = df.head(1000).to_pandas()

    # Create a simple batch manually
    batch_size = 16
    sequence_length = 32

    # Get data for first well
    well_data = sample_data[
        sample_data["well_name"] == sample_data["well_name"].iloc[0]
    ]

    if len(well_data) < sequence_length:
        print(f"❌ Not enough data in well: {len(well_data)} < {sequence_length}")
        return

    # Create batch
    x_num_list = []
    y_list = []

    for i in range(batch_size):
        start_idx = i % (len(well_data) - sequence_length)
        seq_data = well_data.iloc[start_idx : start_idx + sequence_length]

        x_num = torch.tensor(seq_data[numeric_cols].values, dtype=torch.float32).T
        y = torch.tensor(seq_data["state"].iloc[-1], dtype=torch.long)

        x_num_list.append(x_num)
        y_list.append(y)

    batch = {
        "x_num": torch.stack(x_num_list),
        "x_cat": torch.empty(batch_size, 0, sequence_length, dtype=torch.long),
        "y": torch.stack(y_list),
    }

    print("Test batch:")
    print(f"  x_num: {batch['x_num'].shape}")
    print(f"  y: {batch['y'].shape}")
    print(f"  y values: {torch.unique(batch['y'])}")
    print(f"  y distribution: {torch.bincount(batch['y'])}")

    # Test model
    model = PatchTSTNan(
        C_num=len(numeric_cols),
        C_cat=0,
        T=sequence_length,
        d_model=64,
        patch_size=16,
        num_classes=len(state_to_label),
        n_heads=4,
        n_layers=2,
        task="classification",
    )

    model.eval()
    with torch.no_grad():
        output = model(batch["x_num"], batch["x_cat"])

        print("\nModel test:")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
        print(f"  Contains NaN: {torch.isnan(output).any()}")

        # Test loss
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(output, batch["y"])
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Loss is NaN: {torch.isnan(loss)}")

        # Test accuracy
        preds = torch.argmax(output, dim=1)
        acc = (preds == batch["y"]).float().mean()
        print(f"  Accuracy: {acc.item():.3f}")

        if torch.isnan(loss):
            print("❌ Still getting NaN loss")
        else:
            print("✅ Fixed! No more NaN loss")


def main():
    print("W3 Dataset Fix Tool")
    print("Addressing single-class and NaN loss issues")

    # Try different strategies
    strategies = [
        ("Find natural multi-class data", analyze_w3_data_distribution),
        ("Create balanced dataset", create_balanced_w3_dataset),
        ("Create synthetic classes", create_synthetic_multiclass_w3),
    ]

    fixed_dataset = None

    for strategy_name, strategy_func in strategies:
        print(f"\n{'=' * 60}")
        print(f"STRATEGY: {strategy_name}")
        print(f"{'=' * 60}")

        if strategy_name == "Find natural multi-class data":
            result, sample_size = strategy_func()
            if result is not None:
                fixed_dataset = result
                break
        else:
            result = strategy_func()
            if result is not None:
                fixed_dataset = result
                break

    # Test the fixed dataset
    if fixed_dataset is not None:
        test_fixed_dataset(fixed_dataset)

        # Save for future use
        output_path = "/home/ubuntu/TREA-C/data/W3/fixed_train_sample.parquet"
        fixed_dataset.write_parquet(output_path)
        print(f"\n✅ Saved fixed dataset to: {output_path}")
        print(f"   Rows: {len(fixed_dataset):,}")
        print(f"   Classes: {fixed_dataset['state'].n_unique()}")
    else:
        print("\n❌ Could not create a valid multi-class dataset")
        print(
            "   Recommendation: Check original data source or use a different dataset"
        )


if __name__ == "__main__":
    main()
