"""Create W3 dataset with guaranteed multi-class distribution."""

import polars as pol


def find_diverse_state_regions():
    """Find regions of the dataset that contain diverse states."""
    print("=" * 60)
    print("FINDING DIVERSE STATE REGIONS")
    print("=" * 60)

    df_lazy = pol.scan_parquet("/home/ubuntu/TREA-C/data/W3/train.parquet")

    # Strategy: Search systematically for regions with state diversity
    print("Searching for regions with multiple states...")

    diverse_regions = []
    search_positions = range(0, 66_000_000, 1_000_000)  # Every 1M rows
    chunk_size = 100_000

    for pos in search_positions:
        try:
            chunk = (
                df_lazy.slice(pos, chunk_size).select(["state", "well_name"]).collect()
            )

            if len(chunk) > 0:
                # Analyze this chunk
                states = chunk.filter(pol.col("state").is_not_null())["state"]
                unique_states = states.n_unique()
                state_list = states.unique().to_list()

                if unique_states > 1:
                    wells = chunk["well_name"].n_unique()
                    print(
                        f"✅ Position {pos:,}: {unique_states} states "
                        f"{state_list}, {wells} wells"
                    )
                    diverse_regions.append((pos, chunk_size, unique_states, state_list))
                else:
                    print(f"   Position {pos:,}: {unique_states} state {state_list}")

        except Exception as e:
            print(f"   Error at position {pos:,}: {e}")

        # Stop after finding enough diverse regions
        if len(diverse_regions) >= 10:
            break

    return diverse_regions


def create_multiclass_dataset_v1():
    """Strategy 1: Use 'class' column which has proven diversity."""
    print("\n" + "=" * 60)
    print("STRATEGY 1: USING 'CLASS' COLUMN")
    print("=" * 60)

    df_lazy = pol.scan_parquet("/home/ubuntu/TREA-C/data/W3/train.parquet")

    print("Sampling data with 'class' as target...")

    # Sample chunks to get diverse classes
    target_rows = 2_000_000
    chunk_size = 200_000
    n_chunks = target_rows // chunk_size

    samples = []
    step_size = 66_000_000 // n_chunks

    for i in range(n_chunks):
        pos = i * step_size
        print(f"Sampling chunk {i + 1}/{n_chunks} at position {pos:,}...")

        try:
            chunk = (
                df_lazy.slice(pos, chunk_size)
                .filter(
                    pol.col("well_name").str.contains("WELL", literal=False)
                    & pol.col("class").is_not_null()
                )
                .drop("state")  # Drop state, use class instead
                .with_columns(
                    [
                        pol.col("class").alias("target")  # Rename class to target
                    ]
                )
                .drop("class")
                .collect()
            )

            if len(chunk) > 0:
                classes = chunk["target"].unique().to_list()
                print(f"  Sampled: {len(chunk):,} rows, classes: {classes}")
                samples.append(chunk)

        except Exception as e:
            print(f"  Error: {e}")

    if len(samples) == 0:
        print("❌ No samples found")
        return None

    # Combine samples
    combined_df = pol.concat(samples)

    # Analyze class distribution
    class_counts = combined_df["target"].value_counts().sort("target")
    print("\nClass distribution in combined dataset:")
    print(class_counts)

    unique_classes = combined_df["target"].n_unique()
    print(f"Unique classes: {unique_classes}")

    if unique_classes < 2:
        print("❌ Still no class diversity")
        return None

    print(f"✅ Successfully created dataset with {unique_classes} classes!")
    return combined_df


def create_multiclass_dataset_v2():
    """Strategy 2: Create synthetic classes from sensor data."""
    print("\n" + "=" * 60)
    print("STRATEGY 2: SYNTHETIC CLASSES FROM SENSORS")
    print("=" * 60)

    df_lazy = pol.scan_parquet("/home/ubuntu/TREA-C/data/W3/train.parquet")

    print("Creating synthetic classes from pressure sensor (P-PDG)...")

    # Sample data and create classes based on sensor patterns
    sample_size = 1_000_000
    df_sample = (
        df_lazy.head(sample_size)
        .filter(pol.col("well_name").str.contains("WELL", literal=False))
        .drop(["state", "class"])
        .collect()
    )

    print(f"Loaded {len(df_sample):,} rows for synthetic classification")

    # Use P-PDG (pressure) to create classes
    if "P-PDG" in df_sample.columns:
        pressure_values = df_sample.filter(pol.col("P-PDG").is_not_null())["P-PDG"]

        if len(pressure_values) > 1000:
            q25 = pressure_values.quantile(0.25)
            q75 = pressure_values.quantile(0.75)

            print("Creating pressure-based classes:")
            print(f"  Low pressure: < {q25:.2f}")
            print(f"  Medium pressure: {q25:.2f} - {q75:.2f}")
            print(f"  High pressure: > {q75:.2f}")

            # Create synthetic target
            df_with_target = df_sample.with_columns(
                [
                    pol.when(pol.col("P-PDG") < q25)
                    .then(0)
                    .when(pol.col("P-PDG") > q75)
                    .then(2)
                    .otherwise(1)
                    .alias("target")
                ]
            ).filter(pol.col("target").is_not_null())

            # Check distribution
            target_counts = df_with_target["target"].value_counts().sort("target")
            print("Synthetic target distribution:")
            print(target_counts)

            return df_with_target
        else:
            print("❌ Not enough valid pressure data")
            return None
    else:
        print("❌ P-PDG column not found")
        return None


def save_and_test_dataset(df, strategy_name):
    """Save the dataset and create a test version."""
    if df is None:
        return None, None

    print("\n" + "=" * 40)
    print(f"SAVING {strategy_name} DATASET")
    print("=" * 40)

    # Save main dataset
    output_path = (
        f"/home/ubuntu/TREA-C/data/W3/multiclass_{strategy_name.lower()}.parquet"
    )
    df.write_parquet(output_path)

    # Create test version
    test_path = output_path.replace(".parquet", "_test.parquet")
    df_test = df.head(500_000)
    df_test.write_parquet(test_path)

    print("✅ Saved datasets:")
    print(f"   Main: {output_path} ({len(df):,} rows)")
    print(f"   Test: {test_path} ({len(df_test):,} rows)")

    # Analyze final dataset
    target_col = "target"
    unique_targets = df[target_col].n_unique()
    target_dist = df[target_col].value_counts().sort(target_col)

    print("Final dataset quality:")
    print(f"  Unique targets: {unique_targets}")
    print("  Distribution:")
    print(target_dist)

    return output_path, test_path


def main():
    print("W3 Multi-Class Dataset Creator")
    print("Solving the single-class problem with targeted strategies")

    # Find diverse regions first (for understanding)
    diverse_regions = find_diverse_state_regions()

    if len(diverse_regions) > 0:
        print(f"\n✅ Found {len(diverse_regions)} regions with state diversity")
        print("This confirms that multi-class data exists in the dataset")
    else:
        print("\n⚠️  No state diversity found - will use alternative strategies")

    # Try Strategy 1: Use class column
    print("\n" + "=" * 60)
    print("ATTEMPTING STRATEGY 1: CLASS COLUMN")
    print("=" * 60)

    df_class = create_multiclass_dataset_v1()

    if df_class is not None and df_class["target"].n_unique() > 1:
        print("✅ Strategy 1 successful!")
        main_path, test_path = save_and_test_dataset(df_class, "class")

        print("\n🚀 Ready for training!")
        print("Use: train_patchtstnan_improved.py")
        print(f"Update dataset path to: {test_path}")
        return

    # Try Strategy 2: Synthetic classes
    print("\n" + "=" * 60)
    print("ATTEMPTING STRATEGY 2: SYNTHETIC CLASSES")
    print("=" * 60)

    df_synthetic = create_multiclass_dataset_v2()

    if df_synthetic is not None and df_synthetic["target"].n_unique() > 1:
        print("✅ Strategy 2 successful!")
        main_path, test_path = save_and_test_dataset(df_synthetic, "synthetic")

        print("\n🚀 Ready for training!")
        print("Use: train_patchtstnan_improved.py")
        print(f"Update dataset path to: {test_path}")
        return

    # If both fail
    print("\n❌ Both strategies failed")
    print("Recommendation: Switch to regression task")
    print("Use any sensor (e.g., P-PDG) as continuous target")


if __name__ == "__main__":
    main()
