"""Create a diverse 5M row W3 dataset with proper sampling and filtering."""

import time

import numpy as np
import polars as pol


def analyze_full_dataset_structure():
    """Analyze the full dataset to understand structure before sampling."""
    print("=" * 60)
    print("ANALYZING FULL W3 DATASET STRUCTURE")
    print("=" * 60)

    df_lazy = pol.scan_parquet("/home/ubuntu/TREA-C/data/W3/train.parquet")

    print("Step 1: Basic dataset info...")
    try:
        # Get basic info without loading all data
        schema = df_lazy.schema
        print(f"Columns: {list(schema.keys())}")
        print(f"Data types: {schema}")

        # Count total rows efficiently
        total_rows = df_lazy.select(pol.len()).collect().item()
        print(f"Total rows: {total_rows:,}")

    except Exception as e:
        print(f"Error getting basic info: {e}")
        return None

    print("\nStep 2: Analyzing well names...")
    try:
        # Sample well names to understand patterns
        well_sample = df_lazy.select("well_name").head(100_000).collect()
        unique_wells = well_sample["well_name"].unique().to_list()

        print("Sample of well names:")
        for i, well in enumerate(unique_wells[:10]):
            print(f"  {well}")
        print(f"... and {len(unique_wells) - 10} more")

        # Check which wells contain "WELL"
        wells_with_well = [w for w in unique_wells if "WELL" in str(w).upper()]
        wells_without_well = [w for w in unique_wells if "WELL" not in str(w).upper()]

        print("\nWell name analysis:")
        print(f"  Wells containing 'WELL': {len(wells_with_well)}")
        print(f"  Wells NOT containing 'WELL': {len(wells_without_well)}")

        if wells_with_well:
            print(f"  Examples with 'WELL': {wells_with_well[:5]}")
        if wells_without_well:
            print(f"  Examples without 'WELL': {wells_without_well[:5]}")

        return df_lazy, wells_with_well, total_rows

    except Exception as e:
        print(f"Error analyzing wells: {e}")
        return df_lazy, [], 0


def create_diverse_sampling_strategy(df_lazy, target_rows=5_000_000):
    """Create a sampling strategy to get diverse data across the full dataset."""
    print("\n" + "=" * 60)
    print("CREATING DIVERSE SAMPLING STRATEGY")
    print(f"Target: {target_rows:,} rows")
    print("=" * 60)

    # Strategy: Sample from multiple chunks across the dataset
    print("Step 1: Analyzing dataset size...")
    try:
        total_rows = df_lazy.select(pol.len()).collect().item()
        print(f"Total available rows: {total_rows:,}")
    except Exception:
        total_rows = 10_000_000  # Conservative estimate
        print(f"Estimated total rows: {total_rows:,}")

    # Calculate sampling strategy
    if target_rows >= total_rows:
        print(f"Target ({target_rows:,}) >= Total ({total_rows:,})")
        print("Will use entire dataset")
        return [(0, total_rows)]

    # Create multiple chunks for diversity
    num_chunks = 20  # Sample from 20 different parts of the dataset
    chunk_size = target_rows // num_chunks

    print("Sampling strategy:")
    print(f"  Number of chunks: {num_chunks}")
    print(f"  Rows per chunk: {chunk_size:,}")

    # Calculate chunk positions spread across the dataset
    chunk_positions = []
    step_size = total_rows // num_chunks

    for i in range(num_chunks):
        start_pos = i * step_size
        # Add some randomness within each chunk
        random_offset = (
            np.random.randint(0, min(step_size // 4, 100_000))
            if step_size > 100_000
            else 0
        )
        start_pos += random_offset

        # Ensure we don't go past the end
        if start_pos + chunk_size > total_rows:
            start_pos = total_rows - chunk_size

        chunk_positions.append((start_pos, chunk_size))
        print(f"  Chunk {i + 1}: rows {start_pos:,} to {start_pos + chunk_size:,}")

    return chunk_positions


def sample_diverse_data(df_lazy, chunk_positions):
    """Sample data from multiple chunks with filtering."""
    print("\n" + "=" * 60)
    print("SAMPLING DIVERSE DATA")
    print("=" * 60)

    samples = []
    total_sampled = 0

    for i, (start_pos, chunk_size) in enumerate(chunk_positions):
        print(f"\nProcessing chunk {i + 1}/{len(chunk_positions)}...")
        print(f"  Position: {start_pos:,} to {start_pos + chunk_size:,}")

        try:
            # Sample chunk with filtering
            chunk = (
                df_lazy.slice(start_pos, chunk_size)
                .filter(
                    pol.col("well_name").str.contains("WELL", literal=False)
                    & pol.col("state").is_not_null()
                )
                .drop("class")
                .collect()
            )

            if len(chunk) > 0:
                print(f"  Sampled: {len(chunk):,} rows")
                print(f"  Wells: {chunk['well_name'].n_unique()}")

                # Check state diversity in this chunk
                states = chunk["state"].unique().to_list()
                print(f"  States: {states}")

                samples.append(chunk)
                total_sampled += len(chunk)
            else:
                print("  No valid data in this chunk")

        except Exception as e:
            print(f"  Error processing chunk: {e}")
            continue

    print(f"\nCombining {len(samples)} chunks...")
    if len(samples) == 0:
        print("❌ No valid samples found")
        return None

    # Combine all samples
    combined_df = pol.concat(samples)
    print(f"✅ Combined dataset: {len(combined_df):,} rows")

    return combined_df


def analyze_dataset_quality(df):
    """Analyze the quality and diversity of the sampled dataset."""
    print("\n" + "=" * 60)
    print("ANALYZING DATASET QUALITY")
    print("=" * 60)

    print(f"Dataset size: {len(df):,} rows")

    # Well analysis
    wells = df["well_name"].unique().to_list()
    print(f"Unique wells: {len(wells)}")
    print(f"Sample wells: {wells[:10]}")

    # State analysis
    state_counts = df["state"].value_counts().sort("state")
    print("\nState distribution:")
    print(state_counts)

    unique_states = df["state"].n_unique()
    print(f"Unique states: {unique_states}")

    if unique_states < 2:
        print("⚠️  WARNING: Still only 1 unique state!")
        return False

    # Check class balance
    min_count = state_counts["count"].min()
    max_count = state_counts["count"].max()
    balance_ratio = min_count / max_count

    print(f"Class balance ratio: {balance_ratio:.3f}")
    if balance_ratio < 0.1:
        print("⚠️  WARNING: Highly imbalanced classes")
    elif balance_ratio > 0.3:
        print("✅ Good class balance")
    else:
        print("⚠️  Moderate class imbalance")

    # Numeric features analysis
    numeric_cols = [col for col in df.columns if col not in ["state", "well_name"]]
    print(f"\nNumeric features: {len(numeric_cols)}")

    # Check for excessive NaN values
    nan_percentages = []
    for col in numeric_cols[:5]:  # Check first 5 columns
        nan_pct = df[col].null_count() / len(df) * 100
        nan_percentages.append(nan_pct)
        print(f"  {col}: {nan_pct:.1f}% NaN")

    avg_nan_pct = np.mean(nan_percentages)
    if avg_nan_pct > 50:
        print("⚠️  WARNING: High percentage of NaN values")
    else:
        print("✅ Acceptable NaN levels")

    # Check data distribution by well
    well_state_counts = (
        df.group_by("well_name")
        .agg(
            [
                pol.col("state").n_unique().alias("unique_states"),
                pol.len().alias("row_count"),
            ]
        )
        .sort("row_count", descending=True)
    )

    print("\nWell statistics:")
    wells_with_multiple_states = (well_state_counts["unique_states"] > 1).sum()
    print(f"  Wells with multiple states: {wells_with_multiple_states}")
    print(f"  Average rows per well: {well_state_counts['row_count'].mean():.0f}")
    print("  Top wells by size:")
    print(well_state_counts.head(5))

    return unique_states >= 2


def save_improved_dataset(
    df, output_path="/home/ubuntu/TREA-C/data/W3/improved_train_5M.parquet"
):
    """Save the improved dataset."""
    print("\n" + "=" * 60)
    print("SAVING IMPROVED DATASET")
    print("=" * 60)

    try:
        df.write_parquet(output_path)
        print(f"✅ Saved to: {output_path}")
        print(f"   Rows: {len(df):,}")
        print(f"   Wells: {df['well_name'].n_unique()}")
        print(f"   States: {df['state'].n_unique()}")

        # Create a smaller test version too
        test_path = output_path.replace(".parquet", "_test_500K.parquet")
        df_test = df.head(500_000)
        df_test.write_parquet(test_path)
        print(f"✅ Saved test version: {test_path}")
        print(f"   Test rows: {len(df_test):,}")

        return output_path, test_path

    except Exception as e:
        print(f"❌ Error saving: {e}")
        return None, None


def main():
    print("W3 Dataset Improvement Tool")
    print("Goal: Create diverse 5M row dataset with multi-class data")

    # Step 1: Analyze dataset structure
    result = analyze_full_dataset_structure()
    if result is None:
        print("❌ Could not analyze dataset")
        return

    df_lazy, wells_with_well, total_rows = result

    if len(wells_with_well) == 0:
        print("❌ No wells found containing 'WELL' in name")
        print("Continuing with all wells...")
    else:
        print(f"✅ Found {len(wells_with_well)} wells with 'WELL' in name")

    # Step 2: Create sampling strategy
    target_rows = 5_000_000
    chunk_positions = create_diverse_sampling_strategy(df_lazy, target_rows)

    # Step 3: Sample diverse data
    start_time = time.time()
    df_improved = sample_diverse_data(df_lazy, chunk_positions)
    sampling_time = time.time() - start_time

    if df_improved is None:
        print("❌ Failed to create improved dataset")
        return

    print(f"\nSampling completed in {sampling_time:.1f} seconds")

    # Step 4: Analyze quality
    is_good_quality = analyze_dataset_quality(df_improved)

    if not is_good_quality:
        print("⚠️  Dataset quality issues detected")
        print("Consider adjusting sampling strategy or using synthetic labels")

    # Step 5: Save improved dataset
    output_path, test_path = save_improved_dataset(df_improved)

    if output_path:
        print("\n" + "=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print("✅ Created improved W3 dataset")
        print(f"   Main dataset: {output_path}")
        print(f"   Test dataset: {test_path}")
        print("   Ready for training with PatchTSTNan!")

        # Quick test recommendation
        print("\n🚀 Next steps:")
        print("   1. Test with: train_patchtstnan_quick_test.py (using test dataset)")
        print("   2. Full training: train_patchtstnan.py (using main dataset)")
        print("   3. Monitor for NaN loss - should be fixed!")


if __name__ == "__main__":
    main()
