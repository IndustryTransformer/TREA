"""Investigate why W3 dataset has only one state value across 66M rows."""

import polars as pol


def deep_state_analysis():
    """Deep dive into the state column issue."""
    print("=" * 60)
    print("DEEP STATE ANALYSIS - W3 DATASET")
    print("=" * 60)

    df_lazy = pol.scan_parquet("/home/ubuntu/TREA-C/data/W3/train.parquet")

    print("Step 1: Analyzing state column across entire dataset...")

    # Check state distribution across the ENTIRE dataset
    try:
        state_analysis = df_lazy.select(
            [
                pol.col("state").n_unique().alias("unique_states"),
                pol.col("state").min().alias("min_state"),
                pol.col("state").max().alias("max_state"),
                pol.col("state").null_count().alias("null_count"),
                pol.len().alias("total_rows"),
            ]
        ).collect()

        print("Full dataset state analysis:")
        print(state_analysis)

        unique_states = state_analysis["unique_states"][0]
        min_state = state_analysis["min_state"][0]
        max_state = state_analysis["max_state"][0]
        null_count = state_analysis["null_count"][0]
        total_rows = state_analysis["total_rows"][0]

        print("\nDetailed breakdown:")
        print(f"  Total rows: {total_rows:,}")
        print(f"  Unique states: {unique_states}")
        print(f"  State range: {min_state} to {max_state}")
        print(f"  Null states: {null_count:,} ({null_count / total_rows * 100:.1f}%)")

        if unique_states == 1:
            print("❌ PROBLEM: Only 1 unique state in entire 66M row dataset!")
            print("   This explains why sampling didn't help")

    except Exception as e:
        print(f"Error analyzing full dataset: {e}")

    print("\nStep 2: Checking state values in detail...")

    # Get actual state values
    try:
        state_values = df_lazy.select("state").head(100_000).collect()
        unique_vals = state_values["state"].unique().to_list()
        value_counts = state_values["state"].value_counts()

        print("State values sample (first 100K rows):")
        print(f"  Unique values: {unique_vals}")
        print("  Value counts:")
        print(value_counts)

    except Exception as e:
        print(f"Error getting state values: {e}")

    print("\nStep 3: Checking 'class' column for comparison...")

    # Check if 'class' column has more diversity
    try:
        class_analysis = df_lazy.select(
            [
                pol.col("class").n_unique().alias("unique_classes"),
                pol.col("class").min().alias("min_class"),
                pol.col("class").max().alias("max_class"),
                pol.col("class").null_count().alias("null_count"),
            ]
        ).collect()

        print("Class column analysis:")
        print(class_analysis)

        # Sample class values
        class_values = df_lazy.select("class").head(100_000).collect()
        class_counts = class_values["class"].value_counts()
        print("Class value counts (sample):")
        print(class_counts)

    except Exception as e:
        print(f"Error analyzing class column: {e}")

    print("\nStep 4: Relationship between state and class...")

    try:
        relationship = df_lazy.select(["state", "class"]).head(100_000).collect()
        combined_counts = (
            relationship.group_by(["state", "class"]).len().sort("len", descending=True)
        )
        print("State-Class combinations:")
        print(combined_counts)

    except Exception as e:
        print(f"Error analyzing relationship: {e}")


def check_temporal_patterns():
    """Check if states change over time within wells."""
    print("\n" + "=" * 60)
    print("TEMPORAL PATTERN ANALYSIS")
    print("=" * 60)

    df_lazy = pol.scan_parquet("/home/ubuntu/TREA-C/data/W3/train.parquet")

    # Check if there's any temporal structure
    print("Sampling data from different time periods...")

    # Sample from beginning, middle, and end
    positions = [0, 20_000_000, 40_000_000, 60_000_000]
    chunk_size = 50_000

    for i, pos in enumerate(positions):
        print(f"\nTime period {i + 1}: rows {pos:,} to {pos + chunk_size:,}")

        try:
            chunk = df_lazy.slice(pos, chunk_size).collect()

            if len(chunk) > 0:
                # State analysis for this chunk
                state_info = chunk.select(
                    [
                        pol.col("state").n_unique().alias("unique_states"),
                        pol.col("state").min().alias("min_state"),
                        pol.col("state").max().alias("max_state"),
                    ]
                ).row(0)

                wells_in_chunk = chunk["well_name"].n_unique()

                print(f"  Rows: {len(chunk):,}")
                print(f"  Wells: {wells_in_chunk}")
                print(f"  Unique states: {state_info[0]}")
                print(f"  State range: {state_info[1]} to {state_info[2]}")

                # Check state distribution
                states = chunk["state"].value_counts()
                print(f"  State distribution: {states}")

        except Exception as e:
            print(f"  Error: {e}")


def explore_other_classification_options():
    """Explore alternative ways to create classification targets."""
    print("\n" + "=" * 60)
    print("ALTERNATIVE CLASSIFICATION STRATEGIES")
    print("=" * 60)

    df_lazy = pol.scan_parquet("/home/ubuntu/TREA-C/data/W3/train.parquet")

    print("Option 1: Use 'class' column instead of 'state'...")

    try:
        # Analyze class column distribution
        class_sample = df_lazy.select("class").head(500_000).collect()
        class_unique = class_sample["class"].n_unique()
        class_counts = class_sample["class"].value_counts().sort("class")

        print("Class column analysis:")
        print(f"  Unique classes: {class_unique}")
        print("  Distribution:")
        print(class_counts)

        if class_unique > 1:
            print("✅ 'class' column has multiple values - could use this instead!")
        else:
            print("❌ 'class' column also has only 1 unique value")

    except Exception as e:
        print(f"Error analyzing class column: {e}")

    print("\nOption 2: Create synthetic targets from sensor patterns...")

    try:
        # Sample some numeric features to create synthetic targets
        feature_sample = (
            df_lazy.select(["P-PDG", "P-TPT", "T-TPT", "QGL", "QBS"])
            .head(100_000)
            .collect()
        )

        print("Analyzing key sensor features for pattern-based classification:")

        for col in ["P-PDG", "P-TPT", "T-TPT", "QGL", "QBS"]:
            if col in feature_sample.columns:
                values = feature_sample[col].drop_nulls()
                if len(values) > 0:
                    q25, q50, q75 = values.quantile([0.25, 0.5, 0.75]).to_list()
                    print(f"  {col}: Q25={q25:.2f}, Q50={q50:.2f}, Q75={q75:.2f}")

                    print(
                        f"    Could create classes: Low<{q25:.2f}, "
                        f"Med={q25:.2f}-{q75:.2f}, High>{q75:.2f}"
                    )

        print("\n✅ Synthetic classification approach is viable!")

    except Exception as e:
        print(f"Error analyzing features: {e}")

    print("\nOption 3: Well-based classification...")

    try:
        # Use well names as classes
        well_sample = df_lazy.select("well_name").head(100_000).collect()
        unique_wells = well_sample["well_name"].n_unique()
        well_counts = (
            well_sample["well_name"].value_counts().sort("count", descending=True)
        )

        print("Well-based classification:")
        print(f"  Unique wells: {unique_wells}")
        print("  Top wells by frequency:")
        print(well_counts.head(10))

        if unique_wells > 1:
            print("✅ Could classify by well type/performance!")

    except Exception as e:
        print(f"Error analyzing wells: {e}")


def recommend_solutions():
    """Recommend specific solutions based on findings."""
    print("\n" + "=" * 60)
    print("RECOMMENDED SOLUTIONS")
    print("=" * 60)

    print("🔍 Root Cause:")
    print("   The 'state' column in W3 dataset appears to be invariant")
    print(
        "   This suggests it might be a status flag rather than a classification target"
    )
    print("   OR the dataset represents a single operational state")

    print("\n✅ Solution 1: Use 'class' column (if it has diversity)")
    print("   - Replace 'state' with 'class' in your training code")
    print("   - Check if 'class' has multiple values")

    print("\n✅ Solution 2: Create synthetic multi-class targets")
    print("   - Use sensor readings (P-PDG, T-TPT, etc.) to create classes")
    print("   - Example: Pressure-based classes (Low/Medium/High)")
    print("   - Or combine multiple sensors for richer classification")

    print("\n✅ Solution 3: Switch to regression")
    print("   - Predict continuous sensor values instead of classes")
    print("   - Use one sensor as target, others as features")
    print("   - Avoids the classification problem entirely")

    print("\n✅ Solution 4: Well performance classification")
    print("   - Classify wells by production efficiency")
    print("   - Use statistical properties of sensor readings per well")

    print("\n🚀 Immediate Action:")
    print("   1. Run this analysis to understand your data better")
    print("   2. Try Solution 1 (class column) first - easiest fix")
    print("   3. If that fails, implement Solution 2 (synthetic targets)")
    print("   4. Consider Solution 3 (regression) as backup")


def main():
    print("W3 State Investigation Tool")
    print("Understanding why only 1 state exists in 66M rows")

    # Run all analyses
    deep_state_analysis()
    check_temporal_patterns()
    explore_other_classification_options()
    recommend_solutions()

    print("\n" + "=" * 60)
    print("INVESTIGATION COMPLETE")
    print("=" * 60)
    print("This analysis should reveal why your state column is invariant")
    print("and provide actionable solutions for creating a multi-class dataset.")


if __name__ == "__main__":
    main()
