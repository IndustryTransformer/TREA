"""Quick test to debug dataset loading speed."""

import time

import polars as pol


# Test how long it takes to load a single well
print("Testing single well loading speed...")

parquet_path = "/home/ubuntu/TREA-C/data/W3/train.parquet"
df_lazy = pol.scan_parquet(parquet_path)

# Get a well name first
wells = df_lazy.select("well_name").unique().collect()["well_name"].to_list()
test_well = wells[0]
print(f"Testing with well: {test_well}")

start_time = time.time()

well_data = (
    df_lazy.filter(pol.col("well_name") == test_well)
    .filter(pol.col("state").is_not_null())
    .collect()
)

end_time = time.time()
print(f"Loading one well took: {end_time - start_time:.2f} seconds")
print(f"Well contains {len(well_data)} rows")

# Test loading multiple wells at once
print("\nTesting multiple well loading...")
test_wells = wells[:5]

start_time = time.time()

multi_well_data = (
    df_lazy.filter(pol.col("well_name").is_in(test_wells))
    .filter(pol.col("state").is_not_null())
    .collect()
)

end_time = time.time()
print(f"Loading 5 wells took: {end_time - start_time:.2f} seconds")
print(f"Total rows: {len(multi_well_data)}")
