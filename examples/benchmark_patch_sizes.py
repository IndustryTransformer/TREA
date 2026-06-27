"""Benchmark different patch sizes for PatchTSTNan performance optimization."""

import time

import torch

from treac.models import PatchTSTNan


def benchmark_patch_size(patch_size, C_num=20, T=64, batch_size=128, num_batches=10):
    """Benchmark a specific patch size configuration."""
    print(f"\n--- Benchmarking patch_size={patch_size} ---")

    # Create model
    model = PatchTSTNan(
        C_num=C_num,
        C_cat=0,
        T=T,
        d_model=128,
        patch_size=patch_size,
        num_classes=3,
        n_heads=8,
        n_layers=4,
        task="classification",
    )

    # Calculate effective sequence length
    n_patches = T // patch_size
    print(f"Effective transformer sequence length: {n_patches}")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {params:,}")

    # Create sample data
    x_num = torch.randn(batch_size, C_num, T)
    x_cat = torch.empty(batch_size, 0, T, dtype=torch.long)

    # Warmup
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            _ = model(x_num, x_cat)

    # Benchmark forward pass
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_batches):
            model(x_num, x_cat)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()

    total_time = end_time - start_time
    avg_time_per_batch = total_time / num_batches
    samples_per_second = batch_size / avg_time_per_batch

    print(f"Average time per batch: {avg_time_per_batch:.4f}s")
    print(f"Samples per second: {samples_per_second:.1f}")
    speedup = get_speedup_ratio(patch_size, avg_time_per_batch)
    print(f"Speedup vs patch_size=8: {speedup:.2f}x")

    return {
        "patch_size": patch_size,
        "n_patches": n_patches,
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "time_per_batch": avg_time_per_batch,
        "samples_per_second": samples_per_second,
    }


def get_speedup_ratio(patch_size, time_per_batch):
    """Calculate speedup ratio compared to patch_size=8 baseline."""
    # Rough estimation based on transformer complexity O(n^2) where n=sequence_length
    baseline_patches = 64 // 8  # 8 patches for patch_size=8
    current_patches = 64 // patch_size

    # Theoretical speedup is roughly (baseline_patches/current_patches)^2
    theoretical_speedup = (baseline_patches / current_patches) ** 2
    return theoretical_speedup


def main():
    print("PatchTSTNan Patch Size Performance Benchmark")
    print("=" * 60)

    # Test different patch sizes
    patch_sizes = [4, 8, 16, 32]  # 32 gives only 2 patches, might be too extreme
    results = []

    # Estimate dataset characteristics
    print("Estimated W3 dataset characteristics:")
    print("- Sequence length: 64")
    print("- Features: ~20 (estimated)")
    print("- Batch size: 128")
    print("- Dataset size: 5M rows")

    for patch_size in patch_sizes:
        if 64 % patch_size == 0:  # Only test valid patch sizes
            try:
                result = benchmark_patch_size(
                    patch_size, C_num=20, T=64, batch_size=128
                )
                results.append(result)
            except Exception as e:
                print(f"Error with patch_size={patch_size}: {e}")
        else:
            print(
                f"Skipping patch_size={patch_size} "
                f"(64 is not divisible by {patch_size})"
            )

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    header = (
        f"{'Patch Size':<12} {'Patches':<8} {'Params':<10} {'Time/Batch':<12} "
        f"{'Samples/s':<12} {'Speedup':<8}"
    )
    print(header)
    print("-" * 70)

    baseline_time = None
    for result in results:
        if result["patch_size"] == 8:
            baseline_time = result["time_per_batch"]

        speedup = baseline_time / result["time_per_batch"] if baseline_time else 1.0

        row = (
            f"{result['patch_size']:<12} {result['n_patches']:<8} "
            f"{result['params']:,<10} "
            f"{result['time_per_batch']:.4f} "
            f"{result['samples_per_second']:.1f} "
            f"{speedup:.2f}x"
        )
        print(row)

    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    if results:
        best_result = max(results, key=lambda x: x["samples_per_second"])
        print(f"Fastest configuration: patch_size={best_result['patch_size']}")
        if baseline_time:
            speedup = baseline_time / best_result["time_per_batch"]
            print(f"Expected speedup: {speedup:.2f}x faster")
        else:
            print("N/A")

        print("\nFor W3 dataset training:")
        print(f"- Use patch_size={best_result['patch_size']} for maximum speed")
        print(
            f"- This reduces transformer sequence from 8 to "
            f"{best_result['n_patches']} patches"
        )
        print(
            f"- Expected throughput: ~{best_result['samples_per_second']:.0f} "
            f"samples/second"
        )


if __name__ == "__main__":
    main()
