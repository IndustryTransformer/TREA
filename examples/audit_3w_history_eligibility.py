"""Audit label-based window omissions on fixed internal validation only."""

import json

from pathlib import Path

import numpy as np
import pandas as pd


def main():
    protocol = json.loads(
        Path(
            "logs/benchmark_3w_binary/range_confirmation_probe/protocol.json"
        ).read_text()
    )
    manifest = protocol["val"]
    total, unlabeled, invalid, affected = 0, 0, 0, []
    for file in manifest["files"]:
        values = pd.read_parquet(Path(manifest["data_path"]) / file, columns=["class"])[
            "class"
        ].to_numpy()
        values = np.where(np.isfinite(values), values, -1).astype(np.int64)
        values = np.where(values >= 100, values % 100, values)
        n = len(values) // 96
        if n == 0:
            continue
        windows = values[: n * 96].reshape(n, 96)
        valid = windows >= 0
        missing = ~valid.any(axis=1)
        # The existing loader ignores negative labels and drops out-of-range medians.
        eligible = windows[~missing].astype(float)
        eligible[eligible < 0] = np.nan
        median = np.nanmedian(eligible, axis=1)
        bad = int(((median < 0) | (median >= 10)).sum())
        total += n
        unlabeled += int(missing.sum())
        invalid += bad
        if missing.any() or bad:
            affected.append(
                {
                    "file": file,
                    "all_unlabeled_windows": int(missing.sum()),
                    "invalid_median_windows": bad,
                    "omissions_are_prefix": bool(
                        np.array_equal(
                            np.flatnonzero(missing), np.arange(missing.sum())
                        )
                    ),
                }
            )
    result = {
        "scope": "Fixed internal validation only, no consumed or outer test groups",
        "raw_full_windows": total,
        "cached_windows": manifest["n"],
        "all_unlabeled_omissions": unlabeled,
        "invalid_median_omissions": invalid,
        "affected_files": affected,
        "all_omissions_are_prefix": all(f["omissions_are_prefix"] for f in affected),
        "distinct_prefix_lengths": sorted(
            {f["all_unlabeled_windows"] for f in affected}
        ),
    }
    Path("docs/results/3w_history_eligibility_audit.json").write_text(
        json.dumps(result, indent=2)
    )
    print(
        json.dumps({k: v for k, v in result.items() if k != "affected_files"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
