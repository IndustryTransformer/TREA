"""Canonical 3W training entrypoint.

The maintained implementation lives in ``examples.train_3w``. Keep this root-level
wrapper so existing commands continue to work without preserving the older
row-level parquet loader, whose split-local label mapping made validation invalid
on well-disjoint splits.
"""

from examples.train_3w import main


if __name__ == "__main__":
    main()
