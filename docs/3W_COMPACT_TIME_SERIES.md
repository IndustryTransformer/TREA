# Compact time-series recipe on 3W

## Question and locked evaluation

Does the compact recipe selected on single-row NOx data transfer to a true temporal
model on 3W strongly enough to beat XGBoost?

The admissible 3W task remains binary normal-vs-fault classification with a
well-disjoint split. This avoids the known leakage from structural missingness when
files from the same well cross the split. The primary metric is AUC on unseen wells;
macro-F1 is secondary. The data split seed is fixed at 42 and model seeds are
42/43/44.

The two new baseline gates are:

1. `xgb_naive`: 12 evenly spaced raw readings per retained sensor, flattened in
   temporal order. NaNs remain native XGBoost missing values. No window statistics
   or domain crosses are added.
2. `xgb_engineered`: per-sensor mean, standard deviation, minimum, maximum, missing
   rate, first-to-last delta, half-window trend, and energy.

Both XGBoost arms use at most 200,000 training windows per seed, 500 depth-6 trees,
learning rate 0.05, row/column subsampling 0.8, and class balancing. They share the
same 919-train-well / 230-validation-well split and 110,595 validation windows.

The transferred deep recipe is explicit: width 128, four heads, two layers, zero
dropout, and supervised LR `3e-4`. This is a combined-recipe test, not a depth-only
ablation. The earlier large 3W settings remain reconstructible with command-line
arguments. Early stopping now reloads the best validation-loss checkpoint before
evaluation.

## Baselines (2026-09-17)

| Model | AUC mean ± sd | Macro-F1 mean ± sd | Train sec/run |
| --- | ---: | ---: | ---: |
| XGB naïve raw grid | 0.9068 ± 0.0030 | 0.7431 ± 0.0047 | 3.18 |
| XGB engineered | 0.9474 ± 0.0021 | 0.7980 ± 0.0052 | 1.92 |
| Prior RF engineered | 0.9584 ± 0.0036 | 0.8202 ± 0.0002 | 11.93 |
| Prior large TREA + stat tokens | 0.8447 ± 0.0053 | 0.6689 ± 0.0076 | 393.5 |

The compact model therefore needs +0.0621 mean AUC over the prior large model to
clear the naïve XGB gate, and +0.1027 to clear the engineered XGB stretch gate.
The RF result is retained as a stronger contextual bar, but the requested gates are
the two XGBoost arms above.

Exact run rows and protocol are recorded in
`docs/results/3w_compact_xgb_baselines.json`. The runnable entry point is
`examples/benchmark_3w_binary.py`.

## Reproduction

```bash
uv run python examples/benchmark_3w_binary.py \
  --models xgb_naive,xgb_engineered \
  --seeds 42,43,44 \
  --xgb-n-jobs 16 \
  --num-workers 0 \
  --run-name compact_xgb_full \
  --no-show-progress
```

The deep run uses the script defaults and should be run on a functioning CUDA
device. Ensembling, if evaluated, must average independently trained seed
probabilities and be reported separately from single-fit scores.
