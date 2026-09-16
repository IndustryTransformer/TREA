# Basic numerical-encoding reruns — 2026-09-16

**Conclusion:** no general accuracy improvement from the richer numerical bases
under the fixed training recipe. Column-conditioned fusion is a modest exploratory
candidate, but harms the future-year NOx check. Retain the default pending independent
confirmation; these experiments do not support scaling pretraining yet.

## Protocol

15 jobs: three datasets × five configurations × three seeds, giving 45 neural
fits, plus tree baselines. All use the original identity-query first block. Only
numerical basis or fusion changes; no learning-rate/dropout tuning or pretraining.
Each basis keeps the continuous value/present branch. Fusion uses identity encoding.
The protocol was written before training to `logs/basic_numeric_experiments/protocol.json`.

- NOx: 2,219 fit / 554 validation / 4,439 historical holdout / 14,542 future-year
  rows, fixed split 42, model seeds 10/11/12, nine sensors. 128 dimensions, four
  layers, dropout 0.2, AdamW lr 8e-4, max 100 epochs, patience 12, best validation
  checkpoint. Same protocol as the earlier first-block diagnostic.
- Diabetes (442 rows, 10 features) and balance_scale (625 rows, four features):
  shared-fit-v2, seeds 0/1/2, fixed hyperparameters, max 100 epochs, patience 16.
  Unlike NOx, each static seed changes both the data split and initialization.
  Neural width 128, three layers, dropout 0.2, lr 8e-4; classification uses
  logit-adjusted loss and validation macro-F1. All arms fit identical rows.
- CPU, two Torch threads per job, up to four concurrent jobs. The prior NOx
  diagnostic used four threads; the baseline was rerun here because numerical
  differences can affect training and early stopping. Within this experiment,
  runtime settings are matched. Matching seed numbers across architectures does
  not mean their differently shaped parameters have identical initial values.
- These are previously inspected, deliberately small diagnostic datasets, not
  a representative tabular benchmark or untouched confirmation set.

## Results

Mean ± sample standard deviation over three seeds. Lower RMSE and higher macro-F1
are better. The static harness stores population SD; the table recomputes sample
SD from its saved per-seed scores for consistency with NOx.

| Configuration | NOx historical RMSE ↓ | NOx future RMSE ↓ | Diabetes RMSE ↓ | Balance macro-F1 ↑ |
| --- | ---: | ---: | ---: | ---: |
| Current encoder | 5.171 ± 0.181 | 12.270 ± 0.834 | 58.949 ± 2.776 | 0.844 ± 0.006 |
| Piecewise only | 5.380 ± 0.079 | 13.706 ± 0.277 | 63.177 ± 5.600 | 0.733 ± 0.069 |
| Periodic only | 5.598 ± 0.134 | 12.438 ± 0.558 | 65.962 ± 8.891 | 0.841 ± 0.038 |
| Combined bases | 5.587 ± 0.114 | 13.219 ± 0.655 | 63.703 ± 7.056 | 0.791 ± 0.019 |
| Column-conditioned fusion | 5.056 ± 0.104 | 12.657 ± 0.442 | 58.072 ± 1.657 | 0.845 ± 0.028 |
| Tree reference | 4.619 | 12.520 | 59.507 | 0.696 |

NOx reference is XGBoost; static references are the best mean test score across
XGBoost, RF and CatBoost (a descriptive test-score envelope, not a model selected
on validation). Tree scores are identical across encoding variants.

Column-conditioned fusion reduces historical NOx mean RMSE by 2.22%, improving
2/3 seed pairs. It meets the exploratory 2% / 2-of-3 diagnostic threshold, but
future-year RMSE increases 3.15%. Diabetes RMSE improves 1.49%, in 3/3 seed pairs;
this is smaller than the 2% practical margin. Balance macro-F1 changes by only
+0.00054, in 1/3 pairs. This is not evidence for a universal improvement.

The nonlinear bases all have worse mean primary scores than the current encoder.
On NOx, periodic encoding lowers training RMSE from 4.925 to 3.903 while increasing
historical test RMSE from 5.171 to 5.598. This is consistent with a generalization
problem rather than insufficient ability to fit the training rows; it is not a
proof of the underlying mechanism. Basis-specific tuning was not attempted.

## Why the old balance_scale collapse does not count as a new win

The fresh current-encoder baseline reaches 0.844 macro-F1, far above the earlier
~0.23 cited in session notes. To distinguish protocol effects from stale results,
a follow-up control executed `HEAD:scripts/tabular_vs_trees.py` with the unchanged
identity/additive model and original defaults (three seeds, 100 epochs).
This control was added after observing the surprising baseline; it is explanatory,
not a preregistered performance claim.

| Dataset | Previous protocol, current model | Corrected protocol, current model |
| --- | ---: | ---: |
| diabetes | 59.6934 | 58.9489 |
| balance_scale | 0.8621 | 0.8440 |

The old protocol also succeeds on balance_scale, so the ~0.23 collapse does not
reproduce in either control. The old artifact's exact provenance has not been
reconstructed. Do not attribute that large jump to these changes. The controls
also show that the split/scaling corrections alone do not provide a large lift.

## Artifacts and verification

- Runner: `scripts/basic_numeric_experiments.py`; it refuses to overwrite an
  existing experiment directory.
- Logs, per-job JSON, summary and initial protocol:
  `logs/basic_numeric_experiments/` (git-ignored).
- NOx outputs also include exact split indices, per-epoch histories and predictions.
- `legacy_control_protocol.json`, `legacy_diabetes.json`, and
  `legacy_balance_scale.json` preserve the follow-up control.
- All 15 planned jobs completed without failures. All 120 NOx RMSEs were
  recomputed from saved predictions; split indices matched exactly. All static
  variants had identical per-seed split counts and tree scores. Runner lint passes.

A sensible next decision is to keep the simple default, and—if pursuing the small
fusion signal—confirm it on a fresh fixed set before committing further compute.
No industrial kill verdict or production default was changed.
