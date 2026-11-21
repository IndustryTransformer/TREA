# 3W calibration and context follow-up — development evidence

The goal is still active. Neither consumed test has been reevaluated or used to
select new thresholds. The last frozen confirmation remains TREA ensemble AUC
0.96669 / macro-F1 0.76731, below the stronger weighted XGB comparison overall.

## Calibration did not supply an easy fix

Using saved validation predictions, fit decision rules on three source-stratified
group folds and predict the fourth. Four folds keep each well together. These are
diagnostics, not independent model tests: the base checkpoints previously used
the same validation population for early stopping and selection.

TREA cross-fitted macro-F1:

| Policy | Macro-F1 |
|---|---:|
| Ordinary macro-F1 threshold | 0.85042 |
| Class-balanced threshold | 0.76954 |
| Worst-prior threshold over fixed priors 0.1/0.3/0.5/0.7/0.9 | 0.76942 |
| Group-balanced threshold | 0.80363 |
| Separate source thresholds | 0.81008 |
| Source-specific worst-prior thresholds | 0.77833 |
| Source-aware, group-weighted sigmoid calibration | 0.56798 |

Several physical wells have opposite failure modes: some normal windows are
almost all flagged, whereas faults on another well are almost all missed. This
does not support treating the remaining problem as a single global threshold
offset. Robust-prior policies could behave differently under a new mixture, but
these results are not permission to tune the consumed test.

## Source specialists did not improve the selected TREA recipe

Each seed uses the exact same 200,000 sampled training rows, partitioned between
a physical-well expert and simulation expert. Both neural and XGB arms receive
this treatment. Scaling, class balancing and checkpoint selection become
source-specific; this is a joint recipe comparison, not an isolated causal
estimate of negative transfer.

| Specialist model | Mean single-fit AUC | Three-fit pipeline ensemble AUC | Ensemble macro-F1 |
|---|---:|---:|---:|
| TREA | 0.89724 | 0.89824 | 0.82106 |
| TabM PLE | 0.84099 | 0.84304 | 0.69554 |
| XGB | 0.80900 | 0.81024 | 0.73473 |

There are two experts per seed, six trained experts across seeds 42,43,44.
Each input is routed to its own source expert; this is not an average of all six
experts' probabilities. Joint TREA's matched validation AUC was 0.92341, so the
specialist recipe was not promoted.

## Past-only context is a more promising representation

Retain the original eight summary blocks and append four per-sensor blocks:
mean minus first earlier observed mean, mean minus past-only exponential mean,
std divided by past-only exponential std, and reference-available mask. The
exponential update rate is fixed at 0.1. No future rows or other files enter the
history calculation. Tests check prefix invariance, future perturbations, missing
references and file isolation; raw file timestamps are checked for monotonicity.

This is explicitly a **longer-context task**, not an isolated 96-step-window
comparison. Both neural and XGB models receive identical history and sampled
training indices. On internal validation:

| Cached-history model | Mean single-fit AUC | Three-fit ensemble AUC | Ensemble macro-F1 |
|---|---:|---:|---:|
| TREA | 0.91273 | 0.92363 | 0.81308 |
| TabM PLE | 0.93986 | 0.94295 | 0.83746 |
| XGB | 0.90919 | 0.91124 | 0.80275 |

The corresponding joint window-only TabM ensemble, with the same group loss
power 0.5, scored AUC 0.90598. Window-only XGB scored 0.85942. The promising
context result belongs to TabM, not evidence that TREA attention was fixed.

### Raw-sensor history verification before outer scoring

The existing window cache excludes fully unlabeled windows. An independent
fixed-validation audit found 186,636 raw full windows versus 173,464 cached
windows: 13,172 omitted windows. Every omission belongs to a **37-window prefix**;
there are no interior omissions or out-of-range median labels in this audit.

The new reconstruction reads sensor columns first, builds every full 96-step
window and its history using a fixed warm-up of 37 physical-well windows (zero
for simulations), and only then reads annotations to select scored examples.
The warm-up rule comes from the fixed internal-validation audit, not outer data.
Tests show that changing previous annotation availability cannot change a later
input, and that future sensors cannot change earlier inputs.

All four robustness folds have now passed exact raw/cache equivalence:

| Fold / split | Scored windows | Raw sensor windows | Maximum feature difference |
|---|---:|---:|---:|
| 0 / train | 285,243 | 299,895 | 0 |
| 1 / train | 254,320 | 261,424 | 0 |
| 2 / train | 254,238 | 272,405 | 0 |
| 3 / train | 282,821 | 301,950 | 0 |
| Every fold / internal validation | 173,464 | 186,636 | 0 |

Both the eight base summaries and causal-history blocks match exactly. All four
saved verification records pass their configuration and source-hash checks.
No outer data has been opened to investigate this issue. Folds 0 and 2 have
finished all candidates; folds 1 and 3 are training. All fits must finish before
selection is frozen and any outer scoring begins.

The one-shot outer evaluator is implemented. It verifies frozen artifact hashes
and reproduces every selected checkpoint's saved validation probabilities across
all four folds before the first outer read. An integration test has reproduced
six real checkpoint predictions (three window-TREA fits and three causal-XGB
fits) using training/internal-validation inputs only. A failed last-fold check
is unit-tested to prevent any outer-data access. Thresholds remain fitted solely
on internal validation; pooled classification metrics use fixed fold-specific
decisions. Paired uncertainty resamples whole groups within fold/source strata.

## Fixed robustness study now training

Membership and recipes were frozen before its fits. Four outer folds partition
the current training groups: each outer fold has five physical wells and
137–138 total groups. Each fit uses 15 physical training wells and the fixed
eight-well internal validation. Both consumed test partitions—370 groups,
including 12 physical wells—are excluded entirely.

For each fold, compare window-only TREA, causal-context TREA, causal-context
TabM, window-only XGB and causal-context XGB, each with seeds 42,43,44. Select
one neural and one tree recipe using internal-validation ensemble AUC, then
freeze thresholds and artifact hashes. All candidates/all folds must finish
and pass provenance/eligibility checks before outer scoring. No outer-driven
refitting or threshold tuning is permitted.

This is a development-population robustness study, not a newly acquired
external dataset: outer groups have historical fitting exposure. Report each
fold, pooled out-of-fold scores, single fits, ensembles, physical/simulation
breakdowns and grouped uncertainty separately. A strong validation point or an
interval crossing zero does not establish completion of the goal.

Plan/membership: `docs/results/3w_grouped_robustness_v1/`.
Training artifacts: `logs/benchmark_3w_binary/robustness_v1/`.
Exact development bundle: `docs/results/3w_development_followup.json`.
Eligibility audit: `docs/results/3w_history_eligibility_audit.json`.

77 focused and SingleRow regression tests pass. Scoped Ruff and
`git diff --check` pass. Prior frozen model code/artifacts and user edits remain
preserved. No commits or staging were performed. Concurrent-run timings are not
controlled speed measurements.
