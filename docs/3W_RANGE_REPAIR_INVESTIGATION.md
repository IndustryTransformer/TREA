# 3W range-repair continuation — 2026-09-17

## Status

The previous clipped-data investigation remains a historical record. This
continuation found and repaired a separate raw-input information-loss problem,
then refitted models under a new, explicitly frozen grouped protocol. The
current-stage confirmation result clears the AUC milestone but does **not**
complete the objective: TREA's three-fit ensemble reaches AUC **0.96669** with
macro-F1 **0.76731**, versus standard engineered XGB **0.94705 / 0.78436** and
the stronger group-weighted XGB **0.98353 / 0.86207**. The goal remains active.

## What the audit found

The old loader clipped every sensor to ±1,000,000 before normalization. Global
normalization preserves the magnitude information it receives, but cannot
recover variation already destroyed by clipping. In the former training data:

| Sensor | Finite readings | Beyond the old clipping limit |
|---|---:|---:|
| P-ANULAR | 16,110,474 | 15,711,792 |
| P-MON-CKP | 40,562,726 | 37,356,721 |
| P-TPT | 44,675,596 | 43,273,091 |

Sampled medians for several pressure sensors were around 13–21 million.
The same pipeline clipped extreme artifacts such as −1.180116e42 to an ordinary
finite −1,000,000 rather than representing them as missing. P-PDG contained
305,679 readings beyond magnitude 1e12; T-PDG contained 872,003.
Exact counts, sampled quantiles, and frequent extreme values are recorded in
`docs/results/3w_training_raw_audit.json`.

`ThreeWDataset(value_policy="preserve_range")` now masks nonfinite readings or
absolute values above 1e12 before float32 conversion, otherwise preserving range.
The cutoff comes from the training-only audit, not current-stage test metrics.
It is a conservative numerical-artifact rule, not a complete unit-aware physical
quality specification. Other implausible readings may remain. The default
`legacy_clip` is unchanged and tested against the original transformation.

## Protocol and exposure history

The earlier test is excluded entirely. A fixed seed, 20260918, selected eight
physical wells plus 20% of simulations from the former training groups for
confirmation, using identifiers only. The earlier internal validation is retained.

| Partition | Groups | Physical wells | Windows |
|---|---:|---:|---:|
| Current training | 549 | 20 | 358,874 |
| Internal validation | 230 | 8 | 173,464 |
| Confirmation | 140 | 8 | 109,640 |
| Excluded earlier test | 230 | 4 | 113,442 |

Physical group IDs aggregate all files from the same identifiable well.
Simulation groups are individual files, not proven independent generators.
Frozen inventory hashes, canonical file order, complete partition coverage, and
pairwise group disjointness are checked. The new training groups alone determine
sensor selection and normalization. Test signals are not loaded by development
or selection scripts.

Important historical exposure: these confirmation groups were used for fitting
in the previous investigation and in the raw-scale audit. Four preliminary
range-preserving fits also completed before an interruption reached them; their
results were first read only after group membership was frozen. Those fits are
excluded from this comparison. All reported current-stage models were freshly
initialized and trained without the confirmation groups. Thus the current fits
are well-disjoint, but this is not a wholly untouched external dataset or a
pristine replication of every earlier research decision.

The membership and exposure record is
`docs/results/3w_range_confirmation_protocol.json`. It must not be changed.
Checkpoint selection and classification thresholds use internal validation only.
As before, every model uses the same 200,000 sampled training windows per seed;
seeds 42,43,44 change fitting, not group membership. Both independent fits and
explicit arithmetic probability ensembles are reported separately.

## Evidence-driven changes

1. Preserve sensor range and mask extreme invalid readings, behind a compatibility flag.
2. Reduce dominance by long wells using per-window weights proportional to
   group count raised to −0.5 or −1. Normalize separately within physical and
   simulation sources, preserving each source's total weight. Tree and neural
   models can use the same weights; existing class balancing remains separate.
3. Recheck exact eight-summary neural models, variation-only models, direct-stat
   attention, and stats-only attention under the repaired protocol.
4. Test four additional scale-relative summaries while retaining all original
   summaries: relative dispersion, standardized range, delta, and half-window
   trend. They did not improve the selected TREA model.
5. Test the existing piecewise numerical encoder with Hadamard fusion in TREA.
   It did not improve the seed-42 probe.
6. Test dropout 0.1, keeping other TREA settings fixed, because later epochs
   frequently improved training loss while worsening validation. Its three-fit
   validation ensemble underperformed zero dropout.

These results do not attribute performance to model size alone. The selected
TREA still uses two layers, width 128, zero dropout, learning rate 3e-4,
quantile-transformed exact engineered summaries, square-root class balancing,
and square-root inverse-group weighting. It learns interactions through
`SingleRowClassifier`, not the original raw temporal transformer.

## Internal-validation results

| Recipe | Single-fit mean AUC | Three-fit probability ensemble AUC | Ensemble macro-F1 |
|---|---:|---:|---:|
| TREA, group power 0.5 | 0.90269 | 0.92341 | 0.85759 |
| TabM PLE, group power 1 | 0.90641 | 0.90778 | 0.83171 |
| TabM PLE, group power 0.5 | 0.90339 | 0.90598 | see exact JSON |
| TREA, group power 0.5, dropout 0.1 | 0.87798 | 0.90406 | see exact JSON |
| RF, group power 0.5 | 0.89531 | 0.89522 | 0.81301 |
| XGB, group power 0.5 | 0.85336 | 0.85942 | 0.73040 |
| XGB, original window weighting | 0.80180 | 0.81547 | 0.68017 |
| Naive XGB, original window weighting | 0.73881 | 0.74169 | 0.67425 |

TREA's single-fit AUCs are 0.91485, 0.92282, 0.87040: ensembling helps, but seed
variation remains material. Its ensemble scores 0.82466 on physical validation
wells versus 0.98206 on simulation windows. XGB's selected ensemble scores
0.63699 and 0.99279, respectively. Overall metrics therefore conceal a large
physical-versus-simulation gap.

Seed-42 probes: direct-stat attention 0.87784, stats-only 0.89305,
relative-summary TREA 0.85755, piecewise-encoded TREA 0.82681. These are screening
results, not three-seed estimates. Variation-only TabM was promising at seed 42
(0.90834) but its three-fit mean fell to 0.81098, ensemble 0.86082. That route
was rejected rather than selecting the favorable seed.

All validation results include model-selection optimism. Validation macro-F1
also optimizes its threshold on that same partition. The exact development
ledger is `docs/results/3w_range_development.json`.

## Frozen selection and confirmation

Before opening current-stage confirmation signals, `freeze_3w_range.py` saved
selected recipes, source directories, checkpoint/preprocessing/prediction
hashes, and protocol/code hashes in `docs/results/3w_range_selection.json`.
It selected the best complete three-fit ensemble per family by validation AUC.
The primary neural recipe is TREA, with TabM as a secondary comparison. Both
validation-selected weighted XGB and the original unweighted XGB are retained;
the latter is not silently replaced by a weaker baseline.

A single equal TREA/TabM probability mixture scored 0.91805 on validation,
below TREA's 0.92341, and was not selected for confirmation. No mixture weights
were optimized. Evaluation checks all frozen hashes and reproduces every saved
validation prediction before reporting confirmation results. It refuses to
overwrite an existing evaluation.

### Frozen confirmation results

| Model | Single-fit AUC mean ± SD | Single-fit macro-F1 mean | Three-fit ensemble AUC | Ensemble macro-F1 |
|---|---:|---:|---:|---:|
| Selected TREA | 0.93990 ± 0.02755 | 0.70575 | 0.96669 | 0.76731 |
| Secondary TabM PLE | 0.96756 ± 0.00701 | 0.76833 | 0.97256 | 0.75437 |
| Selected weighted XGB | 0.98118 ± 0.00370 | 0.88242 | 0.98353 | 0.86207 |
| Standard engineered XGB | 0.94424 ± 0.00723 | 0.76985 | 0.94705 | 0.78436 |
| Naive XGB | 0.91467 ± 0.01513 | 0.77648 | 0.91904 | 0.82312 |
| Weighted RF | 0.98879 ± 0.00094 | 0.78122 | 0.98971 | 0.81119 |

All thresholds remain those frozen from validation. TREA's independent AUCs
are 0.94310, 0.90467, 0.97193; the corresponding macro-F1 values are 0.79444,
0.56058, 0.76225. The favorable ensemble must not be presented as a single fit.
TabM was the preselected secondary model, not a replacement primary selected
after seeing test scores. The old historical AUC 0.9474 is a milestone, not a
substitute for contemporaneous same-protocol tree comparisons.

Against the stronger weighted XGB, TREA's ensemble AUC difference is −0.01684.
The 500-replicate paired whole-group bootstrap 95% interval is
[−0.03123, +0.00095]. Crossing zero does not prove equivalence. TabM's difference
is −0.01097, interval [−0.06461, +0.00162]. Only eight confirmation groups are
physical wells, limiting real-world uncertainty conclusions.

### Remaining failure: threshold and source transfer

Physical-only TREA AUC is 0.96363, but physical macro-F1 is just 0.56583. Weighted
XGB has physical AUC 0.93331 and macro-F1 0.69182. Across the full confirmation
set, TREA flags 23,136 normal windows as faulty while missing 277 fault windows.
At its validation-selected threshold 0.59145, it is an overly eager detector
on these groups. The predetermined 0.5 threshold is not a fix: macro-F1 is 0.74884.

The anomaly share shifts from 81.2% in validation to 53.9% in confirmation, and
the mixture of physical versus simulation windows also changes. This is evidence
to investigate calibration and source robustness, not proof that prevalence
alone explains the failure. No threshold was fitted or optimized on confirmation
labels. The ranking/classification divergence is an important remaining issue,
not something to hide by reporting AUC alone.

The primary future avenue is grouped out-of-fold calibration/threshold-transfer
analysis within development data, followed by source-aware representation or
training experiments if warranted. Confirmation is now consumed: any future
results on it must be labeled exploratory, not another pristine test. A stronger
confirmation claim requires genuinely independent data or a clearly disclosed
new evaluation design. Existing frozen results and selection must remain intact.

The complete strict-JSON bundle is `docs/results/3w_range_neural_results.json`;
undefined one-class metrics are null. This combines the raw audit, all exact
development results/configurations, selection hashes, confirmation manifests,
per-fit and ensemble results, and uncertainty estimates.

The raw clipping repair, revised training membership, weighting, and model
selection all differ from the initial investigation. The historical temporal
AUC and these confirmation AUCs therefore are **not a controlled before/after
effect size** for any single change. Within-stage tree comparisons are matched.

## Reproduction and checks

From the repository root, representative development command:

```sh
.venv/bin/python -m examples.benchmark_3w_grouped \
  --value-policy preserve_range \
  --group-manifest docs/results/3w_range_confirmation_protocol.json \
  --models single_row --seeds 42,43,44 --preprocess quantile \
  --group-loss-power 0.5 --output-dir NEW_RUN_DIRECTORY
```

Exact original run configurations remain in each run's `config.json` and the
development ledger. Do not rerun into completed result directories. Selection:

```sh
.venv/bin/python -m examples.summarize_3w_range
.venv/bin/python -m examples.freeze_3w_range
.venv/bin/python -m examples.freeze_3w_range --evaluate
```

The selection already exists after this investigation; the freeze command must
refuse to replace it. Existing tests plus new protocol/representation tests:
48 passed. Scoped Ruff and `git diff --check` pass. The user's original modified
binary benchmark and its original added test retain their initial SHA-256 hashes.
No commits or staging were performed. Some development runs shared GPU/CPU
resources, so training times are recorded but are not controlled speed comparisons.
