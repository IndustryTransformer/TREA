# NOx: official FT-Transformer and TabM comparison — 2026-09-16

## Question and protocol

Can established supervised neural models close the same-period NOx gap on exactly
the rows and sensor inputs used in the encoding diagnostics? This is an exploratory
comparison on a repeatedly inspected dataset, not a new industrial kill gate.

- Same nine sensors and fixed seed-42 split: 2,219 fit, 554 validation, 4,439
  historical holdout (2011–2013), 14,542 future-year rows (2014–2015).
- Primary endpoint: historical holdout RMSE in original NOx units. Future-year
  RMSE is a separate frozen-model drift check. A gap within 2% of XGBoost is a
  practical proximity threshold, not statistical equivalence.
- Feature and target standardization use only fit rows. TabM piecewise-linear
  embedding bins also use only fit rows. No CO/year features, pretraining,
  unlabeled-data adaptation, or rolling retraining.
- Official packages: `rtdl-revisiting-models==0.0.2`, `tabm==0.0.3`,
  `rtdl-num-embeddings==0.0.12`. See the
  [FT-Transformer package](https://github.com/yandex-research/rtdl-revisiting-models/tree/main/package)
  and [TabM package](https://github.com/yandex-research/tabm).
- Four presets per family (FT-Transformer, TabM, TREA, XGBoost), trained with seed 0.
  Select the lowest validation RMSE, then evaluate that preset with seeds 10/11/12.
  Tuning runs never infer on either held-out split. All selections are written
  before any held-out evaluation. This matches trial counts, not FLOPs or wall time.
- Neural training: batch 256, maximum 300 epochs, patience 30, restore best
  validation checkpoint, two CPU threads per fit. FT-Transformer and TabM use
  their reference AdamW conventions. TREA retains its AdamW betas, parameter
  groups, 5% warmup, and gradient clipping. This compares practical recipes, not
  an architecture-only intervention.
- TabM uses 32 internal members: optimize the mean of individual-member losses,
  then average predictions at validation/test time. This is distinct from an
  additional ensemble across the three evaluation seeds.
- Every family includes its reference/current configuration among four presets.
  FT also explores smaller models and learning rates; TabM explores smaller
  models and piecewise-linear embeddings; TREA explores shallower models and
  lower dropout; XGBoost explores tree depth and boosting rounds.
- Prior TREA/XGBoost results remain useful fixed-configuration controls. Neural
  budgets here differ from the prior 100-epoch, batch-512 diagnostic.

All exact presets, versions, split indices, targets, histories, selected settings,
and predictions are saved under `logs/nox_reference_models/` (git-ignored).

## Reproduction

```sh
uv sync --group dev
MPLCONFIGDIR=/tmp/trea-mpl .venv/bin/python -m scripts.nox_reference_models \
  --output logs/nox_reference_models_repeat > logs/nox_reference_models_repeat.log 2>&1
```

The runner refuses to overwrite an existing output directory. Implementation:
`scripts/nox_reference_models.py`. Focused tests:
`tests/test_nox_reference_models.py`.

## Results

Version-controlled protocol, summary, and verification:
[`results/nox_reference_models.json`](results/nox_reference_models.json).

**Both reference models reach XGBoost-level same-period performance.** TabM is
1.10% below XGBoost's RMSE and FT-Transformer is 1.03% above it. Both meet the
registered 2% practical-proximity threshold. Retuned TREA remains 3.82% above
XGBoost as a single fit, but an additional three-seed ensemble check closes that
gap too.

Mean ± sample SD over three initialization seeds, lower is better:

| Model | Historical holdout RMSE | Gap vs XGBoost | Future-year RMSE |
| --- | ---: | ---: | ---: |
| TabM | **4.539 ± 0.032** | -1.10% | 14.426 ± 0.402 |
| XGBoost | 4.589 ± 0.046 | reference | **12.574 ± 0.126** |
| FT-Transformer | 4.636 ± 0.037 | +1.03% | 12.983 ± 0.583 |
| TREA, retuned | 4.765 ± 0.102 | +3.82% | 13.203 ± 0.412 |

SD here measures initialization variability on one fixed split, not uncertainty
across datasets or train/test splits. Differences of around 1% do not establish
a general superiority claim.

Selected configurations:

| Family | Selected preset | Seed-0 validation RMSE | Evaluation best epochs |
| --- | --- | ---: | --- |
| FT-Transformer | 2 blocks, width 128, attention dropout .1, FFN dropout .05, lr .0003 | 4.728 | 74 / 48 / 62 |
| TabM | 2 blocks, width 256, dropout .1, lr .002, 32 members, no numerical embeddings | 4.657 | 228 / 200 / 143 |
| TREA | 2 layers, width 128, dropout 0, lr .0003, product+bias, first input norm skipped | 4.666 | 31 / 70 / 33 |
| XGBoost | 800 trees, depth 6, lr .03, subsample/column sampling .8 | 4.752 | fixed tree count |

All neural evaluation fits early-stopped before the 300-epoch cap. The selected
TabM tuning fit reached the cap (best epoch 298); this bounded search should not
be mistaken for exhaustive tuning. Selecting on one seed can also be noisy:
TREA's seed-0 validation score was better than its three evaluation-seed scores
on average.

For context, the previous fixed-setting TREA default scored 4.979 and XGBoost
4.619. TREA improves 4.3% with this combined change in model size, regularization,
learning rate, batch size, and training budget. The experiment does not isolate
which of those changes caused the improvement, and production defaults were not
changed by this comparison.

## Additional ensemble diagnostic

After completing the registered comparison, average **all three** evaluation-seed
predictions with equal weights for each model. No seed exclusion or weight tuning:

| Three-seed prediction average | Historical holdout RMSE | Future-year RMSE |
| --- | ---: | ---: |
| TREA | **4.519** | 13.011 |
| TabM | 4.529 | 14.402 |
| FT-Transformer | 4.541 | 12.877 |
| XGBoost | 4.566 | **12.555** |

TREA's average reduces historical RMSE by 5.2% relative to its mean single-fit
RMSE and falls 1.04% below the equally ensembled tree. This is evidence that
prediction variation between TREA fits is still consequential and that matching
trees is attainable here when extra inference cost is acceptable. This was an
additional diagnostic, not the pre-registered primary endpoint. It neither
overturns the older temporal ensemble finding nor establishes performance on a
new dataset. TabM's three-seed average combines three already internally ensembled
models (32 members each).

## Interpretation and verification

- The neural reference gap was real and testable: FT-Transformer and TabM reach
  the tree's level on these exact labels and inputs, without pretraining.
- Both selected references and TREA use two blocks/layers. More depth is not
  needed for competitive performance in this diagnostic.
- Plain-input TabM won its validation selection; more elaborate numerical
  embeddings were not necessary here.
- TREA's remaining single-fit deficit is not evidence of an inability to
  represent the solution. Its three-fit average is competitive. Optimization,
  generalization, and prediction variance remain plausible targets; this is not
  a causal isolation of attention, pooling, normalization, or the head.
- Future-year results tell a different story: XGBoost beats all three selected
  neural recipes, and TabM has the largest drift penalty. Periodic retraining was
  not tested.
- All 28 fits completed. Recomputed all 80 individual-fit RMSEs from saved
  predictions, verified exact historical split equality/disjointness, no held-out
  inference in tuning, lowest-validation candidate selection, best-checkpoint
  restoration, and summary means. Saved `verification.json` includes ensemble
  diagnostics and epoch counts. Source/data hashes are in `provenance.json`.
- Three focused tests pass, covering the independent TabM member loss and
  validation-only model selection. All 12 neural presets passed forward/backward
  smoke checks. Ruff and diff-whitespace checks pass.

Evaluation fit plus scoring averaged approximately 26 s for FT-Transformer,
116 s for TabM, 29 s for TREA, and 0.35 s for XGBoost. These were concurrent CPU
jobs and are contextual runtime measurements, not an isolated throughput benchmark.
