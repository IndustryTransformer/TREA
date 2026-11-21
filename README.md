# TREA — Triple-Encoded Attention for Industrial Time Series

TREA is the **consolidated** home for the IndustryTransformer line of work: attention-based
neural networks for industrial tabular time series, with native missing-value handling,
column-name semantics, and self-supervised pretraining for **label-efficient** learning and
**cross-dataset transfer**.

It replaces four parallel prototypes — TREA-C, TREA-R, TabNCT, Hephaestus — which are being
archived (made private). The name drops the `-R`/`-C` suffix on purpose: the row vs column
split is merged back into one project.

## The original thesis

On **fully-labeled** industrial benchmarks, gradient-boosted trees and random forests are
hard to beat. The bet was that a deep model has an edge where trees structurally cannot
reach:
- **Label efficiency**: self-supervised pretraining on abundant *unlabeled* sensor data,
  then fine-tuning on few labels.
- **Cross-dataset transfer**: semantic column embeddings let a model trained on one sensor
  schema carry over to another, which index-based models cannot do.

## Verdict (July 2026): the supervised deep-vs-trees line is killed

The label-efficiency curve was run against two pre-registered gates. Both failed
(`docs/LESSONS.md` §3):

- **Turbine NOx, temporal split:** the pretrained model beat the best tree at only 1 of 4
  low-label fractions, and lost to from-scratch at 1% labels.
- **3W binary normal-vs-fault, unseen wells:** RF beat the deep model on every metric
  (AUC 0.958 vs 0.845, macro-F1 0.820 vs 0.669) and trained ~33× faster.

Per the kill clause, there is no v3 protocol, new backbone, or new benchmark to
re-litigate head-to-head accuracy against trees.

## What still stands

- **Semantic column-name transfer.** Validated on 5 of 5 held-out datasets with columns
  the model never saw. This is the cross-schema capability trees lack. It should ship on
  the simplest backbone that works. XGBoost is still the absolute bar on 4 of the 5.
- **Native missing-value mask.** The [value, present]-flag encoding, with no imputation,
  works on any backbone.
- **Real-NOx pretraining, pending replication.** Updated small-TREA ensembles with masked
  pretraining beat the tree at 1/2/5% labels. This comes from one label draw and one SSL
  seed per arm, so it needs replication before anyone builds on it. See
  `docs/NOX_TEMPORAL_PRETRAINING.md`.

The GAUGE foundation-model design was never gated by §3 and is undecided. It needs its own
pre-registered gates if pursued.

Benchmarks report macro-F1 / balanced accuracy (classification) or RMSE (regression),
never raw accuracy.

## Status

- **Seeded from TREA-C** (triple-encoded patch transformer, BERT/frozen-BERT semantic column
  embeddings, multi-dataset training, SSL objectives, and a multi-seed benchmark suite). See
  `docs/trea_original_README.md` for the inherited API.
- Current provisional single-row defaults: `docs/SMALL_MODEL_DEFAULTS.md`.

## Docs

- `docs/LESSONS.md` — empirical findings, evaluation protocol, and the kill verdict.
- `docs/CONSOLIDATION_PLAN.md` — why four repos became one, and the harvest/decision plan.
- `SEMANTIC_COLUMNS_SUMMARY.md` — column-identity findings + the catalogued bug fixes.
- `docs/trea_original_README.md` — inherited TREA-C API/usage reference.

## Note on package naming

The package was renamed `treac` → `trea` to match the repo. Pretrained checkpoints saved
under the old package path may need a shim to load:
`import trea, sys; sys.modules["treac"] = trea` before `load_from_checkpoint`.
