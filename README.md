# TREA — Triple-Encoded Attention for Industrial Time Series

TREA is the **consolidated** home for the IndustryTransformer line of work: attention-based
neural networks for industrial tabular time series, with native missing-value handling,
column-name semantics, and self-supervised pretraining for **label-efficient** learning and
**cross-dataset transfer**.

It replaces four parallel prototypes — TREA-C, TREA-R, TabNCT, Hephaestus — which are being
archived (made private). The name drops the `-R`/`-C` suffix on purpose: the row vs column
split is merged back into one project.

## Why this exists (the thesis)

On **fully-labeled** industrial benchmarks, gradient-boosted trees / random forests are hard
to beat — on 3W, RF scores 0.92 macro-F1 vs ~0.81 for the best deep variant, at a fraction of
the compute. Architecture tinkering has a low ceiling there.

The defensible edge of a deep model is the regime trees **structurally cannot** touch:
- **Label efficiency** — self-supervised pretraining on abundant *unlabeled* sensor data, then
  fine-tune on few labels (early evidence: beats XGBoost until XGBoost has ~80% of the labels).
- **Cross-dataset transfer** — semantic column embeddings let a model trained on one sensor
  schema carry over to another, which index-based models cannot do.

So TREA is optimized for the **expensive-label / multi-dataset** setting, and is benchmarked
honestly against tree baselines, with **macro-F1 / balanced accuracy** (never raw accuracy).

## Status

- **Seeded from TREA-C** (the most mature base: triple-encoded patch transformer, BERT/frozen-BERT
  semantic column embeddings, multi-dataset training, SSL objectives, and a multi-seed
  non-inferiority benchmark suite). See `docs/trea_original_README.md` for the inherited API.
- **In progress / to harvest** (see `docs/CONSOLIDATION_PLAN.md`):
  - TabNCT's first-class intra-row + inter-row attention and column-name-as-tokens.
  - Hephaestus's numeric projection (only if it earns it in an ablation).
  - The bug fixes catalogued in `SEMANTIC_COLUMNS_SUMMARY.md` (NaN-mask preservation,
    class-weight null filter, macro-F1 reporting, the `B*T < 65536` attention limit).

## The one experiment that matters next

A **label-efficiency curve** (RF vs from-scratch vs pretrained) on 3W *and* a turbine dataset.
The crossover point — where the pretrained model overtakes trees as labels shrink — is the thesis.
If pretraining doesn't beat RF at low label counts on ≥2 datasets, the architecture question is
moot and the semantic-column idea should ship on a simpler backbone.

## Docs

- `docs/CONSOLIDATION_PLAN.md` — why four repos became one, and the harvest/decision plan.
- `SEMANTIC_COLUMNS_SUMMARY.md` — column-identity findings + the catalogued bug fixes.
- `docs/trea_original_README.md` — inherited TREA-C API/usage reference.

## Note on package naming

The Python package is still `trea/` (inherited). Renaming `trea` → `trea` is a tracked
follow-up refactor; deferred so imports don't break during consolidation.
