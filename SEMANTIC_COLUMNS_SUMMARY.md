# Feature/Column Identity + Semantic Columns — Findings & Changes

Summary of the 3W benchmarking effort and the column-identity work, written so the
same fixes can be ported to **TREA-C**. Dated 2026-06-26.

## TL;DR

1. **The row encoder was permutation-invariant over columns.** In the default path
   each feature token was just `[value, mask]` pushed through a *single shared*
   `Linear(2 → d_model)`, with no positional/feature encoding and mean pooling.
   Result: `pressure=5` and `temperature=5` produce identical tokens — the model
   could not tell which sensor a value came from. For heterogeneous tabular series
   this is fatal.
2. **Fixing feature identity is the single biggest lever.** Enabling a learned
   per-feature embedding took macro-F1 from **0.464 → 0.678** on 3W (10-class,
   class-imbalanced), and turned previously-missed fault classes (2, 9) from
   ~0 recall to ~0.97–0.99 recall.
3. **"Triple encoding" (value + mask + column) was only 2/3 implemented.** The
   "column" leg (semantic identity from the column *name*) did not exist — the
   `use_column_embeddings` path was a stub: `nn.Embedding(num_features, 1)`, a single
   learned scalar by index, and it *ignored* the `column_names` it required. We
   implemented real semantic column embeddings (frozen text encoder over column
   descriptions + learned projection, added to each feature token).

## The three identity mechanisms (now selectable)

| Mode | Flag | Mechanism | Use |
|---|---|---|---|
| None (old default) | — | shared `Linear([value,mask])`, mean pool → bag of features | the broken baseline |
| Index identity | `use_feature_embeddings` | learned `[num_features, d_model]` added per token | strongest single-dataset fit |
| Semantic identity | `use_semantic_columns` | `proj(frozen_text_emb(description))` added per token | transfer across schemas |

They are **additive and combinable** — turn on both for a hybrid (semantic backbone
+ learned per-column residual), which is the recommended shape for a broader /
multi-dataset model: semantic gives zero-shot transfer, the index residual recovers
single-dataset flexibility when fine-tuning.

## Results on 3W (macro-F1, the metric that matters under imbalance)

| Variant | Accuracy | **Macro-F1** | Balanced acc | Notes |
|---|---|---|---|---|
| Baseline (no identity) | 0.888 | **0.464** | 0.443 | overfit by epoch 3; rare faults → "normal" |
| Index identity | 0.926 | **0.678** | 0.835 | classes 2/9 recovered; stopped overfitting |
| Semantic columns | _pending_ | _pending_ | _pending_ | run in progress; expect ≈ index on a single data-rich dataset |

Accuracy is misleading here: classes 0+4 are ~91% of windows, so a model can score
0.89 accuracy while completely missing the actual fault types. **Always report
macro-F1 / balanced accuracy.**

Expectation for semantic on a single dataset: roughly *comparable to* index identity,
possibly a touch lower (it's a regularized/constrained form of identity), but it may
help the data-starved classes where index identity has nothing to learn from. Its real
payoff is **transfer**, which needs a second dataset to demonstrate (see below).

## How semantic columns are implemented (port this to TREA-C)

- `utils/w3_columns.py` — maps cryptic 3W codes to natural-language descriptions
  (`P-PDG` → "pressure at the permanent downhole gauge"). Generic text encoders get
  no signal from raw codes; descriptions are what carry meaning.
- `trea_r/models/semantic_columns.py` — `SemanticColumnEmbedder`:
  - encodes the F descriptions **once** with a frozen sentence encoder
    (`sentence-transformers/all-MiniLM-L6-v2`, 384-d), registered as a buffer;
  - a learned `Linear(384 → d_model)` projects into model space;
  - `forward()` returns `[num_features, d_model]`.
- `trea_r/models/row_encoder.py` — accepts an optional `feature_embedding [F, d_model]`
  and adds it to each projected feature token (broadcast over batch/time).
- `trea_r/models/trea_r.py` — `use_semantic_columns` + `column_descriptions`; builds
  the embedder and passes its output into the row encoder.
- `utils/w3_datamodule.py` — exposes ordered `feature_names` so descriptions line up
  with the data's feature axis.

Encoded space is meaningful: cos(P-PDG, P-TPT) = 0.73 (two pressures cluster) vs
cos(P-PDG, ESTADO-DHSV) = 0.46 (pressure vs valve state).

**Caveat for a broad model:** transfer quality rides entirely on description quality.
Raw cryptic column codes won't transfer well; you need curated descriptions or an
LLM step to expand codes into descriptions.

## Other fixes worth porting to TREA-C

- **NaN mask is destroyed by the loaders.** Both W3 loaders do
  `np.nan_to_num(..., nan=0.0)` *before* scaling (`utils/w3_dataset.py`), so the model's
  mask channel (`row_encoder.py` `create_triple_encoding`) never sees a real NaN.
  TREA-R's headline missing-data handling is therefore untested on 3W. **Not yet fixed
  here** — preserve the mask: impute/scale the value channel but pass NaN-derived
  masks through. TREA-C likely shares this loader heritage and the same bug.
- **Class-weight bug** (`examples/train_w3.py`): `compute_class_weights` counted the
  `null`-class rows, producing an 11-long weight vector for a 10-class model and
  crashing CrossEntropyLoss. Fixed by filtering null targets before weighting.
- **Per-class metrics callback never printed** (`PerClassMetricsCallback`): it clears
  its prediction buffers in `on_validation_epoch_end`, which fires at the end of the
  `trainer.validate()` it calls in `on_train_end`, so it always reads empty. Use the
  standalone `examples/eval_3w.py` instead.
- **CUDA grid limit in the row encoder.** It batches attention over `B*T`; with
  `batch*seq ≥ 65536` the attention kernel throws `invalid configuration argument`.
  Keep `batch_size * sequence_length < 65536` (e.g. batch 128 × seq 256 = 32768).

## What's next

- Finish the semantic-columns single-dataset run; fill in the table.
- **NaN-mask preservation** as a separate run (isolates that fix).
- **Baselines**: LightGBM on windowed summary stats + a vanilla transformer over
  `[T, F]` — to test whether the dual-stage row encoder actually beats simpler models.
- **Transfer test** (the real validation of semantic columns): train on one schema,
  evaluate on another with renamed columns. 3W ↔ Tennessee Eastman share
  pressure/temperature/flow concepts and are a natural pair.

## Files

New: `examples/prepare_3w_data.py`, `examples/eval_3w.py`, `examples/train_3w_full.py`,
`trea_r/models/semantic_columns.py`, `utils/w3_columns.py`.
Modified: `trea_r/models/trea_r.py`, `trea_r/models/row_encoder.py`,
`utils/w3_datamodule.py`, `examples/train_w3.py`.
