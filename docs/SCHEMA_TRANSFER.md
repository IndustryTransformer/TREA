# Cross-schema column-name transfer (`trea.transfer`)

**Status: release gate not yet run.** Treat the class as experimental until
`scripts/schema_scaling.py` reports PASS. This page is updated with the result.

`SchemaTransferRegressor` is the shipped form of the one deep-model idea that
survived the §3 kill clause (`LESSONS.md` §3e): a model pretrained on a few
schemas gives a table with *unseen* columns a better few-shot start, because each
column's identity comes from its text description.

## Usage

```python
from trea.transfer import SchemaTransferRegressor, Table

sources = [
    Table(df_a, target="price", descriptions={"med_inc": "median household income"}),
    Table(df_b, target="strength", descriptions={"cement": "cement content kg per m3"}),
]
model = SchemaTransferRegressor()            # pinned, validated configuration
model.pretrain(sources, epochs=40)
model.save("checkpoints/schema_transfer.pt")

model = SchemaTransferRegressor.load("checkpoints/schema_transfer.pt")
model.fit(Table(df_new, target="y", descriptions=descr_new), epochs=80)
pred = model.predict(df_new_test)            # original target units
```

- Pass human-readable descriptions. Terse codes embed poorly and trigger a warning.
- Columns are matched by name at `predict`. Order does not matter. A missing
  column is treated as missing values (NaN), not an error.
- `fit` without `pretrain` is the from-scratch baseline.
- Each table is standardized on its own rows; `predict` returns original units.

## What it is and is not good for

| Situation | Expectation | Evidence |
| --- | --- | --- |
| New table, columns never seen in pretraining, ~200 labels | Better than from-scratch deep model | LESSONS §3b (5/5 held-outs) |
| Same columns under new names | No gain over index identities | LESSONS §3a |
| Beating tuned XGBoost | Usually not (lost 4/5 in §3b) | LESSONS §3b |
| More than ~4 source schemas | Gains saturate by K≈2–4 | LESSONS §3b |

Scope: numeric columns, regression. Categorical columns and classification are
untested for transfer and not supported.

## Model

Frozen MiniLM description embedding → trained `Linear + LayerNorm` → column
identity. Cell token = identity + `Linear([value, present])`. Three post-norm
transformer layers (layer 1 cross-attends identities over cells), attention
pooling, one MLP head per table. Defaults are pinned to the run that passed §3b
(`d_model=128`, 4 heads, 3 layers, dropout 0.1, additive fusion, no QK-norm). They
differ from the NOx-tuned single-row defaults on purpose; changing them needs a new
gate.

## Release gate (pre-registered 2026-11-21)

`scripts/schema_scaling.py`: K=4 sources, 200 labels, 3 source seeds × 3
few-shot seeds, 5 held-outs. PASS iff on ≥4/5 held-outs `transfer_semantic` beats
all of `transfer_index`, `scratch_semantic` and `scratch_index`. `scratch_index` is
new relative to §3b. Failing only against it means the class has no user value at
200 labels and does not ship.
