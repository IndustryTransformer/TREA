# NOx scalar-product encoding — 2026-09-16

**Result:** product plus an independent feature bias improves same-period RMSE
by 3.63%, closing about one-third of the current gap to XGBoost. Pure multiplication
substantially underperforms. Neither change is promoted to the production default.

The user explicitly requested testing scalar multiplication, superseding the earlier
work-order prohibition for these optional experimental fusion modes.

## Encodings

For standardized value x, learned column-identity vector e, independent learned
feature bias b, and presence indicator p:

- Existing `additive`: e + Linear([x, p]).
- New `product`: x * e for observed values.
- New `product_sum`: x * e + b for observed values.

Both new modes use e + a dedicated learned missing vector for NaN/infinite cells.
Thus a missing value remains distinguishable from observed zero. Pure-product
observed zero is a zero value token, but the unchanged first-block identity query
still carries the column identity. These modes use learned column identities in
this experiment, not frozen semantic text embeddings. New modes require
`value_encoder=identity`; incompatible combinations fail explicitly.

The independent bias matters: adding the same e instead would give (x+1)*e,
which remains collinear before normalization. The full transformer, original
identity-query first block, and pooling are unchanged. This is a tokenizer test,
not an implementation of the full FT-Transformer architecture.

For paired initialization, both product modes allocate the same bias table;
product-only omits it in forward. Their initial state hashes match for each seed.
The existing continuous projection is unused in the product value paths. Original
fusion defaults and checkpoint structure remain unchanged when using old modes.

## Matched experiment

Same fixed NOx protocol as the basic numeric reruns: nine sensors, 2,219 fit rows,
554 validation, 4,439 historical random-holdout rows (2011–2013), and 14,542
future rows (2014–2015). Seeds 10/11/12, CPU two threads, 128 dimensions, four
layers, lr 8e-4, dropout 0.2, max 100 epochs, patience 12, validation checkpointing.
All three encodings and XGBoost were rerun. Initial protocol is saved in
`logs/nox_scalar_product/protocol.json` before training.

Primary endpoint: historical random holdout. Exploratory positive threshold:
at least 2% mean RMSE improvement and improvement in at least two seed pairs.
Future-year performance is a separate drift diagnostic, not a go/no-go gate for
this experiment. This is an already-inspected dataset, not independent confirmation.

RMSE mean ± sample SD across three seeds, original NOx units:

| Encoding | Fit | Validation | Historical holdout | Future years |
| --- | ---: | ---: | ---: | ---: |
| Existing additive | 4.925 ± 0.387 | 5.474 ± 0.123 | 5.171 ± 0.181 | 12.270 ± 0.834 |
| Product only | 6.953 ± 0.157 | 7.500 ± 0.099 | 6.997 ± 0.043 | 12.258 ± 0.279 |
| Product + independent bias | 4.433 ± 0.136 | 4.957 ± 0.099 | 4.984 ± 0.101 | 13.085 ± 0.489 |
| XGBoost | 1.375 ± 0.011 | 4.871 ± 0.054 | 4.619 ± 0.028 | 12.520 ± 0.201 |

Product-plus-bias passes the exploratory threshold: 3.63% better mean historical
RMSE, with paired differences -0.3054, +0.0353, -0.2924 (negative is better).
It remains 7.9% above XGBoost. Product-only is 35.3% worse than the current encoder.
Product-plus-bias future-year RMSE rises by 6.64% relative to the current encoder.

## Why the sum helps

The first block normalizes value tokens before attention. For positive x,
LayerNorm(x*e) is approximately LayerNorm(e), apart from its epsilon term;
the value magnitude largely disappears. An independent b changes the token
direction with x and can preserve magnitude information through normalization.

A supplementary initialization probe doubled all nine standardized inputs from
1 to 2: normalized pure-product tokens changed by 0.047% in relative L2 norm,
versus 31.66% with an independent bias. Encoder outputs changed by 0.022% versus
51.89%. This supports the mechanism under a controlled input, not a complete
causal account of trained-model performance. Probe results are saved in
`normalization_probe.json` in the run directory.

## Temporal scope

A frozen model extrapolating across years and a periodically retrained model are
different deployment policies. The latter should be evaluated with rolling
training windows, explicit label availability/delay, and a fixed retraining cadence
applied to both neural models and trees. No rolling retraining was run here.
A random holdout measures same-distribution interpolation and can contain temporal
dependence; it does not establish future deployment performance. For the current
static-table objective it remains the primary diagnostic, per the user's request.
Previous industrial verdicts are unchanged.

## Reproduction and checks

```sh
.venv/bin/python -m scripts.nox_encoding_ablation --arms identity_query \
  --threads 2 --fusion product_sum --output logs/nox_product_sum_repeat.json
```

Use `--fusion product` or `--fusion additive` for the other arms. The same flags
are exposed by `scripts/tabular_vs_trees.py`. All defaults stay additive.

Run artifacts: `logs/nox_scalar_product/`, with JSON, logs, exact split indices,
histories, saved predictions and `summary.json`. All 72 reported RMSEs recomputed
successfully from saved predictions; every split matched and product-arm initial
hashes matched. Eleven focused tests pass; full offline suite: 87 pass, one known
CPU ensemble timing failure previously reproduced with the original encoder.
Lint and diff-whitespace checks pass.
