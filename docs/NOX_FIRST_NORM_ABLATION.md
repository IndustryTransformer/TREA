# NOx first attention normalization ablation — 2026-09-16

**Result:** bypassing the first attention-input LayerNorm rescues pure scalar
multiplication. It does not materially help product-plus-bias or the original
additive encoder. This is an encoding/normalization interaction, not evidence
that all LayerNorms should be removed.

## Protocol and intervention

Six arms, three seeds each (18 neural fits): additive, product, product_sum;
each with first attention norm kept or bypassed. All six arms were freshly run.
The existing default remains unchanged.

- Same nine-sensor NOx inputs and fixed split as prior diagnostics: 2,219 fit,
  554 validation, 4,439 historical random-holdout rows (2011–2013), and 14,542
  future-year rows (2014–2015). Seeds 10/11/12; CPU, two threads per process.
- Width 128, four layers, dropout 0.2, AdamW lr 8e-4, batch 512, 5% warmup,
  max 100 epochs, patience 12, restore best validation checkpoint.
- Only first-block `norm1` is bypassed, on its three calls for Q/K/V. First-block
  feed-forward `norm2`, per-head query/key norms, all later block norms, and all
  residual connections remain active. The first query remains column identity.
- An experiment-only forward hook returns norm1's input, preserving the parameter
  dictionary and initial states. Its affine parameters receive no gradient in the
  bypass arm; all other model parameters are initialized identically within a pair.
  This measures accuracy, not the runtime gain from physically deleting the norm.
- Historical holdout is primary. Future-year performance is a secondary frozen-model
  drift check, not a continuation gate. No pretraining, tuning, or rolling retraining.
- Protocol saved before training: at least 2% mean primary RMSE improvement and
  improvement in at least 2/3 seed pairs, separately for each encoding.

## Primary results

Original-unit historical-holdout RMSE, mean ± sample SD over three seeds:

| Encoding | Keep first norm | Skip first norm | Mean RMSE reduction | Improved seed pairs |
| --- | ---: | ---: | ---: | ---: |
| Current additive | 5.171 ± 0.181 | 5.204 ± 0.097 | -0.64% | 2/3 |
| Pure product | 6.997 ± 0.043 | 4.958 ± 0.088 | **29.14%** | **3/3** |
| Product + independent bias | 4.984 ± 0.101 | 4.979 ± 0.078 | 0.08% | 1/3 |
| XGBoost reference | 4.619 ± 0.028 | same | — | — |

Pure-product paired differences (skip minus keep): -2.0642, -1.9458, -2.1061.
It alone passes the registered diagnostic threshold. The new pure-product result
is 4.11% better than the original additive encoder, but remains 7.35% above XGBoost.
Its 0.021 RMSE advantage over product-plus-bias without first norm is small relative
to seed variation and does not establish which of those two is preferable.

## Secondary future-year results

| Encoding | Keep first norm | Skip first norm |
| --- | ---: | ---: |
| Current additive | 12.270 ± 0.834 | 12.405 ± 0.406 |
| Pure product | 12.258 ± 0.279 | 11.796 ± 0.760 |
| Product + independent bias | 13.085 ± 0.489 | 12.233 ± 0.480 |
| XGBoost | 12.520 ± 0.201 | same |

The pure-product skip arm has a lower mean future-year RMSE than XGBoost in this
small diagnostic. This does not revise old industrial verdicts or demonstrate
performance under periodic retraining. The historical and future-year metrics
answer different questions and are reported separately.

## Interpretation

For positive x, LayerNorm(x*e) is approximately LayerNorm(e), so the original
first attention branch receives little magnitude information from pure-product
tokens. Bypassing that input normalization lets projected values carry magnitude;
query/key normalization can remain active. The large, consistent improvement is
strong evidence that the first norm caused much of pure product's earlier failure.

With an independent feature bias, x*e+b changes direction as x varies, so input
normalization need not erase the value in the same way. That arm already worked
reasonably with the first norm and shows negligible primary gain from removing it.

This experiment does not isolate removing the norm on V alone from removing it on
all Q/K/V inputs. It also does not test removal of later norms or additional skip
connections. These are exploratory results on a repeatedly inspected dataset;
confirmation on fresh tables is needed before changing global defaults.

## Reproduction and artifacts

```sh
.venv/bin/python -m scripts.nox_first_norm_ablation
```

The runner refuses to overwrite its output directory. For an individual arm:

```sh
.venv/bin/python -m scripts.nox_encoding_ablation --arms identity_query \
  --fusion product --skip-first-attention-norm --threads 2 \
  --output logs/nox_product_skip_repeat.json
```

Omit `--skip-first-attention-norm` for the matched control. Results, histories,
initial-weight hashes, exact split indices, predictions, and summary are in
`logs/nox_first_norm/` (git-ignored). Main run log: `logs/nox_first_norm.log`.

Verification: all six jobs completed; all 144 recorded RMSEs recomputed from saved
predictions; split indices identical; all nine within-encoding initial-state pairs
match. Twelve focused tests pass, including a hook/gradient test confirming that
only the intended norm is bypassed and other normalization paths remain active.
Lint and diff-whitespace checks pass. No production model defaults were changed.

## Subsequent provisional default adoption

At the user's request after this experiment, single-row encoders, regressors,
classifiers, masked encoders, their Lightning wrappers, and the general tabular
benchmark now default to `fusion="product_sum"`, `first_prenormalization=False`.
The code comment identifies this as a provisional NOx-based choice. First-block
FFN normalization, query/key normalization, all later norms and residuals remain.
The bias is a separate feature lookup for indexed columns; semantic columns use
an independent projection of their name embedding so reordering/schema transfer
does not attach a bias to the wrong column position.

Restore the previous architecture (including when reconstructing old additive
checkpoints) with `fusion="additive", first_prenormalization=True`.
The general benchmark exposes `--fusion additive --first-prenormalization`.
Nonlinear value encoders such as `dual` require an explicit compatible fusion
(`additive` or `hadamard`); scalar-product fusion remains identity-only.

Historical NOx experiment runners explicitly select keep/skip normalization and
their original additive baseline, so previous reproduction commands retain their
meaning. The basic numerical-basis runner also explicitly preserves the old norm.
The result tables above are unchanged. The production default has an exact
forward-equivalence test against the experimental product-plus-bias bypass arm.
