# NOx first-block encoding ablation — 2026-09-16

**Result: replacing identity-only queries with identity-plus-value queries did
not materially close the XGBoost gap.** Historical holdout RMSE improved by only
0.36%; future-year RMSE worsened on average. Retain the production default.

## Intervention

The current encoder starts with `layers[0](col_emb, values, values)`. The alternative
uses `layers[0](values, values, values)`, where each numeric value token already
includes column identity. This changes both attention queries and the first
residual input. It does not independently isolate their contributions, replace
the shared numerical projection, or reproduce a full FT-Transformer.

The experiment uses a parameter-free forward pre-hook in
`scripts/nox_encoding_ablation.py`; production model code is unchanged.

## Fixed protocol

- Nine contemporaneous sensors: at, ap, ah, afdp, gtep, tit, tat, tey, cdp. No
  histories, engineered features, CO, year, or dummy categorical column.
- 2011–2013: 2,219 fit rows, 554 validation rows, 4,439 random holdout rows.
  Unused historical rows are not used for training or pretraining.
- 2014–2015: 14,542 future-year rows evaluated using the same fitted models.
- Split/subset seed 42; paired model seeds 10, 11, 12. Both neural arms have
  identical initial parameter hashes, minibatch orders, and training settings.
  XGBoost receives exactly the same fit rows and nine features.
- Standardization and numeric config fitted only on fit rows.
- Neural: 128 dimensions, 4 heads, 4 layers, dropout 0.2; AdamW at 8e-4,
  betas (0.9, 0.95), weight decay 3e-4 with repository parameter grouping,
  5% linear warmup, batch 512, gradient clip 1.0. At most 100 epochs,
  patience 12, restore lowest-validation-RMSE weights. All six fits stopped
  before the epoch cap. CPU, four threads, deterministic PyTorch operations.
- XGBoost: 400 estimators, depth 6, learning rate 0.05, subsample and column
  sample 0.8; no tuning or test-based selection.
- Criterion persisted before training: at least 2% lower mean historical
  holdout RMSE and improvement in at least 2/3 paired seeds.

## Results

RMSE in original NOx units; mean ± sample standard deviation across three seeds.

| Arm | Fit | Validation | Historical holdout | Future years |
| --- | ---: | ---: | ---: | ---: |
| Current identity query | 4.800 ± 0.059 | 5.343 ± 0.140 | 5.089 ± 0.049 | 12.340 ± 0.909 |
| Value query + residual | 4.696 ± 0.142 | 5.158 ± 0.057 | 5.071 ± 0.024 | 12.729 ± 0.485 |
| XGBoost | 1.375 ± 0.011 | 4.871 ± 0.054 | 4.619 ± 0.028 | 12.520 ± 0.201 |

Paired historical RMSE differences (alternative minus current):

| Seed | Difference |
| --- | ---: |
| 10 | -0.03051 |
| 11 | +0.01391 |
| 12 | -0.03823 |

The alternative wins 2/3 pairs but improves the mean by only 0.01828 RMSE
(0.359%). It fails the registered diagnostic threshold and remains about 9.8%
above XGBoost on the historical holdout. Future-year mean RMSE rises by 0.38909
(3.15%), with worse results in two pairs and better results in one.

Training and validation fit improve modestly, so the intervention is functioning;
the improvement largely fails to carry through to the primary holdout. This is
evidence against this particular first-block change being a major fix under the
tested recipe, not proof that all encoding choices are harmless.

## Limits and interpretation

One fixed subset and three initializations do not establish statistical
equivalence, robustness across sample draws, or performance on other tables.
The random historical holdout can contain temporally related observations and
is an encoding diagnostic, not evidence of deployment generalization. The future
split provides a separate drift check; the original industrial gates are unchanged.

Scores are not directly comparable with old turbine scripts: those scripts used
different row budgets/validation selection, and generated the deep config from
a broader dataframe (including extra columns), while trees explicitly selected
the nine sensors. This experiment explicitly equalizes feature access.

No pretraining was run. The small average future-year advantage of the current
arm over XGBoost here does not reopen the earlier industrial kill verdict.

## Reproduction and verification

```sh
MPLCONFIGDIR=/tmp/trea-mpl .venv/bin/python -m scripts.nox_encoding_ablation \
  --output logs/nox_encoding_ablation_repeat.json \
  > logs/nox_encoding_ablation_repeat.log 2>&1
```

The script refuses to overwrite an existing output JSON. Outputs from this run:

- `logs/nox_encoding_ablation.json`: protocol, arguments, per-epoch validation
  history, paired initialization hashes, final scores and summary.
- `logs/nox_encoding_ablation.splits.npz`: exact source row indices.
- `logs/nox_encoding_ablation.predictions.npz`: predictions in standardized
  target units (the JSON records the target scale used for RMSE).
- `logs/nox_encoding_ablation.log`: run output.

Checks passed: lint and compilation; two-epoch smoke run; paired initial weights;
alternative forward matches a literal value/value/value computation; finite
gradients; all four partitions disjoint; all 36 reported RMSEs recomputed from
saved predictions and reconstructed targets; saved split indices match exactly.
The logs directory is git-ignored; this report preserves the main results.
