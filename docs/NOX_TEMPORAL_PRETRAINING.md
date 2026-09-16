# Updated TREA: temporal split and masked-pretraining ablation — 2026-09-16

## Protocol fixed before the main run

Follow up the same-period reference-model result with the selected TREA recipe:
two layers, width 128, four heads, dropout 0, scalar product plus independent
feature bias, no first attention-input normalization. All other norms and
residuals remain. Supervised lr .0003, batch 256, AdamW (.9, .95), weight decay
.0003 excluding embeddings/norms/biases, 5% warmup, gradient clip 1, maximum 300
epochs, patience 30, restore best validation checkpoint.

- Only the nine NOx sensor inputs; no CO, year, or constant categorical column.
- Historical years 2011–2013, future test years 2014–2015. The last 10% of
  historical rows form a validation pool; every fitting row precedes every
  validation row. Within-year order follows the supplied CSV files, as in v2.
- Label budgets 1%, 2%, 5%, 10%, and 100% of the 22,191 historical rows. Below
  full labels, 20% of each budget is allocated to validation. At full labels,
  fit on the complete early 90%, validate on the complete late 10%.
- Fixed, nested row draws (seed 42) at every fraction; initialization/shuffle
  seeds 10/11/12 vary models, never labeled rows. Report single-fit mean ± sample
  SD and the equal-weight average of all three predictions, with no seed selection.
- Feature standardization uses the early unlabeled historical pool, available
  to all arms. Target mean/std use **only the labeled fitting rows** for the
  relevant fraction. All RMSEs and saved predictions are in original NOx units.
- XGBoost uses the previously selected 800-tree/depth-6/lr-.03/.8-sampling preset
  and fits exactly the same rows as TREA. Validation labels are reserved for
  neural checkpoint/mode selection. Include the constant labeled-fit mean as a
  deployment-valid baseline; do not substitute the future-period target mean.

Arms:

1. `scratch`: updated TREA, supervised only.
2. `pre`: masked pretraining on unlabeled 2011–2013 features.
3. `pre_target`: masked pretraining on unlabeled 2011–2015 features. This is
   explicitly **transductive**; future NOx labels never enter pretraining.
4. `xgb`: tree reference.

Each pretraining arm uses one seed-42 encoder shared across label fractions and
supervised initializations. Pretrain for 60 epochs, batch 512, lr .0008, mask 30%
of numeric cells with `-inf`, reconstruct masked cells only. Randomly reserve
10% of the respective unlabeled corpus for SSL validation; use a fixed validation
mask and select the lowest masked-cell validation MSE. Neither target labels nor
future supervised error select an SSL checkpoint.

At 1/2/5% labels, each pretrained arm compares a frozen-encoder probe (lr .001)
with low-rate finetuning (lr .00008), selecting on temporal validation only. At
10/100%, finetune at .0003. For each initialization, the fresh prediction head is
identical across scratch and pretrained modes before training.

The primary run has 15 tree fits, 15 scratch fits, 48 pretrained probe/finetune
fits, and two SSL fits. Up to three CPU jobs run concurrently, two threads each.

An additional diagnostic, specified after the scratch results and before its
own fits, uses **all labeled rows (fit + neural validation)** for XGBoost. It
retains the same frozen tree preset and three seeds, with no test-based tuning.
This matches the older benchmark's total-label convention and checks whether
withholding later-period validation rows explains the primary tree gap. Its 15
fits and predictions live in `tree_all_labels/`, with a separate protocol.
Reproduce using `python -m scripts.nox_temporal_tree_refit --output <run-directory>`.
It does not replace the primary comparison on identical fitting rows.

This is an exploratory updated-model diagnostic, **not** an exact rerun of the
historical kill criterion. It corrects feature and fit-row asymmetries in the old
scripts, uses a different supervised recipe, and measures initialization variation
on only one fixed label draw per fraction. It does not measure pretraining-seed
variance, split variance, periodic retraining, or sequence modeling. Prior
same-period model selection has also already exposed this dataset to exploration.

## Artifacts and reproduction

Runner: `scripts/nox_temporal_pretraining.py`; focused tests:
`tests/test_nox_temporal_pretraining.py`.

```sh
MPLCONFIGDIR=/tmp/trea-mpl .venv/bin/python -m scripts.nox_temporal_pretraining \
  --output logs/nox_temporal_pretraining_repeat \
  > logs/nox_temporal_pretraining_repeat.log 2>&1
```

The output directory must not already exist. Artifacts in
`logs/nox_temporal_pretraining/` include the locked protocol, exact splits, data
normalization, source/data hashes, SSL encoders and histories, mode-selection
histories, individual predictions, ensemble predictions, and summary metrics.

## Results

Version-controlled protocol, summaries, additional controls, and verification:
[`results/nox_temporal_pretraining.json`](results/nox_temporal_pretraining.json).

**The updated scratch ensemble is competitive on the year split. Masked
pretraining adds a low-label benefit at 1/2/5%, but not at 10%.** With full labels,
only the transductive pretrained ensemble improves over scratch in this run.

Primary comparison, **equal three-initialization ensembles**, future-year RMSE
in original NOx units, lower is better. XGBoost fits the same rows as the neural
models in this table:

| Historical label budget | Fit / validation rows | XGBoost | Scratch | Source-only SSL | Source + future-feature SSL |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1% | 178 / 44 | 13.102 | 12.453 | 11.895 | **11.270** |
| 2% | 355 / 89 | 12.254 | 12.014 | 11.692 | **11.516** |
| 5% | 888 / 222 | 12.775 | 11.903 | **11.492** | 11.682 |
| 10% | 1,775 / 444 | 13.009 | **12.224** | 12.290 | 12.583 |
| 100% | 19,971 / 2,220 | 13.887 | 12.896 | 13.269 | **12.228** |

The scratch ensemble wins all five matched-fit comparisons. With the additional
tree baseline that fits **all** available labels, including reserved validation
labels, the comparison is substantially closer:

| Label budget | XGBoost, all labels | Scratch ensemble | Source-only pretrained ensemble | Transductive pretrained ensemble |
| --- | ---: | ---: | ---: | ---: |
| 1% | 13.196 | 12.453 | 11.895 | **11.270** |
| 2% | 12.057 | 12.014 | 11.692 | **11.516** |
| 5% | 12.181 | 11.903 | **11.492** | 11.682 |
| 10% | 12.250 | **12.224** | 12.290 | 12.583 |
| 100% | 12.798 | 12.896 | 13.269 | **12.228** |

Relative to this all-label tree, scratch is 5.63% better at 1%, 0.36% better at
2%, 2.28% better at 5%, 0.22% better at 10%, and 0.76% worse at full labels.
The small differences at 2/10/100% should be read as comparable performance, not
reliable ranking. Reserving later-period labels explains much of the primary
tree deficit, so the additional baseline materially qualifies the first table.

## Single-fit variability

Future RMSE, mean ± sample SD over the same three initializations. These SDs do
not estimate variation across label draws, pretraining seeds, or datasets.

| Label budget | XGBoost, matched fit | Scratch | Source-only SSL | Source + future-feature SSL |
| --- | ---: | ---: | ---: | ---: |
| 1% | 13.121 ± .133 | 12.700 ± .402 | 12.568 ± 1.801 | 11.530 ± .173 |
| 2% | 12.272 ± .104 | 12.828 ± 1.291 | 12.156 ± 1.230 | 12.083 ± 1.540 |
| 5% | 12.795 ± .136 | 12.057 ± .756 | 12.048 ± 1.266 | 12.028 ± 1.700 |
| 10% | 13.027 ± .014 | 12.297 ± .856 | 12.446 ± .699 | 12.733 ± 1.132 |
| 100% | 13.898 ± .084 | 12.984 ± .667 | 13.273 ± .167 | 12.886 ± .849 |

At full labels, for example, the transductive arm's single fits score
11.906 / 13.385 / 13.368; their equal average scores 12.228. The ensemble result
must not be represented as typical single-fit performance. No seed was excluded.

## Matched random-encoder control

The initial low-label results frequently selected the frozen probe (source-only:
7/9 selections; transductive: 6/9). A second additional diagnostic therefore
reconstructs the **untrained seed-42 SSL encoder**, uses the identical head seeds,
and compares the same frozen-probe/low-rate-finetune choices on validation. This
tests whether pretraining contributes beyond freezing an arbitrary encoder.

| Label budget | Random encoder + matched adaptation | Source-only SSL | Transductive SSL |
| --- | ---: | ---: | ---: |
| 1% | 12.968 | 11.895 | **11.270** |
| 2% | 13.760 | 11.692 | **11.516** |
| 5% | 12.861 | **11.492** | 11.682 |

All entries are equal three-head-seed ensembles. Both pretrained arms outperform
this control at all three fractions, supporting a contribution from pretrained
weights under these adaptation recipes. This does not rule out other, better
scratch recipes. The additional control was specified after the initial low-label
results; it is not a pre-registered primary endpoint. Runner:
`scripts/nox_temporal_random_control.py`; manifest and results:
`random_encoder_control.json`, `random_encoder_summary.json`.

## Interpretation and limitations

- The selected smaller model plus ensembling carries from the same-period result
  into this temporal diagnostic. Extra inference cost is buying useful variance
  reduction; this is not evidence that every single fit beats a tree.
- Source-only pretraining improves ensemble RMSE versus scratch at 1/2/5% by
  roughly 4.5% / 2.7% / 3.5%. It worsens results at 10% and full labels.
- Unlabeled future-feature pretraining is strongest at 1/2% and full labels in
  this run. It is not uniformly better than source-only SSL, and its access to
  target-period covariates must be stated. Its larger corpus also produces more
  optimizer steps at the same epoch count; data scope and compute are not isolated.
- The adaptation protocol changes above 5%: only the low fractions consider a
  frozen probe and low learning rate. The 10% reversal cannot establish an
  intrinsic label-count threshold for pretraining usefulness.
- Source SSL's best masked validation MSE is .2191 (epoch 45); transductive SSL's
  is .1978 (epoch 57). These are different validation corpora, not a controlled
  head-to-head reconstruction comparison. SSL took about 185 and 296 seconds.
- A meaningful next confirmation is more fixed label draws and independent SSL
  seeds with this recipe frozen. This single-dataset, single-draw diagnostic is
  encouraging evidence for real-sensor pretraining, not a reversal of every older
  gate or support for the structurally different synthetic random-table prior.
- The temporal validation pool follows CSV row order within 2013; the year-level
  train/test separation is explicit, but within-year timestamps are unavailable
  in these sensor-only files. No rolling retraining is simulated.

## Baseline correction and verification

The 2011–2013 target mean is 68.7765; predicting it on 2014–2015 gives **13.7584
RMSE**. The older LESSONS claim that target standard deviation (~11.04) was a
constant-predictor baseline was incorrect and has been corrected. The deployable
mean predictor must use historical labels, not the future mean. For these
specific label budgets, the labeled-fit-mean baselines are respectively
13.929 / 13.752 / 13.631 / 13.776 / 13.806.

Completed **113 fits**: 80 primary, 15 additional tree fits, 18 matched
random-encoder adaptation fits. Recomputed **250 prediction metrics**: 207
individual split scores, 23 neural/primary-tree ensemble scores, and 20 additional
tree individual/ensemble scores. Verified disjoint ordered split indices, exact
label counts, fit-only target statistics, validation-only checkpoint/mode choices,
ensemble arithmetic, and SSL corpus/checkpoint scope. Audit:
`logs/nox_temporal_pretraining/verification.json`.

Five focused tests pass across the temporal and reference-model harnesses;
forward/backward/transfer smoke checks, Ruff, and diff-whitespace checks pass.
