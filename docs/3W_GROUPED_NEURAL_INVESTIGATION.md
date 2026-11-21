# 3W grouped neural investigation — 2026-09-17

This investigation was explicitly requested after the historical supervised
deep-model stop decision in `LESSONS.md`. That earlier result is unchanged.
Historical AUCs (XGB engineered 0.9474, naive 0.9068, RF 0.9584) were selected or
reported on the old validation partition, not a pristine final test set.

## Protocol fixed before new model results

- Sorted toolkit file inventory; physical well identity is `WELL-<digits>`.
  Files without such identity are individual simulation groups. These are not
  1,149 independent physical wells. Related simulations may share an unobserved
  generator; this protocol cannot establish disjointness of latent generators.
- Split seed 20260917: reserve 20% of groups for test. Split the remaining groups
  with seed 20260918 into 75% fitting and 25% internal validation (60/20/20 total).
  Test signals and labels are not loaded during model development.
- Fixed 96/96 windows, existing label mapping and sensor clipping retained.
- Sensor selection and sensor mean/std fitted on training only. Apply those same
  parameters to validation/test. Cached manifests record files, groups, counts,
  and normalization. Legacy two-way splitting remains the default loader behavior.
- Every model receives the same 200,000 sampled training-window indices per seed.
  Seeds 42,43,44 change initialization/subsampling, not the split. They are not
  independent estimates of well-split uncertainty.
- Engineered models share exactly the existing eight XGB summary blocks. Neural
  standard or quantile transforms are fitted on the sampled training rows only.
  This is identical source information, with model-appropriate preprocessing.
- Neural checkpoint selection: maximum internal-validation AUC, early stopping.
  Threshold: maximize internal-validation macro-F1, then freeze for test. Report
  both frozen-threshold macro-F1 and macro-F1 at 0.5. Validation F1 is optimized
  in-sample and must not be described as an unbiased generalization estimate.
- Each seed produces saved probabilities and parameters. Explicit seed ensemble
  averages probabilities, selects its own threshold on validation, and accounts
  for all fits. TabM additionally contains shared-weight members within each fit;
  this is distinct from an independent three-fit ensemble.
- Secondary instance metric averages probabilities per source file and predicts
  whether at least half its windows are faulty. Mixed-label instances make this
  a different target from window classification or early event detection.
- A new split of previously studied data is a reserved test for this investigation,
  not an entirely new dataset. Historical aggregate exposure remains a limitation.

## Audit

The loader performs sensor-wide normalization, not per-window normalization.
Absolute operating levels remain available. Both the original temporal model and
the single-row model explicitly encode missingness. Adding another mask is not
an evidence-backed fix for those paths.

**Subsequent raw-value audit correction:** the statement above describes the
normalization step, not the complete raw-data path. The loader first clipped
every numeric sensor to ±1e6. A later training-only raw audit found that this
flattens most valid pressure readings and also converts enormous negative
artifacts into finite readings. The opt-in `preserve_range` continuation fixes
that separate information-loss issue. This report's original results all retain
legacy clipping and remain a frozen historical record; see
`3W_RANGE_REPAIR_INVESTIGATION.md` for the continuation.

Original temporal stat tokens have shared projection, no sensor identity, no
endpoint delta, and no direct output pooling. Their energy divides by observed
count, while XGB's energy includes zero-filled missing timestamps in its divisor.
Their half-window trend also differs when an entire half is missing.

New opt-in `TriplePatchTransformer` flags:

- `stat_sensor_identity`: learned sensor identity after stat-token normalization.
- `stat_include_delta`: eighth summary, final minus first zero-filled reading.
- `stat_exact_engineered`: match XGB summary definitions and use unmixed values.
- `direct_stat_pooling`: concatenate separate temporal and stat-token means.
- `stats_only`: classify only encoded statistical tokens.

Disabled flags preserve parameter names/shapes, initialization, and behavior.
The `stats_only` forward bypasses the unused temporal path; parameter accounting
currently includes its allocated, unused patch parameters.

## Reproduction

Run from the repository root:

```sh
PYTHONPATH=. .venv/bin/python -m examples.benchmark_3w_grouped \
  --models xgb,xgb_naive,rf --seeds 42,43,44 \
  --output-dir logs/benchmark_3w_binary/grouped_baselines

PYTHONPATH=. .venv/bin/python -m examples.benchmark_3w_grouped \
  --models mlp,tabm,single_row --preprocess quantile --seeds 42 \
  --output-dir logs/benchmark_3w_binary/grouped_probe_quantile
```

Do not add `--evaluate-test` during development. All results are written incrementally
to each run's `results.json`; checkpoints, preprocessing, sampled row indices,
protocol manifests, and per-seed probabilities are saved alongside it. Completed
result directories are protected from accidental overwrite.

## Outcome

**The numerical goal is not met.** The selected three-fit SingleRowClassifier
ensemble reaches **0.9349 test AUC / 0.8259 macro-F1**, close to the full engineered
XGB ensemble's **0.9392 / 0.8333**, but below the historical 0.9474 target. The
paired AUC difference is **-0.00437**, with a whole-group bootstrap 95% interval
**[-0.02559, +0.00530]**. This does not establish superiority or equivalence.

This is substantial progress over the historical temporal result, but comparisons
with historical numbers are descriptive: the split, training cap, preprocessing,
selection metric, and training behavior changed together.

The strongest tested TREA representation is **per-sensor standard deviation and
energy**, transformed using training-fitted quantiles, then classified by the compact
SingleRowClassifier. It uses 44 inputs (2 summaries × 22 retained sensors), two
width-128 attention layers, zero dropout, LR 3e-4, square-root inverse-frequency
class weights, and three independent probability-averaged fits. Each fit has
460,034 parameters. Checkpoints are selected by internal-validation AUC, and the
ensemble threshold is fixed at 0.6478634477 using internal validation.

The apparently strongest single-fit family on validation, TabM with piecewise
numeric embeddings, did not generalize as well on test. The frozen test evaluation
was not used to revise models or thresholds.

### Reserved test: single fits and explicit ensembles

Means and population standard deviations are over seeds 42,43,44 on the same split.
They measure fit/subsampling variation, not uncertainty over unseen wells.
Macro-F1 below uses thresholds selected on internal validation.

| Model | Single-fit AUC | Single-fit macro-F1 | Ensemble AUC | Ensemble macro-F1 |
|---|---:|---:|---:|---:|
| Naive XGB | 0.8952 ± 0.0093 | 0.7652 ± 0.0113 | 0.9069 | 0.7597 |
| Full engineered XGB | 0.9344 ± 0.0084 | 0.8356 ± 0.0021 | 0.9392 | 0.8333 |
| Engineered RF | 0.9402 ± 0.0027 | 0.8113 ± 0.0051 | 0.9405 | 0.8053 |
| TREA SingleRow, std/energy | 0.9098 ± 0.0394 | 0.8129 ± 0.0119 | **0.9349** | **0.8259** |
| TabM + piecewise embeddings, all summaries | 0.9079 ± 0.0062 | 0.7262 ± 0.0383 | 0.9132 | 0.7732 |

The TREA single-fit test AUCs are **0.939675, 0.935717, 0.854081**. Seed 44 was good
on validation but weak on test. The ensemble result must not be presented as the
performance of one compact model. The three-fit ensemble stores 1,380,102 parameters
and requires three inference passes. Recorded aggregate training time was 107.7 s;
concurrent jobs make timings indicative rather than a controlled speed benchmark.

At the fixed 0.5 threshold, ensemble macro-F1 is 0.8318 for TREA; this is a secondary
predefined threshold result, not a test-selected replacement for the frozen threshold.

### Same-feature tree comparison

XGB on the same std/energy inputs was also selected before test results were read:
it was the strongest XGB feature-family validation ensemble. Its test ensemble AUC
is **0.8627**, versus TREA's **0.9349**, with macro-F1 0.8197 versus 0.8259. The AUC
advantage is +0.07214, but its group-bootstrap interval **[-0.01492, +0.16518]** is
wide. This supports the usefulness of the neural representation on those inputs;
it does not mean TREA beats the stronger full-feature XGB baseline.

### Validation experiments

All rows below are internal validation, not final-test claims. Macro-F1 threshold
selection and checkpoint selection used this partition.

| Model / representation | Single-fit mean AUC | Three-fit ensemble AUC |
|---|---:|---:|
| Temporal control | 0.8556 | 0.8868 |
| Temporal control, shuffled timestamps | 0.8484 | 0.8800 |
| Direct-stat temporal + separate stat pool | 0.8561 | 0.8620 |
| Stats-only transformer | 0.8716 | 0.8873 |
| SingleRow, std/energy + quantiles | 0.8967 | **0.9265** |
| TabM, all summaries + quantiles | 0.8994 | 0.9052 |
| TabM + piecewise embeddings + quantiles | **0.9129** | 0.9167 |
| Same TabM + full class balancing | 0.9122 | 0.9135 |
| Full engineered XGB | 0.9003 | 0.9024 |
| Engineered RF | 0.9047 | 0.9072 |

Seed-42 representation probes: standard-scaled MLP 0.8166 → quantile-scaled MLP
0.8877; standard-scaled TabM 0.8738 → quantile-scaled TabM 0.9026. SingleRow on all
176 engineered inputs with quantiles reached 0.8826 in its seed-42 probe. Restricting
it to the validation-supported variation family produced the selected ensemble.
These probes change preprocessing/representation, not just model size.

Full class balancing was tested because XGB uses it. Its seed-42 run was still
improving at 30 epochs, so the confirmation allowed 50 epochs with patience 8.
Across three seeds it did not improve on the square-root-balanced configuration;
the latter remained the selected TabM arm.

Temporal controls in this investigation use no column-embedding or pre-patch
feature-attention module, a common 200k training cap, weighted loss instead of a
weighted sampler, BF16 mixed precision, and validation-AUC checkpointing. They
are paired controls for the new stat/shuffle variants, not exact reproductions
of the legacy full-data Lightning benchmark.

### XGB feature-family diagnosis

| Features | Validation mean AUC over three fits |
|---|---:|
| Mean/min/max | 0.8686 |
| Standard deviation/energy | **0.9049** |
| Missing rate only | 0.8064 |
| Endpoint delta/half-window trend | 0.7711 |
| All except delta/trend | 0.8859 |
| All except missing rate | 0.8962 |
| All eight blocks | 0.9003 |

Variability is a strong signal; structural missingness is also predictive. Removing
coarse direction reduces AUC, so the shuffle result is not evidence that all time
information is useless. The temporal shuffle penalty (~0.007 mean AUC) is modest
relative to the overall representation gap, under this specific control.

## Limitations and stopping point

- Train/validation/test contain **689/230/230 groups**, but only **28/8/4 physical
  wells**. The rest are individually grouped simulation instances. Window counts
  are **468,514 / 173,464 / 113,442**, before the per-seed training cap.
- On physical test wells alone, TREA ensemble AUC is **0.7844** versus XGB **0.7894**;
  macro-F1 is **0.7800** versus **0.7439**. TabM falls to AUC **0.4429**. The mixed
  headline is substantially easier than robust real-well deployment.
- At the instance-majority level, TREA test AUC is 1.0 and macro-F1 0.9800, but the
  test has only 260 instances and all 226 simulated instances are majority-fault.
  This secondary target is confounded with source composition and is not a basis
  for declaring the window-classification goal solved.
- Only four physical test wells makes group uncertainty substantial. Bootstrap
  groups include simulation instances with unknown generator dependence.
- Legacy ±1e6 clipping can discard extreme raw magnitudes for both trees and
  neural models; it was preserved to maintain the common input definition.
- The tested direct-stat flags repair representation limitations but do not
  consistently close the full-feature tree gap. No causal claim is made for any
  one flag in the joint variant.
- The strongest requested representation avenues, numeric-embedding extension,
  class-balancing check, and three-seed confirmations have been run. Continuing
  to tune against the now-observed test would invalidate its reserved status.
  The absolute 0.9474 target and reliable cross-well parity remain unresolved.
  The defensible next evaluation is a pre-registered grouped outer-fold study
  emphasizing physical wells, or genuinely new wells, rather than changing this
  split or reporting a lucky seed as success.

## Reproduce the frozen final evaluation

Selection is recorded in `docs/results/3w_grouped_selection.json`. The default
evaluation refuses to overwrite an existing result directory.

```sh
PYTHONPATH=. .venv/bin/python -m examples.evaluate_3w_reserved \
  --runs logs/benchmark_3w_binary/grouped_baselines \
    logs/benchmark_3w_binary/grouped_tabm_confirm::tabm_ple \
    logs/benchmark_3w_binary/grouped_variation_probe::single_row \
    logs/benchmark_3w_binary/grouped_single_row_confirm::single_row \
  --models xgb,xgb_naive,rf,tabm_ple,single_row \
  --output-dir logs/benchmark_3w_binary/grouped_reserved_test

PYTHONPATH=. .venv/bin/python -m examples.evaluate_3w_reserved \
  --runs logs/benchmark_3w_binary/grouped_ablation_variation --models xgb \
  --output-dir logs/benchmark_3w_binary/grouped_reserved_variation

PYTHONPATH=. .venv/bin/python -m examples.summarize_3w_grouped
```

Every restored model was checked against its saved validation probabilities
before reporting test results. The durable result JSON contains all development
configurations, exact seed results, correctly recombined three-fit ensembles,
test results, protocol manifests, package versions, source hashes, and uncertainty.

## Verification and workspace state

- 16 focused benchmark/protocol tests pass.
- 25 existing SingleRow tests pass.
- Ruff passes on all touched Python files; `git diff --check` passes.
- With new flags disabled, initialization, strict checkpoint loading, and outputs
  match the original committed `TriplePatchTransformer` exactly (zero tolerance).
- The full repository suite was not run.
- The original uncommitted benchmark/docs/tests were preserved. Nothing committed.
