# Tabular benchmark and numeric ablation updates — 2026-09-16

These changes improve experimental controls and make existing encoding components
independently testable. They do not establish an accuracy improvement.

## Benchmark protocol: shared-fit-v2

`scripts/tabular_vs_trees.py` now trains all models on the same fit rows. The outer
split remains 70% train / 30% test; 20% of outer training is validation. Deep uses
validation for checkpoint selection. Trees retain fixed hyperparameters and do
not fit validation rows. No model is refitted on fit+validation before scoring.

Feature/target normalization, numeric quantiles, and prediction clipping bounds
come from fit rows only. The result JSON records the protocol version, settings,
and per-seed fit/validation/test counts. Historical results are not directly
comparable because their trees received more training labels and their
preprocessing used the complete outer training partition.

The aggregate no longer calls a result a tie because it is within one neural
seed standard deviation. Defaults are an absolute 0.01 macro-F1 margin and a
relative 2% RMSE margin, configurable with `--classification-margin` and
`--regression-margin`. These are practical tolerances, not significance or
equivalence tests. Seed standard deviations are still reported separately.
The best tree remains a test-score envelope, explicitly labeled as such, rather
than a deployable model selected by validation.

Output defaults to `tabular_vs_trees_results.fair.json`. Existing files are never
overwritten; use a fresh `--output` for every run. Other scripts importing `prep`
retain its legacy default; the updated harness opts into `fit_only=True`.
In particular, old ICL results and `icl_vs_trees.py` retain their original
protocol and must not be presented as matched comparisons with shared-fit-v2.

## Numerical encoding options

- `identity`: existing continuous value + present-flag projection.
- `ple`: continuous branch plus the existing quantile piecewise-linear basis.
- `periodic`: continuous branch plus the existing learned periodic basis.
- `dual`: existing combination of both bases.

All retain the shared projection and native missingness behavior. The existing
`--fusion hadamard` option conditions value vectors on column identity; default
fusion remains additive. No first-attention-block default changes were made.
Existing identity/dual parameter shapes, initialization, and predictions remain
unchanged for finite training data. Invalid encoder/fusion options now raise an
error instead of silently using another path.

Quantile construction now ignores positive/negative infinity, consistent with
the encoder treating nonfinite inputs as missing. Previously infinity could
poison the bin boundaries. Fully missing columns use finite zero boundaries.

Both benchmark runners expose learning rate and dropout for controlled trials;
they do not automatically search hyperparameters or select settings on test data.

## Next NOx experiment

The previous first-block result remains in `docs/NOX_ENCODING_ABLATION.md`.
To test the numerical basis while holding the first block fixed, run separate
three-seed experiments with identical other settings:

```sh
.venv/bin/python -m scripts.nox_encoding_ablation --arms identity_query \
  --value-encoder ple --output logs/nox_basis_ple.json
.venv/bin/python -m scripts.nox_encoding_ablation --arms identity_query \
  --value-encoder periodic --output logs/nox_basis_periodic.json
.venv/bin/python -m scripts.nox_encoding_ablation --arms identity_query \
  --value-encoder dual --output logs/nox_basis_dual.json
```

The original identity-query arm supplies the baseline at default settings. Treat
comparisons against its already-inspected holdout as exploratory; use fresh
held-out data for confirmation after choosing a configuration. The runner's
`paired_iid` summary applies only when both first-block arms are run; comparisons
between separate numerical-basis runs must be reported separately.

These longer numerical-basis experiments have not been run as part of the code
update. Validation includes focused regression tests, backward-compatibility
checks against the prior encoder, and short end-to-end smoke runs.

Validation completed: 8 new regression tests pass; the full offline suite has
84 passes and one existing ensemble CPU runtime-ratio failure. That same runtime
test also fails with the original HEAD encoder restored in memory (10.23× versus
the test's 10× ceiling); no timing threshold was relaxed. Normal online suite
execution first encountered an unavailable Hugging Face network lookup; cached
models work with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

The fair-harness smoke covered two regression and two classification datasets,
two seeds each, with `ple` + `hadamard`. The NOx smoke covered `periodic` +
`hadamard` with the first block held fixed. These short runs verify execution,
not predictive quality. Existing identity/dual × additive/hadamard encoders were
also compared with HEAD: identical initial state dictionaries and outputs.
