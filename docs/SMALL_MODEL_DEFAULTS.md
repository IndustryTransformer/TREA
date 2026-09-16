# Provisional compact single-row defaults — 2026-09-16

Following the NOx reference-model and temporal/pretraining diagnostics, the public
single-row encoder, regressor, classifier, masked encoder, and Lightning wrappers
now default to **two layers, width 128, four heads, and dropout 0**. Supervised
wrappers use **lr 3e-4**; masked pretraining retains the tested **lr 8e-4**.
Product plus independent feature bias and skipping the first attention-input norm
remain defaults. Explicit constructor settings continue to override these choices.
These defaults are provisional for other datasets, particularly classification.

The generic tabular benchmark follows the compact supervised recipe and exposes
`--n-layers`, `--dropout`, `--lr`, `--epochs`, and `--patience`. Its default training
cap is 300 epochs with patience 30, batch 256. The lower-level `build_trainer`
helper retains its existing defaults: specify `max_epochs=300, patience=30` to
match the experiment budget. Historical NOx and numerical-basis diagnostics pin
their original settings rather than inheriting the new benchmark defaults.

**The improvement was not isolated to model size.** With encoding already fixed,
the jointly selected smaller/less-regularized training recipe improved same-period
single-fit RMSE from 4.979 to 4.765. Averaging three independent fits then reached
4.519 (tree ensemble 4.566). The recipe also changed learning rate, batch size,
training cap, and patience. We did not run a depth-only ablation of the final recipe.
The temporal experiment additionally changed validation and label-budget handling.

The reported ensembles are **three independently trained models**, not the
experimental `ensemble_k` adapter. Ensembling stays explicit and is not enabled
by default. Averaging predictions costs three fits/forward passes; a new model
instance does not reproduce an ensemble's score.

The temporal low-label results support the case for extra compute when labels are
scarce: source-only SSL ensemble RMSE at 1/2/5% was 11.895/11.692/11.492 versus
scratch 12.453/12.014/11.903. Gains did not extend to 10/full labels. Unlabeled
future-feature SSL is a separate transductive arm, not evidence of ordinary
train-only pretraining. Confirm across label draws and SSL seeds before scaling.

Use explicit architecture settings when reconstructing older checkpoints manually.
The pre-ablation model settings were `n_layers=4, dropout=0.2, fusion="additive",
first_prenormalization=True`, with supervised `lr=8e-4`. Lightning checkpoints
with saved hyperparameters retain their recorded settings.

Evidence: [reference comparison](NOX_REFERENCE_MODELS.md),
[temporal/pretraining comparison](NOX_TEMPORAL_PRETRAINING.md), and
[durable findings](LESSONS.md#3h-updated-small-trea--ensembles-real-nox-ssl-is-worth-confirming-2026-09-16).
