# 3W row attention on the single-row encoder — pre-registration (2026-11-10)

Written before any fit of this model. It follows the STOP verdict in
`docs/3W_HISTORY_ATTENTION_PREREG.md`, whose result was confounded: that
architecture scored 0.773 AUC with no context, far below the window-only
single-row TREA on the same summaries. This experiment removes the confound by
construction. It is a new experiment, not a rescue of the previous one.

Margins are the same as the previous pre-registration and are frozen now.

## Question

Starting from the existing single-row TREA window model, does adding **learned
attention over the embeddings of the file's earlier windows** improve on the
window model alone, and on the same model given hand-built history blocks?

## Model (fixed before fitting)

`EncoderHistoryClassifier` in `examples/three_w_history_attention.py`.

- Base: exactly the `single_row` model of `examples/benchmark_3w_grouped.py`
  (`SingleRowClassifier`, width 128, four heads, two layers, dropout 0,
  `product_sum` fusion) over the 176 quantile-transformed summary features.
- Each context window is encoded by the **same encoder** into its pooled
  vector, without gradient. Gradient reaches the encoder through the query row.
- One cross-attention layer: the query vector attends over context tokens
  built from `[context vector, query vector − context vector]` plus a slot
  embedding, and an always-present learned null token.
- The attention output is added to the query vector through a
  **zero-initialised** projection before the unchanged classification head.
  At initialisation the model is therefore identical to the base model, and the
  `none` arm is the base model exactly.

No hand-built difference of raw statistics is supplied; any baseline comparison
must be learned in embedding space. Known risk, stated in advance: quantile
transformation plus pooling may compress small within-file drifts that the
hand-built blocks expose directly.

## Context

Unlabeled, same file, strictly earlier cached windows at 1, 2, 4, 8, 16 and 32
windows back, plus the file's first cached window as an anchor: at most seven
rows, a subset of what the hand-built blocks see. Same longer-context caveat as
before (history comes from cached labeled-window rows).

## Shared recipe

Identical to `docs/3W_HISTORY_ATTENTION_PREREG.md`: same group manifest, value
policy, sampled 200,000 rows per seed, seeds 42/43/44, optimiser, batch size,
epoch cap, patience on internal-validation AUC, class and group weighting,
three-fit ensemble and threshold rule.

## Arms

| Arm | Context |
|---|---|
| `history` | Own file's earlier windows |
| `shuffled` | Trained and evaluated with another file's context, same source |
| `none` | No context: the base single-row model |

## Stage 1 gate (internal validation only; a filter, not evidence)

- **Go** if the `history` ensemble AUC ≥ 0.92863 (existing single-row TREA with
  hand-built history, 0.92363, plus 0.005) **and** exceeds both `shuffled` and
  `none` by ≥ 0.005.
- **Stop** otherwise.
- One configuration, one run, no iteration before the verdict.

Caveat fixed in advance: earlier single-row runs on this validation set show
single-fit AUC varying by several hundredths across seeds, so a 0.005 ensemble
margin is within plausible run-to-run noise. A Go is permission to spend the
outer folds, nothing more.

Stage 2 (outer folds) and stage 3 are as defined in the previous
pre-registration and still require Kai's explicit go-ahead.

## Results

### Stage 1 — adjudicated 2026-11-10: **GO** (permission to run stage 2 only)

One run, the pre-declared configuration, no iteration. Outputs:
`logs/benchmark_3w_binary/encoder_history_v1/` (git-ignored). Sampled training
rows are identical to the existing causal arms for all three seeds.

Three-fit ensembles on internal validation (AUC / macro-F1, threshold chosen on
the same validation set):

| Arm | All | Physical wells | Simulations |
|---|---|---|---|
| `history` | 0.93997 / 0.82952 | 0.83427 / 0.76118 | 0.99629 / 0.95232 |
| `shuffled` | 0.92250 / 0.85758 | 0.81944 / 0.82581 | 0.98662 / 0.89138 |
| `none` (base model) | 0.92332 / 0.85759 | 0.82434 / 0.82536 | 0.98204 / 0.88894 |
| TabM PLE causal (existing) | 0.94295 / 0.83746 | 0.82320 / 0.75803 | 0.99988 / 0.98898 |
| Single-row TREA causal (existing) | 0.92363 / 0.81308 | 0.77427 / 0.72101 | 0.99958 / 0.96996 |
| XGB causal (existing) | 0.91124 / 0.80275 | 0.72547 / 0.69235 | 0.99994 / 0.98802 |

Single-fit AUC by seed 42/43/44: `history` 0.93071 / 0.89393 / 0.94013;
`shuffled` 0.91443 / 0.89687 / 0.91073; `none` 0.91488 / 0.92282 / 0.87048.

Gate checks:

- `history` ≥ 0.92863: pass (0.93997).
- `history` − `shuffled` ≥ 0.005: pass (+0.0175).
- `history` − `none` ≥ 0.005: pass (+0.0166).

Observations (post hoc, not evidence):

- Base parity holds: `none` scores 0.92332, matching the earlier window-only
  single-row figure (0.92341). The previous confound is gone.
- `shuffled` ≈ `none`, so another file's context adds nothing here; the gain
  is specific to the query's own file.
- **Macro-F1 moved the wrong way overall**: 0.830 for `history` against 0.858
  for both controls. On physical wells AUC rises only 0.010 (0.824 → 0.834,
  inside seed noise) while macro-F1 falls 0.064. The clear gains are on
  simulations (AUC 0.982 → 0.996, macro-F1 0.889 → 0.952). The pooled AUC gain
  is therefore mostly a simulation and cross-source ranking effect, not better
  detection on physical wells.
- Seed 43 is worse with history (0.894) than without (0.923). Mean single-fit
  AUC is 0.922 / 0.907 / 0.903 for `history` / `shuffled` / `none`.
- `history` is still slightly below TabM with hand-built history on pooled AUC
  (0.940 vs 0.943).
- `history` fits train longer (best epoch 10–13 against 1–3 for `none`), about
  ten times the wall-clock cost per fit.

Stage 2 has not been run and needs Kai's explicit go-ahead. Its pass rule
includes "pooled macro-F1 no worse than matched XGB", which the stage-1 F1
pattern puts at real risk.

### Follow-up diagnosis — 2026-11-11 (exploratory, not pre-registered)

Kai chose to relax the protocol for this. Stage 2 was not run. Seeds 45–49 of
`history` and `none` were added on internal validation
(`logs/benchmark_3w_binary/encoder_history_v1_seeds45_49/`), giving eight seeds
per arm.

Eight seeds, internal validation:

| | `history` | `none` |
|---|---|---|
| Single-fit AUC, mean ± sd | 0.918 ± 0.017 | 0.879 ± 0.036 |
| Single-fit physical AUC, mean ± sd | 0.791 ± 0.044 | 0.762 ± 0.072 |
| Single-fit macro-F1, mean ± sd | 0.821 ± 0.014 | 0.798 ± 0.046 |
| Eight-fit ensemble AUC / macro-F1 | 0.943 / 0.839 | 0.913 / 0.844 |
| Eight-fit ensemble, physical AUC / macro-F1 | 0.840 / 0.775 | 0.810 / 0.804 |
| Eight-fit ensemble, within-file AUC (mixed-label physical files) | 0.907 | 0.534 |

Paired per-seed difference (`history` − `none`): pooled AUC +0.039 (standard
error 0.014, seven of eight seeds positive); physical AUC +0.029 (standard
error 0.027, four of eight positive).

- The original three seeds flattered `none`: seeds 42 and 43 are its two best
  of eight. Its three-fit 0.923 is 0.913 with eight fits.
- The macro-F1 drop is not a threshold artifact (per-source thresholds change
  nothing) and persists in the eight-fit ensemble on physical wells
  (0.775 vs 0.804).
- Internal validation has eight physical wells and two decide the pooled
  physical result (three-fit ensembles): WELL-00022 0.455 → 0.824,
  WELL-00001 0.839 → 0.557. Dropping either flips the sign.
- What history clearly adds is ranking inside a recording: within-file AUC on
  mixed-label physical files goes from chance to 0.84–0.91. Most of those rows
  are WELL-00022's eight long files. The gain appears at ≥ 64 windows into a
  file; earlier positions get worse.
- `history` roughly halves seed variance.

### Frozen-base correction — 2026-11-11 (exploratory): does not work

`examples/benchmark_3w_history_correction.py`, outputs in
`logs/benchmark_3w_binary/history_correction_v1/`. The fitted `none` checkpoint
of each seed is frozen; a small attention module over the same context slots
adds a scalar to its margin (zero at initialisation and with no context;
epoch 0, the untouched base, is a selection candidate). `diff` sees only
standardised embedding differences to the past, `full` also the absolute
embeddings, `diff_shuffled` is `diff` with another file's context.

Eight-fit ensembles on internal validation:

| Arm | AUC | Macro-F1 | Physical AUC | Simulation AUC | Within-file AUC | Physical AUC, < 64 / ≥ 64 windows in |
|---|---|---|---|---|---|---|
| `none` (base) | 0.913 | 0.844 | 0.810 | 0.984 | 0.534 | 0.765 / 0.829 |
| `diff_shuffled` | 0.914 | 0.841 | 0.808 | 0.984 | 0.532 | 0.771 / 0.826 |
| `diff` | 0.917 | 0.826 | 0.774 | 0.996 | 0.560 | 0.663 / 0.840 |
| `full` | 0.916 | 0.826 | 0.778 | 0.998 | 0.579 | 0.647 / 0.844 |
| `history` (joint) | 0.943 | 0.839 | 0.840 | 0.996 | 0.907 | 0.718 / 0.932 |

- The correction reproduces the simulation gain and nothing else. Within-file
  ranking on physical recordings stays near chance, so that ability comes from
  training the encoder jointly with the context, not from the attention module.
- It still damages the wells that joint training damaged (WELL-00001
  0.82 → 0.60, WELL-00010 0.54 → 0.37), even when it only sees differences. The
  between-file loss is therefore caused by the history signal itself, not by
  the encoder drifting. The hypothesis behind this variant was wrong.
- Untested guess: the correction is learned mostly from simulations, where
  history is highly informative, and transfers badly to short physical files.

### Stage 2 — outer folds, scored once 2026-11-11: **FAIL, line stops**

Run with Kai's explicit go-ahead; this is the outer folds' second scoring.
Fits: `logs/benchmark_3w_binary/encoder_history_outer_v1/fold_{0..3}/`
(`history` and `none`, seeds 42/43/44, training rows identical to the September
fold fits). Scoring: `examples/evaluate_3w_history_outer.py`, results in
`.../encoder_history_outer_v1/outer/results.json`. All 24 checkpoints
reproduced their validation scores before outer rows were loaded; thresholds
come from internal validation only. `trea_causal` and `xgb_causal` are the
frozen September outer predictions, on identical rows.

Three-fit ensembles, 358,874 outer windows, 20 physical wells:

| Model | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Mean fold AUC | Pooled AUC | Pooled macro-F1 | Physical AUC / F1 | Within-file AUC |
|---|---|---|---|---|---|---|---|---|---|
| `history` (learned) | 0.9405 | 0.8976 | 0.8470 | 0.9356 | 0.9052 | 0.9112 | 0.7633 | 0.667 / 0.505 | 0.747 |
| `none` (base) | 0.9071 | 0.8604 | 0.8423 | 0.9118 | 0.8804 | 0.8235 | 0.6001 | 0.352 / 0.335 | 0.620 |
| Causal TREA (hand-built history) | 0.9907 | 0.8702 | 0.8761 | 0.9395 | 0.9191 | 0.9047 | 0.7903 | 0.631 / 0.550 | 0.795 |
| Causal XGBoost (hand-built history) | 0.9968 | 0.8739 | 0.8963 | 0.9443 | 0.9278 | 0.8675 | 0.7648 | 0.463 / 0.465 | 0.843 |

Within-file AUC is the rows-weighted AUC inside physical files that contain
both classes. Pooled AUC mixes differently calibrated fold models, as noted in
`docs/3W_ROBUSTNESS_FINDINGS.md`.

Pass rule (beat causal XGBoost on mean fold AUC, lead in three of four folds,
pooled macro-F1 no worse):

- Mean fold AUC 0.9052 vs 0.9278: **fail**.
- Leads in one of four folds (fold 1): **fail**.
- Pooled macro-F1 0.7633 vs 0.7648: **fail**, marginally.

Kill rule: `history` mean fold AUC is 0.014 *below* the existing causal TREA
(0.9191), so learned row attention adds nothing over hand-built history on 3W.
The rule was written as "within 0.005"; being clearly worse is the same
conclusion. **The line stops.** Stage 3 is not run.

Paired whole-group bootstrap, 95% intervals, `history` minus reference:

| Reference | Mean fold AUC | Pooled AUC | Pooled macro-F1 |
|---|---|---|---|
| Causal XGBoost | −0.044 to +0.001 | −0.079 to +0.114 | −0.114 to +0.057 |
| Causal TREA | −0.034 to +0.008 | −0.066 to +0.045 | −0.126 to +0.022 |
| `none` | −0.001 to +0.045 | +0.004 to +0.164 | −0.029 to +0.327 |

What does hold:

- Learned history beats no history in all four folds (mean fold AUC +0.025),
  so the mechanism is real. It simply recovers less than the hand-built
  history blocks do, in both model families.
- The within-file advantage seen on internal validation did not carry over:
  hand-built history ranks windows inside a recording better (0.84 XGBoost,
  0.79 TREA) than learned history (0.75).
- Fold 0 is where it loses most (0.9405 vs 0.99+), driven by WELL-00005
  (AUC 0.30 against 0.82 and 0.95).
- Not a like-for-like information comparison: the hand-built arms use history
  reconstructed from raw sensors before label eligibility, with a 37-window
  physical warm-up; the learned arm sees at most seven cached labeled-window
  rows per query.
