# 3W history attention — pre-registration (2026-11-10)

Written before any history-attention model was fitted. Margins marked
**(proposed)** were chosen by Claude and are open to change by Kai until the
first development fit finishes; after that they are frozen.

## Question

Given the same past-only information, does **learned attention over a file's
earlier windows** beat the hand-built causal-history features
(`examples/three_w_causal_features.py`)?

Motivation, from `docs/3W_CALIBRATION_CONTEXT_PROGRESS.md` and
`docs/3W_ROBUSTNESS_FINDINGS.md`:

- Physical wells fail in opposite directions, so no global threshold repairs it.
- Hand-built past-only history lifted TabM PLE (0.906 → 0.943 internal-validation
  AUC) and XGB (0.859 → 0.911).
- The single-row TREA did not capture it (0.92341 window-only → 0.92363 with the
  same history blocks), and lost to matched XGB on every outer fold
  (0.9191 vs 0.9278 mean AUC).

This is the "sample axis" version of row attention: rows are whole windows of
the same file, not timesteps inside one window and not labeled rows from other
wells (TabICL with cross-well context already lost to RF, LESSONS §3g).

## Information given to the new arm

- Row representation: the existing eight summary blocks per sensor (22 sensors),
  from the same cache as every other arm. No raw 96-step windows.
- Context for a query window: the **32 immediately preceding cached windows of
  the same file** plus the **file's first cached window** as an anchor. This is
  a subset of what the hand-built blocks see (their exponential mean has
  unbounded memory and their "first observed" reference is per sensor).
- Context rows are **unlabeled**. No label of any context row enters the model
  or the loss. No future row and no other file enters the context.
- Same scope caveat as the existing causal arms: history is built from cached
  (labeled-window) rows, so it is the longer-context task, not the isolated
  96-step benchmark.
- Preprocessing is fitted on the sampled training rows only.

## Model (fixed before fitting)

`examples/three_w_history_attention.py`. One token per (row, sensor), built from
`Linear([8 stats, 8 query-minus-row differences, flags])` plus a sensor identity
and a windows-back embedding. Two pre-norm blocks, each: attention across
sensors within a row, then attention across rows within a sensor, then a
feed-forward layer. The query row's sensor tokens are pooled for the binary
head. Width 128, four heads, two layers, dropout 0 — the compact recipe.

The explicit difference terms are deliberate: every working hand feature is a
difference or ratio against a baseline, and the TabR ablation recorded in the
project notes found plain cross-attention to context rows buys nothing without
them.

## Shared recipe (identical to the existing causal arms)

Group manifest `docs/results/3w_range_confirmation_protocol.json`,
`value_policy=preserve_range`, 200,000 sampled training rows per seed using the
same `default_rng(seed).choice` draw, seeds 42/43/44, AdamW lr 3e-4, weight decay
3e-4, betas (0.9, 0.95), batch 512, ≤30 epochs, patience 6 on internal-validation
AUC, `balance=sqrt`, `group_loss_power=0.5`, three-fit probability ensemble,
threshold = exact macro-F1 optimum on internal validation.

## Arms

| Arm | Context | Purpose |
|---|---|---|
| `history` | Own file's earlier windows | The claim |
| `shuffled` | Trained and evaluated with context drawn from a different file of the same source (physical or simulation) | Mechanism control |
| `none` | No context rows | Same architecture, window-only reference |

Existing references on the same internal validation (three-fit ensembles):
TabM PLE causal 0.94295 / 0.83746, single-row TREA causal 0.92363 / 0.81308,
XGB causal 0.91124 / 0.80275 (AUC / macro-F1).

## Stage 1 — development gate (internal validation only)

Internal validation has been used for selection many times; this stage is a
go/no-go filter, **not evidence**.

- **Go** to stage 2 if the `history` ensemble AUC ≥ 0.92863, i.e. existing
  causal TREA + 0.005 **(proposed)**, and it exceeds both `shuffled` and `none`
  by ≥ 0.005 AUC **(proposed)**.
- **Stop** otherwise. If `history` is within 0.005 of `shuffled`, the model is
  not using its own file's context and the result is reported as a mechanism
  failure regardless of the absolute number.
- No architecture or recipe iteration between stage 1 and the verdict. One
  pre-declared configuration, one run.

## Stage 2 — outer folds (requires Kai's explicit go-ahead)

Uses the frozen four outer folds in `docs/results/3w_grouped_robustness_v1/`.
Consumed test partitions stay excluded.

- **Pass:** `history` beats matched causal XGB on mean outer-fold AUC, leads in
  at least three of four folds, and pooled macro-F1 is no worse **(proposed)**.
- **Kill:** `history` mean outer-fold AUC is within 0.005 of the existing causal
  TREA (0.9191) **(proposed)**. Learned row attention then adds nothing over
  hand features on 3W and the line stops.
- Selection and thresholds from internal validation only; outer groups scored
  once after artifacts are frozen; paired whole-group bootstrap as in
  `examples/evaluate_3w_robustness.py`.

Known weaknesses of stage 2, stated in advance: the outer folds were already
scored once for the causal-context study, so this would be their second use;
there are only 20 physical wells; fold 2 fails for both existing model families.
Per-physical-well results are reported as secondary.

## Stage 3 — conditional on a stage-2 pass

Add K labeled windows from the target well's earlier files to the context and
report AUC against K, with XGB given the same windows. Not designed further
until stage 2 is adjudicated.

## Results

### Stage 1 — adjudicated 2026-11-10: **STOP**

One run, the pre-declared configuration, no iteration. Margins were not changed
before the run finished, so they are frozen as proposed. Outputs:
`logs/benchmark_3w_binary/history_attention_v1/` (git-ignored). The sampled
training rows are identical to the existing causal arms for all three seeds.

Three-fit ensembles on internal validation (AUC / macro-F1):

| Arm | All | Physical wells | Simulations |
|---|---|---|---|
| `history` | 0.90703 / 0.77782 | 0.71737 / 0.64735 | 0.99946 / 0.96784 |
| `shuffled` | 0.87904 / 0.75832 | 0.68999 / 0.66287 | 0.98572 / 0.88032 |
| `none` | 0.77264 / 0.62548 | 0.50783 / 0.50775 | 0.96824 / 0.69658 |
| TabM PLE causal (existing) | 0.94295 / 0.83746 | 0.82320 / 0.75803 | 0.99988 / 0.98898 |
| Single-row TREA causal (existing) | 0.92363 / 0.81308 | 0.77427 / 0.72101 | 0.99958 / 0.96996 |
| XGB causal (existing) | 0.91124 / 0.80275 | 0.72547 / 0.69235 | 0.99994 / 0.98802 |

Single-fit AUC by seed 42/43/44: `history` 0.92256 / 0.90635 / 0.88600;
`shuffled` 0.90390 / 0.81830 / 0.76548; `none` 0.78936 / 0.68805 / 0.83474.

Gate checks:

- `history` ≥ 0.92863: **FAIL** (0.90703; below every existing causal arm).
- `history` − `shuffled` ≥ 0.005: pass (+0.028).
- `history` − `none` ≥ 0.005: pass (+0.134).

Verdict: **STOP.** Stage 2 is not run; the outer folds stay untouched.

Observations (post hoc, not evidence):

- The model does use its own file's context: it beats both controls. The claim
  that failed is the absolute one.
- The comparison is confounded by a weak base. With no context, this
  architecture scores 0.773, far below the window-only single-row TREA on the
  same summaries (0.92341). So the run does not isolate "learned row attention
  versus hand-built history"; it shows this particular tokenisation is a poor
  window model that context only partly rescues.
- Every fit overfits fast: best epoch 1–6 for `history` and `none`, with
  training loss falling to about 0.015 (`history`) and 0.10 (`none`). Physical-well AUC for `none` is 0.508,
  i.e. chance on unseen wells, consistent with memorising training groups.
- Seed spread is large (0.886–0.923 for `history`), as in the earlier
  single-row arms.
- `shuffled` beating `none` by 0.106 was not predicted. Another file's context
  may act as a population reference or as regularisation; this run cannot tell
  which.
- The entire shortfall is on physical wells; simulations are near ceiling for
  every arm.

A follow-up that put the same row attention on top of the single-row TREA or
TabM window encoder would be a new experiment needing its own pre-registration.
It is not licensed by this result.
