# TREA Work Orders — Tier 0/1 fixes from the 2026-07 deep dive

Self-contained instructions for delegating implementation to any assistant session.
Paste the **Context block** plus ONE work order into a fresh chat. Work orders are
ordered by priority; WO-1..4 unblock everything else.

---

## Context block (paste this with every work order)

You are working in the TREA repo (`~/Work/TREA`), a PyTorch Lightning library for
tabular + time-series deep learning. Relevant paths:

- `trea/models/single_row.py` — columns-as-tokens single-row transformer
  (TabularEncoder / TabularRegressor / TabularClassifier / MaskedTabularEncoder).
- `trea/training/single_row.py` — LightningModules (SingleRowRegressor,
  SingleRowClassifier, SingleRowMTM), `build_trainer`, `load_best`.
- `trea/utils/single_row_data.py` — SingleRowConfig / TabularDS / collate fns.
- `scripts/tabular_vs_trees.py` — the 89-dataset deep-vs-XGB/RF benchmark harness
  (38 regression + 51 classification, 3 seeds, data in `data/tabular_corpus/`).
- `tests/test_single_row.py`, `tests/test_schema_transfer.py` — existing tests.

House rules (non-negotiable):
1. **Never encode a numeric value as scalar × vector.** Bins, basis expansions,
   concat+Linear, and vector⊙vector Hadamard are all fine.
2. Every behavior change goes behind a config flag, default = old behavior, so we
   get clean ablations. Exception: outright bug fixes (crashes, unfair-loss bugs).
3. NaN/inf inputs must keep working end-to-end (native missingness via present
   flag). Never add imputation.
4. Report benchmark results as mean±std over 3 seeds, macro-F1 (classification) /
   RMSE (regression). Never raw accuracy.
5. Do not start new repos/files-trees; fix in place. Match existing code style.
6. Tests: `uv run pytest tests/ -x -q`. Lint: `uvx ruff format && uvx ruff check --fix`.
   Do NOT run full training scripts inside the chat (tqdm floods output); write the
   code + a smoke test, the human runs the long jobs
   (`uv run python scripts/tabular_vs_trees.py --smoke` is the sanctioned quick check).

Background (why these orders exist): a literature review + local diagnosis found the
classification benchmark results are poisoned by (A) an unfair-loss bug — the deep
model uses plain CrossEntropy while RF gets `class_weight='balanced'`; six datasets'
deep scores exactly equal a constant-majority predictor (pc3, pc4, ilpd,
ozone_level_8hr, blood_transfusion_service_center, connect_4) — and (B) transformer
training instability on wide tables (post-norm + no warmup + fixed lr; e.g.
MiceProtein 0.128 vs RF 0.990 on *balanced* 8-class data). Separately, the numeric
value encoding and the MSE regression loss are behind 2024-26 best practice.

---

## WO-1 — Logit-adjusted classification loss (fixes Mechanism A) [small]

**Goal:** replace plain CE in `SingleRowClassifier` with logit-adjusted CE
(Menon et al., arXiv:2007.07314): during training add `tau * log(class_priors)` to
the logits before CrossEntropy (`tau=1.0` default, configurable). Class priors are
computed from the training labels (pass them in at construction or compute from the
first epoch — constructor arg preferred). Also add a **post-hoc** helper that applies
`logits - tau*log(priors)` at eval time for already-trained checkpoints.

Do NOT combine with class weights or WeightedRandomSampler (documented to hurt).

**Files:** `trea/training/single_row.py` (SingleRowClassifier), small plumbing in
`scripts/tabular_vs_trees.py` to pass train-split priors.

**Done when:** unit test shows (a) with tau=0 behavior identical to current CE;
(b) on a synthetic 90/10 imbalanced toy dataset, the trained classifier is NOT a
constant-majority predictor and macro-F1 beats the tau=0 run. Then the human
re-runs the 6 collapse datasets.

## WO-2 — Transformer stability kit (fixes Mechanism B) [medium]

**Goal:** in the shared `TransformerBlock` in `trea/models/single_row.py`:
(a) **QK-norm** — LayerNorm applied to queries and keys before the attention dot
product; (b) switch post-norm (`norm1(q + attn_out)`) to **pre-norm**;
(c) add **linear LR warmup** (5-10% of steps) to the optimizers in
`trea/training/single_row.py`. All three behind flags
(`qk_norm=True`, `pre_norm=True`, `warmup_frac=0.05` as new defaults is fine, but
keep the old path reachable for ablation).

Also add collapse diagnostics: log per-head attention entropy and the
predicted-class histogram each val epoch; in `scripts/tabular_vs_trees.py`, flag any
seed whose predictions are constant (prediction entropy ≈ 0) as `"collapsed": true`
in the results JSON instead of silently averaging it.

**Done when:** existing tests pass; a smoke run trains; a new test asserts
attention entropy is finite and logged; flags reproduce old behavior exactly when
disabled.

## WO-3 — Optimizer & selection hygiene [small]

**Goal:** in all three LightningModules in `trea/training/single_row.py`:
(a) Adam → **AdamW (decoupled), betas=(0.9, 0.95)**; (b) **parameter groups**: no
weight decay on embeddings, LayerNorm params, or biases; decay (3e-4 default) on
the rest; (c) model selection & early stopping for the classifier on
**val macro-F1** (torchmetrics) instead of val_loss; keep val_loss logged;
(d) optional label smoothing 0.1 flag for classification (off when reporting AUROC);
(e) raise default patience to 16 full epochs (TabM protocol) with best-checkpoint
reversion (already exists via `load_best`).

**Done when:** tests pass; a test asserts the no-decay group contains all
embedding/norm/bias params; classifier checkpointing monitors `val_macro_f1` (max).

## WO-4 — Data/MTM bug fixes (no flags, straight bug fixes) [small]

In `trea/utils/single_row_data.py` and `trea/training/single_row.py` /
`trea/models/single_row.py`:

1. **OOV categoricals:** `cat[c].map(config.token_dict)` yields NaN for unseen
   strings → garbage long. Fix: `.map(...).fillna(config.token_dict["[UNK]"])`
   before the long cast. Add a test: fit config on train df, build TabularDS on a
   df containing an unseen category → no crash, maps to [UNK].
2. **MTM loss over all cells → masked cells only.** In SingleRowMTM `_step`,
   compute MSE/CE only where the mask was applied. Also guard genuine-NaN targets
   (exclude non-finite target cells from the loss instead of poisoning it).
3. **MTM crashes when a dataset has no categorical columns** (`batch.categorical is
   None`). Fix + test.
4. **`mask_tensor` dtype fragility:** branches on `dtype == torch.float32`; make it
   `tensor.is_floating_point()`.

**Done when:** new tests for all four pass; existing tests green.

## WO-5 — HL-Gauss binned regression head + output hygiene [small/medium]

**Goal:** add `head="hl_gauss"` option to SingleRowRegressor (flag, default "mse"):
target (already standardized by the harness) → Gaussian-smoothed histogram over
~100 uniform bins spanning the train target range padded by 3σ, σ = 2× bin width;
loss = cross-entropy against that soft histogram; point prediction =
Σ p_i · bin_center_i (this is what eval consumes). Add optional
`bin_spacing="quantile"` and `symlog=True` variants for skewed targets.
Independent of head choice: **clip test-time regression predictions to the observed
train target range** in `scripts/tabular_vs_trees.py`.

References: arXiv:2403.03950, arXiv:2402.13425 (defaults above are theirs).

**Done when:** unit test: on a smooth 1-D toy function, hl_gauss head reaches RMSE
within 10% of the mse head; on a bimodal-target toy it beats mse. Harness accepts
`--head hl_gauss`.

## WO-6 — Numeric value encoding: dual representation [medium]

**Goal:** upgrade the numeric cell encoder in `trea/models/single_row.py` (currently
`identity_emb + Linear(2)([value, present])`). New (flagged, `value_encoder=
"dual"`): keep the continuous branch `[value, present]`, add a discrete branch —
**shared quantile-PLE**: per-column quantile bins (computed from train data,
~48-64 bins) soft-assigned over a bin-embedding bank **shared across all columns**
(nn.Embedding(64, d)), plus a small periodic basis (16 learned frequencies,
cos(2π·w·v+b)). Concat(continuous, PLE-weighted-sum, periodic) → Linear → d_model,
then fuse with column identity **two ways behind one flag**: `fusion="additive"`
(current) vs `fusion="hadamard"` (`e_col ⊙ MLP(value_vec)` + additive residual,
MedFuse-style; the value goes through an MLP to a d-vector first — never raw
scalar × vector). Missing cells: present=0, PLE/periodic branch replaced by a
learned MISSING vector.

Quantile boundaries must come from train split only and be stored on the module
(buffer) so checkpoints are self-contained.

**Done when:** permutation-invariance and NaN-native tests still pass; new tests
cover (a) value=0 ≠ zero token, (b) missing ≠ any observed value's encoding,
(c) shared bank shape [64, d] used by every column. Ablation entry points exist in
`scripts/tabular_vs_trees.py` (`--value-encoder`, `--fusion`).

## WO-7 — TabM-mini BatchEnsemble [medium]

**Goal:** add parameter-efficient ensembling (TabM-mini, arXiv:2410.24210) to
TabularEncoder behind `ensemble_k` flag (default 0 = off, recommended 32): a single
per-member multiplicative vector adapter `r_k ⊙ x` applied to token embeddings right
after the embedding layer (first adapter init random ±1), everything else shared;
training loss = mean of the k members' losses; inference = average probabilities
(classification) / bin distributions or means (regression). Batch is expanded
k-fold internally (efficient reshape, not a Python loop).

**Done when:** test: with k=4 on the toy dataset, per-member predictions differ
(diversity), ensembled val metric ≥ best single member; runtime overhead < 2× for
k=8 at d_model=128. Harness flag `--ensemble-k`.

## WO-8 — kNN / retrieval probes [small]

**Goal:** two cheap diagnostics, no architecture change:
1. `scripts/knn_probe.py`: for a trained (or SSL-pretrained) encoder, extract row
   embeddings (attention-pooled), run plain kNN (k = min(10·√N, 1000), L2) vs the
   same kNN on raw standardized features. Report both on a few datasets. If
   embedding-space kNN loses to raw-space kNN, the encoder isn't learning metric
   structure — say so loudly in the output.
2. Optional: a ModernNCA-style soft-kNN head (softmax(-||φ(x)-φ(x_i)||²) weighted
   voting, trained with NCA loss, 30-50% candidate subsampling per batch) as an
   alternative head behind a flag (arXiv:2407.03257).

**Done when:** probe runs on one corpus dataset end-to-end in < 2 min (small N)
and prints a comparison table.

## WO-9 — (Human-run) benchmark re-runs

After WO-1..5 land: `uv run python scripts/tabular_vs_trees.py` full re-run
(3 seeds) → this becomes the new honest baseline. Then ablations one flag at a
time (WO-6, WO-7). Kai runs these; assistants should NOT run them in-chat.

---

## Sequencing / parallelism

- WO-1, WO-3, WO-4 touch `trea/training/single_row.py` — do them in ONE session
  (in that order) to avoid merge conflicts.
- WO-2 and WO-6 touch `trea/models/single_row.py` — one session, WO-2 first.
- WO-5 (head + harness), WO-7 (encoder wrapper), WO-8 (new script) are independent
  and safe to parallelize across sessions.
- Commit per work order with message `WO-N: <summary>`; run tests + ruff before
  each commit.
