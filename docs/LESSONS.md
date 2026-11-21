# TREA — Lessons, Evaluation Protocol & Kill Criterion

Durable knowledge carried out of the consolidation. **Read this before changing
architecture or trusting a benchmark number.** Complements `CONSOLIDATION_PLAN.md`
(why four repos became one) and `../SEMANTIC_COLUMNS_SUMMARY.md` (column-identity
findings + bug fixes).

## 1. What we actually know (empirical)

- **Full-label 3W: trees win.** RF on statistical features = 0.92 macro-F1 vs ~0.81 for
  the best deep variant, at ~80× less compute. `trea_triple` is statistically *inferior*
  to plain `patchtstnan`. → In the supervised regime, architecture tinkering has a low
  ceiling. Stop optimizing it there.
- **Feature identity is the biggest single architectural lever found** (TREA-R: macro-F1
  0.464 → 0.678) — but that only un-cripples a broken default; it does **not** beat trees.
- **Semantic columns underperformed badly on single-dataset 3W** (val_acc ~0.5 vs 0.93).
  Likely an embedding-scale / projection-init bug — *and* single-dataset is the wrong test
  anyway (semantic's payoff is transfer). **Do not judge semantic columns on single-dataset
  accuracy; diagnose the scale issue before reusing.**
- **3W missingness is structural, not random.** Whole sensor columns are absent per well
  (sensor configuration differs by well), so the mask channel partly encodes *well identity*.

## 2. Evaluation protocol (the underbuilt part — fix before trusting ANY number)

This is the highest-leverage gap. The architecture debates are premature until the eval is
trustworthy.

- **Use well-disjoint (grouped) splits.** Because missingness encodes well configuration and
  the same well has multiple instances, an instance-level split lets the model shortcut on
  "which well is this." Current 3W splits are per-instance → **likely leaky**. Every existing
  number — *including RF's 0.92* — may be optimistic. Re-run the headline comparisons on
  well-grouped splits before drawing conclusions.
- **Per-instance, not per-window, metrics.** Windows from one instance are correlated;
  per-window scores inflate effective N and over-state confidence.
- **Macro-F1 / balanced accuracy only.** Classes 0+4 are ~91% of windows; raw accuracy lies.
- **Headline benchmark = label-efficiency curve.** macro-F1 vs # labels, with lines for
  RF / from-scratch / pretrained, on ≥2 datasets (3W + turbine). The crossover *is* the thesis.

## 3. Pre-registered success / kill criterion (decide NOW, not after seeing results)

The restart pattern thrives on never committing to a verdict. Pre-commit, **per task**
(the original single macro-F1 criterion can't span a regression + a classification dataset):

- **Turbine NOx (regression, temporal year split):** CONTINUE only if a pretrained→finetuned
  model beats XGBoost on test RMSE at ≤ **10%** of labels, and the pretrained label-efficiency
  curve dominates from-scratch across the sweep.
- **3W binary normal-vs-fault (well-disjoint):** CONTINUE only if the deep model beats RF on
  AUC over **unseen wells**. (Leaky multiclass 3W numbers are inadmissible — see §1/§2.)
- **KILL** the deep-model line if it beats neither baseline in its own regime → ship the
  semantic-column idea on a simpler backbone and move on.

### 3d. 3W binary well-disjoint gate — ADJUDICATED: FAIL (2026-07-03)

`examples/benchmark_3w_binary.py --split-mode well` (well-grouped split: 1,149 wells →
919 train / 230 val, no well spans both; 3 seeds; both `trea_triple` and
`trea_triple_stat_tokens` variants; RF and LOF baselines for context):

| Model | AUC | Anomaly F1 | Macro F1 | Accuracy | Sec/run |
| ----- | --: | ---------: | -------: | -------: | ------: |
| trea_triple_stat_tokens | 0.8447 | 0.8832 | 0.6689 | 0.8076 | 393.5 |
| trea_triple | — | 0.8878 | 0.6658 | 0.8133 | 297.5 |
| **RF** | **0.9584** | **0.9591** | **0.8202** | **0.9275** | **11.9** |
| LOF | — | 0.8126 | 0.5032 | 0.6959 | 0.9 |

**Verdict: FAIL, decisively.** RF wins every metric column, not marginally — AUC
0.958 vs 0.845, macro-F1 0.820 vs 0.669, anomaly-F1 0.959 vs 0.883 — while training
~33× faster (12s vs ~350s/run). Deep's weak point is the Normal class specifically
(normal-F1 0.45–0.47 vs RF's 0.68), i.e. it over-predicts anomaly on unseen wells;
RF's hand-stat features generalize to new wells far better than raw windows do here.
Both `trea_triple` variants (with and without stat tokens) land within noise of each
other — stat tokens did not close the gap.

## 3e. §3 KILL CLAUSE TRIGGERED (2026-07-03)

Both pre-registered §3 gates are now adjudicated FAIL:
- Turbine NOx label-efficiency (§3c/§3b protocol v2, final): **FAIL** — `pre_target`
  beat best-tree at only 1/4 low fractions and lost to scratch at 1%.
- 3W binary well-disjoint AUC (§3d): **FAIL** — RF beats deep on every metric.

Per the pre-registered kill clause: **the supervised deep-model line (from-scratch
and MTM-pretrained `SingleRow*`/`TriplePatchTransformer` models, evaluated head-to-head
against trees on these two benchmarks) is KILLED.** Do not open a v3 protocol, a new
backbone variant, or a new benchmark to re-litigate this — that is exactly the
restart pattern §3 exists to prevent. What survives, per the clause's own instruction
("ship the semantic-column idea on a simpler backbone") and independent evidence
gathered in this project:
- **Semantic column-name transfer** — separately pre-registered and validated
  5/5 (§3b core mechanism) — is NOT killed; it answers a different question
  (cross-schema transfer, not single-benchmark deep-vs-trees accuracy) and should
  ship on whatever backbone is simplest, not necessarily a full transformer.
- **Native missingness encoding** ([value, present]-flag, no imputation) is
  architecture-agnostic and cheap to keep regardless of backbone.
- **The GAUGE foundation-model design** (see memory: tabular-fm-deepdive) rests on a
  different, untested hypothesis — massive synthetic-prior pretraining, not small-corpus
  MTM on one dataset — and was never gated by §3. A kill here raises the evidentiary
  bar for funding it, but does not itself falsify it. Any decision to pursue it should
  be made explicitly, with its own pre-registered MVP gates, not inherited from this one.
- **The turbine salvage finding** (target-period SSL beat train-only SSL at every
  label fraction) is real evidence for transductive adaptation and should inform
  GAUGE's pretraining design even though the turbine benchmark itself is closed.

> **Pre-registered, revised for feasibility before running (2026-06-29):** the turbine set is
> single-turbine NOx **regression** (temporal split, not classification), and 3W has only ~40
> wells so a clean well-disjoint *multiclass* eval is infeasible (rare classes live on 1–2
> wells; RF macro-F1 0.64→0.26 file→well grouped). Hence the per-task criteria above. Locked
> before results; do not revise after seeing them.

### 3a. Semantic-column transfer — pre-registered MISS → KILL (same-schema rename), 2026-06-30

The distinctive semantic-column bet: a model trained on one schema applies to a renamed
schema via text-derived column identity. Test (`scripts/schema_transfer.py`): turbine NOx
split into disjoint Plant A / Plant B, where B is the **same sensors, paraphrased
descriptions + renamed/reordered codes**; frozen MiniLM embeds the descriptions. Semantic
NN recovers **9/9** A↔B correspondences (precondition met). Pre-registered criterion: VALID
iff at ≤10% B-labels **(a)** transfer_semantic < scratch_semantic AND **(b)** transfer_semantic
< transfer_index (a *charitable* index control that transfers the whole attention stack but
not the column identities). Result (3 seeds, Plant-B test RMSE, NOx units):

| frac | scr_idx | scr_sem | transfer_idx | transfer_sem | xgb |
|------|---------|---------|--------------|--------------|-----|
| 0.01 | 9.24 | 9.10 | **8.55** | 8.56 | 8.68 |
| 0.02 | 8.88 | 8.95 | **8.35** | 8.44 | 7.90 |
| 0.05 | 9.27 | 8.17 | **7.72** | 7.83 | 6.76 |
| 0.10 | 6.76 | 7.25 | **6.33** | 6.90 | 6.08 |
| 1.00 | 5.62 | 5.15 | **4.77** | 4.91 | 4.69 |

**(a) PASS** — backbone transfer is the low-label lever (both transfer arms beat their
scratch counterparts through 10%). **(b) FAIL at every ≤10% fraction** (index ahead by
+0.02/+0.09/+0.12/+0.57 — the gap *grows*): **semantic column names add nothing over fresh
learnable index identities.** XGB is best from 2% on; the deep edge is razor-thin and only
at ~1% (and transfer-general, not semantic).

**What this kills / doesn't:** kills "semantic helps renamed-but-identical schemas." Does
NOT test the regime where semantic has a *structural* edge index cannot match — **disjoint /
variable-width schemas** (B has sensors A never saw → index has no transferable row, semantic
places the new column near related ones via text). The same-schema arena favors the backbone
by construction (index just relearns 9 known columns from B's labels). The disjoint-column
test is the decisive follow-up; pre-register a fresh criterion before running it.

### 3b. Multi-schema scaling — semantic mechanism VALIDATED (2026-06-30)

The follow-up the pair test pointed to: semantic column transfer is a *meta-learning* bet
that needs schema *diversity*, so test whether transfer to a held-out dataset improves
when pretraining across K source schemas. Corpus = 15 diverse OpenML/sklearn regression
sets with meaningful column names (`scripts/download_tabular_corpus.py`); one shared
encoder pretrained on K sources, few-shot (n=200) transfer to a held-out dataset with
unseen columns (`scripts/schema_scaling.py`). Pre-registered: VALID iff transfer_semantic
(1) improves with K, (2) < transfer_index at max K, (3) < scratch_semantic.

**Result: 5/5 held-out datasets pass all three.** transfer_semantic beats the index
vocab-floor by large margins (index collapses unseen columns to `[UNK]` → flat, *worse
than scratch*; semantic is the only way to give a never-seen column a working identity)
and beats from-scratch on every dataset. This validates the semantic-column mechanism —
the cross-schema capability trees structurally lack — which the same-schema pair (§3a)
could not reach.

**Two honest caveats.** (1) The benefit **saturates by K≈2–4**; it is *not* a clean
monotone scaling law (california is best at K=4 then regresses; some slopes are within
noise). Claim: "pretraining on a few diverse schemas helps via column semantics," not
"monotone in K." (2) **XGB is still the absolute bar on 4/5** (abalone is the lone deep
win, transfer_semantic 2.37 < XGB 2.47). So this validates the *mechanism* (README pillar
2: cross-dataset transfer), not "deep beats trees." Next: does the semantic edge scale
with corpus size/diversity beyond K≈4, and does it hold with MTM (unlabeled) pretraining?

### 3c. Turbine gate — ADJUDICATED: FAIL as written (2026-07-02)

Full curve (`scripts/turbine_label_efficiency_deep.py`, temporal split train 2011–13 /
test 2014–15, 3 seeds, RMSE in original NOx units, target std **11.04**):

| frac | n_lab  | xgb          | cat          | scratch      | pre          |
| ---- | ------ | ------------ | ------------ | ------------ | ------------ |
| 0.01 | 222    | 12.310±0.273 | 12.059±0.246 | 12.148±0.513 | 13.144±0.625 |
| 0.02 | 444    | 12.615±0.464 | 12.459±0.498 | 12.212±1.021 | 13.172±0.171 |
| 0.05 | 1,110  | 12.877±0.224 | 12.718±0.262 | 13.049±0.289 | 12.802±0.160 |
| 0.10 | 2,219  | 12.314±0.141 | 12.422±0.197 | 13.004±0.462 | 12.182±0.392 |
| 0.25 | 5,548  | 12.264±0.090 | 12.442±0.141 | 13.721±0.461 | 13.162±0.665 |
| 0.50 | 11,096 | 12.645±0.269 | 12.530±0.128 | 13.874±0.040 | 14.076±0.441 |
| 1.00 | 22,191 | 12.631±0.026 | 12.392±0.064 | 14.206±0.606 | 14.174±0.512 |

**Verdict as pre-registered: FAIL.** Pretrained beats best-tree only at frac 0.10
(+0.132); loses at 0.01–0.05 and everywhere above 0.10. Recorded as a miss — the §3
turbine gate is spent. The **§3 KILL clause is NOT triggered** (it requires failing
*both* gates; the 3W binary well-disjoint gate is unadjudicated).

**What the run actually established (read before citing the FAIL):**

1. **Baseline correction (2026-09-16):** target standard deviation (11.04 in the
   earlier notes) is not the future RMSE of a deployable constant predictor.
   Predicting the 2011–2013 training mean on 2014–2015 gives **13.758 RMSE**;
   several results in this table beat that baseline. The earlier claim that nobody
   beat a constant predictor was incorrect. The year shift remains important, but
   the standard-deviation comparison did not establish that labels cannot help.
   See `NOX_TEMPORAL_PRETRAINING.md` for the updated diagnostic and per-budget
   constants calculated using only the labeled fitting rows.
2. **Deep's full-data i.i.d. win was an i.i.d. artifact.** Same stack: deep 4.31 < xgb
   4.47 on a random split vs deep 14.2 > xgb 12.6 temporal. Deep degrades monotonically
   with more labels (in-distribution val selection under shift); trees are flat and far
   more drift-robust. Report i.i.d. and shifted results side-by-side from now on.
3. **Protocol defects identified before the final verdict** (mechanism-based, logged
   mid-run): (a) no frozen-probe/low-LR arm at tiny N — full finetune @ lr 8e-4 on ~180
   rows explains pre<scratch at 1–2%; (b) model selection on an i.i.d. val slice of the
   *source* years explains deep worsening with labels; (c) no arm pretraining on
   unlabeled *target-period* features (legitimate: no labels used; the actual industrial
   deployment story and the one lever trees lack).

**Pre-registered protocol v2 (locked 2026-07-02, BEFORE running — final for turbine, no
further protocol revisions after seeing v2 results):**

- Same temporal split, same fractions, **5 seeds** at fracs ≤ 0.10 (3 above).
- Model selection for all deep arms on a **temporal val slice (last 10% of train time,
  i.e. late 2013)**, not an i.i.d. subsample.
- Deep arms: `scratch`, `pre` (SSL on train-period only), **`pre_target` (SSL on
  train-period + unlabeled test-period features — transductive, labeled as such)**.
- At fracs ≤ 0.05 each pretrained arm also reports a **frozen-encoder probe** (encoder
  frozen, head-only fit, lr 1e-3) and the finetune uses lr 8e-5 (10× lower); the better
  of probe/finetune per arm is that arm's score (chosen on the temporal val slice, never
  on test).
- **v2 success bar (same spirit as v1):** `pre_target` beats best-tree (xgb/cat) at
  ≥3 of the 4 fracs ≤ 0.10 AND never loses to `scratch` there. Anything less = turbine
  stays a tree benchmark; the deep line's continuation then rests entirely on the 3W
  binary gate. *(Typo corrected pre-run, before any v2 result existed: the sweep has
  4 fractions ≤ 0.10, not 3; the bar is ≥3/4 to preserve the intended ~2/3 stringency.)*

**v2 ADJUDICATED (2026-07-03): FAIL — turbine is a tree benchmark; case closed.**
`pre_target` beat best-tree on 1/4 low fracs (0.10, +0.009 ≪ its ±0.854 seed std) and
lost to scratch at 0.01 (bar violated). Full curve (RMSE, `logs/turbine_labeleff_v2.json`):
pre_target 13.82/13.64/13.24/12.32/12.01/12.71/13.21 vs best-tree
12.16/12.27/12.16/12.33/12.23/12.25/12.44 at fracs 1/2/5/10/25/50/100%. **Salvage
finding worth keeping:** target-period SSL beat train-only SSL at EVERY fraction
(e.g. 13.75→12.32 at 10%, 13.42→12.01 at 25%, where it beat all three trees) — the
transductive-SSL-under-drift mechanism is real but its magnitude is insufficient at
low labels, and the curve is an inverted-U: deep is closest to trees at 10–25% labels
and degrades toward full labels (fits the source distribution ever better while the
test drifts). Per the lock: no v3, no re-litigation. The §3 kill clause now rests
entirely on the 3W binary well-disjoint gate.

## 4. Architecture decisions (from the attention analysis)

- **Bottleneck:** the row encoder mean-pools features to one vector per timestep, so the
  temporal stage can't track a single sensor's trajectory.
- **Baseline-first, not axial-first.** A vanilla transformer over `[T,F]` has no bottleneck —
  it's the free reference. Build axial only if that reference shows the bottleneck bites.
- **Concat-project is rejected:** flattening `F·d → d` hard-codes feature count/order and
  **breaks cross-schema transfer** (conflicts with the thesis). Use attention-pool or axial
  (both set-based / transfer-compatible).
- **Axial / factorized** (cf. iTransformer, Crossformer) is the better long-term architecture
  *if* the bottleneck is shown to bite. Cost: ~F× the temporal stage; mind the
  `batch·seq < 65536` row-encoder CUDA limit.
- **Full joint `(T·F)²` attention:** too expensive (~24× current); not the first move.

## 5. ChatGPT-suggestion scorecard

| Suggestion                                                       | Verdict                                                                   |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Make feature identity first-class                                | ✅ Adopted — validated (0.464→0.678)                                       |
| Preserve missingness masks (no `nan_to_num` pre-model)           | ✅ Adopted — loader fixed                                                  |
| Baselines: MLP / TCN / GRU / vanilla transformer (+ trees)       | ✅ Adopted — RF already wins full-label                                    |
| Track macro-F1 / balanced accuracy                               | ✅ Adopted                                                                 |
| Ablations (no row encoder / feature IDs / missingness / context) | ◑ Partial — keep                                                          |
| Axial attention / keep feature tokens alive                      | ◑ Refined — right direction, wrong priority (baseline-first, conditional) |

ChatGPT's critique was directionally right but scoped to **supervised, single-dataset**. It
under-weighted the actual lever: the **low-label / transfer** regime, where trees structurally
can't compete. Keep that scoping correction in mind when taking further suggestions from it.

## 6. Do-not-repeat (bugs + anti-patterns)

- NaN-zeroing before the model (kills the mask channel).
- Class-weight vector counting the `null` class (CrossEntropy crash).
- Per-class metrics callback that clears its buffers before reading (use a standalone eval).
- Reporting accuracy instead of macro-F1.
- `batch·seq ≥ 65536` (row-encoder attention CUDA grid limit).
- **Starting a fresh repo when stuck.** Fix in place. No fifth repo.

### 3f. Seed-variance decomposition on turbine NOx (2026-09-15)

`scripts/turbine_seed_decomposition.py` (temporal split, v1 stack lr 8e-4): 3 fixed
labeled-subset draws × 5 init seeds each, fit/val split held fixed per subset, so
"between" = std of per-subset means (data draw) and "within" = mean std over inits
(model/optimizer). "ens" = RMSE of the 5-init mean prediction (no selection).

| frac | arm     | mean   | between | within | ens    |
| ---- | ------- | ------ | ------- | ------ | ------ |
| 0.01 | xgb     | 12.309 | 0.351   | 0.127  | 12.275 |
| 0.01 | scratch | 11.856 | 0.332   | 0.665  | 11.600 |
| 0.01 | pre     | 13.475 | 0.086   | 0.473  | 13.266 |
| 0.05 | xgb     | 12.903 | 0.276   | 0.152  | 12.863 |
| 0.05 | scratch | 12.967 | 0.096   | 0.660  | 12.769 |
| 0.05 | pre     | 12.522 | 0.170   | 0.667  | 12.408 |
| 0.10 | xgb     | 12.450 | 0.093   | 0.085  | 12.411 |
| 0.10 | scratch | 13.377 | 0.119   | 0.800  | 13.223 |
| 0.10 | pre     | 13.064 | 0.254   | 0.596  | 12.855 |

Findings: (1) deep spread is **init-dominated** at every fraction (within 0.47–0.80
vs between 0.09–0.33); XGB is the reverse (data-draw dominated, within ≤0.15). So the
"seed" in the v1 curve was the model, not the split. (2) Same-subset init range for
deep is 1–3.2 RMSE units (e.g. pre@5% subset 1: 10.79–14.01) — the "oracle seed beats
XGB" statistic was init luck. (3) 5-init averaging recovers 0.1–0.26 RMSE, closes
~1/3 of the init spread, and never beats the tree except scratch@1% (which single
scratch already did; pre@1% is worse than scratch — full-finetune at lr 8e-4 on ~180
rows, the v1 protocol defect). Turbine verdict unchanged (§3c locked); this is the
variance-attribution evidence behind the "ensembling + recipe stabilization before
any low-label claim" rule. Results: `logs/turbine_seed_decomp.{log,json}`.

### 3g. Pretrained-ICL baselines + synthetic-prior spike (2026-09-16)

**TabICLv2 (open, zero training) on the 51-table classification suite**
(`scripts/icl_vs_trees.py`, same splits/trees as `tabular_vs_trees.py`): beats best tree
on **36/51**, ties-or-beats 43/51, median edge +0.010 macro-F1, ~0.7 s/table. From-scratch
TREA on the same tables: 12/51. Largest wins are exactly TREA's failures (balance_scale
0.96 vs tree 0.71 vs TREA 0.23; vehicle +0.11; gesture +0.16). The field claim
("pretrained ICL beats tuned GBDTs") **holds on our corpus** → the from-scratch ceiling
was corpus/objective, not the tasks. Results: `logs/icl_vs_trees.json`.
TabPFN (regression) is gated behind a license token (`TABPFN_TOKEN`); not run.

**Synthetic-prior spike** (`scripts/synthetic_pretrain_spike.py`, pre-registered in the
docstring): pretrain TREA's single-row encoder on TabICL's open SCM prior (1k/10k
tables), few-shot (200 labels) on the 5 schema-scaling held-outs.
- `sup` objective (predict synthetic y from x, no context rows): **loss pinned at 1.00 =
  target variance for the entire run** — the encoder predicts the prior mean. Structural:
  each synthetic table has its own random function; a row-at-a-time model cannot identify
  it from one row, so the Bayes answer is the mean. This is exactly why TabPFN/TabICL need
  in-context rows. (10k run also NaN-diverged; guarded, irrelevant to the conclusion.)
- `mtm` objective (masked cells incl. target as a column): loss flat at 0.99 from step
  250 to 1250, never below the mean-predictor floor. Finetune: beats scratch 3/5
  (concrete −0.6, california −0.03, airfoil −0.2; worse on bike, tie abalone), all within
  1–2 seed std. **Pre-registered verdict: KILL** for plain pretrain→finetune on synthetic
  priors. Criterion (2) "beats real-corpus K=12 transfer_index" passed 5/5 but is
  **vacuous**: `scratch_index` also beats that reference on 5/5 — the K=12 index-transfer
  arm in §3b was itself worse than scratch-index, and scratch_index beats
  scratch_semantic at 200 labels on 4/5 (untrained MiniLM projection hurts few-shot).
- **Conclusion:** synthetic-prior scale cannot reach a row-independent encoder. The
  in-context mechanism (rows attending to labeled rows of the same table) is the
  load-bearing ingredient, not an inference-time add-on. Any GAUGE build must be ICL-first
  (TabPFN/TabICL-style row attention) with TREA's column semantics + missingness + time
  axis layered on, or it should fine-tune an open ICL model instead of training from
  scratch. Results: `logs/synth_spike.json`, `logs/synth_spike_{sup,mtm}.log`.

**3W binary well-disjoint, TabICL on the RF hand-stat features** (`scripts/three_w_icl.py`,
3 seeds): cap 20k context rows → AUC **0.9415** / macro-F1 0.803 (88 s, 2.7 GB); cap 100k
(seed 42 only) → AUC 0.9402 (744 s, 52 GB RSS) — more context does not help. **RF (200k
rows) still wins: AUC 0.9584 / macro-F1 0.820.** So on the one industrial benchmark the
pretrained-ICL model does NOT beat the tree; the ICL edge on the OpenML suite (36/51)
does not carry to 3W hand stats on unseen wells. Consistent with §3d: the well-disjoint
gap is a generalization-to-new-wells problem, not a model-class problem. Results:
`logs/three_w_icl_cap20k.json`, `logs/three_w_icl_cap100k.log`.

**Scope correction to §3g (2026-09-16, Kai's question):** the synthetic-prior KILL applies
to the **single-row tabular encoder** (attends across columns of one row; no time axis,
no cross-sample attention). It does NOT transfer to `TriplePatchTransformer` /
`PatchTSTNan`: a time-series window is its own context, so a temporal synthetic prior
with a forecasting (next-K rows) or masked-span objective IS identifiable from within
the sample (Chronos/TimesFM absorb synthetic series this way with no cross-sample
attention). Sensor-stream pretraining on temporal priors remains untested and live;
cross-sample (in-context) attention would still be needed for *labels* (not visible
inside a window), but not for the pretraining objective itself.

### 3h. Updated small TREA + ensembles; real-NOx SSL is worth confirming (2026-09-16)

The corrected nine-sensor NOx comparison with official reference implementations
finds same-period RMSE TabM **4.539**, XGBoost **4.589**, FT-Transformer **4.636**,
retuned TREA **4.765** (three initialization seeds, four validation presets per
family). The selected TREA uses two layers, dropout 0, lr .0003, product+bias,
and skips only the first attention-input norm. Averaging all three TREA predictions
gives **4.519**, versus **4.566** for the equally ensembled tree. See
`NOX_REFERENCE_MODELS.md`. Extra inference cost can close the remaining gap here.

Follow-up with late-source validation and future-year testing, fixed nested label
draws, three supervised seeds: scratch ensembles remain competitive even when
XGBoost fits all labels including the neural validation rows. At 1/2/5/10/100%
labels, scratch = **12.453/12.014/11.903/12.224/12.896**, all-label tree =
**13.196/12.057/12.181/12.250/12.798**. Source-only masked pretraining improves
the 1/2/5% ensembles to **11.895/11.692/11.492**, but worsens 10/full labels.
Using unlabeled future features gives **11.270/11.516/11.682/12.583/12.228**;
this is explicitly transductive. Both pretrained arms beat a matched untrained
encoder probe/low-rate-finetuning control at 1/2/5%.

These are updated-model diagnostics, **not an exact rerun or reversal of the old
gate**: corrected feature/fit-row fairness, different optimization, one label draw,
one SSL seed per arm, and a low-label adaptation-mode change. Seed variance remains
large, and ensemble gains must not be presented as single-fit results. Freeze the
recipe and replicate across label draws/SSL seeds before scaling pretraining.
The real-sensor SSL finding does not rehabilitate the unrelated synthetic
random-function prior. Full protocol, additional controls, baseline correction,
and verified predictions: `NOX_TEMPORAL_PRETRAINING.md`.

At Kai's request these settings are now the provisional single-row defaults:
two layers, dropout 0, supervised lr .0003 (MTM lr remains .0008). Independent
three-fit ensembling remains opt-in; the evidence does not isolate smaller depth
from the other recipe changes. See `SMALL_MODEL_DEFAULTS.md` for compatibility
and training-budget details. Compact experiment summaries are tracked in `results/`.

### 3i. Schema-transfer release gate — PASS 5/5 (2026-11-21)

The kill clause's "ship on a simpler backbone" is now `trea.transfer.SchemaTransferRegressor`
(single-row encoder pinned to the §3b config). The pre-registered release gate
(`scripts/schema_scaling.py`: K=4, 200 labels, 3 source × 3 few-shot seeds, §3b regression
corpus of 15 tables) adds the `scratch_index` arm §3b lacked. Held-out test RMSE:

| held-out | transfer_sem | transfer_idx | scratch_sem | scratch_idx | xgb |
| --- | ---: | ---: | ---: | ---: | ---: |
| california_housing | **0.803**±0.036 | 1.151 | 0.850 | 1.151 | 0.679 |
| concrete | **9.74**±2.33 | 14.71 | 11.37±1.33 | 14.76 | 6.92 |
| abalone | **2.463**±0.094 | 2.616 | 2.580±0.028 | 2.582 | 2.451 |
| airfoil_self_noise | **4.229**±0.277 | 6.216 | 4.620 | 6.265 | 3.144 |
| Bike_Sharing_Demand | **133.1**±8.1 | 162.7 | 143.2±7.2 | 162.2 | 104.9 |

**Verdict: PASS** — transfer_semantic beats all three gated arms on 5/5 (needed 4).
Ship `trea.transfer` as a feature. Honest caveats, none of which change the verdict:
(1) **XGBoost beats it on 5/5** (abalone ≈ tie); this is a capability for unseen schemas,
not an accuracy win over trees. (2) Several margins are within one seed std (concrete,
abalone). (3) Index pretraining does nothing (transfer_idx ≈ scratch_idx everywhere):
the gain is column-name semantics, not pretraining per se. (4) Source draws overlap
across held-outs, so the five results are not independent. Note: the gate's first run
crashed before producing numbers because the shared manifest had grown to 89 tables
incl. classification; the pool was restricted to the §3b corpus before any result
existed. Results: `docs/results/schema_transfer_gate.json`.
