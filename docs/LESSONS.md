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

1. **The whole curve — trees included — sits above the target std (11.04):** on this
   temporal split *nobody beats a constant mean-predictor*. The task is drift-limited,
   not label-limited; the run measures drift robustness, not label efficiency.
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
