# TREA — Lessons, Evaluation Protocol & Kill Criterion

Durable knowledge carried out of the consolidation. **Read this before changing
architecture or trusting a benchmark number.** Complements `CONSOLIDATION_PLAN.md`
(why four repos became one) and `../SEMANTIC_COLUMNS_SUMMARY.md` (column-identity
findings + bug fixes).

## 1. What we actually know (empirical)

- **Full-label 3W: trees win.** RF on statistical features = 0.92 macro-F1 vs ~0.81 for
  the best deep variant, at ~80× less compute. `treac_triple` is statistically *inferior*
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

The restart pattern thrives on never committing to a verdict. Pre-commit:

- **CONTINUE** the deep-model line only if: pretrained beats RF by ≥ **[X]** macro-F1 at
  ≤ **[Y]%** labels, on ≥2 datasets, on **well-disjoint** splits.
- **KILL** it if it can't beat RF + a vanilla transformer in *any* regime → ship the
  semantic-column idea on a simpler backbone and move on.
- Fill X / Y from the real industrial label budget (how few labels a deployment actually has).

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

| Suggestion | Verdict |
|---|---|
| Make feature identity first-class | ✅ Adopted — validated (0.464→0.678) |
| Preserve missingness masks (no `nan_to_num` pre-model) | ✅ Adopted — loader fixed |
| Baselines: MLP / TCN / GRU / vanilla transformer (+ trees) | ✅ Adopted — RF already wins full-label |
| Track macro-F1 / balanced accuracy | ✅ Adopted |
| Ablations (no row encoder / feature IDs / missingness / context) | ◑ Partial — keep |
| Axial attention / keep feature tokens alive | ◑ Refined — right direction, wrong priority (baseline-first, conditional) |

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
