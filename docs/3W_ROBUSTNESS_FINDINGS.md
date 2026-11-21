# 3W: better F1, but no robust AUC parity

## Technical summary

**The goal is not achieved.** Causal-context TREA reaches mean outer-fold AUC **0.9191**, versus **0.9278** for matched XGBoost; TREA ranks below XGBoost in every fold. Neither mean-fold result reaches the requested 0.9474 AUC bar.

TREA does improve the operating-point result: pooled macro-F1 is **0.7903 versus 0.7648**, with thresholds selected on internal validation only. This does not establish reliable fault detection across physical wells. The model should not be promoted as a completed fix.

## What this comparison measures

The evaluation covers **358,874 windows across 549 groups**, including **20 physical wells** and 529 simulation instances. Every group is held out once across four outer folds; each model trains without its own outer groups. The fixed internal-validation population remains separate. Both previously consumed test partitions are excluded.

Scores are per-window binary normal-versus-fault AUC and two-class macro-F1, not independent event-detection performance. The current comparison uses **longer, strictly past sensor context**, so it is not the original isolated 96-reading benchmark. Historical 0.9068/0.9474 baselines are milestones, not directly comparable measurements on these cohorts.

The outer groups have historical development/fitting exposure. This is a grouped robustness study, **not pristine external confirmation**.

## AUC falls short in every matched fold

The table compares the three-fit probability ensembles within each outer fold; the table preserves exact AUC, macro-F1, and cohort counts. TREA loses a small amount of ranking performance in every fold, while improving macro-F1 in three of four. Fold 2 remains a major failure for both methods. The large fold spread is a generalization warning, not evidence that another lucky seed will solve the task.

### Fold metrics and scored populations

| Fold | Model | AUC | Macro-F1 | Windows | Normal | Fault |
| --- | --- | --- | --- | --- | --- | --- |
| Fold 0 | TREA | 0.9907 | 0.9311 | 73,631 | 15,429 | 58,202 |
| Fold 0 | XGBoost | 0.9968 | 0.9362 | 73,631 | 15,429 | 58,202 |
| Fold 1 | TREA | 0.8702 | 0.8629 | 104,554 | 44,487 | 60,067 |
| Fold 1 | XGBoost | 0.8739 | 0.8321 | 104,554 | 44,487 | 60,067 |
| Fold 2 | TREA | 0.8761 | 0.5671 | 104,636 | 12,651 | 91,985 |
| Fold 2 | XGBoost | 0.8963 | 0.5490 | 104,636 | 12,651 | 91,985 |
| Fold 3 | TREA | 0.9395 | 0.8233 | 76,053 | 8,885 | 67,168 |
| Fold 3 | XGBoost | 0.9443 | 0.7699 | 76,053 | 8,885 | 67,168 |

## Pooling fold scores reverses the AUC ranking

Pooled out-of-fold AUC is **0.9047 for TREA and 0.8675 for XGBoost**, despite TREA losing in every individual fold. Pooling compares probabilities produced by differently calibrated fold models. A read-only positive/negative-pair decomposition reproduces pooled AUC exactly: TREA's within-fold contribution is lower, and its advantage comes entirely from cross-fold score comparisons.

Therefore, **the pooled AUC difference is not evidence that TREA has better within-fold discrimination**. Report it alongside—not instead of—the mean fold AUC. Macro-F1 uses each fold's frozen decisions, so it does not apply an invented common threshold to pooled scores.

## Single fits remain unstable

The table separates seeds 42, 43, and 44 from explicit probability averaging. TREA's single-fit mean fold AUC is **0.9129 ± 0.0068** across fitting seeds; XGBoost's is **0.9272 ± 0.0019**. These are fitting-seed spreads, not confidence intervals or fold-to-fold variation. The ensemble improves TREA's mean fold AUC but does not close the gap.

### Single-fit and explicit ensemble results

| Order | Model | Fit | Mean fold AUC | Pooled AUC | Mean fold F1 | Pooled F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | TREA | Seed 42 | 0.9041 | 0.9178 | 0.7954 | 0.7999 |
| 1 | TREA | Seed 43 | 0.9207 | 0.9163 | 0.7484 | 0.7214 |
| 2 | TREA | Seed 44 | 0.9139 | 0.8679 | 0.7940 | 0.7866 |
| 3 | TREA | Three-fit ensemble | 0.9191 | 0.9047 | 0.7961 | 0.7903 |
| 4 | XGBoost | Seed 42 | 0.9249 | 0.8633 | 0.7718 | 0.7646 |
| 5 | XGBoost | Seed 43 | 0.9271 | 0.8675 | 0.7749 | 0.7697 |
| 6 | XGBoost | Seed 44 | 0.9297 | 0.8707 | 0.7695 | 0.7614 |
| 7 | XGBoost | Three-fit ensemble | 0.9278 | 0.8675 | 0.7718 | 0.7648 |

## Physical wells remain the main weakness

Simulation-instance AUC is near perfect for both models. Physical-well performance is much weaker and uneven. Source-level pooled AUC is also sensitive to cross-fold calibration, so it cannot replace the individual-fold comparisons. The exact source sample sizes and fixed-threshold F1 values are shown below.

### Source-specific pooled ensemble metrics

| Source | Model | Pooled AUC | Macro-F1 | Windows |
| --- | --- | --- | --- | --- |
| Physical wells | TREA | 0.6307 | 0.5502 | 142,445 |
| Simulation instances | TREA | 0.9988 | 0.9463 | 216,429 |
| Physical wells | XGBoost | 0.4632 | 0.4648 | 142,445 |
| Simulation instances | XGBoost | 0.9992 | 0.9568 | 216,429 |

### Large physical wells expose different failure modes

The post-hoc audit identifies WELL-00002 with TREA AUC **0.0831**, so a global threshold adjustment cannot repair its ranking. On WELL-00023 the frozen decision rule misses **all 14,192 fault windows**; WELL-00024 has **21,563 false negatives**. These examples identify failure modes, not new training targets or permission to retune the consumed outer folds. Single-class wells have undefined AUC; their fixed-two-class F1 must not be interpreted as a balanced two-class test.

## Uncertainty does not establish AUC equivalence

Intervals resample whole groups within each fold and physical/simulation stratum, with 500 replicates. They condition on the fitted models; overlapping training folds and historical development exposure add uncertainty beyond these intervals. Positive differences favor TREA.

The macro-F1 interval is positive, but the AUC intervals include zero. **An interval crossing zero is not an equivalence test.** The equal-fold AUC interval is an explicitly post-hoc sensitivity check on unchanged predictions; the original pooled analysis is preserved.

### TREA minus XGBoost: grouped uncertainty

| Metric | Difference | 95% lower | 95% upper | Interpretation |
| --- | --- | --- | --- | --- |
| Pooled AUC | 0.0373 | -0.0178 | 0.0697 | Prespecified; cross-fold calibration sensitive |
| Mean fold AUC | -0.0087 | -0.0234 | 0.0037 | Post-hoc sensitivity; fixed predictions |
| Pooled macro-F1 | 0.0254 | 0.0015 | 0.0411 | Prespecified; fold-specific frozen decisions |

## Experimental design and verification

Five candidates per fold—window TREA, context TREA, context TabM, window XGBoost, and context XGBoost—were fitted at seeds 42, 43, and 44. Internal-validation ensemble AUC selected context TREA and context XGBoost in all four folds. Selection and 221 artifact hashes were frozen before outer scoring.

The models receive identical sampled training indices (cap 200,000 per seed) and the same engineered summaries/history for the matched comparison. Neural preprocessing is fit on training rows only. The compact neural recipe uses two layers, width 128, zero dropout, learning rate 3e-4, and group-loss power 0.5; this is a joint recipe, not an isolated depth result.

History is constructed from raw sensors before annotation eligibility is applied, with a fixed 37-window physical warm-up and no simulation warm-up. Raw reconstruction exactly matches all four folds' training and validation inputs. Every selected checkpoint reproduced its saved validation probabilities before any outer scoring. The post-hoc audit independently verifies all 32 single-fit/ensemble metric rows, actual group coverage, probability averaging, and frozen ensemble thresholds.

## Next steps

- Keep the goal open; do not present this as XGBoost parity or a deployment-ready fault detector.
- Preserve these outer outcomes. Do not select another threshold or model using their labels.
- Prioritize physical-well representation and score portability over further blind tuning. A defensible development-only candidate is a robust initial-sensor reference that can use earlier unlabeled history, tested identically for neural and tree models.
- Any subsequent confirmation must explicitly address the lack of untouched physical wells; obtain independent wells where possible, or label reused-cohort results as exploratory.

## Questions that still matter

Can early sensor history provide stable reference levels across wells without assuming that the initial period is healthy? Are additional independent physical wells available? Can better within-well discrimination be obtained without trading away macro-F1? The present study does not resolve those questions.

## Reproducible evidence

[Exact frozen results](/Users/kailukowiak/Documents/Codex/2026-09-17/referenced-chatgpt-conversation-this-is-an/outputs/3w_robustness_results.json), [selection and hashes](/Users/kailukowiak/Documents/Codex/2026-09-17/referenced-chatgpt-conversation-this-is-an/outputs/3w_robustness_selection.json), and [post-hoc audit](/Users/kailukowiak/Documents/Codex/2026-09-17/referenced-chatgpt-conversation-this-is-an/outputs/3w_robustness_posthoc_audit.json).

The audit recomputes all 32 metric rows, verifies exact outer-group coverage, ensemble averaging and frozen thresholds, and preserves the original evaluation. The focused 3W/SingleRow regression suite passes all 79 tests; scoped Ruff and git diff --check pass. No commits or staging were performed.

The rich-report validator and its prescribed portable HTML fallback both reject Python/file-backed provenance because they require SQL. No source was relabeled or fabricated. This Markdown note and the exact JSON evidence are the available delivery; no rich report was rendered.
