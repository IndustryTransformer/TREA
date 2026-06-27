# IndustryTransformer — Consolidation Plan & Merge/Split Decision

Cross-repo strategy doc spanning the four current attempts. Dated 2026-06-27.

## The situation

Four codebases attack the same idea — attention over industrial tabular time-series,
value+mask missingness, column-name semantics, intra-row (feature) + inter-row
(temporal) attention, SSL pretraining for low-label regimes, benchmarked vs trees:

| Repo | Core bet | SSL/pretrain | Baselines | Maturity |
|---|---|---|---|---|
| `~/Work/IndustryTransformer/Hephaestus` | numeric-projection tabular attention; multi-output | yes (turbine/NOx) | some | churned (modes added then removed) |
| `~/Work/IndustryTransformer/TabNCT` | intra-row + inter-row attention + column-name tokens; BERT/causal | yes (w3) | yes (xgboost) | many instability-chasing variants |
| `~/Work/TREA-C` | patch-based, column-aware (PatchTST-NaN) | yes (masked/temporal/contrastive/causal) | yes (RF, 5-seed non-inferiority stats) | most mature infra |
| `~/Work/TREA-R` | dual-stage row encoder (feature attn → temporal attn) | no | ad-hoc (`eval_3w.py`) | newest, thinnest |

## The core diagnosis

**The restart pattern is the bug, not any architecture.** Each fresh repo re-implements
the same ~80% (data loading, triple encoding, column embeddings, SSL, benchmarking) and
re-introduces the same defects. Three were re-found in TREA-R this week alone:
NaN-zeroing that defeats the mask channel, an 11-vs-10 class-weight crash, and a
per-class metrics callback that never prints. These almost certainly recur across repos.

The ~20% that differs is the attention factorization. **TREA-C's own benchmark shows
that 20% has a low ceiling:**

| Model (3W, 5-seed) | Macro-F1 | Train s/run |
|---|---|---|
| rf_stat_features (Random Forest) | **0.9194** | 12 |
| multidataset_none | 0.8330 | 276 |
| patchtstnan | 0.8278 | 294 |
| multidataset_auto | 0.8198 | 294 |
| treac_triple (fanciest) | 0.8084 | 1006 |

`treac_triple` is statistically **non-inferior = False** vs every comparator, including
plain `patchtstnan`; it trails RF by 0.111 macro-F1 at 80× the compute. In the
**fully-supervised** regime, trees win and architecture tinkering is near-futile.

## Where the value actually is (the thesis)

The only regime where DL structurally beats trees is **low-label + pretraining +
transfer** — trees cannot use unlabeled data; masked SSL can. The turbine/NOx result
(DL-pretrained beats XGBoost until XGBoost has ~80% of labels) is the supporting
evidence. So the project's worth rides entirely on the label-efficiency / transfer
story, which lives in the SSL + multi-dataset machinery — not in the attention design.

## Decision: MERGE / consolidate (do not maintain parallel repos)

"Split on columns vs rows" is a smaller version of the real problem (4-way duplication).
The merged rows+columns architecture being contemplated substantially **already exists**
(TabNCT does intra-row + inter-row + column-name tokens). This is selection +
consolidation, not new architecture.

### Plan

1. **Pick one base; freeze (archive) the other three.** Decide on *least technical debt
   + cleanest SSL/eval infra*, NOT on attention design. Prior: TREA-C (eval/SSL/
   multi-dataset infra is worth more than any variant); TabNCT is architecturally
   closest to the target. → **First action: a quick comparative audit of TabNCT vs
   TREA-C** to choose on evidence.
2. **Harvest the few genuinely distinct ideas** into the base as flagged, ablatable
   components: TabNCT's first-class intra+inter-row attention and column-name-as-tokens;
   Hephaestus's numeric projection *if it earns it* in an ablation. Drop the rest.
3. **No fifth repo.** Fix bugs in the base.
4. **Port the known fixes once** (from `SEMANTIC_COLUMNS_SUMMARY.md`): NaN-mask
   preservation, class-weight null filter, macro-F1/balanced-acc reporting, the
   `B*T < 65536` attention limit.
5. **Run the one decisive experiment:** label-efficiency curves (RF vs from-scratch vs
   pretrained) on W3 **and** turbine. The crossover *is* the thesis. If pretraining does
   not beat RF at low label counts on ≥2 datasets, the architecture question is moot —
   ship the semantic-column idea on a simpler backbone.

### Decision rule for split-vs-merge

Merge. Keep at most an *ablation flag* separate (row vs column attention), never a repo.
The architecture only matters if step 5 first proves the low-label thesis is real.

## Open question to resolve first

Which base — TREA-C (infra-mature) or TabNCT (architecture-closest)? Needs a shallow
comparative audit of both: SSL cleanliness, eval rigor, data pipeline, technical debt,
and how hard each is to extend with the harvested components.
