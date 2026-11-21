# Real-NOx SSL replication — pre-registration and run plan (2026-11-21)

**Status: pre-registered, not run.** Nothing below has been fit on real data.
Lock this file before the first fit. Do not edit the criteria after any result exists.

## Question

LESSONS §3h reports that source-only masked pretraining (`pre`) improved the
three-init ensemble RMSE over `scratch` at 1/2/5% labels by about 4.5% / 2.7% / 3.5%.
That result used **one label draw** (seed 42) and **one SSL seed** (42).
§3h requires replication across label draws and SSL seeds, with the recipe frozen,
before any pretraining is scaled up.

The question is: **does the low-label benefit of source-only SSL survive fresh label
draws and independent SSL seeds?**

## What is frozen

Runner: `scripts/nox_ssl_replication.py`. It imports every recipe element from
`scripts/nox_temporal_pretraining.py` and changes none of them:

- Data: nine NOx sensors; historical 2011–2013, future test 2014–2015.
  Temporal validation pool = last 10% of historical rows.
- Feature standardization on the early 90% unlabeled pool. Target statistics from
  labeled fit rows only.
- Architecture: 2 layers, width 128, 4 heads, dropout 0, product+bias fusion,
  no first attention-input norm.
- Supervised: AdamW (.9, .95), weight decay .0003, 5% warmup, clip 1, batch 256,
  max 300 epochs, patience 30, best-validation checkpoint.
- SSL: 60 epochs, batch 512, lr .0008, 30% masked cells, masked-only MSE,
  checkpoint picked on fixed-mask SSL validation MSE.
- Adaptation: at ≤5% labels, frozen probe (lr .001) vs low-rate finetune (lr .00008),
  picked on temporal validation. At 10%, finetune at lr .0003.
- XGBoost: frozen 800-tree / depth-6 / lr .03 / .8 / .8 preset.

The runner marks a run **not admissible** if epochs, patience, or SSL epochs differ
from these values.

The only code change to the original runner adds a `seed` argument to
`draw_indices` and `pretrain`, plus small refactors (`write_data`, `XGB_PRESET`).
With the defaults, the original runner produces bit-identical encoders, splits,
and predictions. This was checked on synthetic data before and after the change.

## What varies

| Factor | Values | Controls |
| --- | --- | --- |
| Label draw | 101, 102, 103, 104, 105 | Which historical rows are labeled, and which late rows validate. Draws stay nested across fractions within one seed. |
| SSL seed | 42, 43, 44 | Encoder init, training masks, validation mask, batch order. The SSL corpus train/validation split stays at seed 42. |
| Init seed | 10, 11, 12 | Head init and shuffle, as before. Each cell reports the equal three-init ensemble. |

Fractions: 1%, 2%, 5%, 10%. Full labels are excluded. The label draw does not change
the full-label fit set, and the claim under test is about low labels.

**Reproduction cell.** Draw 42 with SSL seed 42 reruns the original cell. It is a
reproduction check only and is **excluded from the verdict**, because its result is
already known. Differences against the original numbers indicate environment drift
(torch / BLAS version, CPU), not a failed replication.

## Arms

| Arm | Description |
| --- | --- |
| `xgb` | Frozen preset, fits the same rows as the neural models |
| `xgb_all` | Frozen preset, fits fit + validation labels (old benchmark convention) |
| `scratch` | Supervised only |
| `pre` | SSL on unlabeled 2011–2013 features (**primary**) |
| `pre_target` | SSL on unlabeled 2011–2015 features (transductive; secondary) |
| `random` | Untrained encoder with the SSL seed's init, same probe/low-rate adaptation (≤5% only) |

## Pre-registered verdict (primary: `pre`)

Unit of analysis: the three-init ensemble future RMSE of one cell.
For draw *d* and SSL seed *s*, the paired delta is
`Δ(d, s) = pre(d, s) − scratch(d)`. Negative is better.

At each fraction in {1%, 2%, 5%}, `pre` **passes** if all three hold:

- **A.** Mean Δ over all 15 draw × SSL cells is < 0.
- **B.** The per-draw mean Δ (averaged over SSL seeds) is < 0 in **at least 4 of 5** draws.
- **C.** Mean of `pre(d, s) − random(d, s)` over all 15 cells is < 0.

Overall verdict:

- **REPLICATED**: passes at 3 of 3 low fractions.
- **PARTIAL**: passes at 2 of 3.
- **NOT REPLICATED**: passes at 1 or 0.

Decision rule:

- REPLICATED → the §3h low-label SSL benefit is confirmed on this dataset.
  Scaling pretraining (more data, more sensors) is justified.
- PARTIAL → record which fraction failed. Do not scale. One more decision is
  needed from Kai; do not open a new protocol to rescue it.
- NOT REPLICATED → record in LESSONS that §3h was a single-draw artifact. Do not
  scale pretraining on this evidence.

No other result changes the verdict. In particular, a win against trees is not
required here, and a loss to trees does not override a REPLICATED verdict.

## Secondary readouts (reported, not gating)

- `pre_target` under the same A/B/C rules (labeled transductive).
- 10% fraction: mean Δ for `pre` and `pre_target` vs scratch. Tests whether the
  original 10% reversal recurs.
- Means over draws for every arm, including `xgb` and `xgb_all`.
- Variance split: SD of per-draw Δ (between draws) and mean SD over SSL seeds
  (within draw).
- Bootstrap 95% interval over draws for mean Δ. Descriptive only.
- Single-fit means and SDs per cell, so ensembles are not mistaken for single fits.
- Reproduction-cell differences against the original numbers.

## Run plan

**Size.** 750 jobs: 1,032 neural fits (scratch 72, random-control 288,
pretrained 672), 144 XGBoost fits, and 6 SSL fits.
`--dry-run` prints the job counts without fitting.

**Cost estimate.** Measured on synthetic data with the real row counts, in a 4-core
cloud container (3 workers × 2 threads). Synthetic fits stopped after 60–160 epochs,
so real-data fits may run longer, up to the 300-epoch cap.

| Job type | Count | Seconds per job (measured) |
| --- | ---: | ---: |
| Neural, ≤5% labels, one mode (scratch) | 54 | 30–40 |
| Neural, ≤5% labels, probe + low-rate finetune (pre, pre_target, random) | 432 | 30–110 |
| Neural, 10% labels, finetune | 114 | ~100 |
| XGBoost | 144 | 3–10 |
| SSL, 60 epochs | 6 | ~330 (`pre`) to ~550 (`pre_target`) |

That sums to about 13 job-hours, or **about 4–5 hours wall-clock with 3 workers**.
If real fits run to the epoch cap, expect up to 2–3× longer. The original SSL fits
took 185 s / 296 s on Kai's machine, about 2× faster than here, so a local run should
take roughly 2–4 hours. More workers scale nearly linearly until cores run out.

**Where to run.** Kai's machine, where `data/nox/gt_20{11..15}.csv` already exist.
The cloud container cannot download the UCI dataset (network policy).

**Command.**

```sh
uv run python -m scripts.nox_ssl_replication --dry-run   # job counts only
uv run python -m scripts.nox_ssl_replication \
  --output logs/nox_ssl_replication \
  > logs/nox_ssl_replication.log 2>&1 &
grep -E "^(VERDICT|  frac|NOT ADMISSIBLE)" logs/nox_ssl_replication.log
```

Each job writes its own log under the output directory, so the main log stays short.
Rerunning the same command resumes: finished jobs are skipped, and the saved
`protocol.json` must match the invocation exactly. `--summarize-only` recomputes
`summary.json` from finished jobs. `--workers` and `--threads` change only speed,
not results.

**Order.** Phase 1 runs trees, scratch, and the random control (no SSL needed).
Phase 2 runs the six SSL fits. Phase 3 runs the pretrained arms. A partial run can be
summarized at any time; the verdict reads `INCOMPLETE` until every primary cell exists.

**Smaller first pass (optional).** To check timing and the reproduction cell first,
run `--draws 101 --ssl-seeds 42 --output logs/nox_ssl_replication_pilot`. A pilot
needs its own output directory, because its protocol differs from the full run.
It is **not** the pre-registered test: criterion B needs 5 draws. Its draw-101 cells
are not reused by the full run.

**After the run.** Append the verdict and the per-fraction table to LESSONS §3h and
to this file. Copy `summary.json` to `docs/results/nox_ssl_replication.json`.
