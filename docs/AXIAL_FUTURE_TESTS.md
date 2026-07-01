# Axial Time-Series Future Tests

Status after commit `9e9ed2c` (`Add axial time-series benchmark sweeps`).

## Current Readout

The axial direction is worth continuing, but the raw-window path is not finished.
Across HAR plus a five-dataset UEA sweep:

- Row-pooled temporal transformers are retired. They consistently underperform and
  should not be included in future sweeps unless needed as a historical control.
- Axial and conv-axial models extract real signal from raw multivariate windows.
- A well-engineered XGBoost over window statistics remains the practical bar.
- Stats-assisted axial often matches or beats XGBoost, which means the backbone can
  use the relevant information when it is exposed.
- Raw axial still trails stats-assisted axial on several datasets, so the raw
  tokenizer/local pattern extractor is the likely bottleneck.

## Benchmark Summary

Three-seed macro-F1 means from `logs/uea_sweep_core/summary.csv`:

| Dataset | XGB+stats | Axial | Conv-axial | Axial+stats | Conv-axial+stats |
|---|---:|---:|---:|---:|---:|
| ArticularyWordRecognition | 0.867 | 0.429 | 0.827 | 0.942 | 0.929 |
| BasicMotions | 0.932 | 0.905 | 0.886 | 0.857 | 0.866 |
| Epilepsy | 0.962 | 0.923 | 0.920 | 0.944 | 0.946 |
| NATOPS | 0.842 | 0.707 | 0.773 | 0.912 | 0.883 |
| RacketSports | 0.794 | 0.762 | 0.799 | 0.765 | 0.724 |

HAR three-seed macro-F1:

| Model | Macro-F1 |
|---|---:|
| XGB+stats | 0.890 |
| Axial | 0.798 |
| Conv-axial | 0.858 |
| Axial+stats | 0.888 |
| Conv-axial+stats | 0.887 |

## Working Hypothesis

Attention is learning useful feature/time routing, but the raw cell token is too weak.
The current tokenization is essentially:

```text
Linear([value, missing_mask]) + feature_id + time_pos
```

Attention can decide which tokens should interact, but richer local/nonlinear
patterns need to be formed before or inside the axial stack. The strongest evidence:
conv-axial improves raw HAR and RacketSports, while stats-assisted axial closes
large gaps on ArticularyWordRecognition and NATOPS.

## Next Tests

Prioritize tests that reduce the `raw model -> stats-assisted model` gap.

1. **Stronger raw tokenizer**
   - Replace single-cell `Linear([value, mask])` with a per-feature temporal
     tokenizer.
   - Start with depthwise temporal conv blocks because the simple conv stem already
     helped HAR and RacketSports.
   - Test kernel sizes 3, 5, 9 and residual vs non-residual stems.

2. **Patch encoder instead of average patching**
   - Current `time_patch_len` average-pools values/masks and loses local shape.
   - Encode each feature patch with a small shared MLP or depthwise Conv1d before
     axial attention.
   - Compare patch lengths 4, 8, 16 on HAR, NATOPS, and ArticularyWordRecognition.

3. **Multi-scale tokens**
   - Feed both fine cell/short-patch tokens and coarse window-summary tokens.
   - Coarse tokens should be learned from raw windows, not hand stats.
   - Goal: recover the benefit of XGB-style stats without manually exposing them.

4. **Nonlinear cross-feature mixer**
   - Add a gated MLP or bilinear-style mixer after feature-axis attention.
   - Motivation: attention supplies weighted sums; the FFN sees each token
     independently afterward. A mixer may help with nonlinear feature-feature
     interactions that stats/tree models capture easily.

5. **Better pooling**
   - Replace final mean pooling with learned query/class-token pooling over
     `[time, feature]` tokens.
   - Test one global query plus optional per-feature queries.

6. **Self-supervised pretraining**
   - Mask random cells, contiguous spans, and whole channels.
   - Pretrain on train windows only, then fine-tune labels.
   - Main check: does pretraining reduce the need for engineered stats?

7. **Dataset expansion**
   - Add more UEA multivariate datasets after the current core five.
   - Suggested next set: `Cricket`, `Handwriting`, `Libras`, `LSST`,
     `StandWalkJump`.
   - Add at least one forecasting/regression dataset later to avoid overfitting the
     conclusion to classification.

## Success Criteria

Use XGB+stats as a practical upper bar, not an enemy to beat on every dataset.

Continue the axial line if a raw model:

- stays within roughly 3-5 macro-F1 points of XGB+stats on most datasets,
- beats XGB+stats on at least some datasets,
- and reduces the gain from adding hand-engineered stats.

Kill or simplify any branch that only works after hand stats are concatenated.
