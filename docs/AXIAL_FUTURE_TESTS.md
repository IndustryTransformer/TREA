# Axial Time-Series Future Tests

Status after the raw-XGBoost control sweep in
`logs/uea_sweep_rawflat_core/summary.csv` and the nonlinear tokenizer/mixer
screen in `logs/uea_sweep_nonlinear_core/summary.csv`.

## Current Readout

The axial direction is worth continuing, but the raw-window path is not finished.
Across HAR plus a five-dataset UEA sweep:

- Row-pooled temporal transformers are retired. They consistently underperform and
  should not be included in future sweeps unless needed as a historical control.
- Add `xgb_raw_flat` to every forward-looking sweep. This is XGBoost over the
  flattened raw window only, with no lag, rolling, or summary variables. It is the
  clean tree baseline for "no pandas/SQL feature engineering."
- Axial and conv-axial models extract real signal from raw multivariate windows.
- A well-engineered XGBoost over window statistics remains the practical bar,
  but raw flattened XGBoost is the better automatic/no-feature-engineering
  baseline.
- Stats-assisted axial often matches or beats XGBoost, which means the backbone can
  use the relevant information when it is exposed.
- Raw axial still trails stats-assisted axial on several datasets, so the raw
  tokenizer/local pattern extractor is the likely bottleneck.
- The first nonlinear tokenizer/mixer screen did not produce a universal replacement
  for `conv_axial`. `patch_mlp_p8` and `tokenconv_k9` are useful targeted leads, but
  neither met the promotion rule for a 20-dataset sweep.

## Benchmark Summary

Three-seed macro-F1 means from `logs/uea_sweep_rawflat_core/summary.csv`:

| Dataset | XGB raw | XGB+stats | Axial | Conv-axial | Axial+stats | Conv-axial+stats |
|---|---:|---:|---:|---:|---:|---:|
| ArticularyWordRecognition | 0.805 | 0.867 | 0.429 | 0.827 | 0.942 | 0.929 |
| BasicMotions | 0.778 | 0.932 | 0.905 | 0.886 | 0.857 | 0.866 |
| Epilepsy | 0.723 | 0.962 | 0.923 | 0.920 | 0.944 | 0.946 |
| NATOPS | 0.813 | 0.842 | 0.707 | 0.773 | 0.912 | 0.883 |
| RacketSports | 0.833 | 0.794 | 0.762 | 0.799 | 0.765 | 0.724 |

HAR three-seed macro-F1:

| Model | Macro-F1 |
|---|---:|
| XGB+stats | 0.890 |
| Axial | 0.798 |
| Conv-axial | 0.858 |
| Axial+stats | 0.888 |
| Conv-axial+stats | 0.887 |

Summary tables should include both XGB controls:

- `xgb_raw_flat`: raw flattened values only; no feature engineering.
- `xgb_stats`: engineered window summary variables; practical expert baseline.

## Nonlinear Screen

Three-seed macro-F1 means from `logs/uea_sweep_nonlinear_core/summary.csv`:

| Dataset | XGB raw | Axial | Conv-axial | PatchMLP p8 | TokenConv k9 | Gated mixer | Query pool |
|---|---:|---:|---:|---:|---:|---:|---:|
| ArticularyWordRecognition | 0.805 | 0.429 | 0.827 | 0.658 | 0.342 | 0.568 | 0.323 |
| BasicMotions | 0.778 | 0.905 | 0.886 | 0.914 | 0.884 | 0.744 | 0.629 |
| Epilepsy | 0.723 | 0.923 | 0.920 | 0.935 | 0.902 | 0.913 | 0.884 |
| NATOPS | 0.813 | 0.707 | 0.773 | 0.577 | 0.791 | 0.686 | 0.500 |
| RacketSports | 0.833 | 0.762 | 0.799 | 0.736 | 0.710 | 0.732 | 0.734 |

Promotion rule was: beat `axial` by at least 3 macro-F1 points on average, or
beat `xgb_raw_flat` on at least 3/5 datasets. No new variant passed. Conclusions:

- `patch_mlp_p8` is the strongest new internal tokenizer. It helps
  ArticularyWordRecognition and Epilepsy but hurts NATOPS and RacketSports.
- `tokenconv_k9` is the best new NATOPS variant and beats `conv_axial` there, but
  it is weak on ArticularyWordRecognition and RacketSports.
- `gated_mixer` alone and `query_pool` should not be promoted in their current form.
- Existing `conv_axial` remains the best raw deep model to carry into larger sweeps.

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

1. **Promote conv-axial, not the new variants, to the 20-dataset sweep**
   - The first nonlinear screen did not beat the existing conv stem broadly.
   - Keep `patch_mlp_p8` and `tokenconv_k9` as targeted ablations, not defaults.

2. **Second-generation tokenizer**
   - Combine the good pieces: conv stem quality with learned patch/token projection.
   - Try residual depthwise Conv1d over `[value, mask]` before a patch MLP, with
     patch lengths 4/8 and kernel sizes 5/9.
   - Add more capacity only inside the tokenizer, not via fixed feature concat.

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
