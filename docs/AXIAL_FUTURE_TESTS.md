# Axial Time-Series Future Tests

Status after the raw-XGBoost control sweep in
`logs/uea_sweep_rawflat_core/summary.csv` and the nonlinear tokenizer/mixer
screen in `logs/uea_sweep_nonlinear_core/summary.csv`. Newer interaction and
axis-patch screens are logged in `logs/uea_sweep_wild_core/summary.csv` and
`logs/uea_sweep_axis_patches/summary.csv`. The first conv-stem combination screen
is logged in `logs/uea_sweep_conv_combo/summary.csv`. A medium-capacity stress
check is logged in `logs/uea_sweep_conv_combo_medium/summary.csv`.

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
- The larger interaction screen shows that "more LLM-like" is not automatically
  better at this size. Global latents help NATOPS, conv/patch tokenization helps
  ArticularyWordRecognition, and bilinear/moment tokens are too unstable as
  standalone defaults.
- Rectangular axis-patch tokens are worth keeping as an ablation. They match or
  improve plain axial on BasicMotions, Epilepsy, and RacketSports, but they do not
  solve NATOPS or the long 25-class ArticularyWordRecognition case.
- Axis/global tokens become much more useful when placed after the conv stem.
  `conv_axial_axis_global` beats XGB+stats on ArticularyWordRecognition, while
  `conv_axial_axis_patches` and `conv_axial_global_latents` beat XGB+stats on
  BasicMotions. NATOPS remains unsolved.
- Scaling from `d_model=32`, 2 layers to `d_model=64`, 4 layers improves the
  Articulary conv-combo route and exposes capacity as a NATOPS bottleneck, but
  NATOPS still needs a different treatment.

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

## Wild Interaction Screen

Three-seed macro-F1 means from `logs/uea_sweep_wild_core/summary.csv`:

| Dataset | XGB raw | XGB+stats | Axial | Conv-axial | ConvPatch p8 | Moment tokens | Global latents | Bilinear mixer | LLM-ish |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ArticularyWordRecognition | 0.805 | 0.867 | 0.429 | 0.827 | 0.604 | 0.463 | 0.459 | 0.383 | 0.697 |
| BasicMotions | 0.778 | 0.932 | 0.905 | 0.886 | 0.885 | 0.864 | 0.799 | 0.812 | 0.779 |
| Epilepsy | 0.723 | 0.962 | 0.923 | 0.920 | 0.898 | 0.802 | 0.793 | 0.912 | 0.855 |
| NATOPS | 0.813 | 0.842 | 0.707 | 0.773 | 0.645 | 0.468 | 0.753 | 0.719 | 0.550 |
| RacketSports | 0.833 | 0.794 | 0.762 | 0.799 | 0.693 | 0.728 | 0.714 | 0.746 | 0.713 |

Conclusions:

- `conv_axial` remains the strongest general raw deep baseline in this screen.
- `axial_global_latents` is the cleanest targeted improvement on NATOPS, but hurts
  BasicMotions, Epilepsy, and ArticularyWordRecognition.
- `axial_llmish` helps ArticularyWordRecognition versus plain axial but still trails
  `conv_axial` and is worse on the other four datasets.
- `axial_bilinear_mixer` is not stable enough to promote. It can be strong on an
  individual seed but has collapses.
- `axial_moment_tokens` in the current always-on form should not be promoted. If
  revisited, it needs gating or conditional use.

## Axis-Patch Screen

`axial_axis_patches` adds rectangular feature-group/time-window tokens alongside the
raw feature tracks. Each patch token summarizes mean, std, start/end delta,
absolute temporal change, between-feature spread, and missing rate for a local
feature group and time patch. This is intended to give attention cheap coarse
tokens like "this sensor group is slowly increasing" without hand-engineered
external variables.

Three-seed macro-F1 means from `logs/uea_sweep_axis_patches/summary.csv`:

| Dataset | Axial | Conv-axial | XGB raw | XGB+stats | Axis patches |
|---|---:|---:|---:|---:|---:|
| ArticularyWordRecognition | 0.429 | 0.827 | 0.805 | 0.867 | 0.517 |
| BasicMotions | 0.905 | 0.886 | 0.778 | 0.932 | 0.896 |
| Epilepsy | 0.923 | 0.920 | 0.723 | 0.962 | 0.926 |
| NATOPS | 0.707 | 0.773 | 0.813 | 0.842 | 0.673 |
| RacketSports | 0.762 | 0.799 | 0.833 | 0.794 | 0.770 |

Readout:

- Axis patches support the user's row/column-patch hypothesis enough to keep them.
  They match plain axial on Epilepsy, nearly match it on BasicMotions, and slightly
  improve RacketSports.
- They do not replace a local convolutional inductive bias. On
  ArticularyWordRecognition, axis patches improve plain axial but are far behind
  `conv_axial`.
- NATOPS remains the strongest sign that coarse summaries alone are insufficient;
  global latents and/or stats-style information matter more there.
- Next version should test gated axis patches, multiple feature patch sizes, and
  axis patches combined with `conv_axial`, not only with `conv_patch_mlp`.

## Conv-Combo Screen

Three-seed macro-F1 means from `logs/uea_sweep_conv_combo/summary.csv`:

| Dataset | XGB raw | XGB+stats | Conv-axial | Axial axis | Conv axis | Conv global | Conv axis+global |
|---|---:|---:|---:|---:|---:|---:|---:|
| ArticularyWordRecognition | 0.805 | 0.867 | 0.827 | 0.517 | 0.873 | 0.841 | 0.885 |
| BasicMotions | 0.778 | 0.932 | 0.886 | 0.896 | 0.951 | 0.949 | 0.840 |
| Epilepsy | 0.723 | 0.962 | 0.920 | 0.926 | 0.916 | 0.902 | 0.913 |
| NATOPS | 0.813 | 0.842 | 0.773 | 0.673 | 0.727 | 0.749 | 0.683 |
| RacketSports | 0.833 | 0.794 | 0.799 | 0.770 | 0.761 | 0.819 | 0.757 |

Readout:

- The row/column patch idea is materially stronger after a conv stem. It no longer
  just adds coarse summaries to weak raw tokens; it summarizes locally denoised
  trajectories.
- `conv_axial_axis_global` is the first raw/no-feature-engineering model in these
  screens to beat XGB+stats on ArticularyWordRecognition.
- `conv_axial_axis_patches` and `conv_axial_global_latents` beat XGB+stats on
  BasicMotions.
- `conv_axial_global_latents` is the best raw deep model on RacketSports in this
  screen, but still trails raw flattened XGB.
- NATOPS remains a failure case. The next NATOPS branch should not be "more tokens";
  it needs either better supervision/pretraining, more capacity, or a different
  inductive bias.
- The fixed `axis+global` combination is not generally stable. The better version
  is likely a learned gate/router over raw, axis-patch, and global-latent streams.

## Medium-Capacity Screen

Three-seed macro-F1 means from
`logs/uea_sweep_conv_combo_medium/summary.csv`. This used `d_model=64`, 4 layers,
4 heads, and the same 60-epoch schedule.

| Dataset | XGB raw | XGB+stats | Conv-axial | Conv axis | Conv axis+global |
|---|---:|---:|---:|---:|---:|
| ArticularyWordRecognition | 0.805 | 0.867 | 0.837 | 0.856 | 0.876 |
| NATOPS | 0.813 | 0.842 | 0.794 | 0.708 | 0.696 |

Readout:

- Medium `conv_axial_axis_global` beats XGB+stats on ArticularyWordRecognition and
  improves over medium plain conv. This supports the scale-with-data hypothesis for
  longer/many-class sequence datasets.
- Medium `conv_axial` briefly beats XGB+stats on one NATOPS seed, but the average
  still trails raw XGB and XGB+stats. The variance is high.
- Axis/global tokens hurt NATOPS at medium size. For NATOPS, treat extra summary
  tokens as harmful until a gate/router or pretraining says otherwise.
- Larger models should be paired with stronger data: public multi-dataset
  pretraining, synthetic simulation windows, or both. Scaling on the current tiny
  supervised splits alone is too noisy.

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

1. **Promote conv-combo variants to the next larger sweep**
   - Carry `conv_axial`, `conv_axial_axis_patches`,
     `conv_axial_global_latents`, and `conv_axial_axis_global`.
   - Keep plain `axial_axis_patches` only as an ablation; it is not competitive on
     long/many-class data without the conv stem.

2. **Second-generation tokenizer**
   - Combine the good pieces: conv stem quality with learned patch/token projection.
   - Try residual depthwise Conv1d over `[value, mask]` before a patch MLP, with
     patch lengths 4/8 and kernel sizes 5/9.
   - Add more capacity only inside the tokenizer, not via fixed feature concat.

3. **Multi-scale tokens**
   - Feed both fine cell/short-patch tokens and coarse window-summary tokens.
   - Coarse tokens should be learned from raw windows, not hand stats.
   - Goal: recover the benefit of XGB-style stats without manually exposing them.
   - First pass implemented as `axial_axis_patches` and
     `conv_axial_axis_patches`; next pass should add gates/router weights.

4. **Nonlinear cross-feature mixer**
   - Add a gated MLP or bilinear-style mixer after feature-axis attention.
   - Motivation: attention supplies weighted sums; the FFN sees each token
     independently afterward. A mixer may help with nonlinear feature-feature
     interactions that stats/tree models capture easily.
   - First bilinear pass is too unstable to promote; revisit only with residual
     scaling, gates initialized near zero, or larger models.

5. **Better pooling**
   - Replace final mean pooling with learned query/class-token pooling over
     `[time, feature]` tokens.
   - Test one global query plus optional per-feature queries.
   - Global latent mean pooling is useful on NATOPS and should be combined with a
     stronger local tokenizer before dismissal.

6. **Self-supervised pretraining**
   - Mask random cells, contiguous spans, and whole channels.
   - Pretrain on train windows only, then fine-tune labels.
   - Main check: does pretraining reduce the need for engineered stats?
   - Masked autoencoder scaffold exists in `scripts/pretrain_axial_autoencoder.py`;
     next step is a real pretrain/fine-tune comparison, not just reconstruction
     smoke tests.

7. **Dataset expansion**
   - Add more UEA multivariate datasets after the current core five.
   - Suggested next set: `Cricket`, `Handwriting`, `Libras`, `LSST`,
     `StandWalkJump`.
   - Add at least one forecasting/regression dataset later to avoid overfitting the
     conclusion to classification.

8. **Scale with data, not before data**
   - The current default model is intentionally small (`d_model=32`, 2 layers) for
     toy UEA screens and limited labels.
   - Larger models are reasonable once pretraining uses many public time-series
     datasets or generated simulation windows.
   - `scripts/sweep_uea_axial.py` now exposes `--d-model`, `--n-head`,
     `--num-layers`, and `--dropout`, so medium screens can be launched without
     editing code.
   - First medium check completed. Keep medium `conv_axial_axis_global` for
     Articulary-like datasets; keep medium plain `conv_axial` for NATOPS-like
     datasets until a learned router exists.
   - Next scale step should not be another supervised-only size bump. It should be
     masked/pretext pretraining across many UEA datasets and generated simulation
     windows, followed by the same five-dataset fine-tune battery.

## Success Criteria

Use XGB+stats as a practical upper bar, not an enemy to beat on every dataset.

Continue the axial line if a raw model:

- stays within roughly 3-5 macro-F1 points of XGB+stats on most datasets,
- beats XGB+stats on at least some datasets,
- and reduces the gain from adding hand-engineered stats.

Kill or simplify any branch that only works after hand stats are concatenated.
