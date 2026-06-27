# 3W Literature Benchmarks (Binary + Full Classification)

This file tracks published 3W results with explicit verification status.

Important: many 3W numbers are not directly comparable to our current setup because of:
- different class sets (often 8 classes, not 10),
- transient-only labeling rules,
- different train/test protocols,
- or binary anomaly detection instead of full multiclass diagnosis.

## Verification Levels

- `Primary`: directly verified from the cited paper/page text.
- `Secondary`: reported by another paper discussing the source.

## Binary Classification (Anomaly Detection)

| Work | Task definition | Metric(s) | Feature engineering | Verification | Notes |
|---|---|---:|---|---|---|
| Fernandes et al. (2024), JPET | One-class anomaly detection (binary normal vs anomaly) on 3W | LOF F1 = **0.920** (simulated, no FE), LOF F1 = 0.915 (simulated, FE), LOF F1 = 0.870 (real, FE), LOF F1 = 0.859 (real, no FE) | Both with/without FE tested | Primary | Values reported in abstract/body of the open-access paper. |
| Vargas et al. (2019) anomaly benchmark (as reported in Fernandes 2024) | Binary anomaly benchmark baseline | Isolation Forest F1 = **0.727**, OCSVM F1 = **0.470** | Not primary focus | Primary (via Fernandes quoting benchmark values) | Fernandes explicitly states these benchmark values when comparing methods. |

## Full Classification (Multiclass Event Diagnosis)

| Work | Task definition | Metric(s) | Feature engineering | Verification | Notes |
|---|---|---:|---|---|---|
| Turan & Jäschke (2021), IEEE Process Control | 8-class transient-stage multiclass classification (one rare class excluded) | Test macro F1 = **0.85**, accuracy = **0.88** (Decision Tree) | Yes (windowing + TSFRESH workflow) | Primary | Directly reported in Table V / conclusion text. |
| Turan & Jäschke (2021), IEEE Process Control | Same study, CV model sweep | CV F1 = **0.91**, accuracy = **0.94** (Random Forest) | Yes | Primary | Cross-validation result in Table IV, not held-out test metric. |
| Marins et al. (2021), JPSE | Fault classification with Random Forest on 3W-derived setup | Accuracy = **0.94** | Yes (statistical measures) | Primary (abstract-level) | Paper states seven fault classes and 94% accuracy; protocol differs from our 10-class setup. |
| Prior-work value cited in Turan & Jäschke (2021) | Random Forest prior result from [8] | Accuracy = **0.97** (reported with 102 trees, max depth 24) | Yes | Secondary | Mentioned by Turan as prior-work tuning result; treat as indirect unless re-verified in original [8] table. |
| This repo (TREA-C current best run) | 10-class 3W window classification | val macro F1 = **0.8453** | No manual FE | Primary | Current repo run on random file-level split. |

## Practical Interpretation

- Binary anomaly-detection papers often report very high F1, but this is a different target than full multiclass diagnosis.
- For multiclass diagnosis, scores around macro-F1 0.85 are competitive in published 3W studies, depending on class-set/protocol choices.
- Strong claims should use protocol-matched benchmarks (same class set, split strategy, and metric).

## Sources

1. Fernandes et al., *Anomaly detection in oil-producing wells: a comparative study of one-class classifiers in a multivariate time series dataset* (open access), 2024.  
   https://link.springer.com/article/10.1007/s13202-023-01710-6
2. Turan & Jäschke, *Classification of undesirable events in oil well operation*, 2021 IEEE Process Control.  
   PDF: https://jaschke.folk.ntnu.no/preprints/2021/TuranClassification_PC/009.pdf  
   DOI: https://doi.org/10.1109/PC52310.2021.9447527
3. Marins et al., *Fault detection and classification in oil wells and production/service lines using random forest*, Journal of Petroleum Science and Engineering (2021 volume; DOI year 2020).  
   https://doi.org/10.1016/j.petrol.2020.107879
4. Vargas et al., *A realistic and public dataset with rare undesirable real events in oil wells*, 2019.  
   https://doi.org/10.1016/j.petrol.2019.106223
