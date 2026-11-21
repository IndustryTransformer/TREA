# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in
this repository.

## Project Overview

TREA (Triple-Encoded Attention) is a PyTorch Lightning-based library for industrial
tabular time series. The core innovation is a triple-encoded architecture that handles
missing values by encoding them as separate channels (value channels, mask channels, and
column embeddings) rather than preprocessing NaNs.

This is a **consolidated** repo: it merges four archived prototypes (TREA-C, TREA-R,
TabNCT, Hephaestus), seeded from TREA-C. The Python package is still named `trea/`.

**Status (read before optimizing anything): the supervised deep-vs-trees line is
KILLED (July 2026).** Both pre-registered gates in `docs/LESSONS.md` §3 failed:

- **Turbine NOx label efficiency (temporal split):** FAIL. The pretrained model beat the
  best tree at 1 of 4 low-label fractions and lost to from-scratch at 1% (§3c, protocol v2).
- **3W binary normal-vs-fault on unseen wells:** FAIL. RF beat deep on every metric:
  AUC 0.958 vs 0.845, macro-F1 0.820 vs 0.669, at ~33× less compute (§3d).

The kill clause fired (§3e). Do NOT open a v3 protocol, a new backbone variant, or a new
benchmark to re-litigate head-to-head deep-vs-trees accuracy. That is the restart pattern
the kill criterion exists to prevent.

**What still stands:**

- **Semantic column-name transfer.** Validated 5/5 on held-out datasets with unseen
  columns (§3b). It is a cross-schema capability that trees and index embeddings lack. It
  should ship on the simplest backbone that works, not necessarily a full transformer.
  Caveats: the benefit saturates by K≈2–4 source schemas, and XGBoost is still the
  absolute bar on 4 of 5 datasets.
- **Native missing-value mask.** The [value, present]-flag encoding, with no imputation,
  is architecture-agnostic and cheap to keep on any backbone.
- **Real-NOx pretraining: pending replication.** Updated small-TREA ensembles with masked
  pretraining beat the tree at 1/2/5% labels (§3h). This is one label draw and one SSL
  seed per arm. It is not a reversal of the gate. Replicate across label draws and SSL
  seeds before scaling pretraining.

Open, not gated by §3: the GAUGE foundation-model design and temporal synthetic-prior
pretraining for `TriplePatchTransformer` / `PatchTSTNan` (§3e, §3g). Each needs its own
pre-registered gates before work starts. Benchmark with macro-F1 / balanced accuracy
(classification) or RMSE (regression), never raw accuracy.

### Read first (orientation docs)

- `docs/LESSONS.md` — empirical findings, **evaluation protocol, and pre-registered kill
  criterion**. Read before changing architecture or trusting any benchmark number.
- `README.md` — the original thesis, the kill verdict, and what survives it.
- `docs/NOX_TEMPORAL_PRETRAINING.md` — the real-NOx pretraining result awaiting replication.
- `docs/SMALL_MODEL_DEFAULTS.md` — current provisional single-row defaults.
- `docs/CONSOLIDATION_PLAN.md` — why four repos became one; what to harvest.
- `SEMANTIC_COLUMNS_SUMMARY.md` — column-identity findings and catalogued bug fixes.
- `docs/trea_original_README.md` — inherited TREA-C API/usage reference.

## Current focus (immediate next tasks)

The label-efficiency curve on 3W + turbine is done, and it triggered the kill (see
Status above). Work that remains open after the verdict:

1. **Replicate the real-NOx pretraining result** (`docs/LESSONS.md` §3h). Freeze the
   recipe, then repeat across label draws and SSL seeds. Report single fits and
   ensembles separately.
2. **Ship semantic column-name transfer on a simple backbone** (§3b, §3e). Keep the
   native missing-value mask.
3. **Decide on GAUGE explicitly** (§3e, §3g). If pursued, it must be ICL-first and carry
   its own pre-registered MVP gates. It does not inherit a pass from this line.

Not in scope: new backbone variants, axial-attention work, or architecture harvests from
the archived prototypes aimed at beating trees on single-benchmark accuracy.

**Anti-thrash rule:** fix bugs in place — do NOT start a fifth repo. The four prototypes
were archived precisely to stop that pattern.

**Checkpoint-loading gotcha:** the package was renamed `treac` → `trea`, so pretrained
checkpoints pickled under the old path may need
`import trea, sys; sys.modules["treac"] = trea` before `load_from_checkpoint`.

## Development Commands

### Setup

```bash
uv sync                    # Install dependencies
uv sync --group dev        # Install with dev dependencies
```

### Code Quality

```bash
uvx ruff format       # Format code
uvx ruff check --fix  # Lint and auto-fix
uvx ty check          # Type check with ty
```

### Testing

```bash
uv run pytest                                         # Run all tests (testpaths=tests/)
uv run pytest tests/test_column_embeddings.py         # Single file
uv run pytest tests/test_column_embeddings.py::test_x # Single test
```

### Running training scripts

Running training/experiment scripts is fine. The one hazard is unbounded output
(tqdm progress bars, per-epoch logs) overflowing the context window, so:

- **Redirect output to a log file and surface only the summary lines** (e.g.
  `uv run python scripts/foo.py > /tmp/foo.log 2>&1 &`, then tail/grep for the result
  rows), and prefer running long jobs in the background.
- Disable progress bars where you can. `build_trainer` in `trea/training/single_row.py`
  already sets `enable_progress_bar=False`; `scripts/tabular_vs_trees.py` also does
  `logging.disable(logging.INFO)` to mute Lightning's per-fit device banner — copy that
  pattern into scripts that still print it.

## Architecture

### Core Models (trea/models/)

**TriplePatchTransformer** (`triple_attention.py`)

- Base transformer for time series with numeric and categorical features
- Triple-patch encoding: value channels + mask channels + column embeddings for NaN handling
- Supports classification and regression tasks
- Key params: `C_num`, `C_cat`, `T` (sequence length), `d_model`, `n_head`, `num_layers`

**MultiDatasetModel** (`multi_dataset_model.py`)

- Unified model with three configurable modes:
  - `'standard'`: Multi-dataset training with column embeddings
  - `'variable_features'`: Handles datasets with different feature counts
  - `'pretrain'`: Self-supervised learning with SSL objectives
- Uses unified feature space with automatic padding/masking
- Supports multiple column embedding strategies: BERT, auto-expanding, simple
- Key params: `max_numeric_features`, `max_categorical_features`, `mode`,
  `column_embedding_strategy`

**PatchTSTNan** (`patchtstnan.py`)

- Patch-based time series transformer with NaN handling
- Efficient temporal modeling using patch-based processing

### Column Embeddings (trea/models/embeddings.py)

Column embeddings provide semantic understanding of feature names:

- **Simple**: Learned embeddings (lightweight)
- **BERT**: Semantic embeddings using pre-trained language models
- **Frozen BERT**: Cached embeddings for multi-dataset efficiency
- **Auto-expanding**: Dynamic vocabularies that grow as new features are encountered

### SSL Objectives (trea/models/ssl_objectives.py)

Self-supervised learning objectives for pretraining:

- Masked patch prediction
- Temporal order prediction
- Contrastive learning

### Data Infrastructure

**Two `utils` locations — do not confuse them:**

- **Root `utils/`** (NOT part of the `trea` package): data handling. Imported as
  `from utils.X import ...`.
  - `dataset_base.py` — `SyntheticTimeSeriesDataset` (configurable missing-value ratios)
    and `TimeSeriesDataset` base classes
  - `datamodule.py` — `TimeSeriesDataModule` (Lightning train/val/test splits)
  - `data_config.py` — `DatasetConfig` metadata classes
  - `multi_dataset_pretrain.py` — `MultiDatasetPretrainDataModule`, `DatasetSource`
  - `three_w.py`, `three_w_columns.py` — 3W dataset loaders + `SENSOR_COLUMNS`
- **`trea/utils/`** (inside the package): package internals only —
  `paths.py` (project-root / output-dir helpers) and `sequence_standardization.py`.

**Dataset downloaders** live in `scripts/download_*.py` (NASA C-MAPSS, turbofan,
bearing) and `examples/download_har_dataset.py` — there is **no** `data/downloaders/`
directory. The `data/` (and `datasets/`) directories are gitignored storage only.

## Triple-Encoded Architecture

The key innovation: instead of preprocessing NaNs, triple the input channels:

1. **Value channels**: Original values with NaNs → 0
2. **Mask channels**: 1 where NaN, 0 where valid
3. **Column channels**: Semantic embeddings for each feature

This eliminates NaN computations while preserving missingness information and semantic
feature understanding.

## Multi-Dataset Training

The model can train on datasets with different schemas:

- Uses `max_numeric_features` to define unified feature space
- Automatically pads/masks datasets with fewer features
- Column embeddings transfer semantic knowledge between datasets
- Example: Dataset A with [temp, humidity, pressure] and Dataset B with [wind, rain,
  temp, light] both map to the same unified space

## Common Workflows

### Creating a basic model

```python
from trea.models import TriplePatchTransformer
model = TriplePatchTransformer(
    C_num=7, C_cat=0, cat_cardinalities=[],
    T=96, d_model=128, task='classification',
    num_classes=3, n_head=8, num_layers=3
)
```

### Multi-dataset with column awareness

```python
from trea.models import MultiDatasetModel
model = MultiDatasetModel.create_column_aware(
    c_in=7, seq_len=96, num_classes=3,
    column_names=['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
    mode='standard', column_embedding_strategy='bert'
)
```

### Pretraining with SSL

```python
pretrain_model = MultiDatasetModel(
    max_numeric_features=10, mode='pretrain',
    ssl_lambda_mask=1.0, ssl_lambda_temporal=0.5, ssl_lambda_contrastive=0.3
)
```

## 3W Petroleum Well Dataset (Petrobras)

### Overview

The 3W dataset is a benchmark for detecting/classifying undesirable events in offshore
oil wells. It contains 2,228 parquet files across 10 event classes (0-9) with three
data sources: real well data (1,119 files), simulated (1,089), and hand-drawn (20).

- **Dataset path**: `/home/kailukowiak/Work/3W` (cloned from petrobras/3W)
- **Version**: 2.0.0
- **Features**: 27 sensor columns (see `SENSOR_COLUMNS` in `utils/three_w.py`)
- **Classes**: Normal (0), Abrupt BSW Increase (1), Spurious DHSV Closure (2),
  Severe Slugging (3), Flow Instability (4), Rapid Productivity Loss (5),
  Quick PCK Restriction (6), Scaling in PCK (7), Hydrate in Production Line (8),
  Hydrate in Service Line (9)
- **Transient labels**: 101-109 map to 1-9 via `label % 100`
- **ThreeWToolkit**: `split` param for train/val/test is NOT implemented (raises
  ValueError). Use `split=None` and split files manually.

### Published Baselines

**Important**: Published baselines are NOT directly comparable to our setup:
- They use **7-8 classes** (exclude hardest classes), we use **all 10**
- They use **handcrafted statistical features** (mean/std/skew/kurtosis/min/max/Q1/Q3
  per sensor → 72 features), then feed tabular data to RF/DT
- Some report CV scores (inflated vs held-out)
- High scores (0.92+) are **binary** anomaly detection, not multiclass

| Method           | Classes | Features   | Metric       | Value    | Split        |
| ---------------- | ------- | ---------- | ------------ | -------- | ------------ |
| Turan (DT, test) | 8       | Stats      | macro F1     | 0.85     | held-out     |
| Turan (RF, CV)   | 8       | Stats      | F1           | 0.91     | CV           |
| Marins (RF)      | 7       | Stats      | accuracy     | 0.94     | held-out     |
| **TREA-C**       | **10**  | **Raw TS** | **macro F1** | **0.83** | **held-out** |

See `docs/3w_literature_benchmarks.csv` for full details with source URLs.

Class 2 (Spurious Closure of DHSV) is notoriously difficult for all models due to
very few real instances (22 real + 16 simulated files).

### Current Training Setup (`train_w3.py` at repo root — the canonical 3W entrypoint)

Note the filename quirk: the dataset is "3W" but the stable entrypoint is
`train_w3.py`. It is a thin compatibility wrapper around the maintained
implementation in `examples/train_3w.py`. Do not revive the older row-level parquet
loader: it built split-local label mappings and made validation invalid on
well-disjoint splits.

This multiclass setup and its 0.83 macro-F1 predate the kill verdict. It used a random
per-file split, so the number is likely leaky. The adjudicated 3W benchmark is the
binary well-disjoint gate (`examples/benchmark_3w_binary.py --split-mode well`).

- Window: T=96, stride=96 (non-overlapping)
- Architecture: d_model=256, n_head=8, num_layers=4, dropout=0.3
- Loss: Plain CrossEntropyLoss (WeightedRandomSampler handles class balance)
- Optimizer: AdamW (weight_decay=1e-2) + OneCycleLR (max_lr=3e-4, 10% warmup)
- Early stopping on val_loss, patience=10
- Training was early-stopped at epoch 5 (val_loss) but F1 was still improving at epoch 15

### Key Lessons Learned (training)

- **Do NOT combine class weights in loss + WeightedRandomSampler** — double/triple
  correction hurts performance (especially class 0 Normal recall)
- **FocalLoss made things worse** when combined with weighted sampling
- **Plain CE + sampler-only balance** gave best results (macro F1 0.822)
- The official folds file is at `dataset/folds/folds_clf_02.csv` (5-fold CV, real data only)
- We currently use a random 80/20 file-level split mixing all data sources

### Evaluation protocol — fix before trusting any number (from `docs/LESSONS.md`)

These rules governed the gates that killed the deep-vs-trees line, and they still apply
to any new experiment. Treat older numbers from per-instance splits (including RF's 0.92
multiclass macro-F1) as optimistic:

- **Well-disjoint (grouped) splits.** 3W missingness is structural — whole sensor
  columns are absent per well, so the mask channel partly encodes well identity. The
  current per-instance split lets the model shortcut on "which well is this" → likely
  leaky. Split by well.
- **Per-instance, not per-window, metrics.** Windows from one instance are correlated;
  per-window scoring inflates effective N.
- **Macro-F1 / balanced accuracy only.** Classes 0+4 are ~91% of windows; raw accuracy lies.
- **Pre-register before running.** The §3 kill criterion in `docs/LESSONS.md` has fired.
  Any new line of work needs its own criterion, written down before results exist.
- **Report shifted and i.i.d. results side by side.** On turbine NOx, deep won on a random
  split and lost on the temporal split (§3c).
- **Do not judge semantic columns on single-dataset accuracy** — their payoff is
  transfer; single-dataset underperformance is a known (likely embedding-scale) bug.

## Project Structure Notes

- **trea/**: Core library (`models/`, `training/`, `utils/` package internals)
- **utils/** (repo root, separate from `trea/utils/`): data handling — datamodules,
  dataset bases, 3W loaders. See "Two `utils` locations" above.
- **train_w3.py** (repo root): canonical 3W training entrypoint, delegating to
  `examples/train_3w.py`
- **examples/**: usage examples, dataset downloaders, and one-off train/diagnose scripts
  (DO NOT RUN these directly — tqdm output overflows the context window)
- **scripts/**: pretraining, fine-tuning, evaluation, and dataset-download utilities
- **docs/**: orientation docs (LESSONS, CONSOLIDATION_PLAN, benchmarks) — see "Read first"
- **data/**, **datasets/**, **checkpoints/**, **logs/**: gitignored storage (not in repo)
- **tests/**: unit tests (currently `test_column_embeddings.py`)

## Important File Naming Conventions

- Model parameters use underscores: `C_num`, `C_cat`, `T` (sequence length)
- Feature counts: `C_num` = numeric channels, `C_cat` = categorical channels
- Architecture params: `d_model`, `n_head`, `num_layers`, `patch_len`, `stride`
