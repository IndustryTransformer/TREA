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

**The thesis (read before optimizing anything):** on fully-labeled benchmarks, trees
(RF/XGBoost) beat deep models — on 3W, RF ≈ 0.92 macro-F1 vs ~0.81 for the best deep
variant, at a fraction of the compute. Architecture tinkering has a low ceiling there.
The defensible edge of a deep model is the regime trees structurally cannot touch:
**(1) label efficiency** via self-supervised pretraining on abundant unlabeled sensor
data, and **(2) cross-dataset transfer** via semantic column embeddings. TREA is
optimized for the expensive-label / multi-dataset setting and benchmarked honestly with
macro-F1 / balanced accuracy (never raw accuracy).

### Read first (orientation docs)

- `docs/LESSONS.md` — empirical findings, **evaluation protocol, and pre-registered kill
  criterion**. Read before changing architecture or trusting any benchmark number.
- `README.md` — the thesis and the one experiment that matters next (label-efficiency curve).
- `docs/CONSOLIDATION_PLAN.md` — why four repos became one; what to harvest.
- `SEMANTIC_COLUMNS_SUMMARY.md` — column-identity findings and catalogued bug fixes.
- `docs/trea_original_README.md` — inherited TREA-C API/usage reference.

## Current focus (immediate next tasks)

Consolidation is mostly done (seeded from TREA-C). Active work, in order:

1. **Harvest from the archived prototypes** (`docs/CONSOLIDATION_PLAN.md`): TabNCT's
   first-class intra-row + inter-row attention and column-name-as-tokens; Hephaestus's
   numeric projection only if it earns it in an ablation.
2. **Port the bug fixes** in `SEMANTIC_COLUMNS_SUMMARY.md` (NaN-mask preservation,
   class-weight null filter, macro-F1 reporting, the `batch·seq < 65536` attention limit).
3. **Build the headline experiment**: the label-efficiency curve (macro-F1 vs # labels;
   RF / from-scratch / pretrained) on 3W + a turbine dataset, on **well-disjoint** splits.

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

### Important: DO NOT run training scripts directly

The model outputs tqdm progress bars which will overflow the context window. Ask the
user to run training scripts instead of running them yourself. Avoid commands like
`uv run python examples/*.py` or `uv run python scripts/*.py`.

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

### Current Training Setup (`train_w3.py` at repo root — the canonical 3W script)

Note the filename quirk: the dataset is "3W" but the script is `train_w3.py`. An older
copy exists at `examples/train_3w.py`; prefer the root `train_w3.py` (kept current).

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

The eval is the underbuilt part; treat existing benchmark numbers (including RF's 0.92)
as possibly optimistic until re-run under these rules:

- **Well-disjoint (grouped) splits.** 3W missingness is structural — whole sensor
  columns are absent per well, so the mask channel partly encodes well identity. The
  current per-instance split lets the model shortcut on "which well is this" → likely
  leaky. Split by well.
- **Per-instance, not per-window, metrics.** Windows from one instance are correlated;
  per-window scoring inflates effective N.
- **Macro-F1 / balanced accuracy only.** Classes 0+4 are ~91% of windows; raw accuracy lies.
- **Headline benchmark = label-efficiency curve** (macro-F1 vs # labels; lines for RF /
  from-scratch / pretrained) on ≥2 datasets. The crossover point is the thesis.
- A pre-registered **kill criterion** exists in `docs/LESSONS.md` §3 — consult it rather
  than re-litigating whether the architecture "works."
- **Do not judge semantic columns on single-dataset accuracy** — their payoff is
  transfer; single-dataset underperformance is a known (likely embedding-scale) bug.

## Project Structure Notes

- **trea/**: Core library (`models/`, `training/`, `utils/` package internals)
- **utils/** (repo root, separate from `trea/utils/`): data handling — datamodules,
  dataset bases, 3W loaders. See "Two `utils` locations" above.
- **train_w3.py** (repo root): canonical 3W training script
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
