# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in
this repository.

## Project Overview

TREA-C (Triple-Encoded Attention for Column-aware analysis) is a PyTorch Lightning-based
library for time series analysis. The core innovation is a triple-encoded architecture
that handles missing values by encoding them as separate channels (value channels, mask
channels, and column embeddings) rather than preprocessing NaNs.

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

### Important: DO NOT run training scripts directly

The model outputs tqdm progress bars which will overflow the context window. Ask the
user to run training scripts instead of running them yourself. Avoid commands like
`uv run python examples/*.py` or `uv run python scripts/*.py`.

## Architecture

### Core Models (treac/models/)

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

### Column Embeddings (treac/models/embeddings.py)

Column embeddings provide semantic understanding of feature names:

- **Simple**: Learned embeddings (lightweight)
- **BERT**: Semantic embeddings using pre-trained language models
- **Frozen BERT**: Cached embeddings for multi-dataset efficiency
- **Auto-expanding**: Dynamic vocabularies that grow as new features are encountered

### SSL Objectives (treac/models/ssl_objectives.py)

Self-supervised learning objectives for pretraining:

- Masked patch prediction
- Temporal order prediction
- Contrastive learning

### Data Infrastructure

**Data Loaders** (`data/downloaders/`)

- Pre-configured loaders for standard datasets: ETTh1, Human Activity, Air Quality,
  Financial Markets, NASA Turbofan, Pump Sensor
- Each loader handles dataset-specific preprocessing and column names

**Dataset Base** (`utils/dataset_base.py`)

- `SyntheticTimeSeriesDataset`: Synthetic data for testing with configurable missing
  value ratios
- Base classes for time series datasets with numeric and categorical features

**DataModule** (`utils/datamodule.py`)

- PyTorch Lightning DataModules for train/val/test splits
- Handles batching and data loading for training

**Data Config** (`utils/data_config.py`)

- Configuration classes for dataset metadata

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
from treac.models import TriplePatchTransformer
model = TriplePatchTransformer(
    C_num=7, C_cat=0, cat_cardinalities=[],
    T=96, d_model=128, task='classification',
    num_classes=3, n_head=8, num_layers=3
)
```

### Multi-dataset with column awareness

```python
from treac.models import MultiDatasetModel
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

| Method | Classes | Features | Metric | Value | Split |
|--------|---------|----------|--------|-------|-------|
| Turan (DT, test) | 8 | Stats | macro F1 | 0.85 | held-out |
| Turan (RF, CV) | 8 | Stats | F1 | 0.91 | CV |
| Marins (RF) | 7 | Stats | accuracy | 0.94 | held-out |
| **TREA-C** | **10** | **Raw TS** | **macro F1** | **0.83** | **held-out** |

See `docs/3w_literature_benchmarks.csv` for full details with source URLs.

Class 2 (Spurious Closure of DHSV) is notoriously difficult for all models due to
very few real instances (22 real + 16 simulated files).

### Current Training Setup (`examples/train_3w.py`)

- Window: T=96, stride=96 (non-overlapping)
- Architecture: d_model=256, n_head=8, num_layers=4, dropout=0.3
- Loss: Plain CrossEntropyLoss (WeightedRandomSampler handles class balance)
- Optimizer: AdamW (weight_decay=1e-2) + OneCycleLR (max_lr=3e-4, 10% warmup)
- Early stopping on val_loss, patience=10
- Training was early-stopped at epoch 5 (val_loss) but F1 was still improving at epoch 15

### Key Lessons Learned

- **Do NOT combine class weights in loss + WeightedRandomSampler** — double/triple
  correction hurts performance (especially class 0 Normal recall)
- **FocalLoss made things worse** when combined with weighted sampling
- **Plain CE + sampler-only balance** gave best results (macro F1 0.822)
- The official folds file is at `dataset/folds/folds_clf_02.csv` (5-fold CV, real data only)
- We currently use a random 80/20 file-level split mixing all data sources

## Project Structure Notes

- **treac/**: Core library with models, embeddings, and training utilities
- **utils/**: Helper utilities for datasets and data modules (gitignored data handling)
- **data/**: Dataset storage (gitignored) with downloaders for public datasets
- **examples/**: Usage examples and training scripts (DO NOT RUN these directly)
- **scripts/**: Utility scripts for pretraining, fine-tuning, and evaluation
- **tests/**: Unit tests

## Important File Naming Conventions

- Model parameters use underscores: `C_num`, `C_cat`, `T` (sequence length)
- Feature counts: `C_num` = numeric channels, `C_cat` = categorical channels
- Architecture params: `d_model`, `n_head`, `num_layers`, `patch_len`, `stride`
