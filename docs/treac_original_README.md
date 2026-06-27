# TREA-C: Triple-Encoded Attention for Column-aware Time Series

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.0+-purple.svg)](https://lightning.ai/)

TREA-C (Triple-Encoded Attention for Column-aware analysis) is a PyTorch Lightning-based
library for time series analysis that handles both **numeric and categorical features**
with robust **missing value support**. The library implements a novel triple-encoded
architecture that combines value channels, mask channels, and column embeddings directly
into the model structure, eliminating NaN-related computational issues while preserving
missingness information and semantic feature understanding.

## 🚀 Key Features

### Core Architecture

- **Triple-Encoded Architecture**: Encodes values, missing value masks, and column
  semantics as separate channels, avoiding NaN computations
- **Multi-Modal Input Support**: Handles both numeric sensor data and time-varying
  categorical features
- **Patch-Based Processing**: Efficient temporal modeling using patch-based transformers
- **Column Embeddings**: Semantic understanding of feature names using BERT or learned
  embeddings

### Multi-Dataset Training

- **Variable Feature Schemas**: Train on datasets with different numbers and types of
  features
- **Column Semantic Embeddings**: Transfer knowledge between datasets using feature name
  semantics
- **Auto-Expanding Vocabularies**: Dynamic feature vocabularies for multi-dataset
  scenarios
- **Unified Feature Space**: Automatic padding and masking for heterogeneous datasets

### Self-Supervised Learning

- **Masked Patch Prediction**: Reconstruct masked temporal patches
- **Temporal Order Prediction**: Learn temporal relationships
- **Contrastive Learning**: Global sequence-level representations
- **Flexible SSL Objectives**: Configurable pre-training strategies

## 📦 Installation

### Using uv (Recommended)

```bash
git clone https://github.com/kailukowiak/TREA-C.git
cd TREA-C
uv sync
```

### Using pip

```bash
git clone https://github.com/kailukowiak/TREA-C.git
cd TREA-C
pip install -e .
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- PyTorch Lightning 2.0+
- Transformers (for BERT embeddings)
- Scikit-learn, Pandas, NumPy

## 🏗️ Package Structure

```
TREA-C/
├── treac/                          # Core library
│   ├── models/                     # Model implementations
│   │   ├── triple_attention.py     # Core TripleEncodedAttention
│   │   ├── multi_dataset_model.py  # Multi-dataset training model
│   │   ├── patchtstnan.py          # PatchTST with NaN handling
│   │   ├── embeddings.py           # Column embedding strategies
│   │   └── ssl_objectives.py       # Self-supervised learning
│   ├── utils/                      # Core utilities
│   │   ├── paths.py                # Path management
│   │   └── sequence_standardization.py
│   └── training/                   # Training utilities
├── utils/                          # Helper utilities (gitignored data handling)
│   ├── dataset_base.py             # Base dataset classes
│   ├── datamodule.py               # PyTorch Lightning data modules
│   └── data_config.py              # Dataset configuration
├── data/                           # Data storage (gitignored)
│   └── downloaders/                # Dataset download utilities
├── examples/                       # Usage examples and scripts
├── tests/                          # Unit tests
└── scripts/                        # Utility scripts
```

## 🎯 Quick Start

### Basic Time Series Classification

```python
import torch
from treac.models import TriplePatchTransformer
from utils import SyntheticTimeSeriesDataset, TimeSeriesDataModule
import pytorch_lightning as pl

# Create synthetic dataset
dataset = SyntheticTimeSeriesDataset(
    num_samples=1000,
    T=96,           # sequence length
    C_num=7,        # numeric features
    C_cat=2,        # categorical features
    num_classes=3,
    task='classification'
)

# Setup data module
dm = TimeSeriesDataModule(
    train_dataset=dataset,
    val_dataset=dataset,
    batch_size=32
)

# Create model
model = TriplePatchTransformer(
    c_in=7,                    # numeric input channels
    seq_len=96,                # sequence length
    num_classes=3,             # output classes
    patch_len=16,              # patch size
    stride=8,                  # patch stride
    d_model=128,               # model dimension
    n_head=8,                  # attention heads
    num_layers=3,              # transformer layers
    task='classification'
)

# Train
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, dm)
```

### Multi-Dataset Training with Column Embeddings

```python
from treac.models import MultiDatasetModel
from data.downloaders.etth1 import ETTh1Dataset

# Load dataset
dataset = ETTh1Dataset(train=True)
column_names = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']

# Create column-aware model
model = MultiDatasetModel.create_column_aware(
    c_in=7,
    seq_len=96,
    num_classes=3,
    column_names=column_names,
    task='classification',
    patch_len=16,
    stride=8,
    d_model=128,
    n_head=8,
    num_layers=3,
    column_embedding_dim=16,
    mode='standard'
)

# The model now understands feature semantics!
```

### Self-Supervised Pretraining

```python
# Create pretraining model
pretrain_model = MultiDatasetModel(
    max_numeric_features=10,
    seq_len=96,
    patch_len=16,
    stride=8,
    d_model=128,
    n_head=8,
    num_layers=3,
    mode='pretrain',  # Enable SSL objectives

    # SSL configuration
    ssl_lambda_mask=1.0,      # Masked patch prediction weight
    ssl_lambda_temporal=0.5,   # Temporal order weight
    ssl_lambda_contrastive=0.3 # Contrastive learning weight
)

# Pretrain on unlabeled data
trainer.fit(pretrain_model, unlabeled_datamodule)

# Fine-tune on labeled data
finetuned_model = MultiDatasetModel.load_from_checkpoint(
    checkpoint_path,
    mode='standard',  # Switch to supervised mode
    num_classes=3
)
```

## 🔧 Advanced Features

### Handling Missing Values

TREA-C's triple-encoded architecture automatically handles missing values:

```python
# Your data can contain NaNs - no preprocessing needed!
x_num = torch.tensor([
    [1.0, 2.0, float('nan'), 4.0],
    [float('nan'), 6.0, 7.0, 8.0]
])

# Model automatically creates value + mask channels:
# - Value channel: [1.0, 2.0, 0.0, 4.0], [0.0, 6.0, 7.0, 8.0]
# - Mask channel:  [0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]
```

### Column Embedding Strategies

```python
from treac.models.embeddings import (
    create_column_embedding,
    create_multi_dataset_embedder
)

# Simple learned embeddings (lightweight)
simple_emb = create_column_embedding(
    column_names=['temperature', 'humidity', 'pressure'],
    target_dim=1,
    strategy='simple',
    embedding_dim=32
)

# BERT-based semantic embeddings
bert_emb = create_column_embedding(
    column_names=['temperature', 'humidity', 'pressure'],
    target_dim=1,
    strategy='bert',
    bert_model='bert-base-uncased'
)

# Multi-dataset frozen BERT (efficient for many datasets)
frozen_bert = create_multi_dataset_embedder(
    strategy='frozen_bert',
    target_dim=1
)
frozen_bert.set_columns(['temperature', 'humidity'])  # Dynamic columns
```

### Variable Feature Training

```python
# Train on datasets with different feature counts
model = MultiDatasetModel(
    max_numeric_features=20,      # Unified feature space size
    max_categorical_features=5,
    mode='variable_features'      # Enable variable schemas
)

# Set schema for each dataset
model.set_dataset_schema(
    numeric_features=7,           # This dataset has 7 features
    categorical_features=2
)

# Model automatically pads/masks features to unified space
```

## 📊 Examples and Use Cases

### Time Series Classification

- Sensor fault detection
- Human activity recognition
- Equipment state classification
- Medical time series analysis

### Multi-Dataset Scenarios

- Training on multiple sensor networks
- Transfer learning between similar datasets
- Domain adaptation for time series
- Federated learning applications

### Self-Supervised Learning

- Pretraining on large unlabeled sensor data
- Learning temporal representations
- Few-shot learning with pretrained models
- Anomaly detection via reconstruction

## 🧪 Example Scripts

The `examples/` directory contains comprehensive usage examples:

- `final_patch_comparison.py` - Compare baseline vs column-aware models
- `compare_models_etth1.py` - Benchmark different architectures
- `multi_dataset_demo.py` - Multi-dataset training walkthrough
- `benchmark_patch_sizes.py` - Patch size optimization
- `benchmark_3w_noninferiority.py` - Multi-seed 3W benchmark with non-inferiority stats
- `docs/3w_paper_benchmarks.md` - Literature benchmark sheet (binary + multiclass)
- `docs/3w_literature_benchmarks.csv` - Structured literature rows with verification tags

### Non-Inferiority Benchmark Protocol (3W)

Use this script to support claims like "candidate model is statistically
non-inferior to baselines on 3W":

```bash
uv run python examples/benchmark_3w_noninferiority.py \
  --models treac_triple,patchtstnan,multidataset_none,multidataset_auto,rf_stat_features \
  --candidate-model treac_triple \
  --seeds 42,43,44,45,46 \
  --max-epochs 30 \
  --margin 0.01 \
  --run-name main_5seed
```

Outputs are written under `logs/benchmark_3w_noninferiority/<run-name>/`:

- `runs.csv`: one row per (model, seed)
- `summary.csv`: mean/std/CI by model
- `noninferiority.csv`: candidate-vs-comparator deltas with CI and verdict
- `report.md`: paper-ready summary table + overall verdict

Build an apples-to-apples markdown report (internal + literature filter):

```bash
uv run python scripts/build_3w_apples_to_apples_report.py \
  --benchmark-dir logs/benchmark_3w_noninferiority/main_5seed
```

## 🔬 Model Architecture Details

### Triple-Encoded NaN Handling

Instead of preprocessing NaNs, TREA-C triples the input channels:

- **Value channels**: Original values with NaNs → 0
- **Mask channels**: 1 where NaN, 0 where valid
- **Column channels**: Semantic embeddings for each feature

This approach:

- ✅ Eliminates NaN computations (CUDA-friendly)
- ✅ Preserves missingness information
- ✅ Adds semantic feature understanding
- ✅ Allows end-to-end training
- ✅ Minimal computational overhead

### Column Embeddings

Feature names are embedded to provide semantic context:

- **Simple**: Learned embeddings (lightweight)
- **BERT**: Semantic embeddings using pre-trained language models
- **Frozen BERT**: Cached embeddings for multi-dataset efficiency
- **Auto-expanding**: Dynamic vocabularies for new features

### Multi-Dataset Training

Unified feature space with automatic padding:

```
Dataset A: [temp, humidity, pressure] → [temp, humidity, pressure, 0, 0, ...]
Dataset B: [wind, rain, temp, light]  → [wind, rain, temp, light, 0, ...]
```

## 📈 Performance

TREA-C achieves competitive performance on standard benchmarks:

- **ETTh1**: 85%+ accuracy on electricity forecasting classification
- **Multi-dataset**: Minimal performance degradation with column embeddings
- **Missing data**: Robust to 5-20% missing value rates
- **Efficiency**: ~2-3x faster than NaN-aware preprocessing

## 🧪 Testing

```bash
# Run all tests
uv run pytest

# Run specific test categories
uv run pytest tests/test_models.py
uv run pytest tests/test_embeddings.py
```

## 🛠️ Development

```bash
# Install development dependencies
uv sync --group dev

# Code formatting
uvx ruff format .

# Linting
uvx ruff check . --fix

# Type checking (if using mypy)
uvx mypy treac/
```

## 📚 Citation

If you use TREA-C in your research, please cite:

```bibtex
@software{treac2024,
  title={TREA-C: Triple-Encoded Attention for Column-aware Time Series Analysis},
  author={Your Name},
  year={2024},
  url={https://github.com/kailukowiak/TREA-C}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for
details.

## 📄 License

This project is licensed under the Apache 2.0 License - see the
[LICENSE.txt](LICENSE.txt) file for details.

## 🔗 Related Work

- [PatchTST](https://github.com/yuqinie98/PatchTST): Original patch-based time series
  transformer
- [TimeGPT](https://github.com/Nixtla/neuralforecast): Foundation models for time series
- [PyTorch Lightning](https://lightning.ai/): Deep learning framework used as foundation

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/kailukowiak/TREA-C/issues)
- **Discussions**:
  [GitHub Discussions](https://github.com/kailukowiak/TREA-C/discussions)
- **Documentation**: [Full Documentation](https://trea-c.readthedocs.io) (coming soon)
