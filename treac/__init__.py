"""TREA-C: Triple-Encoded Attention for Column-aware Time Series Analysis."""

__version__ = "0.1.0"

from treac.models import MultiDatasetModel, PatchTSTNan, TriplePatchTransformer


__all__ = [
    "TriplePatchTransformer",
    "MultiDatasetModel",
    "PatchTSTNan",
]
