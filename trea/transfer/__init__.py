"""Cross-schema column-name transfer on the simple single-row encoder.

See ``trea.transfer.api`` for usage and docs/SCHEMA_TRANSFER.md for the evidence.
"""

from trea.transfer.api import SchemaTransferRegressor, Table
from trea.transfer.model import SchemaFreeEncoder, SchemaTransferNet


__all__ = ["SchemaTransferRegressor", "Table", "SchemaFreeEncoder", "SchemaTransferNet"]
