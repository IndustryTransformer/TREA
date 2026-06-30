"""Config and dataset for the single-row tabular model.

`SingleRowConfig.generate(df, target=...)` builds the token vocabulary from a
DataFrame: special tokens + numeric column names + categorical values + categorical
column names. Numeric/categorical columns are detected by dtype (object => categorical).

Differences from the Hephaestus original: no BERT/`transformers` dependency, and
`object_tokens` are sorted for deterministic, reproducible token indices.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from trea.models.single_row import NumericCategoricalData

SPECIAL_TOKENS = ["[PAD]", "[NUMERIC_MASK]", "[MASK]", "[UNK]", "[NUMERIC_EMBEDDING]"]


@dataclass
class InputsTarget:
    inputs: NumericCategoricalData
    target: Optional[torch.Tensor] = None


@dataclass
class SingleRowConfig:
    tokens: list
    token_dict: dict
    numeric_col_tokens: list
    categorical_col_tokens: list
    object_tokens: list
    n_tokens: int
    n_numeric_cols: int
    n_cat_cols: int
    n_columns: int
    target: Optional[str] = None

    @classmethod
    def generate(cls, df: pd.DataFrame, target: Optional[str] = None) -> "SingleRowConfig":
        numeric_cols = [
            c for c in df.select_dtypes(include="number").columns if c != target
        ]
        cat_cols = list(df.select_dtypes(include="object").columns)
        df_cat = df[cat_cols].astype(str)
        object_tokens = sorted({v for c in cat_cols for v in df_cat[c].unique()})

        tokens = SPECIAL_TOKENS + numeric_cols + object_tokens + cat_cols
        token_dict = {t: i for i, t in enumerate(tokens)}
        return cls(
            tokens=tokens,
            token_dict=token_dict,
            numeric_col_tokens=numeric_cols,
            categorical_col_tokens=cat_cols,
            object_tokens=object_tokens,
            n_tokens=len(tokens),
            n_numeric_cols=len(numeric_cols),
            n_cat_cols=len(cat_cols),
            n_columns=len(numeric_cols) + len(cat_cols),
            target=target,
        )


class TabularDS(Dataset):
    """Rows of a DataFrame as (numeric, categorical, target) tensors."""

    def __init__(self, df: pd.DataFrame, config: SingleRowConfig):
        df = df.copy()
        self.len = len(df)
        self.target_df = (
            pd.DataFrame(df.pop(config.target)) if config.target is not None else None
        )
        cat = df.select_dtypes(include="object").astype(str)
        for c in cat.columns:
            cat[c] = cat[c].map(config.token_dict)
        self.cat = cat
        self.numeric = df.select_dtypes(include="number")

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        num = torch.tensor(self.numeric.iloc[idx].values, dtype=torch.float32)
        cat = (
            torch.tensor(self.cat.iloc[idx].values, dtype=torch.long)
            if not self.cat.empty
            else None
        )
        tgt = (
            torch.tensor(self.target_df.iloc[idx].values, dtype=torch.float32)
            if self.target_df is not None
            else None
        )
        return InputsTarget(NumericCategoricalData(numeric=num, categorical=cat), tgt)


def _stack_targets(batch):
    if batch[0].target is None:
        return None
    t = torch.stack([item.target for item in batch])
    return t.unsqueeze(-1) if t.dim() == 1 else t


def tabular_collate_fn(batch):
    """Collate to InputsTarget (for supervised regression)."""
    numeric = torch.stack([item.inputs.numeric for item in batch])
    categorical = (
        torch.stack([item.inputs.categorical for item in batch])
        if batch[0].inputs.categorical is not None
        else None
    )
    return InputsTarget(
        inputs=NumericCategoricalData(numeric=numeric, categorical=categorical),
        target=_stack_targets(batch),
    )


def masked_tabular_collate_fn(batch):
    """Collate to NumericCategoricalData only (for MTM pretraining; target dropped)."""
    numeric = torch.stack([item.inputs.numeric for item in batch])
    categorical = (
        torch.stack([item.inputs.categorical for item in batch])
        if batch[0].inputs.categorical is not None
        else None
    )
    return NumericCategoricalData(numeric=numeric, categorical=categorical)
