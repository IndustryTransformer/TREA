"""Consolidated embedding module for column semantic information.

This module provides multiple embedding strategies for incorporating column names
into transformer models for time series data:

1. Simple learned embeddings (lightweight)
2. BERT-based embeddings (semantic understanding)
3. Frozen BERT with caching (efficient for multi-dataset training)
4. Auto-expanding embeddings (dynamic vocabulary)
"""

import json
import os
import pickle
import re

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer

from ..utils import get_project_root


# Set tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SimpleColumnEmbedding(nn.Module):
    """Simple learned embeddings for column names.

    Lightweight alternative to BERT embeddings with minimal overhead.
    """

    def __init__(
        self,
        column_names: list[str],
        target_dim: int,
        embedding_dim: int = 32,
    ):
        """Initialize simple column embedding.

        Args:
            column_names: List of column names to encode
            target_dim: Target embedding dimension to match value/mask channels
            embedding_dim: Dimension for learned embeddings
        """
        super().__init__()

        self.column_names = column_names
        self.target_dim = target_dim
        num_columns = len(column_names)

        # Create simple embedding layer
        self.embedding = nn.Embedding(num_columns, embedding_dim)

        # Project to target dimension
        self.projection = nn.Linear(embedding_dim, target_dim)

        # Create column indices
        self.register_buffer(
            "column_indices", torch.arange(num_columns, dtype=torch.long)
        )

    def get_embeddings(self) -> torch.Tensor:
        """Get column embeddings.

        Returns:
            Column embeddings tensor [num_columns, target_dim]
        """
        column_embs = self.embedding(
            self.column_indices
        )  # [num_columns, embedding_dim]
        return self.projection(column_embs)  # [num_columns, target_dim]

    def forward(self, batch_size: int, sequence_length: int) -> torch.Tensor:
        """Create column embeddings for a batch.

        Args:
            batch_size: Batch size
            sequence_length: Sequence length (time dimension)

        Returns:
            Column embeddings shaped [batch_size, num_columns, sequence_length]
        """
        # Get embeddings [num_columns, target_dim]
        column_embs = self.get_embeddings()

        # Expand to match batch and time dimensions
        column_embs = column_embs.unsqueeze(0).unsqueeze(2)
        column_embs = column_embs.repeat(batch_size, 1, sequence_length, 1)
        column_embs = column_embs.permute(0, 1, 3, 2)

        if self.target_dim == 1:
            column_embs = column_embs.squeeze(2)

        return column_embs


class BERTColumnEmbedding(nn.Module):
    """BERT-based column embeddings with semantic understanding.

    Uses BERT to encode column names for better semantic representation.
    Can optionally freeze BERT parameters for efficiency.
    """

    def __init__(
        self,
        column_names: list[str],
        target_dim: int,
        bert_model: str = "bert-base-uncased",
        tokenization_strategy: str = "split_underscore_camel",
        aggregation_strategy: str = "mean",
        freeze_bert: bool = True,
        device: str | None = None,
    ):
        """Initialize BERT column embedding.

        Args:
            column_names: List of column names to encode
            target_dim: Target embedding dimension
            bert_model: BERT model name from HuggingFace
            tokenization_strategy: How to tokenize column names
            aggregation_strategy: How to aggregate multi-token embeddings
            freeze_bert: Whether to freeze BERT parameters
            device: Device to load models on
        """
        super().__init__()

        self.column_names = column_names
        self.target_dim = target_dim
        self.tokenization_strategy = tokenization_strategy
        self.aggregation_strategy = aggregation_strategy
        self.device = device or "cpu"

        # Load BERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.bert = AutoModel.from_pretrained(bert_model)
        self.bert = self.bert.to(self.device)

        # Freeze BERT parameters if requested
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Get BERT embedding dimension
        bert_dim = self.bert.config.hidden_size

        # Project BERT embeddings to target dimension
        self.projection = nn.Linear(bert_dim, target_dim)

        # Pre-compute column embeddings
        self._precompute_embeddings()

    def _process_column_name(self, column_name: str) -> str:
        """Process a column name according to tokenization strategy."""
        if self.tokenization_strategy == "as_is":
            return column_name.lower()

        elif self.tokenization_strategy == "split_underscore":
            tokens = column_name.split("_")
            return " ".join(token.lower() for token in tokens if token)

        elif self.tokenization_strategy == "split_underscore_camel":
            # Split on underscores first
            parts = column_name.split("_")

            # Then split each part on camelCase
            all_tokens = []
            for part in parts:
                if not part:
                    continue

                # Split camelCase using regex
                camel_tokens = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)", part)
                if not camel_tokens:
                    camel_tokens = [part]

                all_tokens.extend(camel_tokens)

            return " ".join(token.lower() for token in all_tokens if token)

        else:
            raise ValueError(
                f"Unknown tokenization strategy: {self.tokenization_strategy}"
            )

    def _precompute_embeddings(self):
        """Pre-compute embeddings for all column names."""
        self.eval()

        with torch.no_grad():
            # Process column names
            processed_texts = [
                self._process_column_name(name) for name in self.column_names
            ]

            # Tokenize
            tokenized = self.tokenizer(
                processed_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )

            # Move to device
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

            # Get BERT embeddings
            outputs = self.bert(**tokenized)
            hidden_states = outputs.last_hidden_state  # [num_cols, seq_len, bert_dim]

            # Aggregate tokens
            if self.aggregation_strategy == "mean":
                attention_mask = tokenized["attention_mask"]
                masked_embeddings = hidden_states * attention_mask.unsqueeze(-1)
                summed = masked_embeddings.sum(dim=1)
                lengths = attention_mask.sum(dim=1, keepdim=True)
                pooled = summed / lengths  # [num_cols, bert_dim]
            elif self.aggregation_strategy == "cls":
                pooled = hidden_states[:, 0, :]  # [num_cols, bert_dim]
            elif self.aggregation_strategy == "max":
                pooled = hidden_states.max(dim=1)[0]  # [num_cols, bert_dim]
            else:
                raise ValueError(
                    f"Unknown aggregation strategy: {self.aggregation_strategy}"
                )

            # Project to target dimension
            projected = self.projection(pooled)  # [num_cols, target_dim]

            # Store as buffer
            self.register_buffer("column_embeddings", projected)

    def get_embeddings(self) -> torch.Tensor:
        """Get column embeddings.

        Returns:
            Column embeddings tensor [num_columns, target_dim]
        """
        return self.column_embeddings

    def forward(self, batch_size: int, sequence_length: int) -> torch.Tensor:
        """Create column embeddings for a batch."""
        # Get embeddings [num_columns, target_dim]
        column_embs = self.get_embeddings()

        # Expand to match batch and time dimensions
        column_embs = column_embs.unsqueeze(0).unsqueeze(2)
        column_embs = column_embs.repeat(batch_size, 1, sequence_length, 1)
        column_embs = column_embs.permute(0, 1, 3, 2)

        if self.target_dim == 1:
            column_embs = column_embs.squeeze(2)

        return column_embs


class FrozenBERTColumnEmbedder(nn.Module):
    """Frozen BERT embeddings with efficient lookup for multi-dataset training.

    Key advantages:
    - Pre-compute embeddings once, reuse across datasets
    - Minimal model size impact (just lookup table + projection)
    - Auto-expand for new columns without retraining
    - Semantic understanding for better transferability
    """

    def __init__(
        self,
        target_dim: int = 1,
        bert_model: str = "bert-base-uncased",
        cache_dir: Path | None = None,
        tokenization_strategy: str = "split_underscore_camel",
        aggregation_strategy: str = "mean",
        device: str | None = None,
    ):
        """Initialize frozen BERT column embedder."""
        super().__init__()

        self.target_dim = target_dim
        self.bert_model = bert_model
        self.tokenization_strategy = tokenization_strategy
        self.aggregation_strategy = aggregation_strategy
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Setup cache directory
        if cache_dir is None:
            cache_dir = get_project_root() / "cache" / "column_embeddings"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache files
        model_name = bert_model.replace("/", "_")
        self.embeddings_cache_file = (
            self.cache_dir / f"bert_embeddings_{model_name}.pkl"
        )
        self.metadata_cache_file = self.cache_dir / f"metadata_{model_name}.json"

        # Initialize BERT (lazy loading)
        self._bert = None
        self._tokenizer = None

        # Load cached embeddings
        self.embedding_cache = {}
        self.metadata = {"bert_dim": 768}  # Default for BERT-base
        self._load_cache()

        # Get BERT dimension and setup projection
        bert_dim = self.metadata["bert_dim"]
        self.projection = nn.Linear(bert_dim, target_dim)

        # Registry for current dataset columns
        self.current_columns: list[str] = []
        self.current_embeddings: torch.Tensor | None = None

    def _load_cache(self):
        """Load cached embeddings and metadata."""
        if self.embeddings_cache_file.exists():
            try:
                with open(self.embeddings_cache_file, "rb") as f:
                    self.embedding_cache = pickle.load(f)
                print(f"Loaded {len(self.embedding_cache)} cached embeddings")
            except Exception as e:
                print(f"Warning: Could not load embedding cache: {e}")
                self.embedding_cache = {}

        if self.metadata_cache_file.exists():
            try:
                with open(self.metadata_cache_file) as f:
                    self.metadata = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load metadata cache: {e}")
                self.metadata = {"bert_dim": 768}

    def _save_cache(self):
        """Save cached embeddings and metadata."""
        try:
            with open(self.embeddings_cache_file, "wb") as f:
                pickle.dump(self.embedding_cache, f)

            with open(self.metadata_cache_file, "w") as f:
                json.dump(self.metadata, f)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")

    def _init_bert(self):
        """Initialize BERT model and tokenizer (lazy loading)."""
        if self._bert is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.bert_model)
            self._bert = AutoModel.from_pretrained(self.bert_model)
            self._bert = self._bert.to(self.device)
            self._bert.eval()

            # Update metadata with actual BERT dimension
            self.metadata["bert_dim"] = self._bert.config.hidden_size

    def _process_column_name(self, column_name: str) -> str:
        """Process column name according to tokenization strategy."""
        if self.tokenization_strategy == "as_is":
            return column_name.lower()

        elif self.tokenization_strategy == "split_underscore":
            tokens = column_name.split("_")
            return " ".join(token.lower() for token in tokens if token)

        elif self.tokenization_strategy == "split_underscore_camel":
            # Split on underscores first
            parts = column_name.split("_")

            # Then split each part on camelCase
            all_tokens = []
            for part in parts:
                if not part:
                    continue

                # Split camelCase using regex
                camel_tokens = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)", part)
                if not camel_tokens:
                    camel_tokens = [part]

                all_tokens.extend(camel_tokens)

            return " ".join(token.lower() for token in all_tokens if token)

        else:
            raise ValueError(
                f"Unknown tokenization strategy: {self.tokenization_strategy}"
            )

    def _compute_bert_embedding(self, column_name: str) -> torch.Tensor:
        """Compute BERT embedding for a single column name."""
        self._init_bert()

        # Check if already cached
        cache_key = (
            f"{column_name}_{self.tokenization_strategy}_{self.aggregation_strategy}"
        )
        if cache_key in self.embedding_cache:
            return torch.tensor(self.embedding_cache[cache_key], device=self.device)

        # Process column name
        processed_text = self._process_column_name(column_name)

        # Tokenize and compute embedding
        with torch.no_grad():
            tokenized = self._tokenizer(
                processed_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

            # Get BERT embedding
            outputs = self._bert(**tokenized)
            hidden_states = outputs.last_hidden_state  # [1, seq_len, bert_dim]

            # Aggregate tokens
            if self.aggregation_strategy == "mean":
                attention_mask = tokenized["attention_mask"]
                masked_embeddings = hidden_states * attention_mask.unsqueeze(-1)
                summed = masked_embeddings.sum(dim=1)
                lengths = attention_mask.sum(dim=1, keepdim=True)
                pooled = summed / lengths  # [1, bert_dim]
            elif self.aggregation_strategy == "cls":
                pooled = hidden_states[:, 0, :]  # [1, bert_dim]
            elif self.aggregation_strategy == "max":
                pooled = hidden_states.max(dim=1)[0]  # [1, bert_dim]
            else:
                raise ValueError(
                    f"Unknown aggregation strategy: {self.aggregation_strategy}"
                )

            # Cache the result
            embedding = pooled.squeeze(0).cpu()  # [bert_dim]
            self.embedding_cache[cache_key] = embedding.numpy()

            return pooled.squeeze(0)  # [bert_dim]

    def set_columns(self, column_names: list[str]) -> None:
        """Set column names for current dataset and precompute embeddings."""
        self.current_columns = column_names

        # Compute embeddings for all columns
        embeddings = []
        new_embeddings_count = 0

        # Get target device from projection layer
        projection_device = next(self.projection.parameters()).device

        for col in column_names:
            cache_key = (
                f"{col}_{self.tokenization_strategy}_{self.aggregation_strategy}"
            )

            if cache_key in self.embedding_cache:
                # Use cached embedding
                emb = torch.tensor(
                    self.embedding_cache[cache_key], device=projection_device
                )
            else:
                # Compute new embedding
                emb = self._compute_bert_embedding(col)
                emb = emb.to(projection_device)
                new_embeddings_count += 1

            embeddings.append(emb)

        # Stack into tensor [num_columns, bert_dim]
        bert_embeddings = torch.stack(embeddings)

        # Project to target dimension
        with torch.no_grad():
            self.current_embeddings = self.projection(
                bert_embeddings
            )  # [num_columns, target_dim]

        # Save cache if we computed new embeddings
        if new_embeddings_count > 0:
            print(f"Computed {new_embeddings_count} new column embeddings")
            self._save_cache()

        print(f"Ready with embeddings for {len(column_names)} columns")

    def get_embeddings(self) -> torch.Tensor:
        """Get embeddings for current columns."""
        if self.current_embeddings is None:
            raise RuntimeError("No columns set. Call set_columns() first.")
        return self.current_embeddings

    def forward(self, batch_size: int, sequence_length: int) -> torch.Tensor:
        """Create column embeddings for a batch."""
        if self.current_embeddings is None:
            raise RuntimeError("No columns set. Call set_columns() first.")

        # Get embeddings and ensure on correct device
        projection_device = next(self.projection.parameters()).device
        column_embs = self.current_embeddings.to(projection_device)

        # Expand to batch and time dimensions
        column_embs = column_embs.unsqueeze(0).unsqueeze(
            2
        )  # [1, num_columns, 1, target_dim]
        column_embs = column_embs.repeat(
            batch_size, 1, sequence_length, 1
        )  # [B, C, T, target_dim]

        # Transpose and squeeze if needed
        column_embs = column_embs.permute(0, 1, 3, 2)  # [B, C, target_dim, T]

        if self.target_dim == 1:
            column_embs = column_embs.squeeze(2)  # [B, C, T]

        return column_embs


class AutoExpandingEmbedder(nn.Module):
    """Auto-expanding embedding layer that grows vocabulary dynamically.

    This embedding layer automatically expands its vocabulary when new column names
    are encountered during multi-dataset training.
    """

    def __init__(
        self,
        initial_vocab_size: int = 1000,
        embedding_dim: int = 32,
        target_dim: int = 1,
        growth_factor: float = 1.5,
    ):
        """Initialize auto-expanding embedder."""
        super().__init__()

        self.embedding_dim = embedding_dim
        self.target_dim = target_dim
        self.growth_factor = growth_factor

        # Initialize embedding table
        self.embedding = nn.Embedding(initial_vocab_size, embedding_dim)
        self.projection = nn.Linear(embedding_dim, target_dim)

        # Column name to index mapping
        self.column_to_idx = {}
        self.idx_to_column = {}
        self.next_idx = 0

        # Current dataset columns
        self.current_columns = []
        self.current_indices = []

    def _expand_vocabulary(self, new_size: int):
        """Expand the embedding vocabulary to new size."""
        old_size = self.embedding.num_embeddings

        if new_size <= old_size:
            return

        print(f"Expanding vocabulary from {old_size} to {new_size}")

        # Create new larger embedding
        new_embedding = nn.Embedding(new_size, self.embedding_dim)

        # Copy old weights
        with torch.no_grad():
            new_embedding.weight[:old_size] = self.embedding.weight
            # Initialize new embeddings with small random values
            nn.init.normal_(new_embedding.weight[old_size:], mean=0, std=0.01)

        # Replace old embedding
        self.embedding = new_embedding

    def set_columns(self, column_names: list[str]) -> None:
        """Set columns for current dataset, expanding vocabulary if needed."""
        self.current_columns = column_names
        self.current_indices = []

        # Check if we need to expand vocabulary
        new_columns = [col for col in column_names if col not in self.column_to_idx]
        required_size = self.next_idx + len(new_columns)

        if required_size > self.embedding.num_embeddings:
            new_vocab_size = max(
                int(self.embedding.num_embeddings * self.growth_factor), required_size
            )
            self._expand_vocabulary(new_vocab_size)

        # Assign indices to new columns
        for col in new_columns:
            self.column_to_idx[col] = self.next_idx
            self.idx_to_column[self.next_idx] = col
            self.next_idx += 1

        # Get indices for current columns
        self.current_indices = [self.column_to_idx[col] for col in column_names]
        print(
            f"Set {len(column_names)} columns "
            f"(vocabulary size: {self.embedding.num_embeddings})"
        )

    def get_embeddings(self) -> torch.Tensor:
        """Get embeddings for current columns."""
        indices = torch.tensor(
            self.current_indices,
            dtype=torch.long,
            device=self.embedding.weight.device,
        )
        embs = self.embedding(indices)  # [num_columns, embedding_dim]
        return self.projection(embs)  # [num_columns, target_dim]

    def forward(self, batch_size: int, sequence_length: int) -> torch.Tensor:
        """Create column embeddings for a batch."""
        # Get embeddings [num_columns, target_dim]
        column_embs = self.get_embeddings()

        # Expand to batch and time dimensions
        column_embs = column_embs.unsqueeze(0).unsqueeze(2)
        column_embs = column_embs.repeat(batch_size, 1, sequence_length, 1)
        column_embs = column_embs.permute(0, 1, 3, 2)

        if self.target_dim == 1:
            column_embs = column_embs.squeeze(2)

        return column_embs


class SemanticColumnEmbedder(nn.Module):
    """Wide semantic column identity for per-feature tokens.

    Encodes each column's natural-language description **once** with a frozen
    text encoder (stored as a buffer) and projects it into model space with a
    **learnable** ``Linear``. Returns ``[num_columns, d_model]`` to be added to
    per-feature tokens (e.g. the statistical summary tokens).

    This differs from :class:`BERTColumnEmbedding` in two ways that matter for
    a real identity pathway:

    1. The projection is learnable — ``BERTColumnEmbedding`` bakes its
       projection output into a frozen buffer at init, so its projection never
       trains.
    2. It targets full ``d_model`` rather than a single scalar channel, so the
       semantic signal is not crushed to a 1-D ordering of columns.

    The text encoder is loaded only to precompute embeddings and is **not**
    retained as a submodule, keeping checkpoints small.
    """

    def __init__(
        self,
        descriptions: list[str],
        d_model: int,
        bert_model: str = "bert-base-uncased",
        aggregation_strategy: str = "mean",
    ):
        """Initialize the semantic column embedder.

        Args:
            descriptions: Natural-language description per column (ordered to
                match the data's feature axis).
            d_model: Target model dimension.
            bert_model: HuggingFace text encoder used to embed descriptions.
            aggregation_strategy: How to pool token embeddings ('mean' or 'cls').
        """
        super().__init__()
        self.descriptions = descriptions
        self.d_model = d_model

        # Encode descriptions once with a frozen encoder; store as a buffer so
        # it moves with the model's device but receives no gradients.
        text_embeddings = self._encode(descriptions, bert_model, aggregation_strategy)
        self.register_buffer("text_embeddings", text_embeddings)  # [F, bert_dim]

        # Learnable projection into model space (this DOES train).
        self.projection = nn.Linear(text_embeddings.shape[1], d_model)

    @staticmethod
    def _encode(
        descriptions: list[str],
        bert_model: str,
        aggregation_strategy: str,
    ) -> torch.Tensor:
        """Embed descriptions with a frozen encoder; return ``[F, bert_dim]``."""
        tokenizer = AutoTokenizer.from_pretrained(bert_model)
        encoder = AutoModel.from_pretrained(bert_model)
        encoder.eval()

        with torch.no_grad():
            tokenized = tokenizer(
                [d.lower() for d in descriptions],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            outputs = encoder(**tokenized)
            hidden_states = outputs.last_hidden_state  # [F, L, bert_dim]

            if aggregation_strategy == "mean":
                attention_mask = tokenized["attention_mask"].unsqueeze(-1)
                summed = (hidden_states * attention_mask).sum(dim=1)
                lengths = attention_mask.sum(dim=1).clamp(min=1)
                pooled = summed / lengths  # [F, bert_dim]
            elif aggregation_strategy == "cls":
                pooled = hidden_states[:, 0, :]  # [F, bert_dim]
            else:
                raise ValueError(
                    f"Unknown aggregation strategy: {aggregation_strategy}"
                )

        # Detach so the encoder graph is fully dropped (encoder is not retained).
        return pooled.detach().clone()

    def get_embeddings(self) -> torch.Tensor:
        """Return projected semantic embeddings ``[num_columns, d_model]``."""
        return self.projection(self.text_embeddings)

    def forward(self) -> torch.Tensor:
        """Alias for :meth:`get_embeddings`."""
        return self.get_embeddings()


# Factory functions


def create_column_embedding(
    column_names: list[str],
    target_dim: int = 1,
    strategy: str = "simple",
    **kwargs: Any,
) -> SimpleColumnEmbedding | BERTColumnEmbedding:
    """Factory function to create a column embedding instance.

    Args:
        column_names: List of column names
        target_dim: Target embedding dimension (usually 1)
        strategy: 'simple' for learned embeddings or 'bert' for BERT embeddings
        **kwargs: Additional arguments for specific embedding type

    Returns:
        Configured embedding instance
    """
    if strategy == "simple":
        return SimpleColumnEmbedding(
            column_names=column_names,
            target_dim=target_dim,
            **kwargs,
        )
    elif strategy == "bert":
        return BERTColumnEmbedding(
            column_names=column_names,
            target_dim=target_dim,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def create_multi_dataset_embedder(
    strategy: str = "frozen_bert", target_dim: int = 1, **kwargs
) -> FrozenBERTColumnEmbedder | AutoExpandingEmbedder:
    """Factory function for multi-dataset column embedders.

    Args:
        strategy: 'frozen_bert' or 'auto_expanding'
        target_dim: Target embedding dimension
        **kwargs: Additional arguments for specific embedder

    Returns:
        Configured embedder instance
    """
    if strategy == "frozen_bert":
        return FrozenBERTColumnEmbedder(target_dim=target_dim, **kwargs)
    elif strategy == "auto_expanding":
        return AutoExpandingEmbedder(target_dim=target_dim, **kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# Legacy compatibility - for backwards compatibility with existing code
ColumnEmbedding = BERTColumnEmbedding


# Example column names for testing
ETTH1_COLUMNS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


if __name__ == "__main__":
    # Test simple embeddings
    print("Testing Simple Column Embeddings...")
    simple_emb = create_column_embedding(ETTH1_COLUMNS, target_dim=1, strategy="simple")
    embeddings = simple_emb(32, 96)
    print(f"Simple embeddings shape: {embeddings.shape}")

    # Test BERT embeddings
    print("\nTesting BERT Column Embeddings...")
    bert_emb = create_column_embedding(ETTH1_COLUMNS, target_dim=1, strategy="bert")
    embeddings = bert_emb(32, 96)
    print(f"BERT embeddings shape: {embeddings.shape}")

    # Test multi-dataset embedders
    print("\nTesting Multi-dataset Embedders...")
    frozen_bert = create_multi_dataset_embedder("frozen_bert")
    frozen_bert.set_columns(ETTH1_COLUMNS)
    embeddings = frozen_bert(32, 96)
    print(f"Frozen BERT embeddings shape: {embeddings.shape}")
