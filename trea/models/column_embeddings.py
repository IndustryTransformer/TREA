"""Column-identity embedders for the single-row tabular model.

The single-row encoder represents every column as a token whose *identity* is
embedded, then cross-attends identity against values. How that identity vector is
produced decides whether a model can transfer across schemas:

  IndexColumnEmbedder    -- a learned row per column *name string* (the original
                            behavior). Schema-specific: a name unseen at train time
                            maps to [UNK], so a model pretrained on schema A carries
                            *no* usable identity for schema B's renamed columns. This
                            is the negative control -- it is exactly what trees also
                            cannot do.
  SemanticColumnEmbedder -- a frozen text-LM embedding of the column's name/description,
                            projected to d_model by a small learned head. The frozen
                            text vectors live in one shared space across schemas, so a
                            projection learned while pretraining on schema A also places
                            schema B's column *descriptions* into the same d_model space.
                            Semantically-matched columns ("ambient temperature" in A,
                            "intake air temp" in B) land near each other -> the encoder's
                            learned identity->value relationship transfers. This is the
                            one thing tree models structurally cannot do.

Both expose ``forward(names: list[str]) -> [n_cols, d_model]`` so ``TabularEncoder``
uses an identical code path for either; only the embedder differs between experiment
arms. Only the projection (+ its LayerNorm) is trainable in the semantic embedder, so
``transfer_encoder`` moves exactly that across schemas; the frozen text backbone is a
shared, non-persistent attribute (kept out of state_dict and the optimizer).

NOTE on input text: terse codes embed poorly ("TIT" vs "turbine inlet temperature"
cos ~0.11). Feed human-readable descriptions, not raw column codes, for the semantic
geometry to mean anything. See ``scripts/schema_transfer.py``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class IndexColumnEmbedder(nn.Module):
    """Learned per-name embedding (schema-specific). Negative control for transfer.

    The vocabulary is fixed at construction from a list of known column names; any
    name not in it (e.g. a renamed column from another schema) resolves to a shared
    ``[UNK]`` row, so nothing transfers across a rename.
    """

    def __init__(self, names: list[str], d_model: int):
        super().__init__()
        self.vocab = {n: i for i, n in enumerate(names)}
        self.unk = len(self.vocab)
        self.embeddings = nn.Embedding(len(self.vocab) + 1, d_model)
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=d_model**-0.5)

    @property
    def device(self):
        return self.embeddings.weight.device

    def forward(self, names: list[str]) -> torch.Tensor:
        idx = torch.tensor(
            [self.vocab.get(n, self.unk) for n in names],
            dtype=torch.long,
            device=self.embeddings.weight.device,
        )
        return self.embeddings(idx)


class SemanticColumnEmbedder(nn.Module):
    """Frozen text-LM embedding of column names/descriptions, projected to d_model.

    The frozen text features are cached per name string in a plain dict (``_feat_cache``,
    not a buffer/parameter) and moved to the projection's device on use, so they stay
    out of state_dict and the optimizer. Only ``proj`` + ``norm`` carry gradients and
    are what ``copy_encoder_weights`` moves between schemas.
    """

    # Class-level cache so the two embedders in a transfer run (schema A and B) share
    # one loaded text backbone instead of loading the ~22M-param model twice.
    _shared_text: dict = {}

    def __init__(
        self,
        names: list[str],
        d_model: int,
        model_name: str = DEFAULT_TEXT_MODEL,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.model_name = model_name
        self._feat_cache: dict[str, torch.Tensor] = {}

        feats = self._encode(names)  # [n, text_dim], frozen; populates _feat_cache
        self.text_dim = feats.size(1)
        # Only the projection (text_dim -> d_model) and its norm train; a transfer
        # target's new names are encoded on demand the first time they are seen.
        self.proj = nn.Linear(self.text_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def _text_model(self):
        if self.model_name not in SemanticColumnEmbedder._shared_text:
            from transformers import AutoModel, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(self.model_name)
            mdl = AutoModel.from_pretrained(self.model_name).eval()
            for p in mdl.parameters():
                p.requires_grad_(False)
            SemanticColumnEmbedder._shared_text[self.model_name] = (tok, mdl)
        return SemanticColumnEmbedder._shared_text[self.model_name]

    @torch.no_grad()
    def _encode(self, names: list[str]) -> torch.Tensor:
        """Mean-pooled, L2-normalized frozen sentence embeddings; cached per name."""
        missing = [n for n in names if n not in self._feat_cache]
        if missing:
            tok, mdl = self._text_model()
            enc = tok(missing, padding=True, truncation=True, return_tensors="pt")
            out = mdl(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = F.normalize(pooled, dim=-1)
            for n, v in zip(missing, pooled, strict=True):
                self._feat_cache[n] = v.detach()
        return torch.stack([self._feat_cache[n] for n in names])

    @property
    def device(self):
        return self.proj.weight.device

    def forward(self, names: list[str]) -> torch.Tensor:
        feats = self._encode(names).to(self.proj.weight.device)
        return self.norm(self.dropout(self.proj(feats)))


def build_column_embedder(strategy: str, names: list[str], d_model: int, **kwargs):
    """Factory: ``'index'`` (negative control) or ``'semantic'`` (text-derived)."""
    if strategy == "index":
        return IndexColumnEmbedder(names, d_model)
    if strategy == "semantic":
        return SemanticColumnEmbedder(names, d_model, **kwargs)
    raise ValueError(
        f"Unknown column-embedder strategy {strategy!r}; use 'index'|'semantic'."
    )
