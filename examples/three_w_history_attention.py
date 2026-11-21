"""Learned attention over a file's earlier windows (unlabeled, past-only).

Pre-registered in ``docs/3W_HISTORY_ATTENTION_PREREG.md``. Rows are whole 96-step
windows represented by the cached eight summary blocks per sensor; the context of a
query window is the preceding windows of the same file plus the file's first window.
No label, future row or other file enters the context (the ``shuffled`` control
deliberately swaps in another file's context to test that the model uses its own).
"""

from __future__ import annotations

import numpy as np
import torch

from torch import nn


N_STATS = 8
MISSING_BLOCK = 4


def file_positions(file_ids):
    """Return (start row of the file, position within the file) for every row.

    Rows of one file must be contiguous and chronological, as in the grouped cache.
    """
    file_ids = np.asarray(file_ids)
    change = np.r_[True, file_ids[1:] != file_ids[:-1]]
    if change.sum() != len(np.unique(file_ids)):
        raise ValueError("Rows of each file must be contiguous")
    start = np.maximum.accumulate(np.where(change, np.arange(len(file_ids)), 0))
    return start, np.arange(len(file_ids)) - start


def history_indices(file_ids, history=32, anchor=True):
    """Row ids ``[N, L]`` of each query's context; ``-1`` marks an absent row.

    Columns are ``[anchor, oldest previous, ..., newest previous, query]`` with
    ``L = history + anchor + 1``. The anchor is the file's first row and is present
    only when it is not already among the previous rows. Every non-negative context
    id is strictly smaller than its query id and belongs to the same file.
    """
    start, pos = file_positions(file_ids)
    rows = np.arange(len(pos))
    columns = []
    if anchor and history > 0:
        columns.append(np.where(pos > history, start, -1))
    for back in range(history, 0, -1):
        columns.append(np.where(pos >= back, rows - back, -1))
    columns.append(rows)
    return np.stack(columns, axis=1).astype(np.int64)


def shuffled_history_indices(indices, file_ids, physical, seed):
    """Replace every query's context by that of a donor row from another file.

    Donors share the query's source (physical well or simulation) so the control
    removes own-file information without changing the kind of context supplied.
    """
    file_ids, physical = np.asarray(file_ids), np.asarray(physical, dtype=bool)
    rng = np.random.default_rng(seed)
    donor = np.empty(len(file_ids), dtype=np.int64)
    for mask in [physical, ~physical]:
        pool = np.flatnonzero(mask)
        if not len(pool):
            continue
        if len(np.unique(file_ids[pool])) < 2:
            raise ValueError("Shuffled control needs two files per source")
        pick = rng.choice(pool, len(pool))
        for _ in range(100):
            same = file_ids[pick] == file_ids[pool]
            if not same.any():
                break
            pick[same] = rng.choice(pool, int(same.sum()))
        else:
            raise RuntimeError("Could not draw donors from other files")
        donor[pool] = pick
    result = indices[donor].copy()
    result[:, -1] = indices[:, -1]
    return result


def to_sensor_stats(features):
    """``[N, 8 * C]`` block layout -> ``[N, C, 8]``."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] % N_STATS:
        raise ValueError("Expected eight summary blocks")
    return np.ascontiguousarray(
        features.reshape(len(features), N_STATS, -1).transpose(0, 2, 1)
    )


def difference_scale(raw, rows, file_ids):
    """Per (sensor, stat) IQR of lag-one within-file differences at ``rows``.

    ``rows`` are the sampled training rows, so the scale never sees validation.
    """
    _, pos = file_positions(file_ids)
    rows = np.asarray(rows)
    rows = rows[pos[rows] >= 1]
    diff = raw[rows] - raw[rows - 1]
    lo, hi = np.quantile(diff, [0.25, 0.75], axis=0)
    scale = hi - lo
    fallback = diff.std(axis=0)
    scale = np.where(scale > 1e-8, scale, fallback)
    return np.where(scale > 1e-8, scale, 1.0).astype(np.float32)


class HistoryInputs:
    """Holds one split on the device and builds token inputs for index batches."""

    def __init__(self, standardized, raw, indices, scale, device):
        self.z = torch.as_tensor(to_sensor_stats(standardized), device=device)
        self.raw = torch.as_tensor(to_sensor_stats(raw), device=device)
        self.indices = torch.as_tensor(indices, device=device)
        self.scale = torch.as_tensor(scale, device=device)
        self.attention_fanout = max(indices.shape[1], self.z.shape[1])

    def __call__(self, rows):
        """Return tokens ``[B, L, C, 18]`` and absent-row mask ``[B, L]``."""
        ids = self.indices[torch.as_tensor(rows, device=self.indices.device)]
        absent = ids < 0
        safe = ids.clamp_min(0)
        z, raw = self.z[safe], self.raw[safe]
        valid = raw[..., MISSING_BLOCK] < 1
        query_raw, query_valid = raw[:, -1:], valid[:, -1:]
        pair = valid & query_valid & ~absent[:, :, None]
        diff = (query_raw - raw) / self.scale
        diff = torch.sign(diff) * torch.log1p(diff.abs())
        diff = torch.where(pair[..., None], diff, torch.zeros_like(diff))
        tokens = torch.cat(
            [z, valid[..., None].float(), diff, pair[..., None].float()], dim=-1
        )
        tokens = torch.where(absent[:, :, None, None], torch.zeros_like(tokens), tokens)
        return tokens, absent


class HistoryBlock(nn.Module):
    """Pre-norm block: across sensors within a row, across rows within a sensor."""

    def __init__(self, d_model, n_head, dropout=0.0):
        super().__init__()
        self.sensor_norm = nn.LayerNorm(d_model)
        self.sensor_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        self.row_norm = nn.LayerNorm(d_model)
        self.row_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x, absent):
        b, length, c, d = x.shape
        h = self.sensor_norm(x).reshape(b * length, c, d)
        h, _ = self.sensor_attn(h, h, h, need_weights=False)
        x = x + h.reshape(b, length, c, d)
        if length > 1:
            h = self.row_norm(x).permute(0, 2, 1, 3).reshape(b * c, length, d)
            mask = absent[:, None, :].expand(b, c, length).reshape(b * c, length)
            h, _ = self.row_attn(h, h, h, key_padding_mask=mask, need_weights=False)
            x = x + h.reshape(b, c, length, d).permute(0, 2, 1, 3)
        return x + self.ff(self.ff_norm(x))


class HistoryAttentionClassifier(nn.Module):
    """Binary classifier for the last (query) row given its unlabeled context rows."""

    def __init__(self, n_sensors, length, d_model=128, n_head=4, layers=2):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(2 * N_STATS + 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.sensor = nn.Embedding(n_sensors, d_model)
        self.position = nn.Embedding(length, d_model)
        for table in [self.sensor, self.position]:
            nn.init.normal_(table.weight, std=0.02)
        self.blocks = nn.ModuleList(
            [HistoryBlock(d_model, n_head) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)

    def forward(self, tokens, absent):
        x = self.embed(tokens) + self.sensor.weight + self.position.weight[:, None, :]
        for block in self.blocks:
            x = block(x, absent)
        return self.head(self.norm(x[:, -1]).mean(dim=1))


def offset_history_indices(file_ids, offsets=(32, 16, 8, 4, 2, 1), anchor=True):
    """Sparse variant of ``history_indices``: chosen windows back, plus the anchor.

    Columns are ``[anchor, offsets..., query]``; ``-1`` marks an absent row. The
    anchor (file's first row) is present unless an offset already selects it.
    """
    start, pos = file_positions(file_ids)
    rows = np.arange(len(pos))
    columns = []
    if anchor:
        columns.append(np.where((pos > 0) & ~np.isin(pos, offsets), start, -1))
    for back in offsets:
        columns.append(np.where(pos >= back, rows - back, -1))
    columns.append(rows)
    return np.stack(columns, axis=1).astype(np.int64)


class EncoderHistoryInputs:
    """Holds one split's standardized rows on the device for index batches."""

    def __init__(self, standardized, indices, device):
        self.z = torch.as_tensor(
            np.asarray(standardized, dtype=np.float32), device=device
        )
        self.indices = torch.as_tensor(indices, device=device)
        self.attention_fanout = max(1, indices.shape[1] - 1)

    def __call__(self, rows):
        """Return query ``[B, F]``, context ``[B, K, F]`` and absent ``[B, K]``."""
        ids = self.indices[torch.as_tensor(rows, device=self.indices.device)]
        context = ids[:, :-1]
        return self.z[ids[:, -1]], self.z[context.clamp_min(0)], context < 0


class EncoderHistoryClassifier(nn.Module):
    """A single-row classifier plus attention over earlier windows' embeddings.

    ``base`` is a ``TabularClassifier``. Context rows are encoded by the same
    encoder without gradient. The context update enters through a zero-initialised
    projection, so at initialisation (and with no context slots) the model is the
    base classifier exactly.
    """

    def __init__(self, base, d_model, slots, n_head=4, chunk=8192):
        super().__init__()
        self.base = base
        self.slots = slots
        self.chunk = chunk
        if slots:
            self.context = nn.Sequential(
                nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
            )
            self.slot = nn.Embedding(slots + 1, d_model)
            nn.init.normal_(self.slot.weight, std=0.02)
            self.query_norm = nn.LayerNorm(d_model)
            self.context_norm = nn.LayerNorm(d_model)
            self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
            self.out_norm = nn.LayerNorm(d_model)
            self.out = nn.Linear(d_model, d_model)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def embed(self, x):
        return self.base.pooling(self.base.tabular_encoder(x, None))

    def forward(self, query, context, absent):
        h = self.embed(query)
        if not self.slots:
            return self.base.head(h)
        b, k, _ = context.shape
        present = ~absent
        encoded = h.new_zeros(b, k, h.shape[-1])
        with torch.no_grad():
            rows = context[present]
            if len(rows):
                parts = [
                    self.embed(rows[i : i + self.chunk])
                    for i in range(0, len(rows), self.chunk)
                ]
                encoded[present] = torch.cat(parts).to(encoded.dtype)
        tokens = self.context(torch.cat([encoded, h[:, None] - encoded], dim=-1))
        tokens = tokens + self.slot.weight[:k]
        null = self.slot.weight[k].expand(b, 1, -1)
        tokens = self.context_norm(torch.cat([null, tokens], dim=1))
        mask = torch.cat([absent.new_zeros(b, 1), absent], dim=1)
        update, _ = self.attn(
            self.query_norm(h)[:, None],
            tokens,
            tokens,
            key_padding_mask=mask,
            need_weights=False,
        )
        return self.base.head(h + self.out(self.out_norm(update[:, 0])))


class HistoryCorrection(nn.Module):
    """A margin correction from earlier windows' embeddings, for a frozen base.

    The base classifier's own logit is left untouched; this returns a scalar that
    is added to its fault-minus-normal margin. The correction is exactly zero at
    initialisation and for any query with no context row.

    With ``absolute=False`` the module sees only standardised differences between
    the query embedding and each context embedding, so it can describe how a
    window departs from its own past but not where the file sits overall. With
    ``absolute=True`` it also sees the query and context embeddings themselves.
    """

    def __init__(self, d_model, slots, mean, scale, n_head=4, absolute=False):
        super().__init__()
        self.absolute = absolute
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        width = 2 * d_model if absolute else d_model
        self.context = nn.Sequential(
            nn.Linear(width, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.slot = nn.Embedding(slots, d_model)
        self.query = nn.Embedding(1, d_model)
        for table in [self.slot, self.query]:
            nn.init.normal_(table.weight, std=0.02)
        if absolute:
            self.query_in = nn.Linear(d_model, d_model)
        self.query_norm = nn.LayerNorm(d_model)
        self.context_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.out_norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h, encoded, absent):
        b, k, _ = encoded.shape
        tokens = (h[:, None] - encoded) / self.scale
        query = self.query.weight.expand(b, 1, -1)
        if self.absolute:
            tokens = torch.cat([(encoded - self.mean) / self.scale, tokens], dim=-1)
            query = query + self.query_in((h - self.mean) / self.scale)[:, None]
        tokens = self.context_norm(self.context(tokens) + self.slot.weight[:k])
        empty = absent.all(dim=1)
        mask = absent.clone()
        mask[empty, 0] = False  # avoid an all-masked row; its output is zeroed below
        update, _ = self.attn(
            self.query_norm(query),
            tokens,
            tokens,
            key_padding_mask=mask,
            need_weights=False,
        )
        delta = self.out(self.out_norm(update[:, 0])).squeeze(-1)
        return torch.where(empty, torch.zeros_like(delta), delta)
