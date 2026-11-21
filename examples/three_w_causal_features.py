"""Past-only per-file summaries; no targets, future rows or other files used."""

import numpy as np

from scipy.signal import lfilter


def causal_history_features(features, file_ids, alpha=0.1):
    """Append four sensor blocks to the exact eight-block window representation.

    Input rows must retain chronological order within each file. Blocks are mean
    minus first earlier observed mean, mean minus past-only exponential mean,
    std relative to past-only exponential std, and reference-available mask.
    No feature uses labels, final file length, future observations or other files.
    """
    features, file_ids = np.asarray(features), np.asarray(file_ids)
    if features.ndim != 2 or features.shape[1] % 8 or len(file_ids) != len(features):
        raise ValueError("Expected eight summary blocks and aligned file IDs")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0,1]")
    c = features.shape[1] // 8
    means, stds, _, _, missing, _, _, _ = np.split(features, 8, axis=1)
    result = np.zeros((len(features), 4 * c), dtype=np.float32)
    for file in np.unique(file_ids):
        rows = np.flatnonzero(file_ids == file)
        mean, std = means[rows].astype(float), stds[rows].astype(float)
        valid = (missing[rows] < 1) & np.isfinite(mean) & np.isfinite(std)
        observed = valid.astype(float)
        # Shift before filtering: the current window cannot enter its reference.
        previous = np.r_[np.zeros((1, c)), observed[:-1]]
        denominator = lfilter([alpha], [1, -(1 - alpha)], previous, axis=0)
        past_mean = lfilter(
            [alpha],
            [1, -(1 - alpha)],
            np.r_[np.zeros((1, c)), np.where(valid, mean, 0)[:-1]],
            axis=0,
        )
        past_std = lfilter(
            [alpha],
            [1, -(1 - alpha)],
            np.r_[np.zeros((1, c)), np.where(valid, std, 0)[:-1]],
            axis=0,
        )
        past_mean /= np.maximum(denominator, 1e-15)
        past_std /= np.maximum(denominator, 1e-15)
        has_past = (np.cumsum(observed, axis=0) - observed > 0) & valid
        # argmax locates the first observed row; has_past masks every row before
        # that observation, so even those outputs cannot depend on future values.
        first = mean[np.argmax(valid, axis=0), np.arange(c)]
        blocks = [
            mean - first,
            mean - past_mean,
            std / (np.maximum(past_std, 0) + 1e-6),
            has_past.astype(float),
        ]
        result[rows] = np.concatenate(
            [np.where(has_past, block, 0) for block in blocks], axis=1
        )
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite causal feature")
    return result
