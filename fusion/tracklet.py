"""Embedding math + tracklet aggregation.

A tracklet (one person within one camera) produces many per-frame embeddings; we collapse
them into ONE quality-weighted vector and remember the best crop as the thumbnail.
"""
from __future__ import annotations

import numpy as np


def l2_normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cosine(a, b) -> float:
    """Cosine similarity (inputs need not be pre-normalized)."""
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))


class TrackletAggregator:
    """Quality-weighted mean of per-frame embeddings → one tracklet vector.

    Also tracks the highest-quality observation so the caller can use it as the
    person's thumbnail / best view.
    """

    def __init__(self):
        self._sum = None
        self._wsum = 0.0
        self.n = 0
        self.best_quality = -1.0
        self.best_ref = None  # e.g. {"frame_idx":.., "bbox":..} of the sharpest/frontal crop

    def add(self, emb, quality: float = 1.0, ref=None) -> None:
        emb = np.asarray(emb, dtype=float)
        if self._sum is None:
            self._sum = np.zeros_like(emb)
        self._sum += emb * float(quality)
        self._wsum += float(quality)
        self.n += 1
        if quality > self.best_quality:
            self.best_quality = quality
            self.best_ref = ref

    def vector(self):
        if self._sum is None or self._wsum == 0:
            return None
        return l2_normalize(self._sum / self._wsum)
