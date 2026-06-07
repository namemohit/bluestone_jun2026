"""Appearance-embedding backends.

`Embedder` is the interface the pipeline depends on. `ReidEmbedder` is the real body-ReID
backend (torchreid OSNet by default; swap to CLIP-ReID weights for the accuracy-first tier
behind the same interface). Synthetic demos/tests don't use this — they generate identity
vectors directly (see fusion/synthetic_visits.py), so the matcher is testable without a GPU.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from fusion.tracklet import l2_normalize


class Embedder(ABC):
    dim: int = 0

    @abstractmethod
    def embed(self, frame, bbox):
        """Return an L2-normalized appearance vector for the crop frame[bbox], or None."""
        ...


class ReidEmbedder(Embedder):
    """Body-ReID embeddings via torchreid's FeatureExtractor (lazy, optional dependency).

    Default model OSNet; for the locked accuracy-first tier, point this at CLIP-ReID weights
    (same interface). Requires `pip install torchreid torch` + a GPU for throughput.
    """

    def __init__(self, model_name: str = "osnet_x1_0", weights: str = "", device=None, dim: int = 512):
        self.model_name = model_name
        self.weights = weights
        self.device = device
        self.dim = dim
        self._extractor = None

    def _load(self):
        from torchreid.utils import FeatureExtractor  # lazy/optional heavy import

        self._extractor = FeatureExtractor(
            model_name=self.model_name,
            model_path=self.weights or "",
            device=self.device or "cpu",
        )

    def embed(self, frame, bbox):
        if self._extractor is None:
            self._load()
        x1, y1, x2, y2 = (int(v) for v in bbox)
        x1, y1 = max(0, x1), max(0, y1)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        feats = self._extractor([crop])  # 1 x D tensor
        return l2_normalize(feats[0].detach().cpu().numpy())


def build_embedder(cfg: dict | None = None):
    """Factory: a real pixel embedder for production, or None for the synthetic/stub path.

    cfg.embedder = 'reid' -> ReidEmbedder (torchreid OSNet; point reid_weights at CLIP-ReID
    for the accuracy-first tier). 'stub' -> None (synthetic identities, no pixel embedding).
    """
    cfg = cfg or {}
    if cfg.get("embedder", "stub") == "reid":
        return ReidEmbedder(
            model_name=cfg.get("reid_model", "osnet_x1_0"),
            weights=cfg.get("reid_weights", ""),
            device=cfg.get("device"),
        )
    return None
