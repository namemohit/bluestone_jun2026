"""Age + gender estimation (per visitor) and aggregation.

`StubDemographicsEstimator` reads ground-truth attrs carried on the appearance — it exists so
the aggregation/reporting plumbing is testable GPU-free. `MiVOLOEstimator` is the real backend
(face+body, robust to low-res CCTV) and is wired lazily for production.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter

AGE_BUCKETS = ["0-12", "13-19", "20-34", "35-54", "55+"]
GENDERS = ["female", "male"]


class DemographicsEstimator(ABC):
    @abstractmethod
    def estimate(self, ctx) -> dict:
        """Return {gender, gender_conf, age_bucket, age_conf}."""
        ...


class StubDemographicsEstimator(DemographicsEstimator):
    """ctx is an attrs dict (e.g. appearance.attrs) carrying ground-truth gender/age_bucket."""

    def estimate(self, ctx) -> dict:
        attrs = ctx if isinstance(ctx, dict) else {}
        return {
            "gender": attrs.get("gender", "unknown"),
            "gender_conf": 1.0,
            "age_bucket": attrs.get("age_bucket", "unknown"),
            "age_conf": 1.0,
        }


class MiVOLOEstimator(DemographicsEstimator):
    """Real age/gender via MiVOLO (lazy). Operates on a face/body crop. Wire weights for prod."""

    def __init__(self, weights: str = "", device=None):
        self.weights = weights
        self.device = device
        self._model = None

    def estimate(self, crop) -> dict:  # pragma: no cover - requires model weights + GPU
        raise NotImplementedError(
            "Install MiVOLO and load weights, then run inference on the crop here."
        )


def aggregate(estimates) -> dict:
    """Aggregate a list of per-visitor estimates into gender + age-bucket counts."""
    gender = Counter()
    age = Counter()
    for e in estimates:
        gender[e.get("gender", "unknown")] += 1
        age[e.get("age_bucket", "unknown")] += 1
    return {
        "n": len(estimates),
        "gender": dict(gender),
        "age_bucket": dict(age),
    }


def build_demographics(cfg: dict | None = None) -> DemographicsEstimator:
    """Factory: 'stub' (GPU-free) or 'mivolo' (real, age+gender)."""
    cfg = cfg or {}
    if cfg.get("backend", "stub") == "mivolo":
        return MiVOLOEstimator(weights=cfg.get("weights", ""), device=cfg.get("device"))
    return StubDemographicsEstimator()
