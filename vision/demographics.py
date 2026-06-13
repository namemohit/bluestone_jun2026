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


def age_bucket(age) -> str:
    """Map a continuous age to one of AGE_BUCKETS."""
    try:
        a = int(age)
    except Exception:
        return "unknown"
    for lo, hi, b in ((0, 12, "0-12"), (13, 19, "13-19"), (20, 34, "20-34"), (35, 54, "35-54")):
        if lo <= a <= hi:
            return b
    return "55+"


class InsightFaceEstimator(DemographicsEstimator):
    """Real age/gender via InsightFace buffalo_l (face-detect + genderage in one). Lazy-loads the model
    (~350 MB, auto-downloaded to ~/.insightface). estimate(crop) accepts an image PATH or a BGR ndarray
    and returns {gender, gender_conf, age, age_bucket, age_conf, face}. CCTV reality: ~half of full-body
    person crops have a usable face (small/distant/back-facing -> face=False, caller shows 'unknown')."""

    def __init__(self, gpu: bool = False, det_size: int = 640):
        self._app = None
        self._gpu = gpu
        self._det = det_size

    def _model(self):
        if self._app is None:
            from insightface.app import FaceAnalysis
            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if self._gpu
                         else ["CPUExecutionProvider"])
            app = FaceAnalysis(name="buffalo_l", providers=providers)
            app.prepare(ctx_id=0 if self._gpu else -1, det_size=(self._det, self._det))
            self._app = app
        return self._app

    def estimate(self, crop, with_emb: bool = False) -> dict:
        # with_emb=True also returns the 512-d ArcFace face embedding (`face_emb`) + its det confidence
        # (`face_score`) — for face-ReID. Default off so demographics caching stays small.
        none = {"gender": "unknown", "gender_conf": 0.0, "age": None,
                "age_bucket": "unknown", "age_conf": 0.0, "face": False}
        img = crop
        if isinstance(crop, str):
            import cv2
            import os
            img = cv2.imread(os.path.normpath(crop))
        if img is None:
            return none
        try:
            faces = self._model().get(img)
        except Exception:
            return none
        if not faces:
            return none
        f = max(faces, key=lambda z: float(getattr(z, "det_score", 0.0)))   # the most confidently-detected face
        age = int(getattr(f, "age", 0) or 0)
        sex = getattr(f, "sex", None)
        gender = "male" if sex == "M" else "female" if sex == "F" else "unknown"
        conf = round(float(getattr(f, "det_score", 0.0)), 3)
        out = {"gender": gender, "gender_conf": conf, "age": age,
               "age_bucket": age_bucket(age), "age_conf": conf, "face": True}
        if with_emb:
            emb = getattr(f, "embedding", None)
            if emb is not None:
                import numpy as _np
                v = _np.asarray(emb, dtype="float32"); n = float(_np.linalg.norm(v))
                out["face_emb"] = (v / n if n > 0 else v).round(5).tolist()   # L2-normalized ArcFace
                out["face_score"] = conf
        return out


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
    """Factory: 'stub' (GPU-free), 'insightface' (real, buffalo_l face+genderage), or 'mivolo'."""
    cfg = cfg or {}
    backend = cfg.get("backend", "stub")
    if backend == "insightface":
        return InsightFaceEstimator(gpu=cfg.get("gpu", False), det_size=cfg.get("det_size", 640))
    if backend == "mivolo":
        return MiVOLOEstimator(weights=cfg.get("weights", ""), device=cfg.get("device"))
    return StubDemographicsEstimator()
