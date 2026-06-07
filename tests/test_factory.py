"""Model factories select the right backend (real backends instantiate lazily, no heavy import)."""
from fusion.embedder import ReidEmbedder, build_embedder
from vision.demographics import (
    MiVOLOEstimator,
    StubDemographicsEstimator,
    build_demographics,
)


def test_build_demographics():
    assert isinstance(build_demographics(), StubDemographicsEstimator)
    assert isinstance(build_demographics({"backend": "stub"}), StubDemographicsEstimator)
    assert isinstance(build_demographics({"backend": "mivolo"}), MiVOLOEstimator)


def test_build_embedder():
    assert build_embedder({"embedder": "stub"}) is None
    # ReidEmbedder constructs without importing torchreid (import is lazy in _load)
    assert isinstance(build_embedder({"embedder": "reid"}), ReidEmbedder)
