"""Cross-camera fusion: co-temporal merge across cameras, no false merges."""
from fusion.mtmc import Tracklet, fuse
from fusion.synthetic_visits import at_cosine, identity_vector
import numpy as np


def _entry_pair(identity, ts, local):
    """An entry-cam + common-cam sighting of the same person at ~the same time."""
    idv = identity_vector(identity)
    rng = np.random.default_rng(identity)
    return [
        Tracklet("entry", "entry", local, ts, at_cosine(idv, 0.97, rng), crossing="in", identity=identity),
        Tracklet("common", "common", local, ts + 1.0, at_cosine(idv, 0.95, rng), identity=identity),
    ]


def test_two_cameras_merge_into_one_appearance():
    tr = _entry_pair(1, 100.0, 1)
    apps = fuse(tr, time_window=5.0, sim_threshold=0.6)
    assert len(apps) == 1
    assert apps[0].n_cameras == 2
    assert apps[0].crossing == "in"


def test_distinct_people_not_merged():
    tr = _entry_pair(1, 100.0, 1) + _entry_pair(2, 101.0, 2)
    apps = fuse(tr)
    assert len(apps) == 2
    # each appearance contains only its own identity
    for a in apps:
        assert len(set(a.identities)) == 1


def test_same_identity_far_apart_not_merged():
    # entry at 100, exit at 700 (> time_window) -> two appearances (session matcher links them)
    idv = identity_vector(5)
    rng = np.random.default_rng(5)
    tr = [
        Tracklet("entry", "entry", 1, 100.0, idv, crossing="in", identity=5),
        Tracklet("exit", "exit", 9, 700.0, at_cosine(idv, 0.97, rng), crossing="out", identity=5),
    ]
    apps = fuse(tr, time_window=5.0)
    assert len(apps) == 2
    assert {a.crossing for a in apps} == {"in", "out"}


def test_same_camera_not_merged():
    # two entry-cam tracklets (same camera) must stay separate even if similar time
    idv = identity_vector(3)
    tr = [
        Tracklet("entry", "entry", 1, 100.0, idv, crossing="in", identity=3),
        Tracklet("entry", "entry", 2, 101.0, idv, crossing="in", identity=3),
    ]
    apps = fuse(tr, time_window=5.0)
    assert len(apps) == 2
