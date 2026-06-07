"""Lock down the geometry sign conventions and segment-intersection behavior."""
from vision.geometry import (
    inside_sign_from_label,
    segments_intersect,
    side,
    to_pixels,
)

# Horizontal line directed left->right along the middle of the frame.
P1 = (0.0, 50.0)
P2 = (100.0, 50.0)


def test_side_sign_convention_image_coords():
    # In image coords (y down): below the line (larger y) is RIGHT (+1) of P1->P2.
    assert side(P1, P2, (50.0, 90.0)) == 1
    # Above the line (smaller y) is LEFT (-1).
    assert side(P1, P2, (50.0, 10.0)) == -1
    # Exactly on the line -> 0.
    assert side(P1, P2, (50.0, 50.0)) == 0


def test_inside_label_mapping():
    assert inside_sign_from_label("left") == -1
    assert inside_sign_from_label("right") == 1
    assert inside_sign_from_label("LEFT") == -1


def test_segments_intersect_true_when_crossing():
    # Vertical movement segment crosses the horizontal line.
    assert segments_intersect((50.0, 90.0), (50.0, 10.0), P1, P2) is True


def test_segments_intersect_false_outside_segment():
    # Same vertical crossing but the line segment only spans x in [0,10];
    # the crossing at x=50 is outside it -> no intersection.
    assert segments_intersect((50.0, 90.0), (50.0, 10.0), (0.0, 50.0), (10.0, 50.0)) is False


def test_segments_intersect_false_when_parallel():
    assert segments_intersect((0.0, 10.0), (100.0, 10.0), P1, P2) is False


def test_to_pixels_scales_normalized_coords():
    pts = to_pixels([[0.5, 0.5], [1.0, 0.25]], width=200, height=100)
    assert pts == [(100.0, 50.0), (200.0, 25.0)]
