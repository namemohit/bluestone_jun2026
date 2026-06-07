"""Layered review stack for the showroom tracker.

L1 - l1_humans : ALL detected humans in a camera, as a scannable digest (recall-first,
                 full of false positives: street-through-glass, staff, repeats).
L2 - l2_entries: filter L1 down to real customer entries (street-mask + line/zone cross
                 + face-size gate + de-dup).   [next layer]

Each layer is a thin, inspectable pass so a human can review its output instead of raw video.
"""
