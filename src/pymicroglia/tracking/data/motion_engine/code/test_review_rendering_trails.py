"""Display trails cannot change identity masks or invent paths through gaps."""

import numpy as np
import pytest

from review_rendering import unique_outline


def test_trail_gradient_gap_and_opt_out():
    labels = np.zeros((6, 20, 24), np.uint16)
    for frame, x in ((0, 3), (1, 6), (2, 9), (4, 15), (5, 18)):
        labels[frame, 9:11, x:x + 2] = 1
    raw = np.zeros_like(labels)
    display = np.full((*labels.shape, 3), 30, np.uint8)
    options = {"raw_display": display, "edge_mask": np.zeros_like(labels, bool),
               "annotations": [[] for _ in labels]}
    previous = unique_outline(raw, labels, **options)
    off = unique_outline(raw, labels, show_motion_trails=False, **options)
    assert np.array_equal(previous, off)
    shown = unique_outline(raw, labels, show_motion_trails=True,
                           trail_frames=3, **options)
    short = unique_outline(raw, labels, show_motion_trails=True,
                           trail_frames=1, **options)
    assert np.array_equal(shown[0], previous[0])
    assert not np.array_equal(shown[2], previous[2])
    assert not np.array_equal(shown[2], short[2])
    assert np.array_equal(shown[4], previous[4])
    assert not np.array_equal(shown[5], previous[5])
    assert np.array_equal(shown[labels > 0], previous[labels > 0])

    with pytest.raises(ValueError, match="positive whole number"):
        unique_outline(raw, labels, trail_frames=0, **options)
