"""Jamie's join rule in PyMicroglia behaves as in Motion, and the solid outlines have no gaps."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from pymicroglia.tracking import correction_rules as cr
from pymicroglia.visualisation.solid_outlines import edges, outline_image, up

MOTION = Path(os.environ.get("PYMICROGLIA_MOTION_ROOT", Path(__file__).resolve().parents[2] / "Motion"))


def disc(shape, cy, cx, r):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def broken_track():
    """Cell 1 moves right; its number ends at frame 4 and number 2 starts beside it at frame 5.
    Cell 3 sits still all video."""
    T, shape = 10, (60, 80)
    lab = np.zeros((T,) + shape, np.uint16)
    for f in range(T):
        lab[f][disc(shape, 20, 10 + 2 * f, 6)] = 1 if f < 5 else 2
        lab[f][disc(shape, 45, 60, 6)] = 3
    return lab


def test_ended_number_continues_under_its_old_name():
    lab = broken_track()
    out, info = cr.solve(lab)
    assert np.array_equal(out > 0, lab > 0)                      # masks never change
    assert set(np.unique(out)) <= set(np.unique(lab))           # no new numbers
    assert set(np.unique(out[5:][lab[5:] == 2])) == {1}        # 2 continues as 1
    assert info["links"] == 1


def test_review_list_reproduces_the_rule_and_denial_restores_it():
    lab = broken_track()
    data = cr.index_frames(cr.build(lab))
    out, _ = cr.solve(lab)
    assert np.array_equal(cr.compose(lab, data, {}), out)
    (item,) = [it for it in data["items"] if it["kind"] == "applied"]
    assert np.array_equal(cr.compose(lab, data, {item["key"]: "deny"}), lab)


def test_patch_fingerprint_is_stable():
    lab = broken_track()
    out, _ = cr.solve(lab)
    patch = cr.patch_from(lab, out)
    assert patch["parent_state_sha256"] == cr.identity_patch_state_sha256(lab, patch["frames"])


def test_outline_is_closed_at_every_zoom():
    lab = np.zeros((40, 40), np.uint16)
    lab[disc((40, 40), 20, 20, 9)] = 5
    for k in (1, 3, 5):
        line = edges(up(lab, k))
        filled = up(lab, k) > 0
        # A closed outline: the interior (cell minus outline) never touches the background.
        inner = filled & ~line
        grown = np.zeros_like(inner)
        grown[1:] |= inner[:-1]; grown[:-1] |= inner[1:]; grown[:, 1:] |= inner[:, :-1]; grown[:, :-1] |= inner[:, 1:]
        assert not (grown & ~filled).any()
        rgb = outline_image(np.full((40, 40), 50, np.uint8), lab, k, colours={5: (0, 220, 255)})
        assert rgb.shape == (40 * k, 40 * k, 3)
        assert (rgb[line] == (0, 220, 255)).all()


@pytest.mark.skipif(not (MOTION / "manual_editing" / "trajectory_review" / "gap_link.py").exists(),
                    reason="Motion checkout not beside PyMicroglia")
def test_parity_with_motion_gap_link():
    here = str(MOTION / "manual_editing" / "trajectory_review")
    sys.path[:0] = [here, str(MOTION / "code")]
    try:
        import gap_link as motion_gap_link
        import review_data as motion_review
    finally:
        sys.path.remove(here)
    rng = np.random.default_rng(3)
    lab = broken_track()
    lab[3:7][disc(lab.shape[1:], 45, 60, 6)[None].repeat(4, 0)] = 4    # a swap-like interruption
    lab[2, 5:8, 70:73] = 9                                              # a speck under the size floor
    for _ in range(2):
        mine, theirs = cr.solve(lab), motion_gap_link.solve(lab)
        assert np.array_equal(mine[0], theirs[0]) and mine[1] == theirs[1]
        assert cr.build(lab) == motion_review.build(lab)
        lab = np.roll(lab, int(rng.integers(1, 4)), axis=2)


def test_open_outline_check_flags_outlines_enlarged_after_drawing():
    from PIL import Image
    from pymicroglia.visualisation.solid_outlines import open_outlines
    lab = np.zeros((80, 80), np.uint16)
    for i, (cy, cx) in enumerate(((20, 20), (20, 58), (58, 30)), 1):
        lab[disc((80, 80), cy, cx, 9)] = i
    grey = np.full(lab.shape, 40, np.uint8)
    colours = {1: (0, 220, 255), 2: (0, 220, 255), 3: (0, 220, 255)}
    solid = outline_image(grey, lab, 3, colours=colours, default=None)
    assert open_outlines(solid, (0, 220, 255)) == {"closed": 3, "open": 0, "open_fraction": 0.0}
    native = outline_image(grey, lab, 1, colours=colours, default=None)
    blurred = np.asarray(Image.fromarray(native).resize((190, 190), Image.BILINEAR))
    assert open_outlines(blurred, (0, 220, 255))["closed"] < 3      # fewer closed outlines than cells drawn
