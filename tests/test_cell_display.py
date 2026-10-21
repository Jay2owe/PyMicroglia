"""Grey decisions alter ink, not identities or scientific inputs."""
import json

import numpy as np
import pytest
import tifffile

from pymicroglia.cell_display import decision_values, import_decisions, resolve, rgb
from pymicroglia.cell_filters import evaluate
from pymicroglia.visualisation.tracked_outlines import ACCEPTED_COLOURS, overlay


def result():
    return evaluate({19: {"missing_pct": 0}, 26: {"missing_pct": 25}, 42: {}},
                    {"field": "missing_pct", "cmp": "lte", "value": 10},
                    mode="keep", unknown="exclude")


def test_decisions_select_recording_preserve_missing_and_refuse_bad_values():
    text = "recording,identity,passes_fixed_circle\none,19,True\none,26,False\ntwo,19,False\n"
    imported = import_decisions(text, "one")
    assert decision_values(imported, "one") == {19: True, 26: False}
    with pytest.raises(ValueError, match="different recording"):
        decision_values(imported, "two")
    for text in ("identity,passed\n19,True\n19,False", "identity,passed\n19,maybe", "identity,passed\n0,true"):
        with pytest.raises(ValueError):
            import_decisions(text, "one")


def test_frozen_filter_json_and_direct_call_have_identical_colours(tmp_path):
    selected = result()
    path = tmp_path / "filter.json"
    path.write_text(json.dumps(selected))
    expected = {26: (170, 170, 170), 42: (170, 170, 170)}
    assert resolve(selected, [19, 26, 42])[0] == expected
    assert resolve(path, [19, 26, 42])[0] == expected
    assert resolve(None, [19, 26, 42]) == ({}, None)
    with pytest.raises(ValueError, match="no decision"):
        resolve(selected, [88])
    assert rgb("#123456") == (18, 52, 86)
    with pytest.raises(ValueError):
        rgb("red")


def test_grey_overrides_original_identity_colour_and_changes_only_outline_pixels(tmp_path):
    labels = np.zeros((2, 12, 12), np.uint16)
    labels[:, 2:4, 2:4] = 19
    labels[:, 7:9, 7:9] = 26
    path = tmp_path / "labels.tif"
    tifffile.imwrite(path, labels, photometric="minisblack")
    before = path.read_bytes()
    painter = overlay(path, cell_filter=result())
    photons = np.arange(12 * 12 * 3, dtype=np.uint8).reshape(12, 12, 3)
    for frame in (0, 1):
        image = painter.paint(photons, frame)
        boundary = painter.boundary[frame]
        np.testing.assert_array_equal(image[boundary == 0], photons[boundary == 0])
        assert (image[boundary == 26] == (170, 170, 170)).all()
        assert (image[boundary == 19] == ACCEPTED_COLOURS[(19 - 1) % len(ACCEPTED_COLOURS)]).all()
        assert set(np.unique(painter.labels)) == {0, 19, 26}
    assert path.read_bytes() == before
    assert painter.report()["label_colours"] == {"26": [170, 170, 170]}
    assert painter.report()["label_colour_evidence"]["grey_identities"] == [26]


def test_frozen_evidence_hash_rejects_changed_decisions():
    import hashlib
    selected = result()
    selected["evidence_sha256"] = hashlib.sha256(json.dumps(selected["rows"], sort_keys=True, allow_nan=False).encode()).hexdigest()
    selected["rows"][0]["included"] = False
    with pytest.raises(ValueError, match="evidence has changed"):
        resolve(selected, [19, 26])


def test_default_render_still_supports_older_shared_outline_constructor(tmp_path, monkeypatch):
    from pymicroglia.visualisation import tracked_outlines
    current = tracked_outlines.OutlineOverlay
    def legacy(*, labels, boundary, source, sha256, discovery, colour, width_px, opacity, colour_by_label, palette):
        return current(labels=labels, boundary=boundary, source=source, sha256=sha256,
                       discovery=discovery, colour=colour, width_px=width_px,
                       opacity=opacity, colour_by_label=colour_by_label, palette=palette)
    monkeypatch.setattr(tracked_outlines, "OutlineOverlay", legacy)
    labels = np.ones((2, 6, 6), np.uint16) * 19
    path = tmp_path / "labels.tif"
    tifffile.imwrite(path, labels, photometric="minisblack")
    assert overlay(path).label_colours == {}
    with pytest.raises(RuntimeError, match="per-identity colour"):
        overlay(path, cell_filter=result())
