"""The shared boundary accepts geometry, and preserves cell ownership."""

import numpy as np
import pytest
import tifffile

from pymicroglia.tracking import mask_proposals, post_refine, refinement_methods


def _proposals():
    labels = np.zeros((2, 20, 20), np.uint16)
    labels[:, 3:9, 4:10] = 42
    outlines = np.zeros_like(labels, bool)
    outlines[:, 3, 4:10] = outlines[:, 8, 4:10] = True
    outlines[:, 3:9, 4] = outlines[:, 3:9, 9] = True
    return [
        {"kind": "labels", "data": labels},
        {"kind": "binary", "data": labels > 0},
        {"kind": "probability", "data": (labels > 0).astype(float), "threshold": 0.5},
        {"kind": "outlines", "data": outlines},
        {"kind": "polygons", "data": [[[(4, 3), (9, 3), (9, 8), (4, 8)]]] * 2},
    ]


def test_all_geometry_types_use_the_same_handoff():
    baseline = np.zeros((2, 20, 20), np.uint16)
    baseline[:, 3:9, 3:9] = 7
    raw = np.ones(baseline.shape, np.float32)
    results = [mask_proposals.attach(raw, baseline, p)[0] for p in _proposals()]
    assert all(np.array_equal(results[0], result) for result in results)
    assert np.all(results[0][:, 3:9, 9] == 7)
    assert not results[0][:, 3:9, 3].any()


@pytest.mark.parametrize("kind", ["binary", "outlines"])
def test_standard_255_foreground_tiffs_are_supported(kind):
    proposal = next(p for p in _proposals() if p["kind"] == kind)
    scaled = {"kind": kind, "data": proposal["data"].astype(np.uint8) * 255}
    a, _ = mask_proposals.normalise(proposal, (2, 20, 20))
    b, _ = mask_proposals.normalise(scaled, (2, 20, 20))
    assert np.array_equal(a, b)


def test_joined_foreground_preserves_two_cells_and_ignores_untracked_object():
    old = np.zeros((1, 24, 24), np.uint16)
    old[:, 5:9, 3:7] = 19
    old[:, 5:9, 11:15] = 26
    fg = np.zeros_like(old, bool)
    fg[:, 4:10, 3:15] = True  # One connected prediction across both cells.
    fg[:, 18:21, 18:21] = True  # No existing identity here.
    result, _, _ = mask_proposals.attach(np.ones_like(old), old, {"kind": "binary", "data": fg})
    assert set(np.unique(result)) == {0, 19, 26}
    assert np.array_equal(result[old > 0], old[old > 0])
    assert not result[:, 18:21, 18:21].any()
    assert np.all(result[:, 5:9, 7:11] > 0)


def test_disappearing_prediction_retains_the_existing_cell():
    old = np.zeros((2, 16, 16), np.uint16)
    old[0, 4:7, 4:7] = 19  # Motion itself loses the cell in the second frame.
    fg = np.zeros_like(old, bool)
    fg[1, 4:7, 4:7] = True  # Predictor cannot bring back an absent identity.
    result, _, report = mask_proposals.attach(old, old, {"kind": "binary", "data": fg})
    assert np.array_equal(result, old)
    assert report["fallback_detections"] == 1


def test_correspondence_is_independent_of_identity_numbers_and_image_origin():
    old = np.zeros((1, 32, 32), np.uint16)
    old[:, 7:12, 6:10] = 19
    old[:, 7:12, 14:18] = 26
    fg = np.zeros_like(old, bool)
    fg[:, 6:13, 5:19] = True
    a = mask_proposals.attach(old, old, {"kind": "binary", "data": fg})[0]
    renamed = np.where(old == 19, 80, np.where(old == 26, 3, 0)).astype("uint16")
    b = mask_proposals.attach(old, renamed, {"kind": "binary", "data": fg})[0]
    recovered = np.where(b == 80, 19, np.where(b == 3, 26, 0))
    assert np.array_equal(a, recovered)
    shifted_old = np.roll(old, (3, 4), axis=(1, 2))
    shifted_fg = np.roll(fg, (3, 4), axis=(1, 2))
    c = mask_proposals.attach(shifted_old, shifted_old,
                             {"kind": "binary", "data": shifted_fg})[0]
    assert np.array_equal(np.roll(a, (3, 4), axis=(1, 2)), c)


@pytest.mark.parametrize("kind", ["binary", "probability", "outlines", "polygons"])
def test_saved_proposals_refine_completed_motion_without_a_detection_stack(tmp_path, kind):
    import json

    proposal = next(p for p in _proposals() if p["kind"] == kind)
    labels = np.zeros((2, 20, 20), np.uint16)
    labels[:, 3:9, 3:9] = 26
    raw_path, label_path = tmp_path / "raw.tif", tmp_path / "labels.tif"
    tifffile.imwrite(raw_path, np.ones((3, 20, 20), np.float32), photometric="minisblack")
    tifffile.imwrite(label_path, labels, photometric="minisblack")
    path = tmp_path / ("proposal.json" if kind == "polygons" else "proposal.tif")
    if kind == "polygons":
        path.write_text(json.dumps([[]] + proposal["data"]))
    else:
        data = np.concatenate([np.zeros_like(proposal["data"][:1]), proposal["data"]])
        tifffile.imwrite(path, data, photometric="minisblack")
    report = post_refine.refine_motion(raw_path, label_path,
        output_dir=tmp_path / "refined", source_frame_offset=1,
        method_options={"path": str(path), "kind": kind})
    result = tifffile.imread(report["outputs"]["refined_labels"])
    assert np.all(result[:, 3:9, 9] == 26)
    assert report["settings"]["baseline"] == "frozen_motion_masks"
    assert "proposal" in report["settings"]["inputs_sha256"]
    assert np.array_equal(tifffile.imread(label_path), labels)


def test_open_outline_and_invalid_probability_are_refused():
    open_line = np.zeros((1, 10, 10), np.uint8)
    open_line[:, 3, 2:7] = 1
    with pytest.raises(ValueError, match="closed contours"):
        mask_proposals.normalise({"kind": "outlines", "data": open_line}, open_line.shape)
    with pytest.raises(ValueError, match="finite"):
        mask_proposals.normalise({"kind": "probability", "data": open_line * np.nan}, open_line.shape)


def test_single_frame_provider_retains_the_exact_accepted_gate(monkeypatch, tmp_path):
    from pymicroglia import mask_methods
    raw = np.zeros((2, 32, 32), np.float32)
    base = np.zeros_like(raw, np.uint16)
    base[:, 10:16, 10:16] = 1
    single = np.roll(base, 1, axis=2)
    raw[single > 0] = 10
    monkeypatch.setattr(mask_methods, "predict", lambda *a, **k: {"cells": single})
    result, _, descriptor = refinement_methods.propose(raw, base,
        weights=tmp_path / "model.pt", options={"cut": 0.8})
    expected = np.stack([post_refine.gate_frame(r, b, p)[0]
                         for r, b, p in zip(raw, base, single)])
    assert np.array_equal(result, expected)
    assert descriptor["proposal"]["policy"] == "photon_support"


def test_cd68_provider_uses_the_same_geometry_interface(monkeypatch):
    from pymicroglia.learned_mask import cd68
    proposal = _proposals()[1]
    base = np.zeros((2, 20, 20), np.uint16)
    base[:, 3:9, 3:9] = 11
    monkeypatch.setattr(cd68, "predict", lambda *a, **k: proposal)
    result, _, descriptor = refinement_methods.propose(base, base,
                                                       method="cd68", weights="example.pt")
    expected = mask_proposals.attach(base, base, proposal)[0]
    assert np.array_equal(result, expected)
    assert descriptor["proposal"]["policy"] == "overlap"


def test_cd68_retains_the_frozen_probability_export_precision():
    from pymicroglia.learned_mask.cd68 import exported_probabilities
    probabilities = np.array([0.0, 0.5, 0.5001, 0.999, 1.0], np.float32)
    expected = np.array([0, 127, 127, 254, 255], np.float32) / 255
    assert np.array_equal(exported_probabilities(probabilities), expected)
