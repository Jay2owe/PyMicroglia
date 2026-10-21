"""Optional suppression uses foreground alone and never decides an identity."""

import csv
import json

import numpy as np
import pytest
import tifffile

from pymicroglia.tracking import mask_proposals, post_refine, refinement_methods
from pymicroglia.tracking.contract import sha256_of


@pytest.mark.parametrize("kind", mask_proposals.KINDS)
def test_intersection_accepts_every_geometry_without_matching_labels(kind):
    old = np.zeros((1, 12, 12), np.uint16)
    old[:, 2:8, 2:5] = 19
    old[:, 2:8, 5:8] = 26
    foreground = np.zeros_like(old, bool)
    foreground[:, 3:7, 3:9] = True
    data = foreground.astype(np.uint16) * 99
    if kind == "probability":
        data = foreground.astype(float)
    elif kind == "outlines":
        data = np.zeros_like(old)
        data[:, 3, 3:9] = data[:, 6, 3:9] = 255
        data[:, 3:7, 3] = data[:, 3:7, 8] = 255
    elif kind == "polygons":
        data = [[[(3, 3), (8, 3), (8, 6), (3, 6)]]]
    result, _, metadata = mask_proposals.attach(
        old, old, {"kind": kind, "data": data}, policy="intersection")
    assert np.array_equal(result, np.where(foreground, old, 0))
    assert set(np.unique(result)) == {0, 19, 26}
    assert metadata["policy"] == "intersection"
    assert metadata["fallback_detections"] == 0


@pytest.mark.parametrize("with_detections", [False, True])
@pytest.mark.parametrize("saved_gate", [False, True])
def test_public_intersection_keeps_history_and_uses_original_motion_pixels(
        tmp_path, with_detections, saved_gate):
    tracked = np.zeros((4, 16, 16), np.uint16)
    tracked[:3, 3:8, 3:8] = 19
    tracked[:, 10:13, 10:13] = 26  # Entirely outside the base detections.
    base = np.zeros((5, 16, 16), np.uint16)
    base[:, 2:9, 2:9] = 1
    prediction = np.zeros_like(base)
    prediction[0] = 99  # Leading raw frame has no corresponding Motion frame.
    prediction[1, 3:5, 3:5] = 5
    prediction[1, 6:8, 6:8] = 6  # Two pieces still carry the same identity.
    prediction[1:4, 10:13, 10:13] = 2
    prediction[2, 0:2, 0:2] = 77  # New foreground cannot create an identity.
    prediction[3, 3:8, 3:8] = 777  # Original identity 19 resumes unchanged.
    prediction[4, 3:8, 3:8] = 888  # Cannot fill an original Motion gap.
    files = {}
    for name, array in (("tracked", tracked), ("base", base),
                        ("raw", np.ones_like(base, np.float32)),
                        ("proposal", prediction)):
        files[name] = tmp_path / f"{name}.tif"
        tifffile.imwrite(files[name], array, photometric="minisblack")
    before = {key: sha256_of(path) for key, path in files.items()}
    kwargs = {"method_options": {"policy": "intersection"}}
    if saved_gate:
        kwargs["gated_masks"] = files["proposal"]
    else:
        kwargs.update(method="saved", method_options={
            "policy": "intersection", "kind": "labels", "path": str(files["proposal"])})
    result = post_refine.run(files["base"] if with_detections else None,
        files["raw"], files["tracked"], output_dir=tmp_path / "out",
        source_frame_offset=1, **kwargs)
    refined = tifffile.imread(result["outputs"]["refined_labels"])
    assert np.array_equal(refined, np.where(prediction[1:] > 0, tracked, 0))
    assert set(np.unique(refined[0])) == {0, 19, 26}
    assert set(np.unique(refined[1])) == {0, 26}
    assert set(np.unique(refined[2])) == {0, 19, 26}
    assert not refined[3].any()
    assert all(sha256_of(files[key]) == digest for key, digest in before.items())
    with open(result["outputs"]["frame_audit"], newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [json.loads(row["suppressed_identities"]) for row in rows] == [[], [19], [], [26]]
    assert [int(row["source_frame"]) for row in rows] == [1, 2, 3, 4]
    assert result["settings"]["suppressed_cell_frames"] == 2
    assert result["settings"]["identity_presence"] == "subset_of_motion"


@pytest.mark.parametrize("method", ["single_frame", "cd68"])
def test_both_network_providers_bypass_identity_correspondence(monkeypatch, tmp_path, method):
    from pymicroglia import mask_methods
    from pymicroglia.learned_mask import cd68

    original = np.zeros((3, 16, 16), np.uint16)
    original[:, 3:8, 3:8] = 19
    prediction = original.copy()
    prediction[1] = 0
    monkeypatch.setattr(mask_methods, "predict", lambda *a, **k: {"cells": prediction})
    monkeypatch.setattr(cd68, "predict", lambda *a, **k: {"kind": "binary", "data": prediction > 0})

    def no_assignment(*args, **kwargs):
        pytest.fail("intersection must not perform correspondence or photon gating")

    monkeypatch.setattr(mask_proposals, "overlap_frame", no_assignment)
    monkeypatch.setattr(post_refine, "gate_frame", no_assignment)
    result, _, _ = refinement_methods.propose(original, original, method=method,
        weights=tmp_path / "model.pt", options={"policy": "intersection"})
    assert np.array_equal(result, prediction)


def test_intersection_has_no_assignment_parameters():
    with pytest.raises(ValueError, match="unknown intersection options"):
        refinement_methods.describe("single_frame", {
            "policy": "intersection", "policy_options": {"max_expansion_px": 8}})


def test_default_policy_still_keeps_a_missed_observation():
    original = np.zeros((1, 12, 12), np.uint16)
    original[:, 2:5, 2:5] = 19
    proposal = {"kind": "binary", "data": np.zeros_like(original)}
    unchanged = mask_proposals.attach(original, original, proposal)[0]
    suppressed = mask_proposals.attach(original, original, proposal, policy="intersection")[0]
    assert np.array_equal(unchanged, original)
    assert not suppressed.any()


def test_gap_measurement_does_not_restore_suppressed_observations(tmp_path):
    from pymicroglia.pipelines._auto_microglia_support import _measure_gaps, _refine_tracking

    labels = np.zeros((5, 64, 64), np.uint16)
    labels[:, 10:14, 10:14] = 19
    labels[[0, 4], 40:44, 40:44] = 26  # A genuine three-frame Motion gap.
    photons = np.full(labels.shape, 0.01, np.float32)
    photons[:, 10:14, 10:14] = photons[:, 40:44, 40:44] = 2
    proposal = labels.copy()
    proposal[1:4, 10:14, 10:14] = 0  # Three deliberately suppressed observations.
    files = {}
    for name, value in (("labels", labels), ("photons", photons),
                        ("cells", np.zeros_like(labels)), ("proposal", proposal)):
        files[name] = tmp_path / f"{name}.tif"
        tifffile.imwrite(files[name], value, photometric="minisblack")
    decisions = tmp_path / "decisions"
    events = decisions / "09_general_merge_bridge" / "out" / "merge_events.csv"
    events.parent.mkdir(parents=True)
    events.write_text("identity_a,identity_b\n", encoding="utf-8")
    manifest = tmp_path / "motion_inputs.json"
    manifest.write_text(json.dumps({"prepared": {"movie": {
        "measurement_raw": str(files["photons"]), "frame_interval_min": 30}}}))
    tracking = {"movie": {"labels": str(files["labels"]), "source_frame_offset": 0,
                           "decisions": {"root": str(decisions)}}}
    handoff = {"outputs": {"motion_inputs": str(manifest)}}
    masks = {str(tmp_path / "movie.tif"): {"outputs": {"mask_cells": str(files["cells"])}}}
    eligibility = {"movie": {"excluded_identities": [], "views": {
        "analysis": {"labels": str(files["labels"]), "filtered": False}}}}
    refined, _, _ = _refine_tracking(tracking, handoff, masks, eligibility,
        tmp_path / "run", method="saved", options={"path": str(files["proposal"]),
        "kind": "labels", "policy": "intersection"})
    assert refined["movie"]["gap_identity_labels"] == str(files["labels"])
    results = _measure_gaps(refined, handoff, masks, tmp_path / "run")
    with open(results["movie"]["outputs"]["measurement_additions"], newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [(int(row["identity"]), int(row["frame_index"])) for row in rows] == [
        (26, 1), (26, 2), (26, 3)]
    assert np.array_equal(tifffile.imread(files["labels"]), labels)
