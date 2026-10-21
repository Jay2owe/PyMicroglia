"""The accepted post-tracking route keeps Motion identities fixed."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from pymicroglia.measure.gaps import accepted_signal_trace, measure_tracked_gaps
from pymicroglia.pipelines._auto_microglia_support import (
    _measure_gaps, _refine_tracking,
)
from pymicroglia.tracking.contract import sha256_of
from pymicroglia.tracking.post_refine import run as refine_after_tracking


def _rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_refiner_preserves_motion_file_and_every_frame_identity(tmp_path):
    old = np.zeros((4, 32, 32), np.uint16)
    old[:, 10:14, 10:14] = 1
    gated = old.copy()
    gated[:, 10:14, 10:14] = 0
    gated[:, 10:14, 11:15] = 1
    tracked = np.zeros((3, 32, 32), np.uint16)
    tracked[:, 10:14, 10:14] = 19
    tracked[:, 25, 25] = 26  # Motion also tracks a cell outside this detection.
    raw = np.ones(old.shape, np.float32)
    paths = {}
    for name, array in (("old", old), ("gated", gated),
                        ("tracked", tracked), ("raw", raw)):
        paths[name] = tmp_path / f"{name}.tif"
        tifffile.imwrite(paths[name], array, photometric="minisblack")
    motion_hash = sha256_of(paths["tracked"])
    result = refine_after_tracking(
        paths["old"], paths["raw"], paths["tracked"],
        output_dir=tmp_path / "refined", source_frame_offset=1,
        gated_masks=paths["gated"])
    refined = tifffile.imread(result["outputs"]["refined_labels"])
    assert sha256_of(paths["tracked"]) == motion_hash
    assert all(set(np.unique(a)) == set(np.unique(b)) == {0, 19, 26}
               for a, b in zip(tracked, refined))
    assert np.all(refined[:, 10:14, 14] == 19)
    assert np.all(refined[:, 25, 25] == 26)
    assert result["settings"]["source_frame_offset"] == 1


def test_gap_three_is_measured_but_four_and_merges_are_not(tmp_path):
    cells = np.zeros((11, 32, 32), np.uint16)
    labels = np.zeros_like(cells)
    labels[0, 12:16, 12:16] = 7
    labels[4, 12:16, 12:16] = 7  # Three empty frames, then reappearance.
    labels[5, 12:16, 12:16] = 8
    labels[10, 12:16, 12:16] = 8  # Four empty frames.
    raw = np.full(cells.shape, 0.01, np.float32)
    raw[:, 12:16, 12:16] = 2.0
    source = tmp_path / "labels.tif"
    native = tmp_path / "cells.tif"
    photons = tmp_path / "photons.tif"
    merges = tmp_path / "merges.csv"
    for path, array in ((source, labels), (native, cells), (photons, raw)):
        tifffile.imwrite(path, array)
    merges.write_text("identity_a,identity_b\n", encoding="utf-8")
    motion_hash = sha256_of(source)
    result = measure_tracked_gaps(
        source, cells_native=native, photons_native=photons,
        merge_events=merges, frame_interval_h=0.5,
        output_dir=tmp_path / "gaps")
    additions = _rows(result["outputs"]["measurement_additions"])
    audit = _rows(result["outputs"]["gap_audit"])
    assert [(int(row["identity"]), int(row["frame_index"]))
            for row in additions] == [(7, 1), (7, 2), (7, 3)]
    assert all(row["measurement_kind"] == "measurement_only"
               for row in additions)
    assert float(additions[0]["signal_mean"]) == pytest.approx(2.0)
    assert [row["reason"] for row in audit] == ["eligible", "gap_too_long"]
    assert sha256_of(source) == motion_hash
    assert not (tmp_path / "gaps" / "cell_frame.csv").exists()

    merges.write_text("identity_a,identity_b\n7,99\n", encoding="utf-8")
    blocked = measure_tracked_gaps(
        source, cells_native=native, photons_native=photons,
        merge_events=merges, frame_interval_h=0.5,
        output_dir=tmp_path / "blocked")
    assert blocked["measurement_only_frames"] == 0
    assert _rows(blocked["outputs"]["gap_audit"])[0]["reason"] == "merged_identity"


def test_accepted_signal_trace_marks_gap_values_without_inventing_motion(tmp_path):
    observed = tmp_path / "cell_frame.csv"
    additions = tmp_path / "additions.csv"
    observed.write_text(
        "stem,identity,frame_index,hours,signal_mean,background_median,corrected_mean,area_px\n"
        "movie,7,0,1,2,0.1,1.9,16\n"
        "movie,7,4,3,2,0.1,1.9,16\n", encoding="utf-8")
    additions.write_text(
        "identity,frame_index,hours,signal_mean,background_median,corrected_mean,area_px,measurement_kind,method,gap_id,left_frame,right_frame\n"
        "7,3,1.5,2,0.1,1.9,16,measurement_only,nearest,G00000,2,6\n",
        encoding="utf-8")
    result = accepted_signal_trace(
        observed, additions, output_dir=tmp_path / "combined",
        source_frame_offset=2)
    rows = _rows(result["path"])
    assert [int(row["source_frame_index"]) for row in rows] == [2, 3, 6]
    assert [row["measurement_kind"] for row in rows] == [
        "tracked_mask", "measurement_only", "tracked_mask"]
    assert "centroid_x" not in rows[1]


def test_accepted_signal_trace_mean_uses_measured_and_accepted_gap_totals(tmp_path):
    observed = tmp_path / "cell_frame.csv"
    additions = tmp_path / "additions.csv"
    observed.write_text(
        "stem,identity,frame_index,hours,signal_mean,background_median,corrected_mean,area_px\n"
        "movie,7,0,0,10,5,5,1\n"
        "movie,7,2,1,10,5,5,3\n", encoding="utf-8")
    additions.write_text(
        "identity,frame_index,hours,signal_mean,background_median,corrected_mean,area_px,measurement_kind,method,gap_id,left_frame,right_frame\n"
        "7,1,0.5,10,5,5,2,measurement_only,nearest,G00000,0,2\n",
        encoding="utf-8")
    result = accepted_signal_trace(
        observed, additions, output_dir=tmp_path / "combined", source_frame_offset=0)
    rows = _rows(result["path"])
    assert [float(row["integrated_density"]) for row in rows] == [10., 20., 30.]
    assert [row["integrated_density_mean3"] for row in rows[::2]] == ["", ""]
    assert float(rows[1]["integrated_density_mean3"]) == 20.


def test_pipeline_views_use_refined_pixels_but_keep_motion_identity_source(tmp_path):
    old = np.zeros((4, 32, 32), np.uint16)
    old[:, 10:14, 10:14] = 1
    gated = old.copy()
    gated[:, 10:14, 10:14] = 0
    gated[:, 10:14, 11:15] = 1
    labels = np.zeros((3, 32, 32), np.uint16)
    labels[:, 10:14, 10:14] = 19
    labels[:, 25, 25] = 26
    files = {}
    for name, value in (("old", old), ("gated", gated),
                        ("raw", np.ones(old.shape, np.float32)),
                        ("labels", labels)):
        files[name] = tmp_path / f"{name}.tif"
        tifffile.imwrite(files[name], value, photometric="minisblack")
    manifest = tmp_path / "motion_inputs.json"
    manifest.write_text(json.dumps({"prepared": {
        "movie": {"measurement_raw": str(files["raw"]),
                  "frame_interval_min": 30}}}), encoding="utf-8")
    original_hash = sha256_of(files["labels"])
    tracking = {"movie": {"labels": str(files["labels"]),
                          "source_frame_offset": 1,
                          "sha256": {"labels": original_hash}}}
    eligibility = {"movie": {
        "excluded_identities": [26],
        "views": {"analysis": {"labels": str(files["labels"]),
                               "filtered": True, "sha256": original_hash},
                  "videos": {"labels": str(files["labels"]),
                             "filtered": False, "sha256": original_hash}}}}
    refined_tracking, views, reports = _refine_tracking(
        tracking, {"outputs": {"motion_inputs": str(manifest)}},
        {str(tmp_path / "movie.tif"): {"outputs": {"mask_cells": str(files["old"])}}},
        eligibility, tmp_path / "run", gated_masks={"movie": str(files["gated"])})
    assert sha256_of(files["labels"]) == original_hash
    assert tracking["movie"]["labels"] == str(files["labels"])
    assert refined_tracking["movie"]["labels"] != str(files["labels"])
    analysed = tifffile.imread(views["movie"]["views"]["analysis"]["labels"])
    reviewed = tifffile.imread(views["movie"]["views"]["videos"]["labels"])
    assert 26 not in np.unique(analysed)
    assert 26 in np.unique(reviewed)
    assert np.all(analysed[:, 10:14, 14] == 19)
    assert reports["movie"]["motion_labels_sha256"] == original_hash


def test_pipeline_gap_stage_exposes_combined_signal_trace(tmp_path):
    cells = np.zeros((5, 32, 32), np.uint16)
    labels = np.zeros_like(cells)
    labels[[0, 4], 12:16, 12:16] = 7
    photons = np.full(cells.shape, 0.01, np.float32)
    photons[:, 12:16, 12:16] = 2
    files = {}
    for name, value in (("cells", cells), ("labels", labels),
                        ("photons", photons)):
        files[name] = tmp_path / f"{name}.tif"
        tifffile.imwrite(files[name], value, photometric="minisblack")
    decisions = tmp_path / "decisions"
    events = decisions / "09_general_merge_bridge" / "out" / "merge_events.csv"
    events.parent.mkdir(parents=True)
    events.write_text("identity_a,identity_b\n", encoding="utf-8")
    manifest = tmp_path / "motion_inputs.json"
    manifest.write_text(json.dumps({"prepared": {"movie": {
        "measurement_raw": str(files["photons"]),
        "frame_interval_min": 30}}}), encoding="utf-8")
    measurement = tmp_path / "measured" / "measure" / "movie"
    measurement.mkdir(parents=True)
    (measurement / "cell_frame.csv").write_text(
        "stem,identity,frame_index,hours,signal_mean,background_median,corrected_mean\n"
        "movie,7,0,0,2,0.01,1.99\n"
        "movie,7,4,2,2,0.01,1.99\n", encoding="utf-8")
    result = _measure_gaps(
        {"movie": {"labels": str(files["labels"]),
                   "source_frame_offset": 0,
                   "decisions": {"root": str(decisions)}}},
        {"outputs": {"motion_inputs": str(manifest)}},
        {str(tmp_path / "movie.tif"): {
            "outputs": {"mask_cells": str(files["cells"])}}},
        tmp_path / "run", measured={"movie": {"folder": str(tmp_path / "measured")}})
    assert result["movie"]["measurement_only_frames"] == 3
    combined = _rows(result["movie"]["accepted_signal_trace"]["path"])
    assert len(combined) == 5
    assert [row["measurement_kind"] for row in combined] == [
        "tracked_mask", "measurement_only", "measurement_only",
        "measurement_only", "tracked_mask"]
