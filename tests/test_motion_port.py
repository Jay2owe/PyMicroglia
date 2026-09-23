"""Frozen-input parity for the self-contained Motion handoff and engine."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

from pymicroglia.tracking import engine, motion_preparation, prepare

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "development/single_frame_mask_unet/mask_to_motion_handoff"


def test_frozen_motion_archive_imports_without_sibling_project(tmp_path):
    root = engine._engine(tmp_path)
    assert (root / "code/pipeline.py").is_file()
    with zipfile.ZipFile(ROOT / "src/pymicroglia/tracking/data/motion_engine.zip") as archive:
        hashes = json.loads(archive.read("source_hashes.json"))["files_sha256"]
        config = json.loads(archive.read("config.json"))
    scientific = {key: config[key] for key in
                  ("input_space", "observation", "anchor", "tracking",
                   "events", "accepted_postprocessing")}
    scientific_hash = hashlib.sha256(json.dumps(
        scientific, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert scientific_hash == "b20ba4a9c7aef16d79c9379c57cb97639e0b5d72f70174bff021b207b5c5b380"
    assert config["dataset"] == "Tracked recording"
    assert config["stems"] == [] and config["pinned_files"] == {}
    assert config["registered_input_dir"] == config["motion_input_dir"] == "."
    assert hashlib.sha256((root / "code/pipeline.py").read_bytes()).hexdigest() == \
        hashes["code/pipeline.py"]
    completed = subprocess.run(
        [sys.executable, str(root / "code/pipeline.py"), "--help"],
        cwd=root, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert "--config" in completed.stdout


def test_masked_counts_leave_background_zero_and_never_drop_a_masked_pixel():
    photons = np.array([[[0.0001, 2.0], [100000.0, -1.0]],
                        [[3.0, 4.0], [5.0, 6.0]]], np.float32)
    cells = np.array([[[1, 1], [0, 0]], [[1, 1], [1, 0]]], np.uint16)
    raw, scale = prepare.masked_raw(photons, cells)
    assert raw.dtype == np.uint16
    assert np.all(raw[cells == 0] == 0)
    assert np.all(raw[cells > 0] >= 1)
    assert scale > 0
    assert photons[0, 0, 1] == 2.0  # original measurement input untouched


def test_preparation_matches_stored_original_motion_inputs():
    """Compare arrays to old files, never by calling the old implementation."""
    stem = "20260721_1417"
    pictures = HANDOFF.parent / "MCG_mask_annotation/sharp" / stem / "sharp_w7.tif"
    cells = (HANDOFF.parent / "MCG_mask_model/runs/stage3_disguise/predictions"
             / f"{stem}_sharp_w7_cells.tif")
    frozen = HANDOFF / "motion_inputs"
    if not all(path.is_file() for path in (pictures, cells, frozen / f"{stem}.tif")):
        pytest.skip("the frozen handoff parity stacks are not in this checkout")
    photons = tifffile.imread(pictures)
    masks = tifffile.imread(cells)
    raw, _ = prepare.masked_raw(photons, masks)
    assert np.array_equal(raw, tifffile.imread(frozen / f"{stem}.tif"))
    params = dict(motion_preparation.DEFAULTS)
    ratio = motion_preparation.lag_ratio(raw)
    gate = motion_preparation.absolute_change_gate(raw, params["delta_noise_sigma"])
    components, table, gain, loss = motion_preparation.connected_neutral_tracks(
        raw, ratio, gate, params)
    trails, ages, _ = motion_preparation.rolling_trails(
        components, table, params["trail_window"])
    channels, _, _ = motion_preparation.composite_channels(
        raw, ratio, gain, loss, components, ages, params)
    expected = {
        "lag01_float": ratio, "neutral_tracks": components,
        "trail_labels": trails, "trail_ages": ages,
        "motion_evidence": np.stack(channels, axis=1).astype(np.uint16),
    }
    for name, array in expected.items():
        assert np.array_equal(array, tifffile.imread(frozen / f"{stem}_{name}.tif")), name
