"""Prove the distant-host shortcut keeps accepted A1 merge decisions identical."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from persistent_merges import carry_identities_through_merges  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    stem = "95_A1"
    run = f"full_vid95_fresh_20260921_{stem}"
    config = json.loads((ROOT / "registered inputs/full_vid95_fresh_20260921_config.json").read_text())
    params = config["accepted_postprocessing"]["persistent_merges"]
    raw = tifffile.imread(ROOT / "registered inputs/full_vid95_fresh_20260921" / f"{stem}.tif")
    lag = tifffile.imread(ROOT / "registered inputs/motion evidence/full_vid95_fresh_20260921/out" / f"{stem}_lag01_float.tif")
    baseline = tifffile.imread(ROOT / "m7_persistence" / run / "out" / f"{stem}.tif")
    old = ROOT / "m8_merge" / run
    old_manifest = json.loads((old / "run.json").read_text())
    if old_manifest.get("status") != "done":
        raise ValueError("Frozen A1 m8 output is not complete")
    started = time.perf_counter()
    labels, events, inferred = carry_identities_through_merges(baseline, raw, lag, params)
    elapsed = time.perf_counter() - started
    old_labels = tifffile.imread(old / "out" / f"{stem}.tif")
    old_inferred = tifffile.imread(old / "mid" / f"{stem}_inferred_merge_pixels.tif")
    if not np.array_equal(labels, old_labels):
        raise AssertionError("Merge labels differ")
    if not np.array_equal(inferred.astype(np.uint8), old_inferred):
        raise AssertionError("Inferred merge pixels differ")
    old_events = (old / "out/persistent_merge_events.csv").read_bytes()
    new_events = events.to_csv(index=False).encode("utf-8")
    if new_events != old_events:
        raise AssertionError("Merge event table differs")
    old_frames = (old / "out/persistent_merge_frame_partitions.csv").read_bytes()
    frame_rows = events.attrs.get("frame_rows")
    new_frames = frame_rows.to_csv(index=False).encode("utf-8")
    if new_frames != old_frames:
        raise AssertionError("Merge frame partition table differs")
    report = {"stem": stem, "scope": "complete 99-frame registered field",
              "frozen_run": run, "shortcut_seconds": round(elapsed, 3),
              "frozen_stage_seconds": old_manifest.get("elapsed_seconds"),
              "labels_identical": True, "inferred_pixels_identical": True,
              "events_identical": True, "frame_partitions_identical": True,
              "frozen_labels_sha256": sha256(old / "out" / f"{stem}.tif")}
    path = ROOT / "reviews/vid95_full_fresh_20260921_outlines/m8_speed_proof.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
