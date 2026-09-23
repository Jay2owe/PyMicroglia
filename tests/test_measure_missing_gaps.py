"""The optional measurement gap action never changes identity or observed rows."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import tifffile

from pymicroglia import knowledge, registry
from pymicroglia.measure.gaps import measure_missing_gaps


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, fields: list[str], values: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def inputs(tmp_path: Path, *, occupied: bool = False,
           merged: bool = False) -> dict:
    baseline = tmp_path / "baseline.csv"
    original = [{"identity": 7, "frame_index": t,
                 "anchor_kind": "accepted" if t in (0, 4) else "unmatched",
                 "anchor_frame": 0 if t <= 2 else 4}
                for t in range(5)]
    write_rows(baseline, list(original[0]), original)
    sparse = np.zeros((2, 32, 32), np.uint16)
    sparse[:, 12:16, 12:16] = 7
    cells = np.zeros((5, 32, 32), np.uint16)
    if occupied:
        cells[2, 13, 13] = 2
    photons = np.full((5, 32, 32), 0.01, np.float32)
    sparse_path = tmp_path / "sparse.tif"
    cells_path = tmp_path / "cells.tif"
    photons_path = tmp_path / "photons.tif"
    tifffile.imwrite(sparse_path, sparse)
    tifffile.imwrite(cells_path, cells)
    tifffile.imwrite(photons_path, photons)
    merge_path = tmp_path / "merges.csv"
    write_rows(merge_path, ["identity_a", "identity_b"],
               [{"identity_a": 7, "identity_b": 8}] if merged else [])
    return {"source": baseline, "accepted_sparse_labels": sparse_path,
            "cells_native": cells_path, "photons_native": photons_path,
            "merge_events": merge_path, "frame_interval_h": .5,
            "excluded_identities": [], "output_dir": tmp_path / "out"}


def test_gap_length_is_a_user_setting_and_only_changes_proposal_rows(tmp_path):
    options = inputs(tmp_path)
    observed = options["source"].read_bytes()
    low = measure_missing_gaps(**{**options, "output_dir": tmp_path / "low"},
                               max_gap_frames=2)
    high = measure_missing_gaps(**{**options, "output_dir": tmp_path / "high"},
                                max_gap_frames=3)
    assert low["measurement_only_frames"] == 0
    assert rows(tmp_path / "low" / "gap_audit.csv")[0]["reason"] == "gap_too_long"
    assert high["measurement_only_frames"] == 3
    assert [int(row["frame_index"]) for row in rows(
        tmp_path / "high" / "measurement_additions.csv")] == [1, 2, 3]
    assert {row["measurement_kind"] for row in rows(
        tmp_path / "high" / "measurement_additions.csv")} == {"measurement_only"}
    assert options["source"].read_bytes() == observed
    assert not (tmp_path / "low" / "cell_frame.csv").exists()


@pytest.mark.parametrize("change,reason", [
    ({"merged": True}, "merged_identity"),
    ({"occupied": True}, "native_component_in_corridor"),
])
def test_merge_or_another_native_cell_cannot_be_filled(tmp_path, change, reason):
    result = measure_missing_gaps(**inputs(tmp_path, **change), max_gap_frames=3)
    assert result["measurement_only_frames"] == 0
    assert rows(tmp_path / "out" / "gap_audit.csv")[0]["reason"] == reason


def test_manual_merge_exclusion_and_invalid_caps(tmp_path):
    options = inputs(tmp_path)
    for value in (0, 16, 2.5, True):
        with pytest.raises(ValueError, match="1 to 15"):
            measure_missing_gaps(**options, max_gap_frames=value)
    result = measure_missing_gaps(**{**options, "excluded_identities": [7]},
                                  max_gap_frames=3)
    assert result["measurement_only_frames"] == 0
    assert rows(tmp_path / "out" / "gap_audit.csv")[0]["reason"] == "merged_identity"
    with pytest.raises(ValueError, match="already exists"):
        measure_missing_gaps(**options)


def test_invalid_time_scale_is_an_input_error(tmp_path):
    options = inputs(tmp_path)
    with pytest.raises(ValueError, match="frame_interval_h"):
        measure_missing_gaps(**{**options, "frame_interval_h": "bad"})


def test_display_only_photons_are_refused(tmp_path):
    options = inputs(tmp_path)
    renamed = tmp_path / "photons_DISPLAY_ONLY.tif"
    options["photons_native"].rename(renamed)
    with pytest.raises(ValueError, match="display-only"):
        measure_missing_gaps(**{**options, "photons_native": renamed})


def test_action_is_live_opt_in_and_describes_the_limit():
    assert registry.REGISTRY.resolve("measure_missing_gaps") is measure_missing_gaps
    details = knowledge.describe("measure_missing_gaps")
    cap = next(row for row in details["params"] if row["name"] == "max_gap_frames")
    assert cap["default"] == 5
    assert "1 to 15" in cap["description"]
    assert details["pending"] is False
    live = {row["name"]: row for row in
            registry.build_registry().action_params("measure_missing_gaps")}
    assert live["source"]["description"].startswith("Accepted per-cell")
    assert live["output_dir"]["required"] is True
    assert live["frame_interval_h"]["required"] is True


@pytest.mark.parametrize("cap,run", [(5, "r01_a001_full"),
                                     (15, "r03_a003_full")])
def test_frozen_photons_match_when_reference_is_available(tmp_path, cap, run):
    root = (Path(__file__).resolve().parents[1] / "development" /
            "single_frame_mask_unet" / "mask_to_motion_handoff" /
            "native_mask_motion_handoff_tuning")
    links = root / "inputs" / "links"
    stem = "20260721_1417"
    reference = (root / "s5_photon_extraction" / run /
                 "measurement_additions.csv")
    source = links / f"{stem}__baseline_cell_frame.csv"
    if not all(path.is_file() for path in (reference, source)):
        pytest.skip("frozen handoff evidence is not installed")
    from json import loads
    manifest = loads((links / f"{stem}__source_manifest.json").read_text("utf-8"))
    measure_missing_gaps(
        source,
        accepted_sparse_labels=links / f"{stem}__accepted_sparse_labels.tif",
        cells_native=links / f"{stem}__cells_native.tif",
        photons_native=links / f"{stem}__photons_native.tif",
        merge_events=links / f"{stem}__merge_events.csv",
        frame_interval_h=float(manifest["frame_interval_h"]),
        excluded_identities=[], output_dir=tmp_path, max_gap_frames=cap)
    actual = rows(tmp_path / "measurement_additions.csv")
    expected = [row for row in rows(reference) if row["stem"] == stem]
    assert len(actual) == len(expected)
    assert [{key: value for key, value in row.items() if key != "stem"}
            for row in expected] == actual
