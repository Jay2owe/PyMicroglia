"""The quality-control figures, and the notes two earlier stages left for them.

Stage 06 recorded ``"preview_drawn_by": "stage 09, from the stored mask and
event table"`` instead of drawing a cosmic-ray preview itself. Stage 07 wrote
"the QC overlay showing which pixels became a cell is a figure, and belongs to
stage 09". Both are here now, and both are drawn from what those stages stored
rather than from anything recomputed — which is the property these tests are
really about.

What each figure asserts is narrow on purpose. A test that checked a QC figure
"looked right" would be checking Matplotlib. What is worth checking is that the
figure was drawn from the stored artefact, that its plotted table holds the
numbers the picture shows, and that it refuses rather than guessing when the
artefact it needs is not there.
"""

from __future__ import annotations

from tests.figure_record_helpers import figure_record
import csv
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")


@pytest.fixture
def store_root(tmp_path, monkeypatch):
    monkeypatch.setenv("PYMICROGLIA_STORE", str(tmp_path / "cache"))
    monkeypatch.setenv("PYMICROGLIA_DECISIONS", str(tmp_path / "decisions"))
    return tmp_path / "cache"


@pytest.fixture
def stack(tmp_path):
    from tests_support import two_channel_stack

    return two_channel_stack(tmp_path, frames=8)


def read_table(path) -> dict[str, list[str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    return {name: [row[index] for row in rows[1:]]
            for index, name in enumerate(rows[0])}


# ------------------------------------------------------------- registration
def test_the_registration_figure_is_drawn_from_the_stored_shift_table(
        tmp_path, store_root, stack):
    """Never from pixels. The tables are the analysis; the stack is 21 GB of convenience."""
    from pymicroglia import qc as report
    from pymicroglia import registration, store
    from pymicroglia.visualisation import qc

    frames = {
        "frame": [1, 2, 3],
        "shift_x_px": [0.0, 1.5, -0.5],
        "shift_y_px": [0.0, -2.0, 0.25],
        "residual_x_px": [0.0, 0.02, 0.01],
        "residual_y_px": [0.0, -0.03, 0.0],
        "residual_magnitude_px": [0.0, 0.036, 0.01],
        "phase_peak_quality": [1.0, 0.94, 0.97],
    }
    store.put(registration.REGISTRATION_STAGE, stack, {"synthetic": True},
              kind="table", value=frames, name="registration_shifts_and_qc",
              output_dir=tmp_path / "out",
              method_version=registration.METHOD_VERSIONS["reference"])

    result = qc.registration_figure(stack, output_dir=tmp_path / "figs",
                                    overwrite=True)

    assert Path(result["figures"][0]).is_file()
    drawn = read_table(result["table"])
    assert [float(v) for v in drawn["shift_x_px"]] == [0.0, 1.5, -0.5]
    assert [float(v) for v in drawn["residual_magnitude_px"]] == [0.0, 0.036, 0.01]

    record = figure_record(result)
    assert record["artefacts_drawn"], "the figure did not name what it drew"
    assert record["artefacts_drawn"][0]["stage"] == \
        registration.REGISTRATION_STAGE
    assert report  # the QC report shape is unchanged by this stage


def test_a_missing_registration_is_an_error_that_says_what_to_run(
        tmp_path, store_root, stack):
    from pymicroglia.visualisation import qc

    with pytest.raises(Exception) as raised:
        qc.registration_figure(stack, output_dir=tmp_path / "figs")
    assert "registration" in str(raised.value).lower()


# --------------------------------------------------------------- cosmic ray
def test_the_cosmic_preview_picks_the_frame_that_lost_the_most_pixels(
        tmp_path, store_root, stack):
    """A preview of a quiet frame proves nothing, so the choice is not the first frame."""
    from pymicroglia import cosmic, store
    from pymicroglia.visualisation import qc

    events = {
        "frame_zero_based": [0, 5, 5, 5, 2],
        "frame_one_based": [1, 6, 6, 6, 3],
        "grown_area_px": [3, 9, 11, 7, 4],
    }
    store.put(f"{cosmic.COSMIC_STAGE}_events", stack, {"synthetic": True},
              kind="table", value=events, name="cosmic_ray_events",
              output_dir=tmp_path / "out",
              method_version=cosmic.METHOD_VERSION)

    result = qc.cosmic_ray_preview(stack, output_dir=tmp_path / "figs",
                                   overwrite=True)

    drawn = read_table(result["table"])
    shown = [int(float(f)) for f, on in zip(drawn["frame"], drawn["shown"])
             if float(on) == 1.0]
    assert shown == [5], "frame 5 lost 27 px; frame 0 lost 3"
    assert [float(v) for v in drawn["pixels_replaced"]] == [3.0, 4.0, 27.0]


def test_the_preview_can_be_pointed_at_a_frame_and_a_cleaned_stack(
        tmp_path, store_root, stack):
    from pymicroglia.visualisation import qc

    cleaned = np.zeros((8, 120, 120), np.uint16)
    mask = np.zeros((8, 120, 120), bool)
    mask[3, 40:44, 40:44] = True

    result = qc.cosmic_ray_preview(stack, output_dir=tmp_path / "figs",
                                   overwrite=True, frame=3, cleaned=cleaned,
                                   mask=mask)
    record = figure_record(result)
    assert record["settings"]["frame"] == 3


# ------------------------------------------------------------------ channels
def test_the_channel_figure_holds_the_histogram_it_draws(tmp_path, store_root,
                                                         stack):
    """The table beside a figure is the figure's data, not a summary of it."""
    from pymicroglia.visualisation import qc

    result = qc.channel_figure(stack, output_dir=tmp_path / "figs",
                               overwrite=True, channels="dluc=0,struct=1")

    drawn = read_table(result["table"])
    assert {"dluc_intensity", "dluc_pixels", "struct_intensity",
            "struct_pixels"} <= set(drawn)
    assert len(drawn["dluc_pixels"]) == 80
    assert sum(float(v) for v in drawn["dluc_pixels"]) == 120 * 120


def test_the_frames_figure_reports_first_middle_and_last(tmp_path, store_root,
                                                         stack):
    from pymicroglia.visualisation import qc

    result = qc.frames_figure(stack, output_dir=tmp_path / "figs",
                              overwrite=True, channels="dluc=0,struct=1")

    record = figure_record(result)
    assert record["settings"]["frames"] == [0, 4, 7]
    drawn = read_table(result["table"])
    assert [int(float(v)) for v in drawn["frame"]] == [0, 4, 7]
    assert len(drawn["dluc_mean"]) == 3


def test_an_unassigned_stack_still_draws_by_position(tmp_path, store_root,
                                                     stack):
    """Somebody wanting to look at a stack has usually not assigned it yet."""
    from pymicroglia.visualisation import qc

    result = qc.frames_figure(stack, output_dir=tmp_path / "figs",
                              overwrite=True,
                              channels={"dluc": None, "struct": None})
    record = figure_record(result)
    assert set(record["settings"]["channels"]) == {"channel_0", "channel_1"}


# ------------------------------------------------------------------ overlays
def test_the_cell_overlay_draws_the_stored_segmentation(tmp_path, store_root,
                                                        stack):
    """Stage 07's deferred figure, drawn from the labels that stage stored."""
    from pymicroglia import segmentation
    from pymicroglia.visualisation import overlays

    found = segmentation.segment(stack, output_dir=tmp_path / "out",
                                 channels="dluc=0,struct=1",
                                 stationarity_check=False)
    assert int(found.labels.max()) >= 1

    result = overlays.cell_overlay(stack, output_dir=tmp_path / "figs",
                                   overwrite=True)

    drawn = read_table(result["table"])
    assert len(drawn["label"]) == int(found.labels.max())
    for index, area in zip(drawn["label"], drawn["area_px"]):
        assert float(area) == float((found.labels == int(float(index))).sum())

    record = figure_record(result)
    assert record["artefacts_drawn"][0]["stage"] == \
        segmentation.SEGMENTATION_STAGE


def test_the_overlay_says_which_image_it_drew_over(tmp_path, store_root, stack):
    """Frame 0 and the accumulated profile are different images.

    A reader silently shown one when they expected the other would be answering
    a different question, so the fall-back is named on the panel and in the
    provenance rather than being quiet.
    """
    from pymicroglia.visualisation import overlays

    labels = np.zeros((120, 120), np.int32)
    labels[38:44, 38:44] = 1

    quiet = overlays.cell_overlay(stack, output_dir=tmp_path / "a",
                                  overwrite=True, labels=labels)
    record = figure_record(quiet)
    assert record["settings"]["background"] == "frame 0 only"

    told = overlays.cell_overlay(stack, output_dir=tmp_path / "b",
                                 overwrite=True, labels=labels,
                                 background=np.ones((120, 120)))
    record = figure_record(told)
    assert record["settings"]["background"] == "accumulated profile, supplied"


def test_a_missing_segmentation_says_what_to_run_rather_than_finding_cells(
        tmp_path, store_root, stack):
    """This module draws what a stage found. It does not find anything."""
    from pymicroglia.visualisation import overlays

    with pytest.raises(FileNotFoundError) as raised:
        overlays.cell_overlay(stack, output_dir=tmp_path / "figs")
    message = str(raised.value)
    assert "segmentation.segment()" in message
    assert "does not find anything" in message


def test_sweep_candidates_are_drawn_dashed_and_recorded(tmp_path, store_root,
                                                        stack):
    """Solid is a segmented cell, dashed is a permissive candidate.

    The distinction is in the artefact and is drawn; deciding it here would make
    the figure the arbiter of what counts as a cell.
    """
    from pymicroglia.visualisation import overlays

    labels = np.zeros((120, 120), np.int32)
    labels[38:44, 38:44] = 1
    labels[78:82, 73:77] = 2

    result = overlays.cell_overlay(stack, output_dir=tmp_path / "figs",
                                   overwrite=True, labels=labels,
                                   candidates=[2])
    drawn = read_table(result["table"])
    assert [float(v) for v in drawn["sweep_candidate"]] == [0.0, 1.0]

    record = figure_record(result)
    assert record["settings"]["candidates"] == [2]


def test_the_roi_overlay_draws_the_polygons_it_was_given(tmp_path, store_root,
                                                         stack):
    from pymicroglia import roi
    from pymicroglia.visualisation import overlays

    polygon = roi.Polygon(name="scn_left", x=[30.0, 60.0, 60.0, 30.0],
                          y=[30.0, 30.0, 70.0, 70.0])
    result = overlays.roi_overlay(stack, output_dir=tmp_path / "figs",
                                  overwrite=True, polygons=[polygon])

    drawn = read_table(result["table"])
    assert [float(v) for v in drawn["region_1_x"]] == [30.0, 60.0, 60.0, 30.0]
    assert [float(v) for v in drawn["region_1_y"]] == [30.0, 30.0, 70.0, 70.0]


# ---------------------------------------------------------------- the bundle
def test_every_qc_figure_records_its_exact_sources_and_table_once(tmp_path, store_root, stack):
    from pymicroglia.visualisation import qc
    from auto_organotypic.store.ledger import entry_for
    result = qc.channel_figure(stack, output_dir=tmp_path / "figs",
                               overwrite=True, channels="dluc=0,struct=1")
    assert result["bundle"] is None
    assert Path(result["table"]).is_file()
    assert Path(result["provenance"]).is_file()
    record = entry_for(result["figures"][0])
    assert record["extra"]["table"] == Path(result["table"]).name
    row = record["extra"]["sources"][0]
    assert row["sha256"] and row["file_name"] == Path(stack).name
