"""What the file states, and the two questions it cannot answer.

Timestamps, pixel size and channel names come out of the OME-XML. Which channel
is which, and which part of the recording is usable, do not — those are
judgement calls, and an answer a person gives is stored as a decision keyed on
the source so nobody is asked twice.
"""

from __future__ import annotations

import numpy as np
import pytest

import fixtures
from pymicroglia import metadata, open_series, store
from pymicroglia.store import budget


@pytest.fixture
def store_root(tmp_path, monkeypatch):
    monkeypatch.setenv("PYMICROGLIA_STORE", str(tmp_path / "cache"))
    monkeypatch.delenv("PYMICROGLIA_DECISIONS", raising=False)
    return tmp_path / "cache"


@pytest.fixture
def stack(tmp_path):
    path = tmp_path / "raw" / "VID52_C1_phase-green-red_timestack.tif"
    fixtures.write_series(path, frames=8, channels=3, height=32, width=24)
    return path


# ------------------------------------------------------------------ OME-XML
def test_timestamps_pixel_size_and_channel_names_come_out_of_the_file(stack):
    with open_series(stack) as series:
        meta = series.meta

    assert meta.times_s is not None
    assert meta.time_source.startswith("OME-XML Plane DeltaT")
    assert meta.um_per_px == pytest.approx(0.65)
    assert meta.um_source == "OME PhysicalSizeX"
    assert meta.channel_names == list(fixtures.CHANNEL_NAMES)


def test_the_frame_times_are_hours_from_the_first_plane(stack):
    with open_series(stack) as series:
        hours = series.meta.times_h

    assert hours[0] == 0.0
    assert hours[1] == pytest.approx(0.5)      # 30 minutes
    assert hours[-1] == pytest.approx(3.5)


def test_a_gap_in_the_acquisition_shows_up_in_the_times(tmp_path):
    path = tmp_path / "gapped.tif"
    fixtures.write_series(path, frames=8, channels=2, height=8, width=6,
                          gap_after=3, gap_hours=6.0)
    with open_series(path) as series:
        hours = series.meta.times_h

    assert hours[4] - hours[3] == pytest.approx(6.5)


def test_a_file_with_no_plane_timestamps_returns_cleanly(tmp_path):
    """Not every converted file carries DeltaT. That is a missing fact, not an
    error: the caller decides whether a uniform spacing is acceptable."""
    path = tmp_path / "no_times.tif"
    fixtures.write_series(path, frames=4, channels=2, height=8, width=6,
                          plane_timestamps=False)
    with open_series(path) as series:
        meta = series.meta

    assert meta.times_s is None
    assert meta.times_h is None
    assert meta.time_source == ""
    assert meta.channel_names == ["phase", "green_biolum"]


def test_a_file_with_no_pixel_size_says_it_is_not_calibrated(tmp_path):
    path = tmp_path / "uncalibrated.tif"
    fixtures.write_series(path, frames=2, channels=2, height=8, width=6,
                          um_per_px=None)
    with open_series(path) as series:
        assert series.meta.um_per_px is None
        assert any("not calibrated" in note for note in series.meta.notes)


def test_parse_ome_on_an_empty_description_returns_empty_fields():
    parsed = metadata.parse_ome("")
    assert parsed == {"t": None, "um": None, "names": None, "t_note": None}


def test_deltat_units_other_than_seconds_are_converted():
    xml = ('<OME><Image><Pixels>'
           '<Plane TheT="0" TheC="0" DeltaT="0" DeltaTUnit="min"/>'
           '<Plane TheT="1" TheC="0" DeltaT="30" DeltaTUnit="min"/>'
           '</Pixels></Image></OME>')
    times = metadata.parse_ome(xml)["t"]

    assert times[1, 0] == pytest.approx(1800.0)      # 30 min in seconds


# -------------------------------------------------------- channel assignment
def test_the_channels_are_identified_from_their_statistics(stack, store_root):
    """Bioluminescence is the one with saturating cosmic-ray spikes;
    brightfield is the flat one. Medians alone would not separate them."""
    with open_series(stack) as series:
        assignment = metadata.assign_channels(series)

    assert assignment.dluc == 1          # green_biolum
    assert assignment.bf == 0            # phase, flat
    assert assignment.struct == 2        # red_mCherry, structured
    assert assignment.source == "inferred"
    assert assignment.confidence == "high"


def test_an_override_is_obeyed_and_recorded_as_a_decision(stack, store_root):
    with open_series(stack) as series:
        assignment = metadata.assign_channels(series, override="dluc=2,bf=1,struct=0")

        assert assignment.dluc == 2 and assignment.source == "override"
        assert store.decision("channel_assignment", series.source)["dluc"] == 2


def test_a_recorded_decision_outranks_the_statistics(stack, store_root):
    with open_series(stack) as series:
        metadata.assign_channels(series, override={"dluc": 2, "bf": 0, "struct": 1})
        again = metadata.assign_channels(series)

    assert again.dluc == 2
    assert again.source == "decision"


def test_a_decision_survives_a_version_bump_and_a_full_eviction(stack,
                                                                store_root):
    """A person answered a question once. A new engine version is not a reason
    to ask them again, and neither is clearing the local cache."""
    with open_series(stack) as series:
        metadata.assign_channels(series, override="dluc=0,bf=1,struct=2")

        budget.evict(cap=0)
        store.manifest.path().unlink()

        after = metadata.assign_channels(series)

    assert after.dluc == 0
    assert after.source == "decision"


def test_an_override_that_does_not_name_the_bioluminescence_channel_is_refused(
        stack, store_root):
    with open_series(stack) as series:
        with pytest.raises(ValueError):
            metadata.assign_channels(series, override="bf=1")


def test_an_unknown_role_in_an_override_is_refused(stack, store_root):
    with open_series(stack) as series:
        with pytest.raises(ValueError) as raised:
            metadata.assign_channels(series, override="lumen=1")
    assert "lumen" in str(raised.value)


def test_a_channel_number_outside_the_file_is_refused(stack, store_root):
    with open_series(stack) as series:
        with pytest.raises(ValueError) as raised:
            metadata.assign_channels(series, override="dluc=9")
    assert "3 channel" in str(raised.value)


def test_a_single_non_bioluminescence_channel_does_both_jobs(tmp_path,
                                                             store_root):
    path = tmp_path / "two_channel.tif"
    fixtures.write_series(path, frames=4, channels=2, height=32, width=24)
    with open_series(path) as series:
        assignment = metadata.assign_channels(series)

    assert assignment.dluc == 1
    assert assignment.bf == assignment.struct == 0
    assert any("two jobs" in reason for reason in assignment.reasons)


def test_a_name_that_disagrees_with_the_statistics_is_reported_not_obeyed(
        tmp_path, store_root):
    """The statistics decide, as they must for files with no names at all. A
    disagreement is worth saying out loud."""
    path = tmp_path / "misnamed.tif"
    data = fixtures.synthetic_stack(4, 3, 32, 24)
    description = fixtures.ome_xml(4, 3, 32, 24,
                                   names=("phase", "widget", "red_mCherry"))
    import tifffile

    tifffile.imwrite(str(path), data.reshape(-1, 32, 24),
                     description=description, metadata=None,
                     photometric="minisblack")
    with open_series(path) as series:
        assignment = metadata.assign_channels(series)

    assert assignment.dluc == 1                       # statistics still win
    assert assignment.confidence == "check"
    assert any("widget" in reason for reason in assignment.reasons)


# ---------------------------------------------------------- the time window
def test_the_longest_continuous_block_is_chosen():
    hours = [0.0, 0.5, 1.0, 1.5, 9.0, 9.5, 10.0, 10.5, 11.0]
    window = metadata.usable_window(hours)

    assert window.blocks == ((0, 4), (4, 9))
    assert window.block_start == 5          # 4, plus the dropped restart frame
    assert window.end == 9
    assert len(window) == 4


def test_the_frame_that_restarts_a_block_is_dropped():
    """It is stray-light contaminated."""
    hours = [0.0, 9.0, 9.5, 10.0, 10.5]
    window = metadata.usable_window(hours)

    assert window.start == 2
    assert any("stray light" in note for note in window.notes)


def test_a_recording_with_no_gaps_keeps_everything():
    hours = [0.0, 0.5, 1.0, 1.5]
    window = metadata.usable_window(hours)

    assert (window.start, window.end) == (0, 4)
    assert window.gaps == ()


def test_an_explicit_window_selects_the_block_containing_it(tmp_path):
    """Not necessarily the longest block. This is what keeps matched
    recordings on matched hours."""
    hours = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 20.0, 20.5, 21.0]
    window = metadata.usable_window(hours, t0=20.0, t1=21.0)

    assert window.block_start == 7          # the short late block, plus restart
    assert window.end == 9
    assert any("explicit window" in note for note in window.notes)


def test_an_explicit_window_that_matches_nothing_is_refused():
    with pytest.raises(ValueError) as raised:
        metadata.usable_window([0.0, 0.5, 1.0], t0=50.0, t1=60.0)
    assert "does not overlap" in str(raised.value)


def test_a_start_that_throws_away_most_of_the_block_says_so():
    hours = list(np.arange(0, 10, 0.5))
    window = metadata.usable_window(hours, t0=8.0)

    assert any("throws away" in note for note in window.notes)


def test_no_timestamps_at_all_is_refused():
    with pytest.raises(ValueError):
        metadata.usable_window([])


def test_the_window_is_json_shaped():
    described = metadata.usable_window([0.0, 0.5, 1.0]).as_dict()
    assert described["frames"] == 3
    assert described["blocks"] == [[0, 3]]
