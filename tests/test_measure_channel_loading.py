"""Lining an extra channel up with the outlines, checked pixel by pixel.

Ported from Motion's ``analysis/test_channel_loading.py``. Every other test
checks what a number means; these check where a number came *from*, which
for an extra channel is the whole problem: the file is right, the arithmetic
is right, and the answer is still about the wrong dye in the wrong frame at
the wrong end of the dish, with nothing anywhere looking broken.

The contract under test is one line::

    analysed[i, y, x] == source[frame_offset + i,
                                crop_origin[0] + y - shift_y[i],
                                crop_origin[1] + x - shift_x[i]]

The refusals are tested as carefully as the successes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tifffile

from pymicroglia.measure.inputs import load_channel
from pymicroglia.measure.spec import ChannelSpec, MovieSpec, _named_list

SOURCE_FRAMES, SOURCE_HEIGHT, SOURCE_WIDTH = 8, 30, 40
FIELD_HEIGHT, FIELD_WIDTH = 10, 12


def _ramp(frames: int = SOURCE_FRAMES, channels: int | None = None) -> np.ndarray:
    """``value = 10000*channel + 1000*frame + 10*y + x``: every pixel says where it is."""
    y, x = np.mgrid[0:SOURCE_HEIGHT, 0:SOURCE_WIDTH]
    plane = (10 * y + x).astype(np.uint32)
    stack = np.stack([plane + 1000 * t for t in range(frames)])
    if channels is None:
        return stack.astype(np.uint16)
    return np.stack(
        [np.stack([stack[t] + 10000 * c for c in range(channels)]) for t in range(frames)]
    ).astype(np.uint16)


def _write(path, array, axes: str | None = None):
    if axes:
        tifffile.imwrite(path, array, imagej=True, metadata={"axes": axes})
    else:
        tifffile.imwrite(path, array)
    return path


def _movie(offset: int = 0) -> MovieSpec:
    return MovieSpec(stem="t", labels="labels.tif", raw="raw.tif", source_frame_offset=offset)


def _load(channel: ChannelSpec, movie: MovieSpec | None = None, frames: int = 4):
    return load_channel(channel, movie or _movie(), frames, FIELD_HEIGHT, FIELD_WIDTH)


# ------------------------------------------------------------------ the happy path

def test_a_plain_stack_already_in_the_label_field_needs_two_settings(tmp_path):
    path = _write(tmp_path / "plain.tif", _ramp())
    stack, record = _load(ChannelSpec(name="green", path=path))

    assert stack.values.shape == (4, FIELD_HEIGHT, FIELD_WIDTH)
    assert np.array_equal(stack.values, _ramp()[:4, :FIELD_HEIGHT, :FIELD_WIDTH])
    assert record["unsampled_fraction"] == 0.0
    assert record["shift_y_range"] == [0, 0]


def test_the_crop_picks_the_field_out_of_a_larger_frame(tmp_path):
    path = _write(tmp_path / "plain.tif", _ramp())
    stack, _ = _load(ChannelSpec(name="green", path=path, crop_origin=(7, 13)))
    assert np.array_equal(
        stack.values, _ramp()[:4, 7:7 + FIELD_HEIGHT, 13:13 + FIELD_WIDTH])


def test_the_frame_offset_is_inherited_from_the_movie_and_overridable(tmp_path):
    path = _write(tmp_path / "plain.tif", _ramp())
    source = _ramp()

    inherited, record = _load(ChannelSpec(name="green", path=path), _movie(offset=3))
    assert np.array_equal(inherited.values[0], source[3, :FIELD_HEIGHT, :FIELD_WIDTH])
    assert record["frame_offset"] == 3

    overridden, record = _load(
        ChannelSpec(name="green", path=path, frame_offset=1), _movie(offset=3))
    assert np.array_equal(overridden.values[0], source[1, :FIELD_HEIGHT, :FIELD_WIDTH])
    assert record["frame_offset"] == 1


def test_the_named_channel_of_a_hyperstack_is_the_one_measured(tmp_path):
    path = _write(tmp_path / "two.tif", _ramp(channels=2), axes="TCYX")
    source = _ramp(channels=2)
    for index in (0, 1):
        stack, record = _load(ChannelSpec(name="green", path=path, channel_index=index))
        assert np.array_equal(
            stack.values, source[:4, index, :FIELD_HEIGHT, :FIELD_WIDTH]), index
        assert record["channel_index"] == index


def test_the_alignment_contract_holds_pixel_for_pixel(tmp_path):
    """``shift_scale`` is negative on purpose: a log usually records the drift, not the fix."""
    path = _write(tmp_path / "plain.tif", _ramp())
    drift = pd.DataFrame({
        "t": range(SOURCE_FRAMES),
        "cum_dy": [0.0, -1.4, -2.6, 1.0, 0.0, 0.0, 0.0, 0.0],
        "cum_dx": [0.0, 3.2, -0.4, 2.5, 0.0, 0.0, 0.0, 0.0],
    })
    shifts = tmp_path / "drift.csv"
    drift.to_csv(shifts, index=False)

    stack, record = _load(ChannelSpec(
        name="green", path=path, crop_origin=(9, 11), shifts=shifts,
        shift_columns=("cum_dy", "cum_dx"), shift_scale=-1.0))

    source = _ramp()
    shift_y = np.rint(-drift["cum_dy"].to_numpy()).astype(int)
    shift_x = np.rint(-drift["cum_dx"].to_numpy()).astype(int)
    for index in range(4):
        top, left = 9 - shift_y[index], 11 - shift_x[index]
        expected = source[index, top:top + FIELD_HEIGHT, left:left + FIELD_WIDTH]
        assert np.array_equal(stack.values[index], expected), index
    assert record["shift_y_range"] == [int(shift_y[:4].min()), int(shift_y[:4].max())]


def test_a_pixel_the_alignment_cannot_reach_is_blank_not_borrowed(tmp_path):
    """A wrapped edge would measure the far side of the field and say nothing."""
    path = _write(tmp_path / "plain.tif", _ramp())
    shifts = tmp_path / "drift.csv"
    pd.DataFrame({"shift_y": [5] * SOURCE_FRAMES,
                  "shift_x": [0] * SOURCE_FRAMES}).to_csv(shifts, index=False)

    stack, record = _load(ChannelSpec(name="green", path=path, crop_origin=(0, 0),
                                      shifts=shifts))

    assert np.isnan(stack.values[:, :5, :]).all()
    assert np.isfinite(stack.values[:, 5:, :]).all()
    assert record["unsampled_fraction"] == pytest.approx(5 / FIELD_HEIGHT)


def test_the_stack_is_kept_narrow_when_the_source_allows_it(tmp_path):
    narrow = _write(tmp_path / "narrow.tif", _ramp())
    wide = _write(tmp_path / "wide.tif", _ramp().astype(np.uint32))

    small, record = _load(ChannelSpec(name="green", path=narrow))
    assert small.values.dtype == np.float32
    assert small.saturation_value == float(np.iinfo(np.uint16).max)
    assert record["storage_dtype"] == "float32"

    large, record = _load(ChannelSpec(name="green", path=wide))
    assert large.values.dtype == np.float64
    assert large.saturation_value == float(np.iinfo(np.uint32).max)
    assert record["storage_dtype"] == "float64"


# ----------------------------------------------------------------- the refusals

def test_a_hyperstack_without_a_channel_index_is_refused(tmp_path):
    path = _write(tmp_path / "two.tif", _ramp(channels=2), axes="TCYX")
    with pytest.raises(ValueError, match="channel_index"):
        _load(ChannelSpec(name="green", path=path))


def test_a_channel_index_on_a_single_channel_file_is_refused(tmp_path):
    path = _write(tmp_path / "plain.tif", _ramp())
    with pytest.raises(ValueError, match="no channel axis"):
        _load(ChannelSpec(name="green", path=path, channel_index=1))


def test_a_channel_index_past_the_end_is_refused(tmp_path):
    path = _write(tmp_path / "two.tif", _ramp(channels=2), axes="TCYX")
    with pytest.raises(ValueError, match="outside"):
        _load(ChannelSpec(name="green", path=path, channel_index=4))


def test_a_crop_that_leaves_the_source_is_refused(tmp_path):
    path = _write(tmp_path / "plain.tif", _ramp())
    with pytest.raises(ValueError, match="crop_origin"):
        _load(ChannelSpec(name="green", path=path, crop_origin=(SOURCE_HEIGHT - 2, 0)))


def test_a_channel_too_short_for_the_labels_is_refused(tmp_path):
    path = _write(tmp_path / "plain.tif", _ramp())
    with pytest.raises(ValueError, match="frame_offset"):
        _load(ChannelSpec(name="green", path=path, frame_offset=6), frames=4)


def test_a_shifts_table_that_does_not_cover_the_movie_is_refused(tmp_path):
    path = _write(tmp_path / "plain.tif", _ramp())
    shifts = tmp_path / "short.csv"
    pd.DataFrame({"shift_y": [0, 0], "shift_x": [0, 0]}).to_csv(shifts, index=False)
    with pytest.raises(ValueError, match="one row per source frame"):
        _load(ChannelSpec(name="green", path=path, shifts=shifts))


def test_a_shifts_table_missing_its_columns_names_what_it_has(tmp_path):
    path = _write(tmp_path / "plain.tif", _ramp())
    shifts = tmp_path / "wrong.csv"
    pd.DataFrame({"dy": [0] * SOURCE_FRAMES, "dx": [0] * SOURCE_FRAMES}).to_csv(
        shifts, index=False)
    with pytest.raises(ValueError, match="dy, dx"):
        _load(ChannelSpec(name="green", path=path, shifts=shifts))


def test_a_missing_file_says_which_channel_it_belongs_to(tmp_path):
    with pytest.raises(FileNotFoundError, match="green"):
        _load(ChannelSpec(name="green", path=tmp_path / "absent.tif"))


# -------------------------------------------------------------- the declaration

def _resolve(value):
    return None if value in (None, "") else value


@pytest.mark.parametrize("name", ["Green", "green channel", "2nd", "", "raw", "labels"])
def test_a_channel_name_that_cannot_be_a_column_or_a_series_is_refused(name):
    with pytest.raises(ValueError):
        ChannelSpec.from_dict({"name": name, "path": "x.tif"}, _resolve)


def test_two_channels_with_one_name_are_refused():
    entries = [{"name": "green", "path": "a.tif"}, {"name": "green", "path": "b.tif"}]
    with pytest.raises(ValueError, match="declared twice"):
        _named_list(entries, "channel", "stem",
                    lambda entry: ChannelSpec.from_dict(entry, _resolve))


def test_the_defaults_are_no_adjustment():
    channel = ChannelSpec.from_dict({"name": "green", "path": "x.tif"}, _resolve)
    assert channel.channel_index is None
    assert channel.frame_offset is None
    assert channel.crop_origin == (0, 0)
    assert channel.shifts is None
    assert channel.shift_scale == 1.0


def test_the_declaration_round_trips_through_the_manifest():
    """``as_dict`` writes the keys ``from_dict`` reads, so a manifest rebuilds it."""
    block = {"name": "green", "path": "x.tif", "channel_index": 1, "crop_origin": [3, 4],
             "shifts": "s.csv", "shift_columns": ["dy", "dx"], "shift_scale": -1.0,
             "sha256": "abc", "description": "d"}
    channel = ChannelSpec.from_dict(block, _resolve)
    assert ChannelSpec.from_dict(channel.as_dict(), _resolve) == channel
    assert channel.as_dict()["sha256"] == "abc"
