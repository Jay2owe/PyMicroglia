"""Lining a set of reference shapes up with the outlines, checked pixel by pixel.

Ported from Motion's ``analysis/test_object_loading.py``. The same problem as
an extra channel and the same contract, with one thing added: an object set
is a *label image*, so a mistake shows up as a shape in the wrong place,
measured confidently, with the right number of objects and the right areas.

    analysed[i, y, x] == source[frame_offset + i,
                                crop_origin[0] + y - shift_y[i],
                                crop_origin[1] + x - shift_x[i]]

The objects module itself is stage 04's; its test returns with it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tifffile

from pymicroglia.measure.inputs import load_object_set
from pymicroglia.measure.spec import MovieSpec, ObjectSetSpec, _named_list

SOURCE_FRAMES, SOURCE_HEIGHT, SOURCE_WIDTH = 8, 30, 40
FIELD_HEIGHT, FIELD_WIDTH = 10, 12


def _ramp(frames: int = SOURCE_FRAMES, channels: int | None = None) -> np.ndarray:
    y, x = np.mgrid[0:SOURCE_HEIGHT, 0:SOURCE_WIDTH]
    plane = (10 * y + x).astype(np.uint32)
    stack = np.stack([plane + 1000 * t for t in range(frames)])
    if channels is None:
        return stack.astype(np.uint16)
    return np.stack(
        [np.stack([stack[t] + 10000 * c for c in range(channels)]) for t in range(frames)]
    ).astype(np.uint16)


def _blobs(frames: int = SOURCE_FRAMES) -> np.ndarray:
    """Two numbered shapes: one that stays put and one that moves."""
    stack = np.zeros((frames, SOURCE_HEIGHT, SOURCE_WIDTH), dtype=np.uint16)
    for frame in range(frames):
        stack[frame, 2:6, 2:5] = 1
        stack[frame, 0:3, 8 + frame:11 + frame] = 2
    return stack


def _write(path, array, axes: str | None = None):
    if axes:
        tifffile.imwrite(path, array, imagej=True, metadata={"axes": axes})
    else:
        tifffile.imwrite(path, array)
    return path


def _movie(offset: int = 0) -> MovieSpec:
    return MovieSpec(stem="t", labels="labels.tif", raw="raw.tif", source_frame_offset=offset)


def _load(spec: ObjectSetSpec, movie: MovieSpec | None = None, frames: int = 4):
    return load_object_set(spec, movie or _movie(), frames, FIELD_HEIGHT, FIELD_WIDTH)


# ------------------------------------------------------------------ the happy path

def test_a_plain_stack_already_in_the_label_field_needs_two_settings(tmp_path):
    path = _write(tmp_path / "plain.tif", _ramp())
    stack, record = _load(ObjectSetSpec(name="vessels", path=path))

    assert stack.values.shape == (4, FIELD_HEIGHT, FIELD_WIDTH)
    assert stack.values.dtype == np.int32
    assert stack.static is False
    assert np.array_equal(stack.values, _ramp()[:4, :FIELD_HEIGHT, :FIELD_WIDTH])
    assert record["unreachable_px"] == 0
    assert record["frames_supplied"] == 4


def test_the_crop_picks_the_field_out_of_a_larger_frame(tmp_path):
    path = _write(tmp_path / "plain.tif", _ramp())
    stack, _ = _load(ObjectSetSpec(name="vessels", path=path, crop_origin=(7, 13)))
    assert np.array_equal(
        stack.values, _ramp()[:4, 7:7 + FIELD_HEIGHT, 13:13 + FIELD_WIDTH])


def test_the_alignment_contract_holds_pixel_for_pixel(tmp_path):
    path = _write(tmp_path / "plain.tif", _ramp())
    drift = pd.DataFrame({
        "t": range(SOURCE_FRAMES),
        "cum_dy": [0.0, -1.4, -2.6, 1.0, 0.0, 0.0, 0.0, 0.0],
        "cum_dx": [0.0, 3.2, -0.4, 2.5, 0.0, 0.0, 0.0, 0.0],
    })
    shifts = tmp_path / "drift.csv"
    drift.to_csv(shifts, index=False)

    stack, record = _load(ObjectSetSpec(
        name="vessels", path=path, crop_origin=(9, 11), shifts=shifts,
        shift_columns=("cum_dy", "cum_dx"), shift_scale=-1.0))

    source = _ramp()
    shift_y = np.rint(-drift["cum_dy"].to_numpy()).astype(int)
    shift_x = np.rint(-drift["cum_dx"].to_numpy()).astype(int)
    for index in range(4):
        top, left = 9 - shift_y[index], 11 - shift_x[index]
        expected = source[index, top:top + FIELD_HEIGHT, left:left + FIELD_WIDTH]
        assert np.array_equal(stack.values[index], expected), index
    assert record["shift_y_range"] == [int(shift_y[:4].min()), int(shift_y[:4].max())]


def test_the_frame_offset_is_inherited_from_the_movie_and_overridable(tmp_path):
    path = _write(tmp_path / "plain.tif", _ramp())
    source = _ramp()

    inherited, record = _load(ObjectSetSpec(name="vessels", path=path), _movie(offset=3))
    assert np.array_equal(inherited.values[0], source[3, :FIELD_HEIGHT, :FIELD_WIDTH])
    assert record["frame_offset"] == 3

    overridden, record = _load(
        ObjectSetSpec(name="vessels", path=path, frame_offset=1), _movie(offset=3))
    assert np.array_equal(overridden.values[0], source[1, :FIELD_HEIGHT, :FIELD_WIDTH])
    assert record["frame_offset"] == 1


def test_the_named_channel_of_a_hyperstack_is_the_one_read(tmp_path):
    path = _write(tmp_path / "two.tif", _ramp(channels=2), axes="TCYX")
    source = _ramp(channels=2)
    for index in (0, 1):
        stack, record = _load(ObjectSetSpec(name="vessels", path=path, channel_index=index))
        assert np.array_equal(
            stack.values, source[:4, index, :FIELD_HEIGHT, :FIELD_WIDTH]), index
        assert record["channel_index"] == index


# ---------------------------------------------------------------------- static

def test_a_static_map_is_stored_once_and_handed_out_for_every_frame(tmp_path):
    single = _blobs(frames=1)[0]
    path = _write(tmp_path / "map.tif", single)
    stack, record = _load(ObjectSetSpec(name="vessels", path=path, static=True))

    assert stack.static is True
    assert stack.values.shape == (1, FIELD_HEIGHT, FIELD_WIDTH)
    assert record["frames_supplied"] == 1
    assert record["frame_offset"] is None
    expected = single[:FIELD_HEIGHT, :FIELD_WIDTH]
    for index in range(4):
        assert np.array_equal(stack.frame(index), expected), index


def test_a_static_map_is_held_fixed_and_does_not_inherit_the_movies_offset(tmp_path):
    single = _blobs(frames=1)[0]
    path = _write(tmp_path / "map.tif", single)
    stack, _ = _load(ObjectSetSpec(name="vessels", path=path, static=True), _movie(offset=3))
    assert np.array_equal(stack.frame(0), single[:FIELD_HEIGHT, :FIELD_WIDTH])


def test_a_static_map_may_be_one_plane_of_a_multi_channel_image(tmp_path):
    plane = _blobs(frames=1)[0]
    both = np.stack([plane, plane * 7])
    path = _write(tmp_path / "map.tif", both, axes="CYX")
    stack, _ = _load(ObjectSetSpec(name="vessels", path=path, static=True, channel_index=1))
    assert np.array_equal(stack.frame(0), (plane * 7)[:FIELD_HEIGHT, :FIELD_WIDTH])


@pytest.mark.parametrize("setting,value", [("frame_offset", 2), ("shifts", "d.csv")])
def test_static_together_with_a_time_setting_is_refused(setting, value):
    def resolve(item):
        return None if item in (None, "") else item

    with pytest.raises(ValueError, match=setting):
        ObjectSetSpec.from_dict(
            {"name": "vessels", "path": "x.tif", "static": True, setting: value}, resolve)


def test_a_static_set_pointed_at_a_whole_movie_is_refused(tmp_path):
    path = _write(tmp_path / "many.tif", _blobs())
    with pytest.raises(ValueError, match="more than one map"):
        _load(ObjectSetSpec(name="vessels", path=path, static=True))


# ----------------------------------------------------------------- the refusals

def test_a_pixel_the_alignment_cannot_reach_is_empty_ground_and_is_counted(tmp_path):
    """A label image has no spare value for "could not look", so it is counted."""
    path = _write(tmp_path / "plain.tif", _blobs())
    shifts = tmp_path / "drift.csv"
    pd.DataFrame({"shift_y": [5] * SOURCE_FRAMES,
                  "shift_x": [0] * SOURCE_FRAMES}).to_csv(shifts, index=False)

    stack, record = _load(ObjectSetSpec(name="vessels", path=path, crop_origin=(0, 0),
                                        shifts=shifts))
    assert (stack.values[:, :5, :] == 0).all()
    assert record["unreachable_px"] == 4 * 5 * FIELD_WIDTH
    assert record["unreachable_fraction"] == pytest.approx(5 / FIELD_HEIGHT)


def test_a_crop_that_leaves_the_source_is_refused(tmp_path):
    path = _write(tmp_path / "plain.tif", _ramp())
    with pytest.raises(ValueError, match="crop_origin"):
        _load(ObjectSetSpec(name="vessels", path=path, crop_origin=(SOURCE_HEIGHT - 2, 0)))


def test_a_set_too_short_for_the_labels_is_refused(tmp_path):
    path = _write(tmp_path / "plain.tif", _ramp())
    with pytest.raises(ValueError, match="frame_offset"):
        _load(ObjectSetSpec(name="vessels", path=path, frame_offset=6), frames=4)


def test_a_hyperstack_without_a_channel_index_is_refused(tmp_path):
    path = _write(tmp_path / "two.tif", _ramp(channels=2), axes="TCYX")
    with pytest.raises(ValueError, match="channel_index"):
        _load(ObjectSetSpec(name="vessels", path=path))


def test_a_negative_value_means_this_is_not_a_label_image(tmp_path):
    path = _write(tmp_path / "signed.tif", (_blobs().astype(np.int16) - 1))
    with pytest.raises(ValueError, match="not a label image"):
        _load(ObjectSetSpec(name="vessels", path=path))


def test_a_missing_file_says_which_set_it_belongs_to(tmp_path):
    with pytest.raises(FileNotFoundError, match="vessels"):
        _load(ObjectSetSpec(name="vessels", path=tmp_path / "absent.tif"))


def test_the_refusal_says_object_set_rather_than_channel(tmp_path):
    path = _write(tmp_path / "plain.tif", _ramp())
    with pytest.raises(ValueError, match="object set 'vessels'"):
        _load(ObjectSetSpec(name="vessels", path=path, crop_origin=(SOURCE_HEIGHT - 2, 0)))


# -------------------------------------------------------------- the declaration

def _resolve(value):
    return None if value in (None, "") else value


@pytest.mark.parametrize("name", ["Vessels", "blood vessels", "2nd", "", "raw"])
def test_a_set_name_that_cannot_be_a_column_value_is_refused(name):
    with pytest.raises(ValueError):
        ObjectSetSpec.from_dict({"name": name, "path": "x.tif"}, _resolve)


def test_two_object_sets_with_one_name_are_refused():
    entries = [{"name": "vessels", "path": "a.tif"}, {"name": "vessels", "path": "b.tif"}]
    with pytest.raises(ValueError, match="declared twice"):
        _named_list(entries, "object set", "stem",
                    lambda entry: ObjectSetSpec.from_dict(entry, _resolve))


def test_the_defaults_are_no_adjustment():
    spec = ObjectSetSpec.from_dict({"name": "vessels", "path": "x.tif"}, _resolve)
    assert spec.static is False
    assert spec.channel_index is None
    assert spec.frame_offset is None
    assert spec.crop_origin == (0, 0)
    assert spec.shifts is None


def test_a_static_declaration_round_trips_through_the_manifest():
    """A static set never writes back the time settings it refuses."""
    spec = ObjectSetSpec.from_dict({"name": "vessels", "path": "x.tif", "static": True,
                                    "channel_index": 1}, _resolve)
    block = spec.as_dict()
    assert "frame_offset" not in block and "shifts" not in block
    assert ObjectSetSpec.from_dict(block, _resolve) == spec
