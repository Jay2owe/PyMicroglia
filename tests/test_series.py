"""Opening a time-lapse without loading it, and registering it without one.

The two claims this stage makes. Opening reads the page directory and nothing
else, so asking a ten-gigabyte stack how many frames it has is a reasonable
thing to do. And a registered frame comes from the source plus a 298 KB table
of shifts, so the 21 GB registered copy is an optimisation rather than a
prerequisite.
"""

from __future__ import annotations

import numpy as np
import pytest
import tifffile
from scipy import ndimage

import fixtures
from pymicroglia import open_series, store
from pymicroglia.series import Series


@pytest.fixture
def store_root(tmp_path, monkeypatch):
    monkeypatch.setenv("PYMICROGLIA_STORE", str(tmp_path / "cache"))
    monkeypatch.delenv("PYMICROGLIA_DECISIONS", raising=False)
    return tmp_path / "cache"


@pytest.fixture
def stack(tmp_path):
    path = tmp_path / "raw" / "VID52_C1_phase-green-red_timestack.tif"
    data = fixtures.write_series(path, frames=8, channels=3, height=32, width=24)
    return path, data


def store_registration(source, shifts, folder, *, params=None,
                       method_version="2026-08-19-test", **kwargs):
    return store.put("registration", source, params or {"reference_channel": 0},
                     kind="table", value=fixtures.shift_table(shifts),
                     name="registration_shifts_and_qc", output_dir=folder,
                     method_version=method_version, **kwargs)


# ------------------------------------------------------------------ opening
def test_opening_reports_the_shape_without_decoding_anything(stack):
    path, data = stack
    series = open_series(path)

    assert series.shape == data.shape
    assert series.dtype == np.uint16
    assert series.meta.channels == 3
    series.close()


def test_opening_does_not_fingerprint_the_file(stack, store_root):
    """A fingerprint is a 24 MB read. Opening a series has to stay cheap
    enough to do casually, so identity is computed on first use."""
    path, _ = stack
    series = open_series(path)

    assert series._source is None
    assert series.source.size > 0
    assert series._source is not None


def test_opening_something_that_is_not_there_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_series(tmp_path / "absent.tif")


def test_a_series_can_be_opened_on_a_path_past_the_windows_limit(tmp_path):
    segment = "a_folder_named_at_length_to_exceed_the_windows_limit"
    folder = tmp_path
    for index in range(6):
        folder = folder / f"{segment}_{index}"
    path = folder / "timestack.tif"
    assert len(str(path)) > 260

    data = fixtures.write_series(path, frames=3, channels=2, height=8, width=6)
    with open_series(path) as series:
        assert series.shape == data.shape
        assert np.array_equal(series.frame(1, 1), data[1, 1])


# ------------------------------------------------------------------- frames
@pytest.mark.parametrize("index", [0, 4, 7])
def test_a_frame_matches_reading_the_whole_file_and_slicing_it(stack, index):
    path, data = stack
    whole = tifffile.imread(str(path)).reshape(data.shape)
    with open_series(path) as series:
        for channel in range(data.shape[1]):
            assert np.array_equal(series.frame(index, channel),
                                  whole[index, channel])


def test_a_frame_outside_the_series_is_an_index_error(stack):
    path, data = stack
    with open_series(path) as series:
        with pytest.raises(IndexError):
            series.frame(data.shape[0], 0)
        with pytest.raises(IndexError):
            series.frame(0, data.shape[1])


def test_a_window_hands_out_one_frame_at_a_time(stack):
    path, data = stack
    with open_series(path) as series:
        frames = list(series.window(2, 5, 1))

    assert len(frames) == 3
    assert np.array_equal(frames[0], data[2, 1])
    assert np.array_equal(frames[-1], data[4, 1])


def test_the_page_map_is_read_from_the_file_when_it_states_one(tmp_path):
    """Converted files do not all interleave channels the same way, so the
    explicit OME plane map is used when there is one."""
    path = tmp_path / "explicit.tif"
    data = fixtures.write_series(path, frames=4, channels=3, height=8, width=6,
                                 explicit_tiffdata=True)
    with open_series(path) as series:
        assert "page order" not in " ".join(series.meta.notes)
        assert np.array_equal(series.frame(3, 2), data[3, 2])


def test_falling_back_to_page_order_is_recorded_as_a_note(stack):
    path, _ = stack
    with open_series(path) as series:
        assert any("page order" in note for note in series.meta.notes)


# -------------------------------------------------------------------- crops
def test_a_crop_matches_the_full_frame_sliced_by_the_same_box(stack):
    path, data = stack
    with open_series(path) as series:
        view = series.crop(4, 3, 10, 9)

        assert view.shape == (data.shape[0], data.shape[1], 9, 10)
        for t, c in ((0, 0), (5, 2)):
            assert np.array_equal(view.frame(t, c),
                                  series.frame(t, c)[3:12, 4:14])


def test_a_crop_of_a_crop_composes(stack):
    path, _ = stack
    with open_series(path) as series:
        once = series.crop(4, 3, 12, 10)
        twice = once.crop(2, 1, 5, 4)

        assert np.array_equal(twice.frame(0, 0),
                              series.frame(0, 0)[4:8, 6:11])


def test_a_crop_outside_the_frame_is_refused(stack):
    path, _ = stack
    with open_series(path) as series:
        with pytest.raises(ValueError):
            series.crop(0, 0, 999, 999)


def test_a_crop_shares_the_open_handle(stack):
    path, _ = stack
    with open_series(path) as series:
        assert series.crop(1, 1, 4, 4)._reader is series._reader


# ------------------------------------------------------------- registration
def test_a_registered_frame_applies_the_stored_shift(stack, store_root,
                                                     tmp_path):
    path, data = stack
    shifts = np.array([[0.0, 0.0], [-1.5, 0.5], [-2.25, 1.0], [1.0, -1.0],
                       [0.0, 2.0], [3.0, 0.0], [-1.0, -2.0], [0.5, 0.5]])
    with open_series(path) as series:
        store_registration(series.source, shifts, tmp_path / "exports")
        got = series.registered(2, 1)

        expected = ndimage.shift(data[2, 1].astype(np.float32),
                                 shift=(-2.25, 1.0), order=1, mode="constant",
                                 cval=0.0, prefilter=False)
    assert np.allclose(got, expected)


def test_a_one_based_shift_table_is_not_read_off_by_one(stack, store_root,
                                                        tmp_path):
    """The engines number frames from one in their CSV and index arrays from
    zero. Getting this wrong shifts a whole recording by one frame and nothing
    downstream notices."""
    path, data = stack
    shifts = np.array([[float(i), 0.0] for i in range(8)])
    with open_series(path) as series:
        store_registration(series.source, shifts, tmp_path / "exports")
        table, _ = series.registration()

    assert table[0, 0] == 0.0 and table[7, 0] == 7.0


def test_two_stored_registrations_raise_rather_than_choosing(stack, store_root,
                                                             tmp_path):
    path, _ = stack
    shifts = np.zeros((8, 2))
    with open_series(path) as series:
        store_registration(series.source, shifts, tmp_path / "run_a",
                           params={"reference_channel": 0})
        store_registration(series.source, shifts, tmp_path / "run_b",
                           params={"reference_channel": 2})

        with pytest.raises(store.AmbiguousArtefact) as raised:
            series.registered(0, 1)

    message = str(raised.value)
    assert "run_a" in message and "run_b" in message
    assert "reference_channel" in message


def test_no_stored_registration_raises_and_explains(stack, store_root):
    path, _ = stack
    with open_series(path) as series:
        with pytest.raises(store.ArtefactMissing):
            series.registered(0, 1)


def test_a_shift_table_of_the_wrong_length_is_refused(stack, store_root,
                                                      tmp_path):
    path, _ = stack
    with open_series(path) as series:
        store_registration(series.source, np.zeros((5, 2)), tmp_path / "exports")
        with pytest.raises(ValueError) as raised:
            series.registered(0, 1)
    assert "5 rows" in str(raised.value)


def test_a_table_with_no_shift_columns_says_what_it_looked_for(stack,
                                                               store_root,
                                                               tmp_path):
    path, _ = stack
    with open_series(path) as series:
        store.put("registration", series.source, {}, kind="table",
                  value={"frame": list(range(1, 9)), "quality": [1.0] * 8},
                  name="registration_shifts_and_qc",
                  output_dir=tmp_path / "exports", method_version="v1")
        with pytest.raises(KeyError) as raised:
            series.registered(0, 1)
    assert "shift_y_px" in str(raised.value)


def test_an_explicitly_named_registration_wins(stack, store_root, tmp_path):
    path, _ = stack
    with open_series(path) as series:
        store_registration(series.source, np.zeros((8, 2)), tmp_path / "run_a",
                           params={"reference_channel": 0})
        chosen = store_registration(series.source, np.full((8, 2), 2.0),
                                    tmp_path / "run_b",
                                    params={"reference_channel": 2})

        table, _ = series.registration(explicit=chosen.path)
    assert float(table[0, 0]) == 2.0


def test_the_registration_is_resolved_once_not_once_per_frame(stack, store_root,
                                                              tmp_path,
                                                              monkeypatch):
    path, _ = stack
    with open_series(path) as series:
        store_registration(series.source, np.zeros((8, 2)), tmp_path / "exports")
        series.registered(0, 1)

        def refuse(*args, **kwargs):
            raise AssertionError("resolve was called again")

        monkeypatch.setattr(store, "resolve", refuse)
        series.registered(1, 1)


def test_a_registration_that_records_a_crop_applies_it(stack, store_root,
                                                       tmp_path):
    path, data = stack
    with open_series(path) as series:
        store_registration(series.source, np.zeros((8, 2)), tmp_path / "exports",
                           params={"reference_channel": 0,
                                   "crop_xyxy": [2, 3, 20, 28]})
        got = series.registered(0, 1)

    assert got.shape == (25, 18)


# ------------------------------------------------------------------- naming
def test_a_display_only_file_says_so(tmp_path):
    path = tmp_path / "stack_DISPLAY_ONLY.tif"
    fixtures.write_series(path, frames=2, channels=2, height=8, width=6)
    with open_series(path) as series:
        assert series.display_only is True


def test_an_ordinary_file_does_not(stack):
    path, _ = stack
    with open_series(path) as series:
        assert series.display_only is False


def test_repr_names_the_file_and_its_shape(stack):
    path, _ = stack
    with open_series(path) as series:
        text = repr(series)
    assert "timestack.tif" in text and "T=8" in text


def test_describe_is_json_shaped(stack):
    path, _ = stack
    with open_series(path) as series:
        described = series.describe()

    assert described["shape"] == [8, 3, 32, 24]
    assert described["channels"] == 3
    assert described["display_only"] is False


def test_a_series_is_a_series(stack):
    path, _ = stack
    with open_series(path) as series:
        assert isinstance(series, Series)
