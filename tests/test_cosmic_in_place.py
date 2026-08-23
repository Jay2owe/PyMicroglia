"""One method, two doors: cleaning a file and cleaning an array agree.

The single-cell dLuc pipeline has no file at the point it removes cosmic rays —
it has a registered, windowed channel in a memory-mapped array — so it enters
the same method through ``clean_stack_in_place``. Two doors into one method is
a fine thing right up until they stop agreeing, and the whole of that risk is
in this file.

The first test is the load-bearing one: clean a stack as a file, clean the same
pixels as an array, and require every value to match. If it ever fails, the two
have become two methods and the shared ``METHOD_VERSION`` is a false claim
about every dLuc run.

The second is the flaw the in-place form replaces. Repairing an array where it
sits means the frame just rewritten is one of the next frame's two reference
frames, so a naive walk uses repaired values to judge originals and its answer
depends on the direction it walked in. Cleaning the stack backwards must give
the reversed result of cleaning it forwards, exactly.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import tifffile

from pymicroglia import cosmic

#: Frames, and small ones. Under 80 frames the noise sampler takes every frame,
#: so the two doors sample identically and any difference is the method's.
FRAMES, SIDE = 12, 40
#: Well clear of 12 noise units above the neighbours, and nowhere near the top
#: of a 16-bit camera's range, so nothing here is censored as saturated.
SPIKE = 2000


@pytest.fixture
def store_root(tmp_path, monkeypatch):
    monkeypatch.setenv("PYMICROGLIA_STORE", str(tmp_path / "cache"))
    monkeypatch.delenv("PYMICROGLIA_DECISIONS", raising=False)
    return tmp_path / "cache"


def _spiked(tmp_path, *, seed: int = 7):
    """A two-channel stack with four single-frame spikes on the signal channel."""
    rng = np.random.default_rng(seed)
    data = np.zeros((FRAMES, 2, SIDE, SIDE), np.uint16)
    for frame in range(FRAMES):
        data[frame, 0] = np.clip(1500 + rng.normal(0, 18, (SIDE, SIDE)), 0, 65535)
        data[frame, 1] = np.clip(2000 + rng.normal(0, 20, (SIDE, SIDE)), 0, 65535)
    for frame, y, x in ((2, 11, 9), (5, 30, 22), (5, 7, 33), (9, 18, 18)):
        data[frame, 1, y, x] = data[frame, 1, y, x] + SPIKE

    source = Path(tmp_path) / "in_place.ome.tif"
    tifffile.imwrite(source, data, imagej=False, photometric="minisblack",
                     metadata={"axes": "TCYX"}, ome=True)
    return source, data


def _in_place(data, source, **overrides):
    """The signal channel as a float array, cleaned where it sits."""
    array = np.array(data[:, 1], np.float32)
    options = dict(source=str(source), measured_dtype=np.uint16)
    options.update(overrides)
    summary = cosmic.clean_stack_in_place(array, **options)
    return array, summary


# ------------------------------------------------------- the two doors agree
def test_cleaning_an_array_gives_what_cleaning_the_file_gives(tmp_path,
                                                              store_root):
    """Every pixel, not a tolerance — and the counted numbers with them.

    The file path rounds to the camera's integers on the way out and the array
    path keeps its floats, so the comparison rounds the array the same way. That
    is the only difference the two are allowed.
    """
    source, data = _spiked(tmp_path)

    written = cosmic.remove_cosmic_rays(
        source, output_dir=tmp_path / "out", signal_channel=2,
        write_preview=False)
    from_file = tifffile.imread(written.path)[:, 1]

    array, summary = _in_place(data, source)

    np.testing.assert_array_equal(np.rint(array).astype(np.uint16), from_file)
    for key in ("pixel_frames_replaced", "head_pixel_frames",
                "track_pixel_frames", "censored_pixel_frames",
                "connected_events", "temporal_sigma_counts",
                "percent_of_selected_channel", "method_version"):
        assert summary[key] == written.summary[key], key


def test_interleave_replacement_agrees_through_both_doors(tmp_path, store_root):
    """The selectable adjacent-frame checkerboard is one method in both paths."""
    source, data = _spiked(tmp_path)

    written = cosmic.remove_cosmic_rays(
        source, output_dir=tmp_path / "out", signal_channel=2,
        replacement="interleave", write_preview=False)
    from_file = tifffile.imread(written.path)[:, 1]

    array, summary = _in_place(data, source, replacement="interleave")

    np.testing.assert_array_equal(np.rint(array).astype(np.uint16), from_file)
    assert summary["replacement"] == written.summary["replacement"] == "interleave"


def test_the_two_doors_carry_the_same_method_version():
    """A shared version string is a claim that the results are interchangeable.

    The test above is what makes it true; this is what makes it visible, so a
    change to one door that forks the version fails here rather than in six
    months when two runs turn out not to be comparable.
    """
    assert cosmic.stack.METHOD_VERSION == cosmic.METHOD_VERSION


# ------------------------------------------- a repaired frame is not a reference
def test_cleaning_backwards_gives_the_reversed_result(tmp_path, store_root):
    """The flaw the in-place form was written to avoid.

    Reversing a stack maps each frame's two neighbours onto the same two
    frames, reflected ends included, so the answer must be the reverse of the
    forward answer — value for value. It only is if every reference is built
    from measured pixels. A walk that let the frame it had just repaired stand
    in as a reference would drift in the direction it walked.
    """
    source, data = _spiked(tmp_path)

    forwards, _ = _in_place(data, source)
    backwards, _ = _in_place(data[::-1], source)

    np.testing.assert_array_equal(backwards[::-1], forwards)


def test_a_repaired_pixel_takes_the_mean_of_its_original_neighbours(tmp_path,
                                                                    store_root):
    """The one value that would move if a repaired frame leaked into a reference.

    Two spikes were planted in frame 5. The pixel repaired in frame 5 must take
    the mean of frames 4 and 6 **as they were measured**, which is a number this
    test works out from the input rather than from the run.
    """
    source, data = _spiked(tmp_path)
    array, _ = _in_place(data, source)

    y, x = 30, 22
    expected = (float(data[4, 1, y, x]) + float(data[6, 1, y, x])) / 2.0
    assert array[5, y, x] == pytest.approx(expected, abs=1e-4)


# --------------------------------------------------- full scale is the camera's
def test_a_float_arrays_brightest_pixel_is_not_saturation(tmp_path, store_root):
    """The trap the ``measured_dtype`` argument exists for.

    A registered stack is float32 because it has been shifted, and a float array
    has no full scale of its own — so the rule reads its brightest pixel as the
    top of the range and censors it. Told what the camera measured in, it
    censors nothing, which is the truth about a stack whose brightest pixel is
    4000 counts out of 65535.
    """
    source, data = _spiked(tmp_path)

    _, told = _in_place(data, source)
    _, guessing = _in_place(data, source, measured_dtype=None)

    assert told["censored_pixel_frames"] == 0
    assert guessing["censored_pixel_frames"] > 0
    assert told["full_scale_counts"] == 65535.0


# ------------------------------------------------------------- what it refuses
def test_the_control_is_not_offered_here():
    """``mirror_placebo`` writes numbers and no pixels.

    Offered on this door it would hand a pipeline an uncleaned stack while
    reporting a successful cosmic-ray removal, so it is not offered. The control
    belongs to the file entry point, where producing nothing is visible.
    """
    parameters = inspect.signature(cosmic.clean_stack_in_place).parameters
    assert "mirror_placebo" not in parameters
    assert "mirror_placebo" in inspect.signature(cosmic.remove_cosmic_rays).parameters


def test_two_frames_are_refused(tmp_path, store_root):
    """An outlier is defined against the frames either side; two has no inside."""
    array = np.ones((2, 8, 8), np.float32)
    with pytest.raises(ValueError, match="three frames"):
        cosmic.clean_stack_in_place(array)


def test_a_display_only_source_is_refused(tmp_path, store_root):
    """The same guard the file door calls, on the name it was handed."""
    from pymicroglia import guards

    array = np.ones((5, 8, 8), np.float32)
    with pytest.raises(guards.DisplayOnlyInput):
        cosmic.clean_stack_in_place(array, source="registered_DISPLAY_ONLY.tif")


def test_the_summary_says_the_array_was_modified(tmp_path, store_root):
    """A summary naming an output file, from a run that wrote none, is a lie."""
    source, data = _spiked(tmp_path)
    _, summary = _in_place(data, source)

    assert summary["source_modified"] is True
    assert not summary["output"].endswith(".tif")
    assert summary["pixel_frames_replaced"] > 0
