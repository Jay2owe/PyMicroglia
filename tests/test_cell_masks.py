"""The learned mask as a pipeline step: off by default, and a branch when on.

``test_learned_mask.py`` covers what the mask *is*. This covers what happens
when a run asks for one, and every claim here is one that would cost somebody a
day if it quietly stopped being true:

**Off unless asked for.** A default run writes no mask and logs no stage. The
step needs torch, needs weights that are an experimental result rather than
something shipped, and carries a limit worth consenting to, so a run that did
not ask for it must not get it by accident.

**On does not fork the run.** The mask changes no measured number, so two runs
that differ only in whether they asked for it are the same analysis and belong
in the same folder. Putting ``learned_mask`` into the settings hash would split
them, and the split would not be noticed until somebody wondered why their
traces had been computed twice.

**It runs after the measurement, never before.** Enforced at run time by
``check_stage_order``, because a comment saying so cannot stop an edit.

**What it was fed is what the weights were trained on.** The window is a
duration of light, not a count of files, and the rolling mean that assembles it
is the accepted detector's function frame for frame -- the weights were trained
on stacks that function produced.

**A refusal says what to install or where to point.** The two ways this step
cannot run are both somebody's five-second fix, and only if the sentence names
it.

The network itself is never run here: ``apply`` is replaced with a model that
returns a shape. That is deliberate -- these are claims about the wiring, and a
test that needed 40 MB of weights to check that a default is ``False`` would be
skipped on every machine that matters.
"""

from __future__ import annotations
from pymicroglia._results import read_document

import json
from pathlib import Path

import numpy as np
import pytest

from pymicroglia import pipelines
from pymicroglia.learned_mask import scale as scaling
from pymicroglia.pipelines import cell_masks

SMALL = {"t0": None, "t1": None, "baselines": (6.0,), "detrends": ("cubic",),
         "ndecoy": 40, "skip_videos": True, "skip_control": True}


# ── a network that is not a network ─────────────────────────────────────────
class FakeModel:
    """Enough of a model for the wiring: a pixel range and nothing else."""

    pixel_range = (scaling.NATIVE_UM_PER_PX, scaling.NATIVE_UM_PER_PX)


@pytest.fixture
def stub_network(monkeypatch, tmp_path):
    """``apply`` with the four torch-shaped calls replaced.

    Returns the list the stub appends each call's pictures to, so a test can
    assert what the network was shown rather than only what was written.
    """
    from pymicroglia.learned_mask import apply as masking

    shown: list[np.ndarray] = []
    fake_weights = tmp_path / "model.pt"
    fake_weights.write_bytes(b"not really weights")

    def mask_pictures(model, images, *, um_per_px=None, cut=masking.ACCEPTED_CUT,
                      grow_to=None, min_area=None, max_area=0, source="recorded"):
        images = np.asarray(images, np.float32)
        shown.append(images.copy())
        cells = np.zeros(images.shape, np.uint16)
        cells[:, 2:6, 2:6] = 1                       # one region, every frame
        return {"probability": (images / max(images.max(), 1.0)).astype(np.float32),
                "cells": cells, "mask": cells > 0,
                "warning": masking.scale_warning(model, um_per_px, source),
                "settings": {"cut": cut, "grow_to": 0.6, "um_per_px": um_per_px,
                             "min_area": 32, "max_area": max_area}}

    monkeypatch.setattr(masking, "weights", lambda: fake_weights)
    monkeypatch.setattr(masking, "load_model", lambda path: FakeModel())
    monkeypatch.setattr(masking, "hurry", lambda model, threads=0: model)
    monkeypatch.setattr(masking, "mask_pictures", mask_pictures)
    return shown


class FakePrepared:
    """The three attributes ``mask_run`` reads off a prepared run."""

    def __init__(self, dluc, *, frame_interval_h=0.5548, um_per_px=2.0,
                 path=r"X:\Wells\MCG 04 - 1 - 595.tif"):
        self.dluc = dluc
        self.frame_interval_h = frame_interval_h
        self.um_per_px = um_per_px
        self.source = _SourceLike(path)


class _SourceLike:
    """A store identity: a path on an attribute, and a ``str`` that is a repr."""

    def __init__(self, path):
        self.path = path

    def __repr__(self):
        return f"SourceId(size=9, sample='ab', path={self.path!r}, mtime_ns=0)"


# ── the window is a duration, not a count of files ──────────────────────────
def test_the_window_is_the_trained_duration_on_this_recording():
    """Seven frames natively, and half as many when the camera ran half as often.

    ``WINDOW_HOURS`` is what the weights saw. A recording that samples every two
    hours reaches that duration in fewer frames, and using seven of *its* frames
    would hand the network a 14 h exposure it has never seen.
    """
    assert cell_masks.frames_for(scaling.WINDOW_HOURS / 7) == 7
    assert cell_masks.frames_for(0.5548) == 7
    assert cell_masks.frames_for(0.25) == 15
    assert cell_masks.frames_for(2.0) == 1


def test_the_window_is_always_odd_so_the_record_is_not_a_lie():
    """A centred mean takes the same number of frames each side of the middle.

    So the counts it can honour are 1, 3, 5, ...: asking for four would silently
    average five, and a manifest saying four would then be wrong about the one
    thing it was written down to record.
    """
    one_frame = np.zeros((81, 1, 1), np.float32)
    one_frame[40] = 1.0
    for interval in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9, 1.3, 1.9, 3.0):
        window = cell_masks.frames_for(interval)
        assert window % 2 == 1, interval
        # One frame of light, spread over exactly the frames that averaged it:
        # counting them is how many frames the mean really spanned, whatever
        # number was asked for.
        spread = cell_masks.rolling_mean(one_frame, window)
        assert int(np.count_nonzero(spread)) == window, interval


def test_a_recording_that_never_said_how_often_it_sampled_gets_one_frame():
    """No interval means no way to turn hours into frames. One frame, and said so."""
    assert cell_masks.frames_for(0.0) == 1
    assert cell_masks.frames_for(-1.0) == 1
    assert cell_masks.frames_for(None) == 1


# ── the rolling mean is the accepted one ────────────────────────────────────
def _accepted_rolling_mean(cube, window):
    """``MCG_masking_tuning/code/common.rolling_mean``, transcribed.

    Transcribed rather than imported: that folder is the experiment's, not this
    package's, and it is not on a test machine's path. If the two ever disagree
    this test is the place the disagreement shows up.
    """
    values = np.asarray(cube, np.float32)
    frames = values.shape[0]
    cumulative = np.concatenate(
        [np.zeros((1,) + values.shape[1:], np.float64), np.cumsum(values, 0)])
    half = window // 2
    out = np.empty_like(values)
    for index in range(frames):
        lo, hi = max(0, index - half), min(frames, index + half + 1)
        out[index] = (cumulative[hi] - cumulative[lo]) / (hi - lo)
    return out


def test_the_mean_is_the_accepted_detectors_own_function():
    """The weights were trained on stacks that function wrote.

    Anything this does differently is a picture the network was never shown, so
    the bar is equality and not similarity.
    """
    rng = np.random.default_rng(11)
    stack = rng.normal(40.0, 6.0, (23, 9, 11)).astype(np.float32)
    for window in (3, 7, 15):
        assert np.array_equal(cell_masks.rolling_mean(stack, window),
                              _accepted_rolling_mean(stack, window))


def test_the_ends_are_short_means_and_not_repeated_frames():
    """Two wrong ways to end a rolling mean, and this is neither of them.

    Padding repeats real frames. Sliding the window inwards to keep it full
    length makes the first few frames identical to one another, which in a mask
    going on to identity tracking reads as a cell that did not move.
    """
    ramp = np.arange(9, dtype=np.float32).reshape(9, 1, 1)
    out = cell_masks.rolling_mean(ramp, 5)

    middle = out[:, 0, 0]
    assert middle[0] == pytest.approx(1.0)      # mean of 0,1,2
    assert middle[1] == pytest.approx(1.5)      # mean of 0,1,2,3
    assert middle[2] == pytest.approx(2.0)      # the first full window
    assert middle[0] != middle[1], "the first frames must not repeat"
    assert middle[-1] == pytest.approx(7.0)


def test_the_mean_is_over_time_and_never_over_space():
    """The one filter that must not be in here.

    The accepted display look applies a 1.6 px Gaussian, and it suppresses
    exactly the pixel noise the network's scaling divides by. A spatial filter
    creeping in here would arrive at the model as a wrong scale rather than as a
    visible mistake.
    """
    stack = np.zeros((5, 7, 7), np.float32)
    stack[:, 3, 3] = 100.0
    out = cell_masks.rolling_mean(stack, 3)
    assert float(out[2, 3, 3]) == pytest.approx(100.0)
    assert float(out[2, 3, 4]) == 0.0
    assert float(out[2, 2, 3]) == 0.0


def test_a_window_of_one_returns_the_frames_unchanged():
    stack = np.arange(24, dtype=np.float32).reshape(4, 3, 2)
    out = cell_masks.rolling_mean(stack, 1)
    assert np.array_equal(out, stack)
    assert out is not stack


# ── what mask_run writes, and what it says about it ─────────────────────────
def test_the_three_stacks_are_named_for_the_recording_not_its_identity(
        stub_network, tmp_path):
    """``prepared.source`` is a store identity whose ``str`` is a dataclass repr.

    Naming a file after that string would read as nonsense, and on Windows the
    colon in it would be refused outright — so the recording is named from the
    path the identity carries.
    """
    prepared = FakePrepared(np.full((4, 12, 12), 5.0, np.float32))
    out = cell_masks.mask_run(prepared, tmp_path / "run")

    assert set(out["outputs"]) == {"mask_probability", "mask_cells", "mask"}
    for path in out["outputs"].values():
        assert Path(path).is_file()
        assert Path(path).name.startswith("MCG 04 - 1 - 595_learned_")
        assert "SourceId" not in Path(path).name


def test_the_stacks_carry_the_types_the_next_project_reads(stub_network, tmp_path):
    """A probability is a float, a label image is an integer, a mask is a byte.

    The Motion project reads these back to track identities through, and a label
    image written as float32 is one it cannot use as labels.
    """
    import tifffile

    prepared = FakePrepared(np.full((3, 12, 12), 5.0, np.float32))
    out = cell_masks.mask_run(prepared, tmp_path / "run")

    assert tifffile.imread(out["outputs"]["mask_probability"]).dtype == np.float32
    assert tifffile.imread(out["outputs"]["mask_cells"]).dtype == np.uint16
    assert tifffile.imread(out["outputs"]["mask"]).dtype == np.uint8


def test_the_model_is_shown_the_mean_and_not_the_raw_frames(stub_network, tmp_path):
    """What is fed is the assembled exposure, at the window that was recorded."""
    rng = np.random.default_rng(5)
    frames = rng.normal(20.0, 4.0, (9, 12, 12)).astype(np.float32)
    prepared = FakePrepared(frames, frame_interval_h=0.5548)
    out = cell_masks.mask_run(prepared, tmp_path / "run")

    assert out["settings"]["window_frames"] == 7
    assert len(stub_network) == 1
    assert np.array_equal(stub_network[0],
                          _accepted_rolling_mean(frames, 7))


def test_the_settings_say_what_the_window_was_read_from(stub_network, tmp_path):
    """A window with no interval beside it cannot be checked by the next reader."""
    prepared = FakePrepared(np.full((5, 12, 12), 5.0, np.float32),
                            frame_interval_h=0.25)
    settings = cell_masks.mask_run(prepared, tmp_path / "run")["settings"]

    assert settings["frame_interval_h"] == pytest.approx(0.25)
    assert settings["window_frames"] == 15
    assert settings["window_hours"] == pytest.approx(scaling.WINDOW_HOURS)
    assert settings["cells_found"] == 1
    assert settings["pixel_range"] == [2.0, 2.0]


def test_a_matched_recording_says_nothing_and_opens_no_question(
        stub_network, tmp_path):
    from pymicroglia import review

    notes = review.Review()
    out = cell_masks.mask_run(FakePrepared(np.full((3, 12, 12), 5.0, np.float32)),
                              tmp_path / "run", notes)

    assert out["warning"] is None
    assert notes.counts().get("check", 0) == 0
    assert notes.open_questions() == []


def test_a_coarser_recording_is_written_anyway_and_the_review_says_why(
        stub_network, tmp_path):
    """Silence is the failure mode, so the mask is written and the warning stands.

    Refusing to write it would send somebody off to mask the recording another
    way; writing it without a word would let them count cells from it.
    """
    from pymicroglia import review

    notes = review.Review()
    prepared = FakePrepared(np.full((3, 12, 12), 5.0, np.float32), um_per_px=4.0)
    out = cell_masks.mask_run(prepared, tmp_path / "run", notes)

    assert Path(out["outputs"]["mask_cells"]).is_file()
    assert "MISSING CELLS" in out["warning"]
    question = notes.open_questions()
    assert len(question) == 1 and question[0].gate == "learned_mask"
    assert question[0].changes_result is True
    assert "Resample" in question[0].remedy


def test_an_unrecorded_frame_interval_is_a_warning_of_its_own(
        stub_network, tmp_path):
    """One exposure is not the exposure the weights learned on, and says so.

    The pixel size matches here, so nothing else would have spoken up; a frame
    masked on its own is about two and a half times noisier than the 3.9 h mean
    the network was trained through, and the mask is quietly worse for it.
    """
    from pymicroglia import review

    notes = review.Review()
    prepared = FakePrepared(np.full((3, 12, 12), 5.0, np.float32),
                            frame_interval_h=0.0)
    out = cell_masks.mask_run(prepared, tmp_path / "run", notes)

    assert out["settings"]["window_frames"] == 1
    assert "frame interval not recorded" in out["warning"]
    assert len(notes.open_questions()) == 1


# ── the stage order ─────────────────────────────────────────────────────────
def test_the_learned_mask_is_a_branch_off_the_measurement():
    assert pipelines.CELL_MASK_BRANCH in pipelines.BRANCHES
    pipelines.check_stage_order(["register", "cosmic_rays", "measure",
                                 "display", "cell_masks"])


def test_masking_before_measuring_is_refused():
    """The reason display smoothing is refused, plus one more.

    The mask answers a different question from the measurement — what is a cell
    in *this* picture, with no time axis — so a number computed after it is a
    number computed from something nobody measured.
    """
    with pytest.raises(pipelines.StageOrderError) as caught:
        pipelines.check_stage_order(["register", "cell_masks", "measure"])
    assert "learned cell mask" in str(caught.value)


def test_the_pipeline_declares_the_stage_it_can_run():
    from pymicroglia.pipelines import dluc_single_cell

    assert "cell_masks" in dluc_single_cell.STAGES
    pipelines.check_stage_order(dluc_single_cell.STAGES)


# ── the two refusals ────────────────────────────────────────────────────────
def test_asking_for_a_mask_without_torch_names_the_extra(monkeypatch):
    """This only runs when somebody asked for it, so it must not skip quietly."""
    from pymicroglia.pipelines import cell_masks as module
    from pymicroglia.pipelines import dluc_single_cell

    def no_torch(*args, **kwargs):
        raise ImportError("No module named 'torch'")

    monkeypatch.setattr(module, "mask_run", no_torch)
    with pytest.raises(ImportError) as caught:
        dluc_single_cell._cell_masks(None, Path("."), None)
    assert 'pip install "PyMicroglia[mask]"' in str(caught.value)


def test_asking_for_a_mask_without_weights_names_the_variable(monkeypatch):
    """Weights are an experimental result, so nothing is shipped to fall back to."""
    from pymicroglia.pipelines import cell_masks as module
    from pymicroglia.pipelines import dluc_single_cell

    def no_weights(*args, **kwargs):
        raise FileNotFoundError("no weights given")

    monkeypatch.setattr(module, "mask_run", no_weights)
    with pytest.raises(FileNotFoundError) as caught:
        dluc_single_cell._cell_masks(None, Path("."), None)
    assert "PYMICROGLIA_MASK_WEIGHTS" in str(caught.value)


# ── a real run, with and without ────────────────────────────────────────────
@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("PYMICROGLIA_STORE", str(tmp_path / "cache"))
    monkeypatch.setenv("PYMICROGLIA_DECISIONS", str(tmp_path / "decisions"))
    return tmp_path


def _source(folder):
    """The same synthetic recording the run tests use, and for the same reason.

    A whole run has to get through it: a stack with no tissue mask stops at the
    background and never reaches the branch this file is about.
    """
    from tests_support import oscillating_stack

    return oscillating_stack(folder)


def test_a_default_run_writes_no_mask_and_logs_no_stage(isolated):
    """Opt-in, asserted where it matters: on a run that did not ask."""
    from pymicroglia.pipelines import dluc_single_cell

    manifest = dluc_single_cell.run(_source(isolated),
                                    output_dir=isolated / "out",
                                    if_exists="error", **SMALL)

    assert "cell_masks" not in [entry["stage"] for entry in manifest["stages"]]
    assert "learned_mask" not in manifest["summary"]
    folder = Path(manifest["folder"])
    assert list(folder.glob("*_learned_*")) == []


def test_asking_for_the_mask_does_not_fork_the_run(isolated, stub_network):
    """The same analysis, so the same folder.

    The mask changes no number this run reports. If ``learned_mask`` reached the
    settings hash the two runs would land in different folders and every trace
    would be computed twice — noticed, if at all, months later.
    """
    from pymicroglia.pipelines import dluc_single_cell

    source = _source(isolated)
    first = dluc_single_cell.run(source, output_dir=isolated / "out",
                                 if_exists="error", **SMALL)
    # ``error`` refuses a folder that already exists, so this raising *is* the
    # assertion that the second run resolved to the same label as the first.
    with pytest.raises(FileExistsError) as caught:
        dluc_single_cell.run(source, output_dir=isolated / "out",
                             if_exists="error", learned_mask=True, **SMALL)
    assert Path(first["folder"]).name in str(caught.value)


def test_a_run_that_asks_for_it_gets_a_mask_after_the_measurement(
        isolated, stub_network):
    """The whole opt-in path: the stage, the files, and the caveat in the record."""
    from pymicroglia.pipelines import dluc_single_cell

    manifest = dluc_single_cell.run(_source(isolated),
                                    output_dir=isolated / "out",
                                    if_exists="error", learned_mask=True,
                                    **SMALL)

    order = [entry["stage"] for entry in manifest["stages"]]
    assert order.index("measure") < order.index("cell_masks")
    pipelines.check_stage_order(order)

    assert set(manifest["outputs"]) >= {"mask_probability", "mask_cells", "mask"}
    for key in ("mask_probability", "mask_cells", "mask"):
        assert Path(manifest["outputs"][key]).is_file()

    learned = manifest["summary"]["learned_mask"]
    assert learned["cells_found"] == 1
    assert learned["window_frames"] >= 1
    assert "warning" in learned

    stored = read_document(Path(manifest["folder"]) / "manifest.json")
    assert stored["summary"]["learned_mask"]["cut"] == learned["cut"]
    assert any(item["gate"] == "learned_mask" for item in stored["review"])


def test_the_run_records_that_it_was_asked_for(isolated, stub_network):
    """Not in the settings, but not lost either.

    A reproduction script that did not carry ``learned_mask`` would replay a run
    that wrote no mask, and the record would say it had.
    """
    from pymicroglia.pipelines import dluc_single_cell

    manifest = dluc_single_cell.run(_source(isolated),
                                    output_dir=isolated / "out",
                                    if_exists="error", learned_mask=True,
                                    **SMALL)
    records = sorted((Path(manifest["folder"]) / ".analysis-kit" / "records")
                     .glob("*.json"))
    if not records:                      # analysis_kit is optional
        pytest.skip("analysis_kit is not installed in this interpreter")
    stored = read_document(records[0])
    assert stored["params"]["learned_mask"] is True


# ── doing it only once ──────────────────────────────────────────────────────
def _count_calls(monkeypatch):
    """How many times the network was actually asked for a mask."""
    from pymicroglia.learned_mask import apply as masking

    calls = []
    real = masking.mask_pictures
    monkeypatch.setattr(masking, "mask_pictures",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    return calls


def test_a_second_run_over_the_same_pictures_does_not_mask_them_again(
        stub_network, tmp_path, monkeypatch):
    """The network pass is minutes per recording; the answer is already there.

    Re-running a folder is the ordinary case -- one recording failed, a later
    stage was added, somebody wants the traces again -- and paying for every
    mask each time is what makes people stop re-running.
    """
    prepared = FakePrepared(np.random.default_rng(0).random((8, 16, 16), np.float32))
    calls = _count_calls(monkeypatch)

    first = cell_masks.mask_run(prepared, tmp_path)
    assert first["reused"] is False
    again = cell_masks.mask_run(prepared, tmp_path)
    assert again["reused"] is True
    assert len(calls) == 1, "the network ran a second time"
    assert again["outputs"] == first["outputs"]
    assert again["settings"] == first["settings"]


def test_reuse_is_refused_when_the_pictures_are_not_the_same_pictures(
        stub_network, tmp_path, monkeypatch):
    """Keyed on what the network is shown, not on the file it came from.

    The same recording prepared differently -- another window, no cosmic-ray
    step -- is a different set of pictures. A key naming only the recording
    would hand back the previous mask for them, which is the quiet kind of
    wrong this whole module is written against.
    """
    rng = np.random.default_rng(0)
    first = FakePrepared(rng.random((8, 16, 16), np.float32))
    calls = _count_calls(monkeypatch)
    cell_masks.mask_run(first, tmp_path)

    # Same path, same settings, same shape -- different pixels.
    other = FakePrepared(rng.random((8, 16, 16), np.float32))
    assert cell_masks.mask_run(other, tmp_path)["reused"] is False
    assert len(calls) == 2


@pytest.mark.parametrize("changed", [{"cut": 0.5}, {"grow_to": 0.3},
                                     {"min_area": 8}, {"max_area": 4000}])
def test_changing_any_setting_that_moves_a_boundary_masks_again(
        stub_network, tmp_path, monkeypatch, changed):
    """Every setting in the recipe earns its place by being tested."""
    prepared = FakePrepared(np.random.default_rng(1).random((6, 16, 16), np.float32))
    calls = _count_calls(monkeypatch)
    cell_masks.mask_run(prepared, tmp_path)
    assert cell_masks.mask_run(prepared, tmp_path, **changed)["reused"] is False
    assert len(calls) == 2


def test_reuse_false_masks_again_even_when_everything_matches(
        stub_network, tmp_path, monkeypatch):
    """The way out, for a cache somebody has stopped trusting."""
    prepared = FakePrepared(np.random.default_rng(2).random((6, 16, 16), np.float32))
    calls = _count_calls(monkeypatch)
    cell_masks.mask_run(prepared, tmp_path)
    assert cell_masks.mask_run(prepared, tmp_path, reuse=False)["reused"] is False
    assert len(calls) == 2


def test_a_recipe_whose_files_have_gone_is_not_honoured(
        stub_network, tmp_path, monkeypatch):
    """The record can outlive what it names, and then it is not an answer."""
    prepared = FakePrepared(np.random.default_rng(3).random((6, 16, 16), np.float32))
    calls = _count_calls(monkeypatch)
    written = cell_masks.mask_run(prepared, tmp_path)
    Path(written["outputs"]["mask_cells"]).unlink()
    assert cell_masks.mask_run(prepared, tmp_path)["reused"] is False
    assert len(calls) == 2


def test_the_two_masks_never_stand_in_for_each_other(
        stub_network, tmp_path, monkeypatch):
    """Different pictures, different question, and separate recipes.

    The still mask is the mean of every frame; the run mask is a rolling
    window. Handing either back for the other would be a mask of something
    nobody asked about.
    """
    prepared = FakePrepared(np.random.default_rng(4).random((6, 16, 16), np.float32))
    calls = _count_calls(monkeypatch)
    cell_masks.mask_run(prepared, tmp_path)
    still = cell_masks.mask_still(prepared, tmp_path)
    assert still["reused"] is False
    assert len(calls) == 2
    assert cell_masks.mask_still(prepared, tmp_path)["reused"] is True
    assert len(calls) == 2


def test_the_reused_still_mask_returns_the_labels_and_not_an_empty_array(
        stub_network, tmp_path):
    """It is read back off disk, and the measurement runs on what comes back."""
    prepared = FakePrepared(np.random.default_rng(5).random((6, 16, 16), np.float32))
    first = cell_masks.mask_still(prepared, tmp_path)
    again = cell_masks.mask_still(prepared, tmp_path)
    assert again["reused"] is True
    assert np.array_equal(again["labels"], first["labels"])
    assert again["labels"].max() > 0
