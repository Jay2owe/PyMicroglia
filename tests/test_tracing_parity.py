"""Traces, detrends, and the dF/F definition that stops a +634 % spike.

The load-bearing test here is the third one. ``dluc_pipeline.py`` divides dF/F
by each trace's **window mean**, not by its instantaneous rolling baseline, and
the reason is written into its header: one cell's rolling baseline falls to 5 %
of its mean, and the textbook definition turned that into a +634 % spike that
flattened its whole panel. This file reproduces exactly that trace and asserts
the spike does not appear.

The comparison against the reference dataset's stored traces needs the same
registered, windowed, cleaned array as stage 07's gate 1, and is deferred to
stage 11 with it.
"""

from __future__ import annotations
from pymicroglia._results import read_document

import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from pymicroglia import tracing


@pytest.fixture
def store_root(tmp_path, monkeypatch):
    monkeypatch.setenv("PYMICROGLIA_STORE", str(tmp_path / "cache"))
    monkeypatch.setenv("PYMICROGLIA_DECISIONS", str(tmp_path / "decisions"))
    return tmp_path / "cache"


# ------------------------------------------------- regression: the dF/F divisor
@pytest.fixture
def trace_with_a_collapsing_baseline():
    """The reference cell: a rolling baseline that falls to 5 % of its mean.

    Built to be exactly the shape that broke the textbook definition — a bright
    start, a deep trough, and a rhythm riding on top. At the trough the rolling
    baseline is near zero, so dividing by it is dividing by nothing.
    """
    times = np.arange(0.0, 120.0, 0.5)
    # tuned so the 24 h ROLLING baseline — not just the trace — bottoms out at
    # about 5 % of the window mean, which is what the reference cell did
    trend = 1050.0 - 1049.0 * np.exp(-((times - 60.0) ** 2) / (2 * 45.0 ** 2))
    rhythm = 40.0 * np.sin(2 * np.pi * times / 24.0)
    return times, trend + rhythm


def test_the_rolling_baseline_really_does_collapse(
        trace_with_a_collapsing_baseline):
    """The fixture is only a regression test while it reproduces the failure."""
    times, values = trace_with_a_collapsing_baseline
    length = tracing.window_length(24.0, times)
    baseline = tracing.rolling_baseline(values, length)

    assert baseline.min() / values.mean() < 0.075, (
        f"the baseline bottoms out at {baseline.min() / values.mean():.2%} of "
        "the window mean; the reference cell's fell to about 5 %, and this is "
        "only a regression test while it reproduces that")


def test_window_mean_dff_produces_no_spike(trace_with_a_collapsing_baseline):
    """Gate 3. The textbook form gives +634 %; this must not.

    Both are computed here rather than only the good one, so the test says what
    it is preventing rather than merely asserting a bound.
    """
    times, values = trace_with_a_collapsing_baseline
    length = tracing.window_length(24.0, times)
    baseline = tracing.rolling_baseline(values, length)

    textbook = (values - baseline) / baseline            # the singular version
    settled = tracing.window_mean_dff(values, times, 24.0)[0]

    # the reference cell reached +634 %; this trace reaches a few hundred per
    # cent, which is the same failure at a slightly different depth
    assert np.abs(textbook).max() > 1.0, (
        "the textbook definition should blow up on this trace; if it does not, "
        "the fixture no longer reproduces the failure")
    assert np.abs(settled).max() < 1.0, (
        f"window-mean dF/F reached {100 * np.abs(settled).max():.0f} %, which "
        "is the spike this definition exists to prevent")


def test_the_divisor_is_one_number_per_trace(trace_with_a_collapsing_baseline):
    """The property, not just the symptom.

    A test on the peak alone would pass for any definition that happened to be
    bounded. This asserts what the definition *is*.
    """
    times, values = trace_with_a_collapsing_baseline
    length = tracing.window_length(24.0, times)
    baseline = tracing.rolling_baseline(values, length)

    got = tracing.window_mean_dff(values, times, 24.0)[0]
    expected = (values - baseline) / values.mean()
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_a_non_positive_window_mean_gives_nan_not_a_large_number():
    """A trace whose ring placement is wrong has no meaningful dF/F.

    Returning a huge number instead would put it on a panel and let somebody
    read it.
    """
    times = np.arange(0.0, 48.0, 0.5)
    values = np.full_like(times, -5.0)
    result = tracing.window_mean_dff(values, times, 24.0)
    assert np.isnan(result).all()


# -------------------------------------------------------------- the detrends
@pytest.fixture
def detrendable():
    times = np.arange(0.0, 168.0, 0.5)
    drift = 900.0 + 3.0 * times - 0.02 * times ** 2
    return times, drift + 60.0 * np.sin(2 * np.pi * times / 24.0)


@pytest.mark.parametrize("method", ["24h", "48h", "cubic", "poly6"])
def test_every_detrend_removes_the_drift_and_keeps_the_rhythm(detrendable,
                                                              method):
    """All four land on one axis, which is why the pipeline computes three."""
    times, values = detrendable
    result = tracing.detrend(values, times, method)[0]

    assert np.abs(np.median(result)) < 0.02, "the drift is still there"
    # the 24 h swing survives: peak-to-peak of at least half the input's
    assert np.ptp(result) * values.mean() > 0.5 * 2 * 60.0


def test_the_polynomial_baseline_is_conditioned(detrendable):
    """Raw hours to 168 at degree 6 is a badly conditioned Vandermonde matrix.

    Centring and scaling time first is not cosmetic: numpy warns about the
    unconditioned fit and then solves it badly.
    """
    times, values = detrendable
    with np.errstate(all="raise"):
        baseline = tracing.polynomial_baseline(times, values, 6)
    assert np.isfinite(baseline).all()
    assert np.abs(baseline - values).max() < 200.0


def test_detrend_aliases_match_the_engines_command_line(detrendable):
    times, values = detrendable
    np.testing.assert_allclose(tracing.detrend(values, times, "bicubic"),
                               tracing.detrend(values, times, "cubic"))
    np.testing.assert_allclose(tracing.detrend(values, times, "degree6"),
                               tracing.detrend(values, times, "poly6"))


def test_an_unknown_detrend_says_what_is_available(detrendable):
    """The refusal lists the methods, so a typo does not become a search.

    The unknown name is nonsense on purpose. This test used to pass "savgol",
    which was unknown when it was written and is now an alias for
    ``savitzky_golay`` -- Auto-Organotypic added it, and the test started
    asserting nothing while still passing. A name that cannot ever become a
    method is the only kind that keeps testing the refusal.
    """
    times, values = detrendable
    with pytest.raises(ValueError) as raised:
        tracing.detrend(values, times, "no_such_detrend_exists")
    message = str(raised.value)
    assert "no_such_detrend_exists" in message
    assert "cubic" in message, "the refusal stopped listing what is available"


# ------------------------------------------------------- baseline validation
def test_a_baseline_longer_than_the_record_is_refused():
    """A window with no fully supported centre sample is all edge reflection.

    The trace that comes out of one looks plausible, which is why this refuses
    rather than warning.
    """
    times = np.arange(0.0, 20.0, 0.5)
    with pytest.raises(ValueError) as raised:
        tracing.validate_baseline_windows([48.0], times)
    message = str(raised.value)
    assert "48 h needs" in message
    assert "too short" in message


def test_a_workable_baseline_passes():
    tracing.validate_baseline_windows([24.0, 48.0], np.arange(0.0, 240.0, 0.5))


# --------------------------------------------------------------- the ring
def test_the_ring_excludes_every_other_object():
    """A cell sits in its neighbour's scattered light; the ring must not."""
    from scipy import ndimage

    labels = np.zeros((120, 120), np.int32)
    labels[50:60, 40:50] = 1
    labels[50:60, 62:72] = 2

    ring = tracing.ring_of(labels == 1, labels == 2, labels.shape)
    assert ring.sum() >= tracing.RING_MIN
    assert not (ring & (labels == 2)).any()
    assert not (ring & (labels == 1)).any()
    # and it is an annulus, not a disc: the RING_IN gap is real
    inner = ndimage.binary_dilation(labels == 1, np.ones((3, 3), bool),
                                    iterations=tracing.RING_IN)
    assert not (ring & inner).any()


def test_the_ring_widens_rather_than_shrinking():
    """A cell wedged against neighbours gets a wider annulus, not forty pixels.

    Measuring against a too-small ring gives a number that looks like the
    others and is not one.
    """
    from scipy import ndimage

    element = np.ones((3, 3), bool)
    labels = np.zeros((200, 200), np.int32)
    labels[95:105, 95:105] = 1
    mask = labels == 1
    # neighbours filling everything from the RING_IN gap out to 20 px, so the
    # ordinary RING_OUT annulus is entirely blocked
    labels[ndimage.binary_dilation(mask, element, iterations=20)
           & ~ndimage.binary_dilation(mask, element, iterations=tracing.RING_IN)] = 2
    occupied = labels == 2

    inner = ndimage.binary_dilation(mask, element, iterations=tracing.RING_IN)
    narrow = (ndimage.binary_dilation(mask, element, iterations=tracing.RING_OUT)
              & ~inner & ~occupied)
    assert narrow.sum() < tracing.RING_MIN, (
        "this fixture is only meaningful while RING_OUT is too small to use")

    ring = tracing.ring_of(mask, occupied, labels.shape)
    assert ring.sum() >= tracing.RING_MIN
    reach = np.where(ring.any(axis=1))[0]
    # it reached past RING_OUT to find that many pixels
    assert (reach.max() - reach.min() + 1) > 2 * tracing.RING_OUT


# ---------------------------------------------------- traces end to end
def test_traces_are_extracted_stored_and_reused(tmp_path, store_root):
    from tests_support import two_channel_stack

    from pymicroglia import segmentation

    source = two_channel_stack(tmp_path, frames=12)
    segmentation.segment(source, output_dir=tmp_path / "out",
                         channels="dluc=0,struct=1", stationarity_check=False)

    first = tracing.extract_traces(source, output_dir=tmp_path / "out",
                                   channels="dluc=0,struct=1",
                                   baselines=(2.0,), detrends=("cubic",))
    assert len(first) > 0
    assert first.processed.shape == (len(first), len(first.times_h))
    assert first.artefacts["traces"].record["extra"]["dff_denominator"] \
        == "each trace's window mean"

    again = tracing.extract_traces(source, output_dir=tmp_path / "out",
                                   channels="dluc=0,struct=1",
                                   baselines=(2.0,), detrends=("cubic",))
    assert again.artefacts.get("cached") is True
    np.testing.assert_allclose(again.processed, first.processed)


def test_the_processed_trace_is_the_mask_minus_its_ring(tmp_path, store_root):
    """Not the mask mean, and not a whole-field background."""
    from tests_support import two_channel_stack

    from pymicroglia import segmentation, series

    source = two_channel_stack(tmp_path, frames=10)
    found = segmentation.segment(source, output_dir=tmp_path / "out",
                                 channels="dluc=0,struct=1",
                                 stationarity_check=False)
    traces = tracing.extract_traces(source, output_dir=tmp_path / "out",
                                    channels="dluc=0,struct=1",
                                    baselines=(2.0,), detrends=("cubic",))

    label = int(traces.labels[0].split("_")[-1])
    mask = found.labels == label
    ring = tracing.ring_of(mask, (found.labels > 0) & ~mask, found.labels.shape)

    with series.open_series(source) as opened:
        planes = [np.asarray(opened.frame(t, 0), np.float32)
                  for t in range(opened.shape[0])]
    raw, processed = tracing.trace_of(planes, mask, ring)

    np.testing.assert_allclose(traces.raw[0], raw)
    np.testing.assert_allclose(traces.processed[0], processed)
    assert not np.allclose(raw, processed), "the ring subtracted nothing"


def test_extract_traces_says_what_is_missing(tmp_path, store_root):
    from tests_support import two_channel_stack

    source = two_channel_stack(tmp_path)
    with pytest.raises(FileNotFoundError) as raised:
        tracing.extract_traces(source, output_dir=tmp_path / "out")
    assert "segment()" in str(raised.value)


# ------------------------------- against the reference dataset's own traces
REFERENCE = Path("reference-data") / "dluc-pipeline"

BUNDLE = "dluc_registered_windowed_clean.npy"
HOW_TO_BUILD = (
    "Set PYMICROGLIA_DLUC_REGISTERED to the .npy in a bundle written by\n"
    "    from pymicroglia.pipelines.registered import parity_bundle\n"
    "    parity_bundle(r'<...>/MCG_04 - 1 - 595.tif', r'<folder>')")


def _bundle():
    override = os.environ.get("PYMICROGLIA_DLUC_REGISTERED")
    if not override:
        return None
    path = Path(override)
    folder = path if path.is_dir() else path.parent
    return folder if (folder / BUNDLE).is_file() else None


def test_the_reference_traces_match(store_root):
    """Gates 1 and 2, deferred to stage 11 with segmentation's gate 1.

    A trace is a mean over a mask minus a mean over its ring, and the mask comes
    from the label image stage 07 could not reproduce without the pipeline's
    registered, windowed, cleaned array. Comparing traces taken from different
    masks would say nothing about either, which is why both gates close on one
    run or neither does.

    The engine's own label image is used here rather than a freshly segmented
    one — deliberately. The segmentation parity test next door already asserts
    that the two label images are identical; taking the engine's here means a
    failure in *this* test is a failure in the tracing and nothing else.
    """
    import json

    import tifffile

    folder = _bundle()
    if folder is None:
        pytest.skip("gate 1 needs the registered, windowed, cosmic-cleaned "
                    "dLuc array that dluc_pipeline.py traces, which stage 11 "
                    "assembles. " + HOW_TO_BUILD)
    if not REFERENCE.is_dir():
        pytest.skip(f"the reference run is not here: {REFERENCE.name}")

    from pymicroglia import io

    labels = tifffile.imread(REFERENCE / "cell_masks_labels.tif")
    summary = read_document(REFERENCE / "summary.json")
    stack = np.asarray(np.load(folder / BUNDLE, mmap_mode="r"))
    frames = stack.shape[0]
    planes = [stack[index] for index in range(frames)]

    # the engine's own ring geometry: every object's halo excluded from every
    # other object's annulus, which is what `occ` is in dluc_pipeline.py
    occupied = ndimage.binary_dilation(labels > 0, np.ones((3, 3), bool),
                                       iterations=3)

    raw_traces, processed_traces, order = [], [], []
    for record in summary["objects"]:
        label = int(record["label"])
        mask = labels == label
        ring = tracing.ring_of(mask, occupied, labels.shape)
        assert int(ring.sum()) == int(record["ring_px"]), \
            f"object {label}: ring is {int(ring.sum())} px, not {record['ring_px']}"
        raw, processed = tracing.trace_of(planes, mask, ring)
        raw_traces.append(raw)
        processed_traces.append(processed)
        order.append(label)
        assert raw.mean() == pytest.approx(record["raw_mean"], rel=1e-9), \
            f"object {label} raw mean"
        assert processed.mean() == pytest.approx(record["proc_mean"], rel=1e-9), \
            f"object {label} processed mean"
        # ``proc_mean`` and NOT ``mean_counts``, which is a different number in
        # the same record. The engine measures admissibility against the halo of
        # *every* object and the traces against the halo of the *kept* ones, so
        # an object the decoy test excluded shrinks its neighbours' annuli
        # between the two stages. ``mean_counts`` belongs to admissibility and
        # is asserted in the segmentation parity test, where that geometry is
        # the one in use.

    # and the traces themselves, column for column, out of the engine's CSV.
    #
    # The tolerance below is the *file's* precision, not a slack allowance.
    # ``dluc_pipeline.py`` writes these with ``fmt="%.6f"``, so a stored value
    # is the true one rounded to six decimals and the largest honest
    # disagreement is half a unit in that place. Asserting anything tighter
    # would be asserting against digits the engine never wrote down.
    STORED_DECIMALS = 1e-6
    rows = io.read_csv(REFERENCE / "traces_24h.csv")
    times = np.asarray([float(row["hours"]) for row in rows], float)
    assert len(times) == frames

    for position, label in enumerate(order):
        kind = "cell" if summary["objects"][position]["kind"] == "cell" \
            else "object"
        stem = f"{kind}_{label}"
        expected_raw = np.asarray([float(row[f"{stem}_raw"]) for row in rows])
        expected_processed = np.asarray(
            [float(row[f"{stem}_processed"]) for row in rows])
        np.testing.assert_allclose(raw_traces[position], expected_raw,
                                   rtol=0, atol=STORED_DECIMALS,
                                   err_msg=f"{stem} raw trace")
        np.testing.assert_allclose(processed_traces[position],
                                   expected_processed, rtol=0,
                                   atol=STORED_DECIMALS,
                                   err_msg=f"{stem} processed trace")

    # gate 2: every requested detrend reproduces the engine's own dF/F
    values = np.asarray(processed_traces, float)
    for method, filename in (("24h", "traces_24h.csv"),
                             ("48h", "traces_48h.csv"),
                             ("cubic", "traces_cubic.csv"),
                             ("poly6", "traces_poly6.csv")):
        table = io.read_csv(REFERENCE / filename)
        mine = tracing.detrend(values, times, method)
        for position, label in enumerate(order):
            kind = "cell" if summary["objects"][position]["kind"] == "cell" \
                else "object"
            expected = np.asarray(
                [float(row[f"{kind}_{label}_dFF"]) for row in table])
            np.testing.assert_allclose(
                mine[position], expected, rtol=0, atol=STORED_DECIMALS,
                err_msg=f"{method} detrend of {kind} {label}")
