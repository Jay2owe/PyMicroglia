"""The four claims the learned mask makes, written so undoing one fails here.

This route was settled over a long series of proving rounds, and the rounds that
*failed* are the expensive knowledge in it. A docstring saying "do not do X" does
not stop the next person doing X. These do, on synthetic data small enough to
reason about by hand, on any machine, with no weights and no recording:

**The picture is put on one scale before the network sees it.** A glow across the
well inflates a frame-wide noise estimate, which then shrinks every cell in the
picture. That was real: the naive rule read 4.2x the pixel noise on one recording
and 8.4x on another purely because the wells glow differently.

**A mismatched pixel size is announced.** The network counts in pixels, so a
coarser recording quietly loses a third to a half of its cells while returning a
mask that still looks reasonable. Silence is the failure mode, so the warning is
tested, not the mask.

**The cut is a seed, not the answer.** A process is dim by nature and never
clears a cut strict enough to keep false blobs down, but the low cut must apply
only *next to* something the high cut believed in -- otherwise it is a lower
threshold over the whole field, which is the thing it was invented to avoid.

**The fast filters are the same filters.** OpenCV is a speed-up and nothing else.
If it disagrees with SciPy by more than float32 rounding on this machine, it was
built differently here and must not be trusted here.

The last test needs weights and torch and skips without them, the way the
source-verification tests in this suite skip without ``Protocols``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import types

import numpy as np
import pytest

from pymicroglia.learned_mask import filters, refine, scale as scaling
from pymicroglia.learned_mask import network


def _picture(glow: float = 0.0, seed: int = 3) -> np.ndarray:
    """A field of photons: a few bright somata, noise, and an optional glow.

    ``glow`` adds a smooth ramp across the field -- a well that is brighter at
    one edge than the other.
    """
    rng = np.random.default_rng(seed)
    height = width = 96
    rows, columns = np.mgrid[:height, :width]
    photons = np.full((height, width), 4.0, np.float32)
    for y, x in ((24, 24), (24, 70), (68, 30), (70, 72)):
        photons += 40.0 * np.exp(-(((rows - y) ** 2 + (columns - x) ** 2)
                                   / (2 * 2.0 ** 2)))
    photons = rng.poisson(photons).astype(np.float32)
    if glow:
        photons = photons + glow * (columns / width)
    return photons


# --- the picture is put on one scale -----------------------------------------

def test_a_glow_across_the_well_cannot_set_the_noise_scale():
    """The band-passed spread ignores a glow; the frame-wide spread does not.

    White noise of a known size with a ramp laid over it, so the right answer is
    known: the pixel noise did not change, and an estimator that says it did is
    wrong. Measured away from the edges, where the reflected border turns the
    ramp into a crease the band-pass legitimately sees -- on a real 498 px field
    that margin is a twentieth of the picture, and here it is excluded outright
    rather than tolerated.
    """
    rng = np.random.default_rng(7)
    size, margin = 256, 64
    columns = np.mgrid[:size, :size][1]
    field = rng.normal(0.0, 1.0, (size, size)).astype(np.float32)
    lit = field + 30.0 * (columns / size).astype(np.float32)
    inner = (slice(margin, -margin), slice(margin, -margin))

    accepted = (network._mad(refine.band_pass(lit)[inner])
                / network._mad(refine.band_pass(field)[inner]))
    naive = network._mad(lit) / network._mad(field)

    assert accepted == pytest.approx(1.0, rel=0.02), (
        f"the band-passed spread moved by {accepted:.2f}x for a glow that "
        "changed no pixel's noise, so it is reading structure as noise")
    # And the estimator it replaced really does fail, so the assertion above is
    # not passing because the case is too easy to fail.
    assert naive > 10.0, (
        f"the frame-wide spread only moved {naive:.1f}x, so this case has "
        "stopped exercising the failure it was written for")


def test_one_threshold_means_the_same_in_a_dim_well_and_a_bright_one():
    """What the square root buys: noise the same size at any brightness.

    Two fields of pure photon counting, one twenty-five times brighter. Their
    raw spreads differ five-fold -- root twenty-five, exactly as Poisson says --
    and that is the reason a single noise figure for a frame is wrong in both
    directions at once.
    """
    rng = np.random.default_rng(7)
    dim = rng.poisson(np.full((128, 128), 20.0)).astype(np.float32)
    bright = rng.poisson(np.full((128, 128), 500.0)).astype(np.float32)

    raw = network._mad(bright) / network._mad(dim)
    squared_up = (network._mad(network.anscombe(bright))
                  / network._mad(network.anscombe(dim)))

    assert raw > 4.0, (
        f"the two wells differ by only {raw:.1f}x, so this case no longer poses "
        "the problem the square root exists to solve")
    assert squared_up == pytest.approx(1.0, rel=0.05), (
        f"after the square root the two wells still differ by {squared_up:.2f}x")


def test_the_background_is_a_surface_and_not_one_number():
    lit = _picture(glow=300.0)
    surface = network.background(network.anscombe(lit))
    left = surface[:, :10].mean()
    right = surface[:, -10:].mean()
    assert right > left * 1.5, (
        "the glow was removed as a single number, so a bright corner keeps a "
        f"pedestal under its cells: left {left:.2f}, right {right:.2f}")


# --- a mismatched pixel size is announced ------------------------------------

def _weights_claiming(low: float, high: float | None = None):
    """Stand-in for a loaded model: only its pixel claim is under test."""
    return types.SimpleNamespace(pixel_range=(low, low if high is None else high))


def test_the_native_pixel_size_passes_without_comment():
    from pymicroglia.learned_mask import apply as masking

    assert masking.scale_warning(_weights_claiming(2.0), 2.0) is None


def test_a_coarser_recording_is_warned_about_rather_than_refused():
    from pymicroglia.learned_mask import apply as masking

    warning = masking.scale_warning(_weights_claiming(2.0), 4.0)
    assert warning and "MISSING CELLS" in warning, (
        "a 2x pixel mismatch must say so loudly; it costs a third to a half of "
        "the cells and nothing else notices")


def test_an_unrecorded_pixel_size_is_not_quietly_assumed_to_be_right():
    from pymicroglia.learned_mask import apply as masking

    warning = masking.scale_warning(_weights_claiming(2.0), None)
    assert warning and "not recorded" in warning


def test_a_model_trained_across_a_range_accepts_that_range():
    from pymicroglia.learned_mask import apply as masking

    covering = _weights_claiming(*scaling.TRAIN_UM_PER_PX)
    assert masking.scale_warning(covering, 4.0) is None
    assert masking.scale_warning(covering, 6.0)


# --- the cut is a seed, not the answer ---------------------------------------

def _soma_with_an_arm(seed: int = 11):
    """A bright soma, a dim arm reaching off it, and a dim patch on its own.

    The arm and the patch are equally dim, equally certain and the same length.
    The only difference between them is that one touches a seed, which is the
    whole of what ``grow`` is supposed to care about. Both are inside
    ``GROW_PX`` of where they start, so nothing here turns on reach.
    """
    rng = np.random.default_rng(seed)
    picture = rng.normal(0.0, 0.3, (80, 80)).astype(np.float32)
    probability = np.zeros((80, 80), np.float32)

    picture[18:26, 18:26] += 30.0          # soma
    probability[18:26, 18:26] = 0.97
    picture[21:24, 26:34] += 6.0           # its arm, dim but real
    probability[21:24, 26:34] = 0.62
    picture[60:63, 60:68] += 6.0           # the same thing, touching nothing
    probability[60:63, 60:68] = 0.62
    return picture, probability


def test_a_process_next_to_a_seed_is_picked_up():
    picture, probability = _soma_with_an_arm()
    cells = refine.refine(picture, probability, high=0.80, low=0.60,
                          min_area=32, max_area=0)
    assert cells[22, 30] != 0, "the arm was left out of its own cell"
    assert cells[22, 30] == cells[22, 22], "the arm became a second cell"


def test_the_same_dimness_away_from_every_seed_is_not():
    picture, probability = _soma_with_an_arm()
    cells = refine.refine(picture, probability, high=0.80, low=0.60,
                          min_area=32, max_area=0)
    assert cells[61, 64] == 0, (
        "the low cut reached a region no seed touches, which makes it a lower "
        "threshold over the field rather than a growth rule")


def test_the_size_floor_drops_a_speck_and_keeps_a_cell():
    probability = np.zeros((60, 60), np.float32)
    probability[10:18, 10:18] = 0.95       # 64 px: a cell
    probability[40:43, 40:44] = 0.95       # 12 px: a speck
    marked = refine.seeds(probability, high=0.80, min_area=32)
    assert marked[14, 14] != 0
    assert marked[41, 41] == 0
    assert marked.max() == 1


def test_splitting_is_off_unless_the_answer_is_a_count():
    picture, probability = _soma_with_an_arm()
    merged = refine.refine(picture, probability, high=0.80, low=0.60,
                           min_area=32)
    assert merged[22, 22] != 0, (
        "the default dropped a cell, so max_area is no longer 0 and a merged "
        "lobe is being cut apart in a route that only makes a mask")


# --- sizes are facts about the cell, not about the camera --------------------

def test_a_finer_recording_gets_the_same_cell_in_more_pixels():
    native = refine.settings(scaling.NATIVE)
    fine = refine.settings(scaling.Scale(um_per_px=1.0))
    assert fine["grow_px"] == 2 * native["grow_px"]
    assert fine["min_area"] == 4 * native["min_area"]


# --- the fast filters are the same filters -----------------------------------

def test_opencv_agrees_with_scipy_on_this_machine():
    if filters._opencv() is None:
        pytest.skip("OpenCV is not installed; the SciPy fallback is in use")
    report = filters.agrees()
    for sigma in (2.0, 15.0):
        difference = report[f"blur_{sigma:g}_max_difference"]
        spread = report[f"blur_{sigma:g}_spread"]
        assert difference < 1e-5 * spread, (
            f"the sigma {sigma:g} blur differs from SciPy by {difference:.2e} on "
            f"a spread of {spread:.2f}, which is larger than float32 rounding: "
            "OpenCV is built differently here and must not be trusted here")
    assert report["reach_identical"], (
        "the distance transform is not bit-identical, which means the 5x5 "
        "chamfer approximation is being used instead of DIST_MASK_PRECISE")


def test_the_blur_window_is_the_one_every_sigma_assumes():
    assert filters.radius(2.0) == 8       # SciPy truncates at four sigma
    assert filters.radius(15.0) == 60


# --- importing costs nothing --------------------------------------------------

def test_importing_the_package_does_not_import_torch():
    """The claim that lets a machine without torch still import pymicroglia.

    Run in a fresh interpreter because torch may well already be imported by the
    time this test runs, which would make an in-process check pass for the wrong
    reason.
    """
    proof = ("import sys; import pymicroglia.learned_mask as m; "
             "assert m.refine and m.scale and m.filters; "
             "print('torch' in sys.modules)")
    done = subprocess.run([sys.executable, "-c", proof], capture_output=True,
                          text=True, cwd=str(os.path.dirname(os.path.dirname(
                              os.path.abspath(__file__)))))
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "False", (
        "importing learned_mask pulled torch in; the lazy __getattr__ for "
        "apply/network has been undone")


# --- the whole route, when there are weights to run it with -------------------

def test_the_route_is_deterministic_on_the_fast_path():
    """Two runs of the fast memory layout agree byte for byte.

    The fast path is not bit-identical to the exact one -- it moves the
    probability by about 2e-6 -- and that trade was made deliberately. What it
    must never be is unrepeatable.
    """
    named = os.environ.get("PYMICROGLIA_MASK_WEIGHTS")
    if not named:
        pytest.skip("set PYMICROGLIA_MASK_WEIGHTS to a trained run's model.pt")
    pytest.importorskip("torch")
    from pymicroglia.learned_mask import apply as masking

    model = masking.hurry(masking.load_model(masking.weights()))
    pictures = _picture()[None]
    first = masking.mask_pictures(model, pictures, um_per_px=2.0)
    second = masking.mask_pictures(model, pictures, um_per_px=2.0)
    assert np.array_equal(first["probability"], second["probability"])
    assert np.array_equal(first["cells"], second["cells"])
    assert first["warning"] is None
