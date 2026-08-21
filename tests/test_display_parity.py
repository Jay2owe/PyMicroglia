"""Do the two display methods produce the engines' pixels, to the last bit?

Stage 06's exit gate 4. Both engines were settled by eye — one on a 90-filter
sweep, the other on a parameter study confirmed by an injection gate — so
"looks about right" is not a check that can distinguish a working port from a
broken one. What can is that the ported code reproduces a stack the engine
already wrote, pixel for pixel.

No engine is executed. The reference is
``Tmem_Cry_leaktest_magenta_display_2026-08-17``, written on 2026-08-17 and
2026-08-18 by ``microglia_bioluminescence_display.py`` and
``microglia_static_background_removal.py``, together with the ``report.json``
each of them left beside its output. The settings come out of those reports, so
this run is the engine's run and not an approximation of it.

It reads a 344 MB input and takes about 45 seconds, so it runs whenever the
files are on this machine and skips cleanly when they are not — the same rule
``test_real_stack.py`` follows. Point it somewhere else with:

    $env:PYMICROGLIA_DISPLAY_REFERENCE = "...\\Tmem_Cry_leaktest_..._2026-08-17"
"""

from __future__ import annotations

import json
import os
import stat as stat_module
from pathlib import Path

import numpy as np
import pytest

from pymicroglia import display

#: Optional local reference. Override with ``PYMICROGLIA_DISPLAY_REFERENCE``.
DEFAULT_REFERENCE = (Path("reference-data") /
                     "Tmem_Cry_leaktest_magenta_display_2026-08-17")

#: The one recording of the eight in that folder this compares. One is enough:
#: the methods have no per-file branch, so a second file tests the same code
#: again at the cost of another 344 MB read.
STEM = ("20260710_Tmem_Cry_BSL_leaktest_1713_Multichannel Time Lapse_"
        "20260721_1417.ome_registered_translation_cosmic_cleaned")

_OFFLINE = getattr(stat_module, "FILE_ATTRIBUTE_OFFLINE", 0x1000)
_RECALL_ON_ACCESS = 0x00400000


def is_placeholder(path: Path) -> bool:
    attributes = getattr(path.stat(), "st_file_attributes", 0)
    return bool(attributes & (_OFFLINE | _RECALL_ON_ACCESS))


def reference_root() -> Path:
    override = os.environ.get("PYMICROGLIA_DISPLAY_REFERENCE")
    root = Path(override) if override else DEFAULT_REFERENCE
    if not root.is_dir():
        pytest.skip(
            f"the display reference run is not here ({root.name}). Set "
            "PYMICROGLIA_DISPLAY_REFERENCE to a folder holding "
            "cosmic_cleaned/, display/ and static_removed/ to close this gate.")
    return root


def _local(path: Path, what: str) -> Path:
    if not path.exists():
        pytest.skip(f"{what} is not here: {path.name}")
    if is_placeholder(path):
        pytest.skip(
            f"{path.name} is a Dropbox online-only placeholder "
            f"({path.stat().st_size / 1024 ** 2:.0f} MB). Reading it would "
            "download it. Make it available offline and run this again.")
    return path


def _engine_report(folder: Path) -> dict:
    """The ``report.json`` the engine wrote beside its output.

    Found by search rather than named, because the engine timestamps its
    quality-control folder to the second and nothing else knows that stamp.
    """
    found = sorted(folder.glob(f"*{STEM[-40:]}*/*/report.json"))
    if not found:
        found = sorted(folder.glob("**/report.json"))
    if not found:
        pytest.skip(f"no engine report.json under {folder.name}")
    return json.loads(found[0].read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def reference():
    root = reference_root()
    source = _local(root / "cosmic_cleaned" / f"{STEM}.tif", "the input stack")
    return {
        "source": source,
        "display_output": _local(
            root / "display" / f"{STEM}_display_DISPLAY_ONLY.tif",
            "the engine's display output"),
        "display_report": _engine_report(root / "display"),
        "static_output": _local(
            root / "static_removed" / f"{STEM}_static_removed_DISPLAY_ONLY.tif",
            "the engine's static-removed output"),
        "static_report": _engine_report(root / "static_removed"),
    }


@pytest.fixture(autouse=True)
def local_store(tmp_path, monkeypatch):
    monkeypatch.setenv("PYMICROGLIA_STORE", str(tmp_path / "cache"))
    monkeypatch.setenv("PYMICROGLIA_DECISIONS", str(tmp_path / "decisions"))


# ----------------------------------------------------- bioluminescence display
@pytest.fixture(scope="module")
def display_run(reference, tmp_path_factory):
    """One run of the port, with the settings the engine's report records."""
    settings = reference["display_report"]["settings"]
    return display.bioluminescence_display(
        reference["source"],
        output_dir=tmp_path_factory.mktemp("display"),
        signal_channel=reference["display_report"]["signal_channel"],
        black_point_pct=settings["black_point_pct"],
        white_point_pct=settings["white_point_pct"],
        pool_px=settings["pool_px"],
        sharpness=settings["sharpness"],
        noise_multiple=settings["noise_multiple"],
        pad_frames=settings["pad_frames"],
        spatial_sigma_px=settings["spatial_sigma_px"],
        frame_interval_h=settings["frame_interval_h"])


def test_the_display_stack_is_identical_to_the_engines(display_run, reference):
    """Every one of 241 x 504 x 504 pixels. No tolerance: both write uint16."""
    import tifffile

    got = tifffile.imread(display_run.path)
    expected = tifffile.imread(reference["display_output"])
    assert got.shape == expected.shape
    np.testing.assert_array_equal(got, expected)


def test_the_display_range_is_identical_to_the_engines(display_run, reference):
    """The black and white points to the last bit of a float.

    These are the two settings the method is *for*: the black point is what
    decides whether the channel merges without a seam, and it is derived from
    the data, so agreeing on it means agreeing on the whole filtered stack's
    distribution and not only on its pixels.
    """
    expected = reference["display_report"]["display_range"]
    assert display_run.black == expected["black"]
    assert display_run.white == expected["white"]


def test_the_merge_readiness_numbers_are_identical(display_run, reference):
    """The numbers somebody reads before putting this in a composite."""
    got = display_run.report["merge_readiness"]
    expected = reference["display_report"]["merge_readiness"]
    for name in ("field_at_black_percent", "field_median_screen_level",
                 "clipped_at_white_percent"):
        assert got[name] == expected[name], name


def test_the_filter_does_not_move_any_pixels_average_brightness(display_run,
                                                                reference):
    """The claim the method rests on, checked against the engine's own figure.

    A temporal filter that shifted a pixel's whole-record average would change
    how bright a cell looks, which is the one thing a display must not do. The
    engine measures the drift and so does this; on this recording it is under
    four thousandths of a count.
    """
    got = display_run.report["measured"]
    expected = reference["display_report"]["measured"]
    assert (got["mean_brightness_drift_max_counts"]
            == expected["mean_brightness_drift_max_counts"])
    assert got["mean_brightness_drift_max_counts"] < 0.01
    assert (got["frame_to_frame_noise_counts_after"]
            == expected["frame_to_frame_noise_counts_after"])


# ------------------------------------------------------- static background
@pytest.fixture(scope="module")
def static_run(reference, tmp_path_factory):
    settings = reference["static_report"]["settings"]
    return display.remove_static_background(
        reference["source"],
        output_dir=tmp_path_factory.mktemp("static"),
        signal_channel=reference["static_report"]["signal_channel"],
        band_edge_period_h=settings["band_edge_period_h"],
        spatial_sigma_px=settings["spatial_sigma_px"],
        black_point_pct=settings["black_point_pct"],
        white_point_pct=settings["white_point_pct"],
        frame_interval_h=settings["frame_interval_h"],
        k_extra=settings["k_extra"],
        gain=settings["gain"], offset=settings["offset"],
        fluctuation_only=settings["fluctuation_only"])


def test_the_static_removed_stack_is_identical_to_the_engines(static_run,
                                                              reference):
    import tifffile

    got = tifffile.imread(static_run.path)
    expected = tifffile.imread(reference["static_output"])
    assert got.shape == expected.shape
    np.testing.assert_array_equal(got, expected)


def test_the_basis_is_the_same_basis(static_run, reference):
    """NW, K and the noise factor.

    The basis is where this method's whole frequency response comes from, and
    it is computable before any pixel is read. A port that agreed on the output
    but built a different basis would have got there by accident.
    """
    got = static_run.report["basis"]
    expected = reference["static_report"]["basis"]
    assert got["K"] == expected["K"]
    assert got["NW"] == expected["NW"]
    assert got["noise_factor"] == expected["noise_factor"]


def test_the_static_display_range_and_measurements_are_identical(static_run,
                                                                 reference):
    expected_range = reference["static_report"]["display_range"]
    assert static_run.black == expected_range["black"]
    assert static_run.white == expected_range["white"]

    got = static_run.report["measured"]
    expected = reference["static_report"]["measured"]
    for name in ("background_noise_scale_counts", "input_noise_scale_counts",
                 "peak_sigma"):
        assert got[name] == expected[name], name


def test_the_declared_response_keeps_the_circadian_band(static_run):
    """What the basis does to a 24 h sinusoid, from the basis alone.

    The engine's rule is that the band edge goes at *half* the shortest period
    that must survive, so 18 h for 24 h biology. This is that rule paying off:
    at 24 h almost everything comes through, and at 3 h — the noise the method
    exists to remove — almost nothing does.
    """
    response = {row["period_h"]: row for row in static_run.report["declared_response"]}
    assert response[24.0]["energy_retained"] > 0.95
    assert response[3.0]["energy_retained"] < 0.05


# ------------------------------------------------- and they are not the same
def test_the_two_methods_disagree_on_this_recording(display_run, static_run):
    """Same input, same channel, two genuinely different stacks.

    If a refactor ever folded these into one function with different defaults,
    this is the assertion that would fail — on real data, where the difference
    is the point.
    """
    import tifffile

    shown = tifffile.imread(display_run.path)
    cleaned = tifffile.imread(static_run.path)
    assert shown.shape == cleaned.shape
    assert not np.array_equal(shown, cleaned)
    # and not by a rounding-sized margin either
    assert int(np.abs(shown.astype(np.int64)
                      - cleaned.astype(np.int64)).max()) > 100
