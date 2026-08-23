"""The one-rule cosmic-ray engine's own test cases, run against the port.

Every case here is **copied** from
``Protocols/Analysis/test_microglia_cosmic_ray_removal.py``. That file is not
run, not imported and not modified — it keeps testing the engine, this keeps
testing the package, and the two agreeing is the claim.

The cases are translated from the engine's ``Settings``/``process`` pair to
``cosmic.remove_cosmic_rays``, and nothing else about them changes: the same
arrays, the same seeds, the same numbers, the same assertions. Several are
arithmetic whose right answer can be worked out on paper, which is what makes
them worth copying rather than approximating with a tolerance.

This replaced ``test_filtering_parity.py``'s subject on 2026-08-20, when the
protocol replaced the matched-line method with this one. That file still tests
the older method, which two callers inside the package still use.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import tifffile

from pymicroglia import cosmic


@pytest.fixture(autouse=True)
def store_root(tmp_path, monkeypatch):
    """A store of its own, so a run here never reads the real machine's."""
    monkeypatch.setenv("PYMICROGLIA_STORE", str(tmp_path / "cache"))
    monkeypatch.delenv("PYMICROGLIA_DECISIONS", raising=False)
    return tmp_path / "cache"


def synthetic_stack(frames=24, size=64, level=1000.0, noise=40.0, seed=7):
    generator = np.random.default_rng(seed)
    stack = generator.normal(level, noise, (frames, size, size))
    return np.clip(stack, 0, 65535).astype(np.uint16)


def write_stack(path: Path, stack: np.ndarray, axes: str = "TYX") -> Path:
    """An OME-TIFF, because that is what this package opens.

    The engine writes a plain TIFF and reads it with its own reader; the only
    difference here is the metadata that lets ``open_series`` find the axes.
    The pixels are identical.
    """
    tifffile.imwrite(path, stack, ome=True, metadata={"axes": axes})
    return path


def clean(source, output_dir, **overrides):
    options = dict(write_preview=False)
    options.update(overrides)
    return cosmic.remove_cosmic_rays(source, output_dir=output_dir, **options)


# -- step 1: the reference ---------------------------------------------------
def test_the_reference_is_the_two_frames_either_side():
    assert cosmic.reference_window(10, 5, 2) == [4, 6]


def test_an_end_frame_reflects_and_keeps_the_duplicate():
    """Frame 0 has no frame -1, so it takes frame 1 twice and still gets a
    reference of the same width as every other frame's."""
    assert cosmic.reference_window(10, 0, 2) == [1, 1]
    assert cosmic.reference_window(10, 9, 2) == [8, 8]


def test_the_mean_reference_is_unbiased_and_the_max_reference_is_not(tmp_path):
    """max(a, b) = (a + b) / 2 + |a - b| / 2, so a max reference carries a
    positive half-normal term that has to be cancelled by an offset."""
    from pymicroglia import series

    source = write_stack(tmp_path / "plain.ome.tif", synthetic_stack())
    with series.open_series(source) as opened:
        mean_centre, _ = cosmic.measure_noise(opened, 0, "mean2", 0)
        max_centre, _ = cosmic.measure_noise(opened, 0, "max2", 0)

    assert abs(mean_centre) < abs(max_centre)
    assert max_centre < 0


# -- step 2: the one statistic ----------------------------------------------
def test_a_bright_pixel_is_replaced_by_the_mean_of_its_neighbours(tmp_path):
    stack = synthetic_stack()
    stack[10, 32, 32] = 60000
    source = write_stack(tmp_path / "spike.ome.tif", stack)
    result = clean(source, tmp_path / "out")

    cleaned = tifffile.imread(result.path)
    expected = 0.5 * (float(stack[9, 32, 32]) + float(stack[11, 32, 32]))
    assert abs(float(cleaned[10, 32, 32]) - expected) <= 1.0
    assert result.summary["head_pixel_frames"] == 1


def test_interleave_replacement_alternates_real_adjacent_pixels(tmp_path):
    stack = synthetic_stack()
    frame, y, x = 10, 32, 32
    stack[frame, y, x] = 60000
    source = write_stack(tmp_path / "interleave.ome.tif", stack)
    result = clean(source, tmp_path / "out", replacement="interleave")

    cleaned = tifffile.imread(result.path)
    take_following = bool((y + x + frame) & 1)
    neighbour = frame + 1 if take_following else frame - 1
    assert cleaned[frame, y, x] == stack[neighbour, y, x]
    assert result.summary["replacement"] == "interleave"


def test_the_growth_replaces_the_skirt_as_well_as_the_core(tmp_path):
    """Growth of 2 px dilates one pixel to 5x5 — 25, counted by hand."""
    stack = synthetic_stack()
    stack[10, 32, 32] = 60000
    source = write_stack(tmp_path / "skirt.ome.tif", stack)
    result = clean(source, tmp_path / "out", growth_px=2, write_mask=True)

    assert result.mask[10, 32, 32]
    assert int(result.mask[10].sum()) == 25


def test_a_combined_z_is_the_same_number_as_a_mean_in_standard_errors():
    values = np.array([3.0, 4.0, 5.0, 6.0])
    standard_errors = values.mean() / (1.0 / math.sqrt(len(values)))
    assert cosmic.combined_z(values) == pytest.approx(standard_errors)


def test_nothing_is_a_constant_in_camera_counts(tmp_path):
    """The same recording at half the gain must have the same pixels repaired.

    This is what makes the method portable. Every cut is a multiple of the
    recording's own noise, so scaling the data scales the cut with it. An
    absolute threshold in camera counts would clean one of these and not the
    other.
    """
    stack = synthetic_stack()
    stack[10, 32, 32] = 40000
    stack[15, 20, 44] = 30000
    bright = write_stack(tmp_path / "bright.ome.tif", stack)
    dim = write_stack(tmp_path / "dim.ome.tif", (stack // 2).astype(np.uint16))

    loud = clean(bright, tmp_path / "bright_out")
    quiet = clean(dim, tmp_path / "dim_out")

    assert loud.summary["head_pixel_frames"] == quiet.summary["head_pixel_frames"]
    assert (loud.summary["pixel_frames_replaced"]
            == quiet.summary["pixel_frames_replaced"])
    assert loud.summary["temporal_sigma_counts"] == pytest.approx(
        2 * quiet.summary["temporal_sigma_counts"], rel=0.02)


# -- step 3: the line test ---------------------------------------------------
def test_a_round_hit_never_becomes_a_track():
    """The shape gate is not an optimisation.

    A round hit's principal axis points in an arbitrary direction, so searching
    around it and keeping the best score is a search for the best line through
    noise. Without the gate it finds one.
    """
    generator = np.random.default_rng(3)
    z = generator.normal(0.0, 1.0, (128, 128))
    z[60:66, 60:66] = 20.0                      # a round, very bright hit

    assert cosmic.track_components(z > 12.0, z, cosmic.Settings()) == []


def test_a_real_line_is_found_and_its_band_is_as_wide_as_itself():
    generator = np.random.default_rng(4)
    z = generator.normal(0.0, 1.0, (128, 200))
    z[64, 20:60] = 20.0                         # the head
    # The track sits below grow_z, so it never joins the head component: it is
    # found only because the whole run of pixels is scored together.
    z[64, 60:190] = 1.5

    settings = cosmic.Settings()
    components = cosmic.track_components(z > 12.0, z, settings)
    assert len(components) == 1

    match = cosmic.match_track(components[0], z, settings)
    assert match is not None
    assert match["z"] > settings.seed_z
    # The band is the hit measured down to grow_z — three pixels tall here —
    # plus the two pixels of growth every hit gets on each side. It is never a
    # width somebody chose.
    assert match["band_px"] == 7
    assert match["candidates_tried"] == 11


def test_the_line_is_judged_by_the_same_cut_as_a_pixel():
    generator = np.random.default_rng(5)
    z = generator.normal(0.0, 1.0, (128, 200))
    z[64, 20:60] = 20.0
    z[64, 60:190] = 0.2                         # a continuation that is not there

    settings = cosmic.Settings()
    components = cosmic.track_components(z > 12.0, z, settings)
    assert cosmic.match_track(components[0], z, settings) is None


# -- step 4: the bleed off a censored pixel ----------------------------------
def test_the_bleed_fit_recovers_a_planted_amplitude_and_decay():
    generator = np.random.default_rng(6)
    rows, reach = 400, 60
    saturated = generator.integers(1, 6, rows).astype(float)
    distance = np.arange(1, reach + 1, dtype=float)[None, :]
    truth = 300.0 * saturated[:, None] * np.exp(-distance / 20.0)
    values = truth + generator.normal(0.0, 30.0, truth.shape)

    model = cosmic.fit_tail(saturated, values, 65535.0, 30.0, 1.0)
    assert model["alpha_counts_per_censored_pixel"] == pytest.approx(300.0, rel=0.05)
    assert model["decay_px"] == pytest.approx(20.0, rel=0.05)
    assert model["alpha_share_of_full_scale"] == pytest.approx(
        300.0 / 65535.0, rel=0.05)


def test_a_row_with_nothing_censored_has_nothing_subtracted():
    """The model is forced through the origin, so it cannot invent a bleed."""
    model = {"alpha_counts_per_censored_pixel": 300.0, "decay_px": 20.0}
    assert float(cosmic.predict_tail(model, np.array([0.0]), 30).sum()) == 0.0


def test_censoring_is_a_share_of_full_scale_not_a_count():
    settings = cosmic.Settings(saturation_fraction=0.99)
    scale = cosmic.full_scale(np.dtype(np.uint16), 0.0)
    assert scale == 65535.0

    frame = np.array([[64878, 64880]], np.uint16)
    assert cosmic.censored(frame, scale, settings).tolist() == [[False, True]]


# -- the file contract -------------------------------------------------------
def test_the_source_is_never_modified(tmp_path):
    import hashlib

    stack = synthetic_stack()
    stack[10, 32, 32] = 60000
    source = write_stack(tmp_path / "audit.ome.tif", stack)
    before = hashlib.sha256(source.read_bytes()).hexdigest()

    result = clean(source, tmp_path / "out")

    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert result.summary["source_modified"] is False
    assert result.summary["registered_input_required"] is True


def test_an_existing_output_is_refused_without_overwrite(tmp_path):
    source = write_stack(tmp_path / "twice.ome.tif", synthetic_stack())
    clean(source, tmp_path / "out")

    with pytest.raises(FileExistsError):
        clean(source, tmp_path / "out", reuse=False)
    clean(source, tmp_path / "out", reuse=False, overwrite=True)


def test_the_border_crop_trims_every_edge(tmp_path):
    source = write_stack(tmp_path / "crop.ome.tif", synthetic_stack(size=64))
    result = clean(source, tmp_path / "out", border_crop_px=7)

    assert tifffile.imread(result.path).shape == (24, 50, 50)


def test_a_smaller_label_image_is_centred_in_the_frame(tmp_path):
    labels = np.zeros((50, 50), np.uint16)
    labels[0, 0] = 3
    path = tmp_path / "labels.tif"
    tifffile.imwrite(path, labels)

    settings = cosmic.Settings(exclude_labels=path, exclude_label_ids=(3,))
    excluded = cosmic.load_exclusion(settings, 64, 64)

    assert excluded[7, 7]                        # (64 - 50) // 2 == 7
    assert int(excluded.sum()) == 1


def test_other_channels_are_copied_through_untouched(tmp_path):
    generator = np.random.default_rng(11)
    stack = generator.normal(1000, 40, (12, 2, 48, 48)).astype(np.uint16)
    stack[5, 0, 24, 24] = 60000
    stack[5, 1, 24, 24] = 60000
    source = write_stack(tmp_path / "twochannel.ome.tif", stack, axes="TCYX")

    result = clean(source, tmp_path / "out", signal_channel=1)
    cleaned = tifffile.imread(result.path)

    assert cleaned.shape == stack.shape
    assert cleaned[5, 0, 24, 24] < 60000         # the selected channel was cleaned
    assert cleaned[5, 1, 24, 24] == 60000        # the other channel was not


@pytest.mark.parametrize("bad", [
    {"minimum_aspect": 1.0},
    {"saturation_fraction": 1.5},
    {"slope_search_band_px": 10},
    {"grow_z": 99.0},
    {"tail_loss_scale_noise": 0.0},
])
def test_a_bad_setting_is_refused_before_anything_is_written(tmp_path, bad):
    source = write_stack(tmp_path / "bad.ome.tif", synthetic_stack())
    folder = tmp_path / "out"

    with pytest.raises(ValueError):
        clean(source, folder, **bad)
    assert not folder.exists() or not list(folder.glob("*_cosmic_cleaned.tif"))


def test_the_bleed_correction_can_be_turned_off_without_touching_hits(tmp_path):
    """A recording whose placebo fails still gets its hits and tracks repaired.

    Turning the bleed correction off must change nothing about detection: the
    same pixels are replaced, and the only difference is that no censored row is
    reduced.
    """
    stack = synthetic_stack()
    stack[10, 32, 32] = 65535
    stack[10, 32, 33:60] = 3000
    source = write_stack(tmp_path / "bleed.ome.tif", stack)

    on = clean(source, tmp_path / "on")
    off = clean(source, tmp_path / "off", bleed_correction=False)

    assert on.summary["pixel_frames_replaced"] == off.summary["pixel_frames_replaced"]
    assert on.summary["censored_pixel_frames"] == off.summary["censored_pixel_frames"]
    assert off.summary["bleed_counts_removed"] == 0
    assert off.summary["bleed_model"]["alpha_counts_per_censored_pixel"] == 0.0
    assert off.summary["bleed_model"]["model"] == "bleed correction turned off"


# -- what this package adds on top -------------------------------------------
def test_the_placebo_writes_numbers_and_never_a_product(tmp_path):
    """A control that produced a cleaned stack would be a product.

    Its removal figure has to come back near zero: it fits the bleed on the side
    nothing bleeds toward, so a large number there says the model is fitting
    noise and the real run's figure cannot be believed.
    """
    stack = synthetic_stack()
    stack[10, 32, 32] = 65535
    stack[10, 32, 33:60] = 3000
    source = write_stack(tmp_path / "placebo.ome.tif", stack)

    control = clean(source, tmp_path / "out", mirror_placebo=True)

    assert control.summary["mirror_placebo"] is True
    assert not list((tmp_path / "out").glob("*_cosmic_cleaned.tif"))
    assert "placebo" in control.artefacts
    assert control.summary["fitted_on_side"] in {"low", "high"}


def test_a_repeat_run_is_a_cache_hit_and_says_so(tmp_path):
    source = write_stack(tmp_path / "again.ome.tif", synthetic_stack())
    first = clean(source, tmp_path / "out")
    second = clean(source, tmp_path / "out")

    assert second.artefacts.get("cached") is True
    np.testing.assert_array_equal(np.asarray(first.mask), np.asarray(second.mask))


def test_the_action_is_registered_and_carries_the_new_method_version():
    from pymicroglia import describe

    entry = describe("remove_cosmic_rays")
    assert entry["pending"] is False
    assert entry["binds_to"] == "cosmic.remove_cosmic_rays"
    assert entry["method_version"] == "2026-08-21-selectable-replacement"
    names = {row["name"] for row in entry["params"]}
    assert {
        "replacement", "seed_z", "grow_z", "saturation_fraction",
        "mirror_placebo"
    } <= names
