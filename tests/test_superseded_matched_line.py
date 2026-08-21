"""The **superseded** matched-line cosmic-ray method, and its own test cases.

Every case here was copied from the engine's test file as it stood at
``2026-08-16-matched-line-neighbour-blend``, including the stored engine run in
``fixtures/cosmic_ray_reference`` — those fixtures are that method's, not the
one running now. On 2026-08-20 the protocol replaced the method wholesale; the
port of its replacement, and the copies of its new cases, are in
``test_cosmic_parity.py``.

Nothing calls this method any longer. The file stays for the reason the method
does: a run record's equivalent script says ``filtering.remove_cosmic_rays``,
and a record that no longer reproduces its run is a note rather than a result.
These cases are what says it still reproduces it — an untested method kept for
replay is a method nobody can trust a replay from.

The cases are translated from ``unittest`` to ``pytest`` and from the engine's
``Settings``/``process`` pair to ``remove_cosmic_rays``, and nothing else about
them changes: the same arrays, the same numbers, the same assertions.

Three of them test the arithmetic directly, with hand-built arrays whose right
answer can be worked out on paper — which is what makes them worth copying
rather than approximating with a tolerance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from pymicroglia.superseded import matched_line


@pytest.fixture
def store_root(tmp_path, monkeypatch):
    """A store of its own, so a run here never reads the real machine's."""
    monkeypatch.setenv("PYMICROGLIA_STORE", str(tmp_path / "cache"))
    monkeypatch.delenv("PYMICROGLIA_DECISIONS", raising=False)
    return tmp_path / "cache"


# ---------------------------------------------------------------- detection
def test_persistent_signal_is_not_a_single_frame_hit():
    """A cell that is bright in every frame is not a spike, however bright.

    This is the case the whole design exists for. An absolute count threshold
    calls this a hit; the temporal test does not, because the neighbours are
    just as bright.
    """
    current = np.full((9, 9), 500, np.float32)
    reference = np.full((9, 9), 500, np.float32)
    mask, _ = matched_line.detect_cosmic_pixels(current, reference, 0, 2, 12, 0)
    assert not mask.any()


def test_two_pixel_growth_makes_a_five_by_five_mask():
    """Growth of 2 px dilates one pixel to 5x5 — 25, counted by hand."""
    current = np.full((11, 11), 100, np.float32)
    reference = current.copy()
    current[5, 5] = 1000
    mask, _ = matched_line.detect_cosmic_pixels(current, reference, 0, 2, 12, 2)
    assert int(mask.sum()) == 25


def test_matched_line_extends_a_bright_head_across_the_frame():
    """A bright head plus a faint trail: the trail is repaired, and separately.

    ``point_mask`` staying False at the far end is the part that matters. The
    trail is blended rather than replaced with the neighbour maximum, because
    a maximum would print the whole faint band into the result.
    """
    reference = np.zeros((80, 160), np.float32)
    current = reference.copy()
    current[39:42, :45] = 20
    current[40, 45:] = 1

    mask, point_mask, _, matches = matched_line.detect_cosmic_pixels_with_lines(
        current, reference, centre=0, temporal_sigma=1, threshold_sigma=12,
        mask_growth_px=2, line_band_width=11, line_score_minimum=8,
        line_weak_threshold=2, line_minimum_length=25, line_minimum_aspect=6)

    assert len(matches) == 1
    assert mask[40, -1]
    assert not point_mask[40, -1]


def test_neighbour_blend_is_unbiased_and_uses_both_frames():
    """Half the pixels from each neighbour, so the blend adds no bias.

    Deterministic, not random: the same frame always splits the same way, so a
    re-run reproduces the cleaned stack byte for byte.
    """
    previous = np.zeros((4, 4), np.float32)
    following = np.full((4, 4), 10, np.float32)

    blended = matched_line.neighbour_blend(previous, following, frame=4)

    assert float(blended.mean()) == 5.0
    assert set(np.unique(blended)) == {0.0, 10.0}


# --------------------------------------------------------------- end to end
@pytest.fixture()
def spiked_stack(tmp_path):
    """Nine frames, two channels, one 5000-count spike in channel 2 frame 4."""
    rng = np.random.default_rng(7)
    data = np.clip(rng.normal(100, 3, size=(9, 2, 32, 32)), 0, 65535
                   ).astype(np.uint16)
    original = data.copy()
    data[4, 1, 16, 17] = 5000

    source = tmp_path / "synthetic_registered.ome.tif"
    delta_t = [float(value) for frame in range(9)
               for value in (frame * 60, frame * 60 + 1)]
    tifffile.imwrite(
        source, data, ome=True,
        metadata={
            "axes": "TCYX",
            "PhysicalSizeX": 2.0, "PhysicalSizeXUnit": "um",
            "PhysicalSizeY": 2.0, "PhysicalSizeYUnit": "um",
            "Channel": {"Name": ["reference", "signal"]},
            "Plane": {"DeltaT": delta_t, "DeltaTUnit": ["s"] * len(delta_t)},
        },
    )
    return source, data, original, delta_t


def _clean(source, tmp_path, **overrides):
    options = dict(output_dir=tmp_path / "AI_Exports" / "cosmic_test",
                   output_name="runner_named_cleaned.tif",
                   signal_channel=2, threshold_sigma=12, mask_growth_px=0,
                   sample_frames=9, write_preview=False)
    options.update(overrides)
    return matched_line.remove_cosmic_rays(source, **options)


def test_end_to_end_preserves_other_channel_and_geometry(spiked_stack, tmp_path,
                                                         store_root):
    source, data, original, delta_t = spiked_stack
    result = _clean(source, tmp_path)

    output = tifffile.imread(result.path)
    assert output.shape == data.shape

    # the untouched channel comes through byte for byte
    np.testing.assert_array_equal(output[:, 0], data[:, 0])

    # the spike is replaced with the larger of its two neighbours
    expected = max(int(data[3, 1, 16, 17]), int(data[5, 1, 16, 17]))
    assert int(output[4, 1, 16, 17]) == expected

    assert bool(result.mask[4, 16, 17])
    assert int(result.summary["pixel_frames_replaced"]) > 0
    assert result.summary["source_modified"] is False

    # the source file is not touched
    np.testing.assert_array_equal(tifffile.imread(source), data)
    assert not np.array_equal(original, data)


def test_rerunning_refuses_rather_than_overwriting(spiked_stack, tmp_path,
                                                   store_root):
    """A second run into the same name leaves the first output alone.

    The engine raises ``FileExistsError`` here and so does this. Refusing is
    what stops a re-run destroying the stack somebody is comparing against.

    ``reuse=False`` is what makes this reach the write at all: an identical
    re-run reads the stored mask and returns without touching the file, which
    is the store doing its job. The refusal is for the case where the work is
    genuinely redone.
    """
    source, _, _, _ = spiked_stack
    first = _clean(source, tmp_path)
    before = first.path.read_bytes()

    with pytest.raises(FileExistsError):
        _clean(source, tmp_path, reuse=False)

    assert first.path.read_bytes() == before


def test_geometry_channel_names_and_timestamps_survive(spiked_stack, tmp_path,
                                                       store_root):
    """Pixel size, channel names and per-plane times are still true of the output.

    They describe the recording, not the filtering, so the port carries the
    source's own description across rather than writing a new one.
    """
    source, _, _, delta_t = spiked_stack
    result = _clean(source, tmp_path)

    with tifffile.TiffFile(result.path) as handle:
        pixels = tifffile.xml2dict(handle.ome_metadata)["OME"]["Image"]["Pixels"]

    assert float(pixels["PhysicalSizeX"]) == 2.0
    assert [channel["Name"] for channel in pixels["Channel"]] == ["reference",
                                                                  "signal"]
    assert [float(plane["DeltaT"]) for plane in pixels["Plane"]] == delta_t


# ------------------------------------------------------- what the store adds
def test_the_mask_records_the_registration_it_was_derived_from(spiked_stack,
                                                               tmp_path,
                                                               store_root):
    """``upstream`` is what makes this artefact downstream of *one* registration.

    Nothing registered this synthetic file, so the list is empty here — the
    assertion is that the field exists and is carried into the key, which is
    the property gate 7 turns on.
    """
    source, _, _, _ = spiked_stack
    result = _clean(source, tmp_path)

    record = result.artefacts["mask"].record
    assert "upstream" in record
    assert list(record["upstream"]) == list(result.upstream)


def test_a_second_identical_run_reads_the_stored_mask(spiked_stack, tmp_path,
                                                      store_root):
    """Same source, same parameters, same METHOD_VERSION: no work done twice."""
    source, _, _, _ = spiked_stack
    first = _clean(source, tmp_path)
    again = _clean(source, tmp_path, overwrite=True)

    assert again.artefacts.get("cached") is True
    np.testing.assert_array_equal(again.mask, first.mask)


def test_a_changed_threshold_misses(spiked_stack, tmp_path, store_root):
    """The setting that decides what counts as data is part of the key."""
    source, _, _, _ = spiked_stack
    _clean(source, tmp_path)
    other = _clean(source, tmp_path, threshold_sigma=8, overwrite=True)

    assert other.artefacts.get("cached") is not True


# ------------------------------- against the engine's own stored output
REFERENCE = Path(__file__).parent / "fixtures" / "cosmic_ray_reference"


@pytest.fixture
def engine_run():
    """The engine's input, its output and the settings it used.

    Copied byte for byte from a real run; see the README beside them. No engine
    is executed here — running one would test that the engine is installed, not
    that the port agrees with it.
    """
    import json

    if not REFERENCE.is_dir():
        pytest.skip("engine reference fixtures not present")
    return json.loads((REFERENCE / "engine_run.json").read_text(encoding="utf-8"))


def test_the_cleaned_stack_is_identical_to_the_engines(engine_run, tmp_path,
                                                       store_root):
    """Every pixel, not a tolerance.

    Both write uint16 counts, so there is no floating-point slack to allow for:
    the two stacks either are the same array or they are not.
    """
    result = matched_line.remove_cosmic_rays(
        REFERENCE / "synthetic_registered.ome.tif", output_dir=tmp_path / "out",
        signal_channel=engine_run["signal_channel_one_based"],
        threshold_sigma=engine_run["threshold_robust_sigma"],
        mask_growth_px=engine_run["mask_growth_px"],
        sample_frames=engine_run["sample_frames"],
        series=engine_run["series_zero_based"], write_preview=False)

    expected = tifffile.imread(REFERENCE / "engine_cleaned.tif")
    np.testing.assert_array_equal(tifffile.imread(result.path), expected)


def test_the_replacement_mask_is_identical_to_the_engines(engine_run, tmp_path,
                                                          store_root):
    """The mask is the record of what changed, so it has to match exactly too.

    The engine writes it as 0 or 255 in a TIFF; this package stores it packed
    as bits. Same pixels either way, which is what is compared.
    """
    result = matched_line.remove_cosmic_rays(
        REFERENCE / "synthetic_registered.ome.tif", output_dir=tmp_path / "out",
        signal_channel=engine_run["signal_channel_one_based"],
        threshold_sigma=engine_run["threshold_robust_sigma"],
        mask_growth_px=engine_run["mask_growth_px"],
        sample_frames=engine_run["sample_frames"],
        series=engine_run["series_zero_based"], write_preview=False)

    expected = tifffile.imread(REFERENCE / "engine_replacement_mask.tif")
    np.testing.assert_array_equal(result.mask.astype(np.uint8) * 255, expected)


def test_the_measured_noise_and_the_event_match_the_engines(engine_run,
                                                            tmp_path,
                                                            store_root):
    """The numbers that went into the decision, not only the decision.

    If the port replaced the same pixels for a different reason, the cleaned
    stack could still match on this one input and diverge on the next.
    """
    result = matched_line.remove_cosmic_rays(
        REFERENCE / "synthetic_registered.ome.tif", output_dir=tmp_path / "out",
        signal_channel=engine_run["signal_channel_one_based"],
        threshold_sigma=engine_run["threshold_robust_sigma"],
        mask_growth_px=engine_run["mask_growth_px"],
        sample_frames=engine_run["sample_frames"],
        series=engine_run["series_zero_based"], write_preview=False)

    assert (result.summary["pixel_frames_replaced"]
            == engine_run["pixel_frames_replaced"])
    assert result.summary["connected_events"] == engine_run["connected_events"]
    assert (result.summary["temporal_difference_centre"]
            == engine_run["temporal_difference_centre"])
    assert round(result.summary["temporal_sigma"], 4) == pytest.approx(
        engine_run["temporal_sigma"], abs=0)

    engine_event = _engine_events()[0]
    events = result.events
    assert events["frame_one_based"][0] == int(engine_event["frame_one_based"])
    assert events["y_peak"][0] == int(engine_event["y_peak"])
    assert events["x_peak"][0] == int(engine_event["x_peak"])
    assert events["grown_area_px"][0] == int(engine_event["grown_area_px"])
    assert (float(events["replacement_value"][0])
            == float(engine_event["replacement_value"]))


def _engine_events():
    import csv

    with (REFERENCE / "engine_events.csv").open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


# ------------------------------------------------ the reason it is still here
def test_the_name_a_record_calls_still_resolves():
    """``filtering.remove_cosmic_rays`` is this function and not a new one.

    Checked as an identity rather than by calling it: what a stored script asks
    for is that name, and the failure this guards against is somebody pointing
    it at the replacement, which would replay the run with a different method
    and report the same run id.
    """
    from pymicroglia import filtering

    assert filtering.remove_cosmic_rays is matched_line.remove_cosmic_rays
    assert (filtering.remove_cosmic_rays_in_place
            is matched_line.remove_cosmic_rays_in_place)
    assert filtering.remove_cosmic_rays.__module__.endswith("matched_line")


def test_a_script_written_before_the_replacement_still_runs(spiked_stack,
                                                            tmp_path,
                                                            store_root):
    """The promise executed, not asserted about.

    This is the shape of the equivalent script a record carried before
    2026-08-20: the import line, the module attribute, resolved arguments. It
    is run in a namespace of its own, the way replaying a record runs it.
    """
    source, _, _, _ = spiked_stack
    script = (
        "from pymicroglia import filtering\n"
        "filtering.remove_cosmic_rays(\n"
        f"    {str(source)!r},\n"
        f"    output_dir={str(tmp_path / 'replayed')!r},\n"
        "    signal_channel=2,\n"
        "    threshold_sigma=12.0,\n"
        "    mask_growth_px=0,\n"
        "    sample_frames=9,\n"
        "    write_preview=False,\n"
        ")\n"
    )
    exec(compile(script, "<record>", "exec"), {})  # noqa: S102 - the point

    written = sorted((tmp_path / "replayed").glob("*.tif"))
    assert written, "the replayed script produced no cleaned stack"
