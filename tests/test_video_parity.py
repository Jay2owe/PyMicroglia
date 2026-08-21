"""Do the ported renderers draw the frames the engines drew?

Stage 10's exit gate 1, and it needs a method, because a movie cannot be
compared the way a TIFF can. `ffmpeg` output is not bit-reproducible across
versions, so two correct renders of the same frames produce different bytes.
Comparing file hashes would fail on a toolchain upgrade and pass on a broken
port; comparing decoded frames with a loose tolerance would pass on either.

So the comparison separates the two sources of difference:

    my RGB   vs  my own decode      -> compression error alone
    my RGB   vs  the engine's decode -> compression error + any render error

If those two numbers agree, the render is exact and everything left is the
encoder. On the validated recording they agree to three decimal places — 2.760
against 2.758 mean absolute levels out of 255 — which is why the assertion here
is on their *difference* rather than on either one's size.

Container geometry is compared exactly: resolution, frame count and frame rate
are decisions, not approximations.

No engine is executed. The references are movies the engines wrote in July and
August 2026, beside the stacks they were rendered from. Each test skips cleanly
when its reference is not on this machine, the same rule
``test_display_parity.py`` follows, and every one of them is a Dropbox
online-only placeholder until somebody makes it available offline.
"""

from __future__ import annotations

import json
import os
import stat as stat_module
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("imageio_ffmpeg")

from pymicroglia import video
from pymicroglia.video import annotate, encode, exports, luts, render

EXPERIMENTS = Path("reference-data") / "video"

#: The bioluminescence recording used everywhere in this project as the
#: reference: 241 frames, 504 x 504, half-hourly, rendered at 12 experimental
#: hours per second and so at 24 fps.
STACK_TO_MP4_STEM = (
    "20260710_Tmem_Cry_BSL_leaktest_1713_Multichannel Time Lapse_"
    "20260721_1417.ome_registered_translation_cosmic_cleaned_display"
    "_DISPLAY_ONLY")

#: One two-channel recording feeds both the red-only and the composite movie,
#: which is what makes the pair worth testing together: the same stack drawn
#: two ways, and only one of them corrects the green channel.
TWO_CHANNEL_STEM = ("VID95_A1_green-red_timestack_registered_cropped_unmixed"
                    "_2.5pct_both_channels")

_OFFLINE = getattr(stat_module, "FILE_ATTRIBUTE_OFFLINE", 0x1000)
_RECALL_ON_ACCESS = 0x00400000

#: The whole point of the two-way comparison above: a render difference shows
#: up as a *gap* between the two means, and compression shows up in both
#: equally. A tenth of a level out of 255 is far below anything a wrong display
#: range, a shifted band or a mis-rounded lookup table could produce.
RENDER_TOLERANCE = 0.10


def is_placeholder(path: Path) -> bool:
    attributes = getattr(path.stat(), "st_file_attributes", 0)
    return bool(attributes & (_OFFLINE | _RECALL_ON_ACCESS))


def local(path: Path, what: str) -> Path:
    root = Path(os.environ.get("PYMICROGLIA_EXPERIMENTS", EXPERIMENTS))
    target = root / path if not path.is_absolute() else path
    if not target.exists():
        pytest.skip(f"{what} is not here: {target.name}. Set "
                    f"PYMICROGLIA_EXPERIMENTS to the folder holding these "
                    f"recordings to close this gate.")
    if is_placeholder(target):
        pytest.skip(
            f"{target.name} is a Dropbox online-only placeholder "
            f"({target.stat().st_size / 1024 ** 2:.0f} MB). Reading it would "
            f"download it. Make it available offline and run this again.")
    return target


def decode(path, indices) -> tuple[dict[int, Any], dict]:
    """Specific frames of a movie, plus what the container says about itself."""
    import imageio.v2 as imageio

    wanted, found = set(indices), {}
    reader = imageio.get_reader(str(path))
    try:
        meta = dict(reader.get_meta_data())
        for position, frame in enumerate(reader):
            if position in wanted:
                found[position] = np.asarray(frame)
            if len(found) == len(wanted):
                break
    finally:
        reader.close()
    return found, meta


def mean_levels(a, b) -> float:
    """Mean absolute difference in 8-bit levels."""
    return float(np.abs(np.asarray(a, np.int16)
                        - np.asarray(b, np.int16)).mean())


# --------------------------------------------- the contract, without any file
def test_the_frame_rate_comes_from_hours_per_second():
    """Gate 3. The control people reason about; fps is derived, never set."""
    assert encode.frame_rate(0.5, 6.0) == 12.0
    assert encode.frame_rate(0.5, 12.0) == 24.0
    assert encode.frame_rate(1.0 / 120.0, 0.5) == 60.0

    with pytest.raises(ValueError) as raised:
        encode.frame_rate(0.0, 12.0)
    assert "frame_interval_h" in str(raised.value)

    with pytest.raises(ValueError) as raised:
        encode.check_frame_rate(encode.frame_rate(0.01, 12.0))
    assert "wrong speed, not a fast movie" in str(raised.value)


def test_no_function_under_video_writes_a_tiff():
    """Gate 4. Rendering is not producing scientific pixels.

    Two of the engines exported a registered stack and a movie from one call.
    That is the conflation this stage exists to undo, so the check is on the
    code rather than on a convention.
    """
    import ast

    root = Path(video.__file__).parent
    offenders = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in (
                    "imwrite", "imsave", "memmap"):
                offenders.append(f"{path.name}:{node.lineno} {node.attr}")
    assert not offenders, offenders


def test_nothing_here_shells_out_to_ffmpeg():
    """Three of the four engines did. A PATH nobody recorded is not a dependency.

    ``imageio-ffmpeg`` ships its own binary instead, which is what lets
    ``doctor`` say whether video export will work before a run starts rather
    than after it.
    """
    import ast

    root = Path(video.__file__).parent
    offenders = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            if {"subprocess", "runpy"} & set(names):
                offenders.append(f"{path.name}:{node.lineno} imports {names}")
    assert not offenders, offenders


def test_the_two_purple_maps_are_kept_apart():
    """Gate 7, and the finding behind it.

    ``tiff_stack_to_mp4.py`` and ``cry1_dluc_photon_pipeline.py`` each define a
    map called ``dluc_purple`` and they are not the same map — five stops
    against three. Merging them would silently repaint one of the two sets of
    movies, so they are named apart instead.
    """
    assert luts.LUT_ANCHORS["dluc_purple"] == [
        "#000000", "#260050", "#7c22b7", "#d47cff", "#ffffff"]
    assert luts.LUT_ANCHORS["dluc_purple_photon"] == [
        (0.0, 0.0, 0.0), (0.20, 0.0, 0.35), (0.65, 0.10, 1.0)]
    assert luts.LUT_ANCHORS["dluc_purple"] != luts.LUT_ANCHORS["dluc_purple_photon"]

    black = luts.paint(np.zeros((2, 2), np.float32), "dluc_purple")
    white = luts.paint(np.ones((2, 2), np.float32), "dluc_purple")
    assert black.tolist() == [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]
    assert white.min() == 255


def test_a_ramp_and_an_anchored_map_round_differently_on_purpose():
    """The two branches disagree by a level, and both are somebody's movies."""
    half = np.full((1, 1), 0.5, np.float32)
    assert luts.paint(half, "red")[0, 0, 0] == 128       # rounded
    assert luts.paint(half, "dluc_purple")[0, 0].tolist() != [0, 0, 0]


def test_the_elapsed_labels_match_the_engines_wording():
    assert annotate.elapsed_label(0, 0.5) == "Time: 0 h 00 min"
    assert annotate.elapsed_label(7, 0.5) == "Time: 3 h 30 min"
    assert annotate.elapsed_label(100, 0.5) == "Time: 50 h 00 min"
    assert annotate.clock_label(0, 30.0) == "Time: 00:00:00"
    assert annotate.clock_label(1017, 30.0) == "Time: 08:28:30"


def test_a_movie_records_itself_as_display_only(tmp_path, monkeypatch):
    """Gate 5. A measurement handed a movie's record raises rather than reads it."""
    monkeypatch.setenv("PYMICROGLIA_STORE", str(tmp_path / "cache"))
    monkeypatch.setenv("PYMICROGLIA_DECISIONS", str(tmp_path / "decisions"))

    from tests_support import two_channel_stack

    from pymicroglia import guards, store

    source = two_channel_stack(tmp_path, frames=6)
    result = video.red_only(source, output_dir=tmp_path / "out",
                            overwrite=True, frame_interval_minutes=30.0)
    assert result["display_only"] is True
    assert Path(result["output"]).is_file()

    found = store.resolve(exports.VIDEO_STAGE, source, required=True,
                          display_only=True)
    assert guards.is_display_only(found)
    with pytest.raises(guards.DisplayOnlyInput):
        guards.require_measurement(found)


def test_there_is_no_way_to_render_a_measurement_from_a_movie():
    """No keyword anywhere here turns the display-only mark off."""
    import inspect

    escapes = {"force", "as_measurement", "display_only", "skip_guard",
               "allow_measurement"}
    for name in ("stack_to_mp4", "red_only", "timestamped_composite",
                 "phase_green_red"):
        parameters = set(inspect.signature(getattr(video, name)).parameters)
        assert not parameters & escapes, f"{name} grew {parameters & escapes}"


def test_it_refuses_to_replace_a_movie_unless_told_to(tmp_path, monkeypatch):
    monkeypatch.setenv("PYMICROGLIA_STORE", str(tmp_path / "cache"))
    monkeypatch.setenv("PYMICROGLIA_DECISIONS", str(tmp_path / "decisions"))

    from tests_support import two_channel_stack

    source = two_channel_stack(tmp_path, frames=4)
    video.red_only(source, output_dir=tmp_path / "out")
    with pytest.raises(FileExistsError):
        video.red_only(source, output_dir=tmp_path / "out")


def test_doctor_says_whether_video_export_will_work():
    """Better learnt before a six-hour run than at the end of one."""
    from pymicroglia import knowledge

    report = knowledge.doctor()
    assert "video_export_available" in report
    assert report["video_export_available"] is encode.available()


# ------------------------------------------------------------- stack_to_mp4
@pytest.fixture(scope="module")
def bioluminescence():
    folder = (EXPERIMENTS / "Tmem119-CreERT2/Tests/Cry1-DIO-dLuc/AI_Exports"
              / "Tmem_Cry_leaktest_magenta_display_2026-08-17")
    return {
        "stack": local(folder / "display" / f"{STACK_TO_MP4_STEM}.tif",
                       "the display stack"),
        "movie": local(folder / "videos"
                       / f"{STACK_TO_MP4_STEM}_dluc_purple_12hps.mp4",
                       "the engine's movie"),
        "report": json.loads(local(
            folder / "videos"
            / f"{STACK_TO_MP4_STEM}_dluc_purple_12hps_report.json",
            "the engine's report").read_text(encoding="utf-8")),
    }


def test_the_purple_movie_matches_the_engines(tmp_path, bioluminescence):
    """Gate 1 for ``tiff_stack_to_mp4``, and the method the docstring describes.

    24 frames rather than 241: the render has no per-frame state at these
    settings, so a longer window tests the same arithmetic again at the cost of
    ten times the encode.
    """
    import tifffile

    settings = bioluminescence["report"]["settings"]
    ranges = bioluminescence["report"]["display_range"]
    picks = [0, 12, 23]

    result = video.stack_to_mp4(
        bioluminescence["stack"], output_dir=tmp_path, output_name="mine",
        overwrite=True, frames=24, lut=settings["lut"], crf=settings["crf"],
        hours_per_second=settings["hours_per_second"],
        timestamp_band_px=settings["timestamp_band_px"])

    # the display range and the frame rate came from the file, as the engine's did
    assert result["black"] == ranges["black"]
    assert result["white"] == ranges["white"]
    assert result["fps"] == bioluminescence["report"]["playback"]["fps"]
    assert result["frame_interval_source"] == "tiff"

    with tifffile.TiffFile(bioluminescence["stack"]) as handle:
        planes = {p: handle.pages[p].asarray() for p in picks}
        meta = handle.imagej_metadata or {}
    interval = float(meta["finterval"]) / 3600.0

    mine, mine_meta = decode(Path(result["output"]), picks)
    theirs, their_meta = decode(bioluminescence["movie"], picks)

    assert tuple(mine_meta["size"]) == tuple(their_meta["size"])
    assert mine_meta["fps"] == their_meta["fps"] == 24.0

    for index in picks:
        screen = render.scale_to_screen(planes[index][None], ranges["black"],
                                        ranges["white"])[0]
        rgb = annotate.pad_to_even(annotate.timestamp_band(
            luts.paint(screen, "dluc_purple"), frame_index=index,
            interval_h=interval,
            band_height=settings["timestamp_band_px"]))

        compression = mean_levels(rgb, mine[index])
        against_engine = mean_levels(rgb, theirs[index])
        assert abs(compression - against_engine) < RENDER_TOLERANCE, (
            f"frame {index}: my render differs from the engine's by more than "
            f"compression explains — {compression:.3f} against my own encode, "
            f"{against_engine:.3f} against theirs")


def test_the_container_matches_frame_for_frame(bioluminescence):
    """Resolution, frame count and duration are decisions, not approximations."""
    report = bioluminescence["report"]
    _, meta = decode(bioluminescence["movie"], [0])
    assert tuple(meta["size"]) == (report["width"],
                                   report["height"] + 48)
    assert meta["fps"] == report["playback"]["fps"]
    assert encode.frame_rate(report["settings"]["frame_interval_h"],
                             report["settings"]["hours_per_second"]) == meta["fps"]


# ------------------------------------------------------- the two-channel pair
@pytest.fixture(scope="module")
def two_channel():
    base = (EXPERIMENTS / "Tmem119-CreERT2/Tests/CD68-DIO-mCherry with HCQ"
            / "AI_Exports/registered_unmixed_2026-07-23")
    return {
        "stack": local(base / "timestamped_red_only_2p5pct_30min"
                       / "Registered_two_channel_stacks"
                       / f"{TWO_CHANNEL_STEM}.tif", "the registered stack"),
        "red_only": local(
            base / "timestamped_red_only_2p5pct_30min"
            / ("VID95_A1_green-red_timestack_registered_cropped_unmixed_2.5pct"
               "_red_only_timestamped_30min.mp4"),
            "the engine's red-only movie"),
        "composite": local(
            base / "timestamped_red_green_2p5pct_green_endpoint_corrected_30min"
            / f"{TWO_CHANNEL_STEM}_red-green_timestamped_30min"
              "_green_endpoint_corrected.mp4",
            "the engine's composite movie"),
    }


def test_the_red_only_movie_matches_the_engines(tmp_path, two_channel):
    """Gate 1 for ``microglia_red_only_video_export``.

    Also the separation this stage is for: the stack handed in already holds
    the unmixed channel, and this action does not unmix. The manifest still
    records the coefficient, because a movie has to be able to say what it was
    drawn from.
    """
    stack, _ = exports._two_channel(two_channel["stack"])
    corrected = stack[:, 1]
    picks = [0, corrected.shape[0] // 2, corrected.shape[0] - 1]

    result = video.red_only(two_channel["stack"], output_dir=tmp_path,
                            output_name="mine", overwrite=True,
                            frame_interval_minutes=30.0, fps=10.0)
    assert result["display_max"] == 198.0        # the manifest's value exactly
    assert result["unmix_applied_here"] is False
    assert result["unmix_formula"] == "max(C2 - 0.025*C1, 0)"

    mine, mine_meta = decode(Path(result["output"]), picks)
    theirs, their_meta = decode(two_channel["red_only"], picks)
    assert tuple(mine_meta["size"]) == tuple(their_meta["size"]) == (644, 676)
    assert mine_meta["fps"] == their_meta["fps"] == 10.0

    height, width = corrected.shape[1:]
    for index in picks:
        rgb = np.zeros((height, width, 3), np.uint8)
        rgb[:, :, 0] = render.linear_uint8(corrected[index], 198.0)
        rgb = annotate.pad_into(annotate.timestamp_band(
            rgb, frame_index=index, interval_h=0.5, band_height=48),
            (height + 48) + (height + 48) % 2, width + width % 2)
        assert abs(mean_levels(rgb, mine[index])
                   - mean_levels(rgb, theirs[index])) < RENDER_TOLERANCE


def test_the_composite_reads_the_display_range_the_stack_carries(tmp_path,
                                                                 two_channel):
    """Gate 1 for ``microglia_timestamped_composite_video_export``.

    The stack carries the display range it was registered at, in its ImageJ
    ``Ranges`` metadata. Recomputing it from percentiles instead produced a
    green channel a few percent off and a mean difference six times larger —
    the single change that took this movie from close to exact.
    """
    stack, meta = exports._two_channel(two_channel["stack"])
    stored = exports._stored_ranges(meta)
    assert stored is not None, "the reference stack should carry Ranges"

    green, red = stack[:, 0], stack[:, 1]
    picks = [0, green.shape[0] // 2, green.shape[0] - 1]

    result = video.timestamped_composite(
        two_channel["stack"], output_dir=tmp_path, output_name="mine",
        overwrite=True, frame_interval_minutes=30.0, fps=10.0)

    # every number the engine's manifest quotes, reproduced
    assert result["display_range_source"] == "tiff"
    assert result["green_mask_pixels"] == 93447
    assert result["green_start_median"] == 1043.0
    assert result["green_end_median_before"] == 536.0
    assert round(result["green_end_gain"], 12) == 1.945895522388
    assert result["green_display_min"] == pytest.approx(316.592542725)
    assert result["green_display_max"] == pytest.approx(1187.926740234)
    assert result["red_display_min"] == 0.0
    assert result["red_display_max"] == 198.0
    assert result["endpoint_relative_error"] == pytest.approx(0.0, abs=1e-9)

    mask = render.stable_tissue_mask(green)
    _, gains, _ = render.endpoint_gain_curve(green, mask)
    mine, mine_meta = decode(Path(result["output"]), picks)
    theirs, their_meta = decode(two_channel["composite"], picks)
    assert tuple(mine_meta["size"]) == tuple(their_meta["size"])
    assert mine_meta["fps"] == their_meta["fps"] == 10.0

    height, width = green.shape[1:]
    for index in picks:
        rgb = render.rgb_composite(green[index], red[index],
                                   gain=float(gains[index]), ranges=stored)
        rgb = annotate.pad_into(annotate.timestamp_band(
            rgb, frame_index=index, interval_h=0.5, band_height=48),
            (height + 48) + (height + 48) % 2, width + width % 2)
        assert abs(mean_levels(rgb, mine[index])
                   - mean_levels(rgb, theirs[index])) < RENDER_TOLERANCE


def test_the_endpoint_gain_is_display_only_and_says_so(two_channel):
    """The most consequential thing in this stage that looks like a measurement."""
    stack, _ = exports._two_channel(two_channel["stack"])
    green = stack[:, 0]
    mask = render.stable_tissue_mask(green)
    before, gains, after = render.endpoint_gain_curve(green, mask)

    assert gains[0] == 1.0, "the curve must start at exactly no correction"
    assert after[0] == pytest.approx(after[-1]), (
        "the whole point is that the corrected endpoints match")
    assert np.all(np.diff(gains) >= 0) or np.all(np.diff(gains) <= 0), (
        "a log-linear ramp is monotonic; a step would be a bug")
    assert "display-only" in render.__doc__.lower()
    assert "never reach a trace" in render.endpoint_gain_curve.__doc__


# ------------------------------------------------------------ organotypic
@pytest.fixture(scope="module")
def organotypic():
    folder = (EXPERIMENTS / "Syn-RCamp + hIba1a/11+/AI_Exports"
              / "VID52_B6_registered_30s")
    return {
        "stack": local(
            folder / "VID52_B6_phase-green-red_timestack_registered"
                     "_red_neuronal_translation.tif", "the registered stack"),
        "manifest": json.loads(local(folder / "video_manifest.json",
                                     "the engine's manifest")
                               .read_text(encoding="utf-8")),
        "folder": folder,
    }


def test_the_organotypic_numbers_match_the_engines(organotypic):
    """Gate 1 for ``phase_green_red_video_export``, on its numbers.

    Its videos are 1,018 frames each and the engine renders four; re-rendering
    them to compare pixels costs an evening. What decides whether the port is
    right is the display geometry — the gain, the mask, the three ranges — and
    those are in the manifest to the twelfth decimal.
    """
    import tifffile

    manifest = organotypic["manifest"]
    with tifffile.TiffFile(organotypic["stack"]) as handle:
        series = handle.series[0]
        count, channels = int(series.shape[0]), int(series.shape[1])
        sampled = np.unique(np.linspace(0, count - 1, 41).round().astype(int))
        green_samples = np.stack([
            handle.pages[int(i) * channels + 1].asarray() for i in sampled])
        red_samples = np.stack([
            handle.pages[int(i) * channels + 2].asarray() for i in sampled])
        mask = render.largest_tissue_mask(np.median(green_samples, axis=0),
                                          clip=False)
        metrics = np.array([
            np.percentile(handle.pages[frame * channels + 1].asarray()[mask], 75)
            for frame in range(count)])

    window = min(20, max(3, count // 10))
    start = float(np.median(metrics[:window]))
    end = float(np.median(metrics[-window:]))
    gain = start / end

    assert start == manifest["start_green_metric"]
    assert end == manifest["end_green_metric"]
    assert gain == pytest.approx(manifest["end_display_gain"], rel=1e-12)

    gains = np.exp(np.linspace(0.0, np.log(gain), count))
    green_display = np.concatenate([
        green_samples[position][mask].astype(np.float32) * gains[frame]
        for position, frame in enumerate(sampled)])
    red_display = red_samples[:, mask].astype(np.float32).ravel()
    detail_display = np.concatenate([
        render.local_contrast(green_samples[position].astype(np.float32)
                              * gains[frame], 8.0)[mask]
        for position, frame in enumerate(sampled)])

    assert list(np.percentile(green_display, (0.5, 99.7))) == pytest.approx(
        manifest["green_display_range"], rel=1e-9)
    assert list(np.percentile(red_display, (0.5, 99.7))) == pytest.approx(
        manifest["red_display_range"], rel=1e-9)
    assert list(np.percentile(detail_display, (70.0, 99.8))) == pytest.approx(
        manifest["green_local_contrast_display_range"], rel=1e-9)


def test_the_organotypic_geometry_matches(organotypic):
    """Resolution and frame count, from the manifest the engine wrote."""
    for entry in organotypic["manifest"]["videos"]:
        if entry["file"] == "registration_before_after.mp4":
            continue            # side by side, and not rendered by this action
        assert entry["width"] == 608
        assert entry["height"] == 408
        assert entry["nb_read_frames"] == "1018"
        assert entry["r_frame_rate"] == "10/1"
        assert entry["pix_fmt"] == "yuv420p"


def test_the_four_organotypic_videos_are_the_ones_the_engine_wrote(organotypic):
    from pymicroglia.video import composites

    expected = {entry["file"][:-4] for entry in organotypic["manifest"]["videos"]
                if entry["file"] != "registration_before_after.mp4"}
    assert {name for name, _, _ in composites.PHASE_GREEN_RED_VIDEOS} == expected


# ------------------------------------------------------------ synthetic run
def test_a_short_synthetic_render_round_trips(tmp_path, monkeypatch):
    """The whole path with no reference folder: render, encode, decode, read back.

    Runs everywhere, so a machine without the recordings still exercises the
    encoder, the band, the padding and the record.
    """
    monkeypatch.setenv("PYMICROGLIA_STORE", str(tmp_path / "cache"))
    monkeypatch.setenv("PYMICROGLIA_DECISIONS", str(tmp_path / "decisions"))

    from tests_support import two_channel_stack

    source = two_channel_stack(tmp_path, frames=8, height=64, width=63)
    result = video.timestamped_composite(source, output_dir=tmp_path / "out",
                                         overwrite=True, fps=4.0,
                                         frame_interval_minutes=30.0)
    assert result["frames"] == 8
    # 63 is odd, and yuv420p halves both dimensions
    assert result["width"] % 2 == 0
    assert result["height"] % 2 == 0

    frames, meta = decode(Path(result["output"]), [0, 7])
    assert meta["fps"] == 4.0
    assert tuple(meta["size"]) == (result["width"], result["height"])
    assert frames[0].shape[:2] == (result["height"], result["width"])
    assert result["first_time_label"] == "Time: 0 h 00 min"
    assert result["last_time_label"] == "Time: 3 h 30 min"
