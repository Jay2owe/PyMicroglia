"""Single-channel exports, and the reading and recording every export shares.

The two multi-channel ones are in ``composites.py``. Splitting them is not
bookkeeping: a composite has to decide what green means relative to red, and a
single-channel render has no such decision in it, so the two families read
differently and fail differently.

Each takes a stack somebody else produced and renders it. None of them writes
scientific pixels — that separation is the reason this stage exists. Two of the
engines these come from exported a registered TIFF *and* a movie from one call:

    before                                after
      red_only_video_export(...)            filtering.unmix(...)      -> artefact
        unmixes                             registration.apply(...)   -> artefact
        applies shifts                      video.red_only(...)       -> MP4
        writes a registered TIFF stack
        renders a red-on-black MP4        the caller composes the three

So ``unmixing_coefficient`` is still accepted here and still recorded in the
manifest — the movie has to be able to say what it was drawn from — but it is
never *applied*. ``filtering.unmix`` is a keyed measurement step, and doing it
twice in two places is how two tools stop agreeing about what a corrected pixel
is.

Every output is display-only. The MP4 keeps the name its engine gave it, so a
saved link still resolves; the *record* of it goes into the store as a
display-only artefact, which is what makes ``guards.require_measurement`` refuse
it if anything ever tries to measure from a movie.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from . import annotate as _annotate
from . import encode as _encode
from . import luts as _luts
from . import render as _render

__all__ = [
    "METHOD_VERSIONS",
    "VIDEO_STAGE",
    "stack_to_mp4",
    "red_only",
]

#: One per engine, carried into the record so a movie says which port made it.
METHOD_VERSIONS = {
    "stack_to_mp4": "2026-08-18",
    "red_only": "2026-07-23-timestamps-stacks",
    "timestamped_composite": "2026-07-23-composite-green-endpoint-correction",
    "phase_green_red": "2026-07-23-phase-green-red-display",
}

VIDEO_STAGE = "video"


# ------------------------------------------------------------------ reading
def _display_stack(path, signal_channel: int, first_frame: int, frames: int):
    """A plane series as ``(t, y, x)``, plus what the TIFF says about display.

    Any of the black point, white point or frame interval comes back as
    ``None`` when the file does not carry it, which is the caller's cue to fall
    back to an argument or to refuse rather than to invent one.
    """
    import numpy as np
    import tifffile

    with tifffile.TiffFile(path) as handle:
        meta = handle.imagej_metadata or {}
        pages = len(handle.pages)
        channels = 1
        try:
            series = handle.series[0]
            axes = getattr(series, "axes", "").upper()
            if "C" in axes:
                channels = int(series.shape[axes.index("C")])
        except (IndexError, AttributeError, ValueError):
            channels = 1
        channels = max(channels, 1)
        if signal_channel < 1 or signal_channel > channels:
            raise ValueError(f"signal_channel={signal_channel} outside "
                             f"1..{channels}")
        # Multi-channel files interleave planes, so every nth page is the same
        # channel. A single-channel file reads straight through.
        indices = list(range(signal_channel - 1, pages, channels))[first_frame:]
        if frames > 0:
            indices = indices[:frames]
        if not indices:
            raise ValueError("no frames selected")
        stack = np.stack([handle.pages[index].asarray() for index in indices])

    black = float(meta["min"]) if "min" in meta else None
    white = float(meta["max"]) if "max" in meta else None
    interval_h = None
    if meta.get("finterval"):
        unit = str(meta.get("tunit", "s")).lower()
        value = float(meta["finterval"])
        interval_h = value / 3600.0 if unit.startswith("s") else value
    return stack, black, white, interval_h


def _two_channel(path, first_frame: int = 0, frames: int = -1):
    """A ``TCYX`` two-channel uint16 stack and its ImageJ metadata.

    The metadata matters: a stack written by the registration export carries a
    ``Ranges`` field holding the display range each channel was chosen to be
    shown at, and a movie that recomputed the range from percentiles would draw
    a different contrast from every other view of the same file.
    """
    import numpy as np
    import tifffile

    with tifffile.TiffFile(path) as handle:
        array = np.asarray(handle.asarray())
        meta = dict(handle.imagej_metadata or {})
    if array.ndim != 4 or array.shape[1] < 2:
        raise ValueError(f"{Path(path).name}: expected a two-channel TCYX "
                         f"stack, found {array.shape}")
    end = None if frames is None or frames < 0 else first_frame + frames
    return array[first_frame:end], meta


def _stored_ranges(meta: dict[str, Any]) -> tuple[float, ...] | None:
    """The four display bounds a saved hyperstack carries, or ``None``.

    ImageJ stores them as a flat ``(min, max)`` pair per channel. Two channels
    means four numbers; anything shorter is a file that does not carry them.
    """
    values = meta.get("Ranges")
    if not isinstance(values, (tuple, list)) or len(values) < 4:
        return None
    return tuple(float(value) for value in values[:4])


def _record(source, name: str, path: Path, payload: dict[str, Any],
            *, output_dir: Path, method_version: str) -> Any:
    """The movie's own record, stored display-only so nothing can measure it.

    The MP4 itself is not an artefact — it is 14 MB of pixels nobody will diff.
    What is worth keeping keyed is the row that says which stack it came from,
    at what display range, at what frame rate, under which encoder. That row is
    marked ``display_only``, so a measurement handed it raises rather than
    quietly treating a movie as data.
    """
    from .. import store

    return store.put(
        VIDEO_STAGE, source, {"video": name, **_keyable(payload)},
        kind="scalars", value={**payload, "video_file": str(path),
                               "display_only": True},
        name=f"{name}_video", output_dir=output_dir,
        method_version=method_version, display_only=True)


def _keyable(payload: dict[str, Any]) -> dict[str, Any]:
    """The settings that decide what the movie looks like, for the cache key.

    Deliberately not the whole payload: the encoder's ffmpeg version and the
    output byte count belong in the record but must not change the key, or a
    toolchain upgrade would silently invalidate every stored movie.
    """
    keys = ("fps", "crf", "lut", "hours_per_second", "frame_interval_h",
            "black", "white", "display_max", "profile", "gain")
    return {key: payload[key] for key in keys if key in payload}


def _default_output_dir(source, suffix: str = "_videos") -> Path:
    path = Path(source).resolve()
    for parent in (path.parent, *path.parents):
        if parent.name.lower() == "ai_exports":
            return parent / f"{path.stem}{suffix}"
    return path.parent / "AI_Exports" / f"{path.stem}{suffix}"


# ------------------------------------------------------------ stack_to_mp4
def stack_to_mp4(source, *, output_dir=None, output_name=None,
                 overwrite: bool = False, input_glob: str = "*.tif",
                 hours_per_second: float = 12.0, frame_interval_h: float = 0.0,
                 lut: str = "dluc_purple", black_point: float = -1.0,
                 white_point: float = -1.0, timestamp: int = 1,
                 timestamp_band_px: int = 48, label: str = "",
                 smooth_frames: int = 0, smooth_sigma_px: float = 0.0,
                 floor_off_tissue_pct: float = 0.0, crf: int = 14,
                 signal_channel: int = 1, first_frame: int = 0,
                 frames: int = -1, max_fps: float = 60.0,
                 min_fps: float = 1.0, glob: str = "*.tif",
                 python: str = "python",
                 python_engine: str = "") -> dict[str, Any]:
    """Render a TIFF time-lapse as an MP4 at a stated experimental speed.

    The general renderer, and the one the display stage's output is meant for:
    that stage writes the black point, the white point and the frame interval
    into the TIFF, so this draws exactly the contrast it chose without being
    told again.

    ``glob``, ``python`` and ``python_engine`` come from the Fiji front end's
    parameter block and are accepted so a saved macro call still validates.
    They are recorded and ignored: this package does not launch an interpreter.
    """
    source = Path(source)
    stack, stored_black, stored_white, stored_interval = _display_stack(
        source, int(signal_channel), int(first_frame), int(frames))
    stack = _render.smooth_for_display(stack, int(smooth_frames),
                                       float(smooth_sigma_px))

    off_tissue_fraction = None
    black = black_point if black_point >= 0 else stored_black
    if floor_off_tissue_pct > 0:
        # Measured after smoothing, so the floor sits against the field as it
        # will actually be drawn rather than against the unsmoothed noise.
        black, off_tissue_fraction = _render.off_tissue_floor(
            stack, float(floor_off_tissue_pct))
    white = white_point if white_point >= 0 else stored_white
    if black is None or white is None:
        raise ValueError(f"{source.name}: the TIFF stores no display range; "
                         f"give black_point and white_point.")
    if white <= black:
        raise ValueError(f"white point {white} must exceed black point {black}")

    interval_h = frame_interval_h if frame_interval_h > 0 else stored_interval
    if not interval_h or interval_h <= 0:
        raise ValueError(f"{source.name}: the TIFF stores no frame interval; "
                         f"give frame_interval_h.")
    fps = _encode.check_frame_rate(
        _encode.frame_rate(interval_h, hours_per_second),
        minimum=float(min_fps), maximum=float(max_fps),
        what=f"{source.name} at {hours_per_second:g} h/s on a "
             f"{interval_h:g} h frame interval")

    folder = Path(output_dir) if output_dir else source.parent
    stem = str(output_name) if output_name else \
        f"{source.stem}_{lut}_{hours_per_second:g}hps"
    target = folder / f"{stem}.mp4"

    screen = _render.scale_to_screen(stack, black, white)

    def painted():
        for index in range(screen.shape[0]):
            rgb = _luts.paint(screen[index], lut)
            # The caption goes on before the band, so it sits inside the image
            # the way the review videos have it rather than in the margin.
            if label:
                rgb = _annotate.corner_label(rgb, label)
            if timestamp:
                rgb = _annotate.timestamp_band(
                    rgb, frame_index=int(first_frame) + index,
                    interval_h=interval_h, band_height=int(timestamp_band_px))
            yield _annotate.pad_to_even(rgb)

    _encode.write_video(target, painted(), fps=fps, profile="grainy_display",
                        crf=int(crf), overwrite=overwrite)

    count, height, width = stack.shape
    report = {
        "display_only": True,
        "input": str(source), "output": str(target),
        "frames": int(count), "height": int(height), "width": int(width),
        "signal_channel": int(signal_channel),
        "hours_per_second": float(hours_per_second),
        "frame_interval_h": float(interval_h),
        "frame_interval_source": "argument" if frame_interval_h > 0 else "tiff",
        "lut": lut, "crf": int(crf), "profile": "grainy_display",
        "timestamp": bool(timestamp),
        "timestamp_band_px": int(timestamp_band_px), "label": label,
        "smooth_frames": int(smooth_frames),
        "smooth_sigma_px": float(smooth_sigma_px),
        "floor_off_tissue_pct": float(floor_off_tissue_pct),
        "black": float(black), "white": float(white),
        "display_range_source": (
            "off_tissue_percentile" if floor_off_tissue_pct > 0
            else "argument" if black_point >= 0 else "tiff"),
        "off_tissue_fraction_of_frame": off_tissue_fraction,
        "fps": float(fps),
        "record_hours": float(count * interval_h),
        "playback_seconds": float(count / fps),
        "seconds_per_biological_day": float(24.0 / hours_per_second),
        "ffmpeg": _encode.ffmpeg_version(),
        "ignored_wrapper_settings": {"glob": glob, "python": python,
                                     "python_engine": python_engine,
                                     "input_glob": input_glob},
    }
    if timestamp:
        report["remove_timestamp_band"] = (
            f"ffmpeg -i in.mp4 -vf crop=iw:ih-{int(timestamp_band_px)}:0:"
            f"{int(timestamp_band_px)} out.mp4")
    _record(source, "stack_to_mp4", target, report, output_dir=folder,
            method_version=METHOD_VERSIONS["stack_to_mp4"])
    return report


# --------------------------------------------------------------- red only
def red_only(source, *, output_dir=None, output_name=None,
             overwrite: bool = False, input_glob: str = "*.tif",
             unmixing_coefficient: float = 0.025, fps: float = 10.0,
             frame_interval_minutes: float = 0.0,
             time_label_band_height: int = 48,
             display_percentile: float = 99.9, crf: int = 18,
             file_lock_retry_seconds: float = 60.0,
             timestamp_font_fallback: str = "DejaVuSans-Bold.ttf",
             timestamp_font_height_fraction: float = 0.54,
             timestamp_font_min_size: int = 14) -> dict[str, Any]:
    """The unmixed red channel of a registered stack, red on black.

    ``source`` is the registered two-channel stack whose C2 is already the
    unmixed mCherry — the engine's own README says C2 "is the direct source of
    each MP4". ``unmixing_coefficient`` is recorded so the movie can say what it
    was drawn from, and is never applied: unmixing is ``filtering.unmix``'s, and
    doing it in two places is how two tools stop agreeing.
    """
    import numpy as np

    source = Path(source)
    stack, _ = _two_channel(source)
    corrected = stack[:, 1]
    display_max = max(float(np.percentile(corrected, display_percentile)), 1.0)

    folder = Path(output_dir) if output_dir else _default_output_dir(source)
    suffix = (f"_timestamped_{frame_interval_minutes:g}min"
              if frame_interval_minutes > 0 else "")
    stem = str(output_name) if output_name else \
        f"{source.stem}_red_only{suffix}"
    target = folder / f"{stem}.mp4"

    height, width = corrected.shape[1:]
    band = int(time_label_band_height) if frame_interval_minutes > 0 else 0
    canvas_height = height + band
    padded = (canvas_height + canvas_height % 2, width + width % 2)

    def painted():
        for index, frame in enumerate(corrected):
            intensity = _render.linear_uint8(frame, display_max)
            rgb = np.zeros((height, width, 3), np.uint8)
            rgb[:, :, 0] = intensity
            if band:
                rgb = _annotate.timestamp_band(
                    rgb, frame_index=index,
                    interval_h=float(frame_interval_minutes) / 60.0,
                    band_height=band)
            yield _annotate.pad_into(rgb, *padded)

    _encode.write_video(target, painted(), fps=float(fps), profile="standard",
                        crf=int(crf), overwrite=overwrite)

    report = {
        "display_only": True,
        "input": str(source), "output": str(target),
        "frames": int(corrected.shape[0]),
        "height": int(padded[0]), "width": int(padded[1]),
        "red_lut": "linear red on black",
        "display_min": 0.0, "display_max": float(display_max),
        "display_percentile": float(display_percentile),
        "fps": float(fps), "crf": int(crf), "profile": "standard",
        "frame_interval_minutes": float(frame_interval_minutes),
        "elapsed_time_first_frame": _annotate.elapsed_label(0, 0.0),
        "elapsed_time_last_frame": _annotate.elapsed_label(
            int(corrected.shape[0]) - 1, float(frame_interval_minutes) / 60.0),
        "timing_calibrated": frame_interval_minutes > 0,
        "unmix_formula": f"max(C2 - {unmixing_coefficient:g}*C1, 0)",
        "unmix_applied_here": False,
        "unmix_note": ("recorded from filtering.unmix; this action renders C2 "
                       "as it found it and never unmixes"),
        "ffmpeg": _encode.ffmpeg_version(),
        "ignored_wrapper_settings": {
            "input_glob": input_glob,
            "file_lock_retry_seconds": file_lock_retry_seconds,
            "timestamp_font_fallback": timestamp_font_fallback,
            "timestamp_font_height_fraction": timestamp_font_height_fraction,
            "timestamp_font_min_size": timestamp_font_min_size},
    }
    _record(source, "red_only", target, report, output_dir=folder,
            method_version=METHOD_VERSIONS["red_only"])
    return report
