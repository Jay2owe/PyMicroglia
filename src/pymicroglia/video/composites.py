"""The two multi-channel exports: green over red, and the organotypic route.

Both have to decide what one channel means relative to another, which is where
a display correction becomes tempting and where it has to be labelled loudest.
Both of these carry one: green fades over a long recording, and a viewer shown
an uncorrected green channel reads bleaching as a biological decline.

The correction is a smooth multiplicative ramp fitted so the start and end
brightness inside one fixed tissue mask match. It is display-only, it never
touches a saved stack, and it must never reach a trace. The engines' own
manifests call it "display-only multiplicative log-linear endpoint correction",
and that wording is kept.

The two measure their endpoints differently and both ways are kept:

* the composite exporter takes the **median** of the first and last frame, on a
  101-frame recording where one frame is a fair sample;
* the organotypic exporter takes the **75th percentile** over a window at each
  end, on a 1,018-frame recording where one frame is not.

Making them agree would move one set of movies for no reason beyond tidiness.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from . import annotate as _annotate
from . import encode as _encode
from . import render as _render
from .exports import (METHOD_VERSIONS, _default_output_dir, _record,
                      _stored_ranges, _two_channel)

__all__ = ["timestamped_composite", "phase_green_red",
           "PHASE_GREEN_RED_VIDEOS"]


# ------------------------------------------------------ timestamped composite
def timestamped_composite(source, *, output_dir=None, output_name=None,
                          overwrite: bool = False, input_glob: str = "*.tif",
                          fps: float = 10.0,
                          frame_interval_minutes: float = 30.0,
                          time_label_band_height: int = 48, crf: int = 18,
                          mask_sample_frames: int = 11,
                          mask_erosion_iterations: int = 3,
                          mask_min_pixels: int = 256,
                          mask_min_area_fraction: float = 0.01,
                          green_display_percentiles: Sequence[float] = (1.0, 99.9),
                          red_display_percentiles: Sequence[float] = (0.0, 99.9)
                          ) -> dict[str, Any]:
    """Green and red on one frame, with the green channel's fade corrected.

    The correction is display-only and is the thing about this movie most
    likely to be mistaken for a measurement: a smooth multiplicative ramp,
    fitted so the start and end medians inside one fixed tissue mask match, so
    a viewer is not misled into reading bleaching as a biological decline. It
    never touches the saved stack and never reaches a trace.
    """
    import numpy as np

    source = Path(source)
    stack, meta = _two_channel(source)
    green, red = stack[:, 0], stack[:, 1]

    mask = _render.stable_tissue_mask(
        green, sample_frames=int(mask_sample_frames),
        erosion_iterations=int(mask_erosion_iterations),
        min_pixels=int(mask_min_pixels),
        min_area_fraction=float(mask_min_area_fraction))
    before, gains, after = _render.endpoint_gain_curve(green, mask)

    stored = _stored_ranges(meta)
    ranges = _widen(stored if stored is not None else (
        *(float(v) for v in np.percentile(green, green_display_percentiles)),
        *(float(v) for v in np.percentile(red, red_display_percentiles))))

    folder = Path(output_dir) if output_dir else _default_output_dir(source)
    stem = str(output_name) if output_name else (
        f"{source.stem}_red-green_timestamped_{frame_interval_minutes:g}min"
        f"_green_endpoint_corrected")
    target = folder / f"{stem}.mp4"

    height, width = green.shape[1:]
    band = int(time_label_band_height)
    canvas_height = height + band
    padded = (canvas_height + canvas_height % 2, width + width % 2)

    def painted():
        for index in range(stack.shape[0]):
            rgb = _render.rgb_composite(green[index], red[index],
                                        gain=float(gains[index]), ranges=ranges)
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
        "frames": int(stack.shape[0]),
        "height": int(padded[0]), "width": int(padded[1]),
        "channels": "C1 green + C2 red",
        "green_correction": ("display-only multiplicative log-linear endpoint "
                             "correction"),
        "green_mask_pixels": int(np.count_nonzero(mask)),
        "green_start_median": float(before[0]),
        "green_end_median_before": float(before[-1]),
        "green_end_gain": float(gains[-1]),
        "green_end_median_after": float(after[-1]),
        "endpoint_relative_error": float(abs(after[-1] - after[0]) / after[0]),
        "green_display_min": ranges[0], "green_display_max": ranges[1],
        "red_display_min": ranges[2], "red_display_max": ranges[3],
        "display_range_source": "tiff" if stored is not None else "percentiles",
        "fps": float(fps), "crf": int(crf), "profile": "standard",
        "frame_interval_minutes": float(frame_interval_minutes),
        "first_time_label": _annotate.elapsed_label(0, 0.0),
        "last_time_label": _annotate.elapsed_label(
            int(stack.shape[0]) - 1, float(frame_interval_minutes) / 60.0),
        "ffmpeg": _encode.ffmpeg_version(),
        "ignored_wrapper_settings": {"input_glob": input_glob},
    }
    _record(source, "timestamped_composite", target, report,
            output_dir=folder,
            method_version=METHOD_VERSIONS["timestamped_composite"])
    return report


def _widen(ranges: Sequence[float]) -> tuple[float, float, float, float]:
    """A zero-width display range widened by one count rather than dividing by nought."""
    green_low, green_high, red_low, red_high = (float(v) for v in ranges)
    if not green_high > green_low:
        green_high = green_low + 1.0
    if not red_high > red_low:
        red_high = red_low + 1.0
    return green_low, green_high, red_low, red_high


# ------------------------------------------------------------ phase/green/red
#: What the organotypic exporter renders, and what each one is for. Four
#: videos, because one of them alone would not settle the question the folder
#: exists to ask — whether a green object is a microglial process or diffuse
#: autofluorescence — and the engine's own report says as much.
PHASE_GREEN_RED_VIDEOS = (
    ("timestamped_green_microglia_red_neurons", "composite",
     "hIba1a green  |  Syn-RCamp red"),
    ("timestamped_green_microglia_only", "green", ""),
    ("timestamped_green_local_contrast_candidates", "detail", ""),
    ("timestamped_green_detail_red_neurons", "detail_composite",
     "green local contrast  |  Syn-RCamp red"),
)


def phase_green_red(source, *, output_dir=None, output_name=None,
                    overwrite: bool = False,
                    frame_interval_seconds: float = 30.0, fps: float = 10.0,
                    crf: int = 18, mask_smooth_sigma_px: float = 3.0,
                    mask_closing_iterations: int = 3,
                    mask_dilation_iterations: int = 2,
                    gain_sample_frames: int = 41,
                    gain_metric_percentile: int = 75,
                    gain_window_max: int = 20, gain_window_min: int = 3,
                    gain_window_divisor: int = 10,
                    detail_blur_sigma_px: float = 8.0,
                    green_display_percentiles: Sequence[float] = (0.5, 99.7),
                    red_display_percentiles: Sequence[float] = (0.5, 99.7),
                    detail_display_percentiles: Sequence[float] = (70.0, 99.8),
                    timestamp_band_height: int = 48,
                    timestamp_font_size: int = 23, legend_font_size: int = 16,
                    panel_title_font_size: int = 13,
                    font_directory: str = "C:/Windows/Fonts",
                    font_regular: str = "arial.ttf",
                    font_bold: str = "arialbd.ttf") -> dict[str, Any]:
    """The organotypic three-channel route: four movies from one registered stack.

    Green is hIba1a, red is Syn-RCamp neurons, and the green channel fades over
    the recording, so it carries the same display-only endpoint gain the
    composite exporter uses — measured here on a percentile rather than a
    median, over a window at each end rather than one frame, because 1,018
    frames at 30 seconds make a single endpoint frame noisy.

    The local-contrast videos subtract a Gaussian background for display only.
    That makes green objects legible against diffuse fluorescence; it is not
    evidence that they are microglial, and this acquisition has no independent
    marker that could settle it.
    """
    import numpy as np
    import tifffile

    source = Path(source)
    with tifffile.TiffFile(source) as handle:
        series = handle.series[0]
        axes = getattr(series, "axes", "").upper()
        shape = series.shape
        if axes != "TCYX" or len(shape) != 4 or shape[1] < 3:
            raise ValueError(f"{source.name}: expected a three-channel TCYX "
                             f"stack, found {shape} {axes}")
        count, channels, height, width = (int(v) for v in shape)

        # Streamed, never loaded whole. The validated recording is 1,018
        # frames of three 408x608 channels — 1.5 GB as float32 — and holding
        # it would put the memory ceiling, not the science, in charge of how
        # long a recording may be.
        sampled = np.unique(np.linspace(0, count - 1,
                                        int(gain_sample_frames)).round()
                            .astype(int))
        green_samples = np.stack(
            [_page(handle, int(index), 1, channels) for index in sampled])
        red_samples = np.stack(
            [_page(handle, int(index), 2, channels) for index in sampled])

        mask = _render.largest_tissue_mask(
            np.median(green_samples, axis=0),
            smooth_sigma_px=float(mask_smooth_sigma_px),
            closing_iterations=int(mask_closing_iterations),
            dilation_iterations=int(mask_dilation_iterations),
            clip=False)

        metrics = np.zeros(count, dtype=np.float64)
        for frame in range(count):
            plane = _page(handle, frame, 1, channels)
            metrics[frame] = np.percentile(plane[mask],
                                           float(gain_metric_percentile))

    window = int(min(int(gain_window_max),
                     max(int(gain_window_min),
                         count // int(gain_window_divisor))))
    start_metric = float(np.median(metrics[:window]))
    end_metric = float(np.median(metrics[-window:]))
    end_gain = start_metric / max(end_metric, 1e-12)
    gains = np.exp(np.linspace(0.0, np.log(max(end_gain, 1e-12)), count))

    # Display ranges come from the sampled frames *inside the mask*, not from
    # the whole stack. Outside the tissue there is only background, and letting
    # it into the percentile would set the contrast from the empty field.
    green_display = np.concatenate(
        [green_samples[position][mask].astype(np.float32) * gains[frame]
         for position, frame in enumerate(sampled)])
    red_display = red_samples[:, mask].astype(np.float32).ravel()
    detail_display = np.concatenate([
        _render.local_contrast(
            green_samples[position].astype(np.float32) * gains[frame],
            float(detail_blur_sigma_px))[mask]
        for position, frame in enumerate(sampled)])

    green_low, green_high = (float(v) for v in
                             np.percentile(green_display, green_display_percentiles))
    red_low, red_high = (float(v) for v in
                         np.percentile(red_display, red_display_percentiles))
    detail_low, detail_high = (float(v) for v in
                               np.percentile(detail_display,
                                             detail_display_percentiles))
    if green_high <= green_low or red_high <= red_low:
        raise ValueError("the display range came out empty; the tissue mask "
                         "probably found no tissue")

    folder = Path(output_dir) if output_dir else _default_output_dir(source)
    band_kwargs = {"band_height": int(timestamp_band_height),
                   "font_directory": font_directory,
                   "font_regular": font_regular, "font_bold": font_bold,
                   "timestamp_size": int(timestamp_font_size),
                   "legend_size": int(legend_font_size),
                   "title_size": int(panel_title_font_size)}
    canvas = (height + int(timestamp_band_height), width)
    canvas = (canvas[0] + canvas[0] % 2, canvas[1] + canvas[1] % 2)

    written = []
    for name, kind, legend in PHASE_GREEN_RED_VIDEOS:
        stem = f"{output_name}_{name}" if output_name else name
        target = folder / f"{stem}.mp4"
        frames_out = _phase_frames(
            source, kind=kind, legend=legend, channels=channels, count=count,
            gains=gains, ranges=(green_low, green_high, red_low, red_high),
            detail_range=(detail_low, detail_high),
            blur_sigma_px=float(detail_blur_sigma_px),
            interval_seconds=float(frame_interval_seconds),
            band_kwargs=band_kwargs, canvas=canvas)
        _encode.write_video(target, frames_out, fps=float(fps), profile="fast",
                            crf=int(crf), overwrite=overwrite)
        written.append({"file": target.name, "path": str(target), "kind": kind})

    report = {
        "display_only": True,
        "input": str(source), "videos": written,
        "frames": int(count), "height": int(canvas[0]), "width": int(canvas[1]),
        "frame_interval_seconds": float(frame_interval_seconds),
        "fps": float(fps), "crf": int(crf), "profile": "fast",
        "start_green_metric": start_metric,
        "end_green_metric": end_metric,
        "end_display_gain": float(end_gain),
        "gain_window_frames": int(window),
        "green_mask_pixels": int(np.count_nonzero(mask)),
        "green_display_range": [green_low, green_high],
        "red_display_range": [red_low, red_high],
        "green_local_contrast_display_range": [detail_low, detail_high],
        "green_local_contrast_operation": (
            f"max(green - GaussianBlur(green,sigma={detail_blur_sigma_px:g}px), 0); "
            f"video display only"),
        "scientific_stack_intensity_normalization": "none",
        "registration_before_after": (
            "not rendered here: it needs the unregistered source stack and the "
            "saved crop, neither of which this action takes. Stage 11's "
            "pipeline holds both."),
        "ffmpeg": _encode.ffmpeg_version(),
    }
    _record(source, "phase_green_red", folder, report, output_dir=folder,
            method_version=METHOD_VERSIONS["phase_green_red"])
    return report


def _page(handle, frame: int, channel: int, channels: int):
    """One plane of an interleaved hyperstack, read without decoding the rest."""
    import numpy as np

    return np.asarray(handle.pages[frame * channels + channel].asarray())


def _phase_frames(source, *, kind: str, legend: str, channels: int, count: int,
                  gains, ranges, detail_range, blur_sigma_px: float,
                  interval_seconds: float, band_kwargs, canvas):
    """One video's frames, generated as the file is read a plane at a time."""
    import numpy as np
    import tifffile

    green_low, green_high, red_low, red_high = ranges
    detail_low, detail_high = detail_range
    with tifffile.TiffFile(source) as handle:
        for frame in range(count):
            green = _page(handle, frame, 1, channels).astype(np.float32)
            corrected = green * float(gains[frame])
            if kind in ("detail", "detail_composite"):
                corrected = _render.local_contrast(corrected, blur_sigma_px)
                low, high = detail_low, detail_high
            else:
                low, high = green_low, green_high

            if kind in ("composite", "detail_composite"):
                red = _page(handle, frame, 2, channels).astype(np.float32)
                rgb = _rgb_two(corrected, low, high, red, red_low, red_high)
            else:
                rgb = _rgb_one(corrected, low, high, plane=1)

            rgb = _annotate.caption_band(
                rgb, _annotate.clock_label(frame, interval_seconds),
                legend=legend, **band_kwargs)
            yield _annotate.pad_into(rgb, *canvas)


def _scale_u8(image, low: float, high: float):
    """Rounded rather than truncated, which is what this engine's ``scale_u8`` does."""
    import numpy as np

    scaled = (np.asarray(image).astype(np.float32) - low) * (255.0 / (high - low))
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)


def _rgb_one(image, low: float, high: float, *, plane: int):
    import numpy as np

    rgb = np.zeros((*np.shape(image), 3), np.uint8)
    rgb[:, :, plane] = _scale_u8(image, low, high)
    return rgb


def _rgb_two(green, green_low: float, green_high: float,
             red, red_low: float, red_high: float):
    import numpy as np

    rgb = np.zeros((*np.shape(green), 3), np.uint8)
    rgb[:, :, 0] = _scale_u8(red, red_low, red_high)
    rgb[:, :, 1] = _scale_u8(green, green_low, green_high)
    return rgb
