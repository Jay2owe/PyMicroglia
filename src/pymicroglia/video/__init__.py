"""Compatibility actions over PySCNSlice's consolidated video renderer.

Rendering moved to :mod:`pyscnslice.video`. PyMicroglia keeps the four action
names already stored in run records and agent catalogues, but each now only
translates its established arguments into the single upstream
``stack_to_video`` call. No pixel operation is implemented here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from pyscnslice import video as _moved

encode = _moved.encode
stack_to_video = _moved.stack_to_video
available = _moved.available

__all__ = [
    "encode",
    "stack_to_video",
    "available",
    "stack_to_mp4",
    "red_only",
    "timestamped_composite",
    "phase_green_red",
]


def _one_source(source: Any, pattern: str) -> Path:
    """Resolve an old folder-plus-glob request to one explicit recording."""
    path = Path(source)
    if path.is_file() or not path.is_dir():
        return path
    matches = sorted(candidate for candidate in path.glob(pattern) if candidate.is_file())
    if len(matches) != 1:
        raise ValueError(
            f"{path}: expected one input matching {pattern!r}, found {len(matches)}"
        )
    return matches[0]


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
    """Translate the established general-video action to ``stack_to_video``."""
    del floor_off_tissue_pct, python, python_engine
    path = _one_source(source, input_glob or glob)
    explicit_range = (
        (float(black_point), float(white_point))
        if float(black_point) >= 0 and float(white_point) >= 0
        else "stored"
    )
    filters = None
    if int(smooth_frames) or float(smooth_sigma_px):
        filters = [("smooth", {
            "frames": int(smooth_frames),
            "sigma_px": float(smooth_sigma_px),
        })]
    return stack_to_video(
        path,
        output_dir=output_dir,
        output_name=output_name,
        overwrite=overwrite,
        channels=int(signal_channel),
        first_frame=int(first_frame),
        frames=int(frames),
        hours_per_second=float(hours_per_second),
        frame_interval_h=(float(frame_interval_h) if frame_interval_h else None),
        min_fps=float(min_fps),
        max_fps=float(max_fps),
        filters=filters,
        lut=lut,
        display_range=explicit_range,
        timestamp=bool(timestamp),
        timestamp_band_px=int(timestamp_band_px),
        label=label,
        container="mp4",
        crf=int(crf),
    )


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
    """Render the already-unmixed second channel in red."""
    del (unmixing_coefficient, file_lock_retry_seconds,
         timestamp_font_fallback, timestamp_font_height_fraction)
    return stack_to_video(
        _one_source(source, input_glob),
        output_dir=output_dir,
        output_name=output_name,
        overwrite=overwrite,
        channels=2,
        fps=float(fps),
        frame_interval_h=(float(frame_interval_minutes) / 60.0
                          if frame_interval_minutes else None),
        lut="red",
        display_range="auto",
        auto_black_percentile=0.0,
        auto_white_percentile=float(display_percentile),
        timestamp=True,
        timestamp_size=int(timestamp_font_min_size),
        timestamp_band_px=int(time_label_band_height),
        container="mp4",
        crf=int(crf),
    )


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
    """Render the first two channels as a green/red composite."""
    del (mask_sample_frames, mask_erosion_iterations, mask_min_pixels,
         mask_min_area_fraction, red_display_percentiles)
    return stack_to_video(
        _one_source(source, input_glob),
        output_dir=output_dir,
        output_name=output_name,
        overwrite=overwrite,
        channels=(1, 2),
        fps=float(fps),
        frame_interval_h=float(frame_interval_minutes) / 60.0,
        lut=("green", "red"),
        display_range="auto",
        auto_black_percentile=float(green_display_percentiles[0]),
        auto_white_percentile=float(green_display_percentiles[-1]),
        timestamp=True,
        timestamp_band_px=int(time_label_band_height),
        container="mp4",
        crf=int(crf),
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
    """Render the established four views through the generic upstream route."""
    del (mask_smooth_sigma_px, mask_closing_iterations,
         mask_dilation_iterations, gain_sample_frames,
         gain_metric_percentile, gain_window_max, gain_window_min,
         gain_window_divisor, legend_font_size, panel_title_font_size,
         font_directory, font_regular, font_bold)
    path = _one_source(source, "*.tif")
    base = str(output_name) if output_name else path.stem
    common = {
        "output_dir": output_dir,
        "overwrite": overwrite,
        "fps": float(fps),
        "frame_interval_h": float(frame_interval_seconds) / 3600.0,
        "timestamp": True,
        "timestamp_size": int(timestamp_font_size),
        "timestamp_band_px": int(timestamp_band_height),
        "container": "mp4",
        "crf": int(crf),
    }
    specifications = (
        ("green", 2, "green", None, green_display_percentiles),
        ("red", 3, "red", None, red_display_percentiles),
        ("green-red", (2, 3), ("green", "red"), None,
         green_display_percentiles),
        ("green-detail", 2, "green",
         [("local_contrast", {"sigma_px": float(detail_blur_sigma_px)})],
         detail_display_percentiles),
    )
    reports = []
    for suffix, channels, lut, filters, percentiles in specifications:
        reports.append(stack_to_video(
            path,
            output_name=f"{base}-{suffix}",
            channels=channels,
            lut=lut,
            filters=filters,
            display_range="auto",
            auto_black_percentile=float(percentiles[0]),
            auto_white_percentile=float(percentiles[-1]),
            **common,
        ))
    return {
        "display_only": True,
        "videos": [report["output"] for report in reports],
        "display_ranges": [report.get("drawn") for report in reports],
        "reports": reports,
    }
