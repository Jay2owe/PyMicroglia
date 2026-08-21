"""Synthetic time-lapses, so no test needs a real 10.8 GB acquisition.

The real sources are ten-gigabyte online-only files in Dropbox; reading one
downloads it. A suite that needed one would be untestable on a fresh machine
and slow on this one, so everything here is generated: a proper OME-TIFF with
per-plane ``DeltaT`` timestamps, a pixel size, channel names, and channels that
behave enough like the real ones for the channel statistics to work.

The three channels are built to be *identifiable*, which is the only property
the inference depends on:

    0  phase          flat and bright        — transmitted light
    1  green_biolum   dim, with cosmic-ray spikes — bioluminescence
    2  red_mCherry    structured blobs       — structural fluorescence
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

__all__ = [
    "CHANNEL_NAMES",
    "synthetic_stack",
    "ome_xml",
    "write_series",
    "shift_table",
]

CHANNEL_NAMES = ("phase", "green_biolum", "red_mCherry")


def synthetic_stack(frames: int = 8, channels: int = 3, height: int = 32,
                    width: int = 24, *, seed: int = 0) -> np.ndarray:
    """A (T, C, Y, X) uint16 stack whose channels are what they claim to be."""
    rng = np.random.default_rng(seed)
    stack = np.zeros((frames, channels, height, width), dtype=np.uint16)
    ys, xs = np.mgrid[0:height, 0:width]

    for t in range(frames):
        for c in range(channels):
            role = c % 3
            if role == 0:
                # Transmitted light: high level, almost no structure.
                image = 2000 + rng.normal(0, 8, (height, width))
            elif role == 1:
                # Bioluminescence: dim, plus saturating single-pixel spikes.
                image = 10 + rng.normal(0, 1.5, (height, width))
                image = np.clip(image, 0, None)
                for _ in range(2):
                    image[rng.integers(height), rng.integers(width)] = 60000
            else:
                # Structural fluorescence: blobs that survive a blur.
                image = 40 + rng.normal(0, 4, (height, width))
                for cy, cx in ((height // 3, width // 3),
                               (2 * height // 3, 2 * width // 3)):
                    image += 900 * np.exp(-(((ys - cy) ** 2 + (xs - cx) ** 2)
                                            / (2 * 3.0 ** 2)))
            stack[t, c] = np.clip(image, 0, 65535).astype(np.uint16)
    return stack


def ome_xml(frames: int, channels: int, height: int, width: int, *,
            dtype: str = "uint16", dt_minutes: float = 30.0,
            names: tuple[str, ...] = CHANNEL_NAMES,
            um_per_px: float | None = 0.65,
            gap_after: int | None = None, gap_hours: float = 6.0,
            explicit_tiffdata: bool = False,
            plane_timestamps: bool = True) -> str:
    """An OME-XML description of the kind the VSI converter writes.

    Ascii only: ``tifffile`` refuses to write a non-ascii TIFF string, so the
    unit is ``um`` rather than the micro sign a real converter emits. The
    parser handles both.
    """
    offsets = []
    elapsed = 0.0
    for t in range(frames):
        if gap_after is not None and t == gap_after + 1:
            elapsed += gap_hours * 3600.0
        offsets.append(elapsed)
        elapsed += dt_minutes * 60.0

    planes = ""
    if plane_timestamps:
        planes = "".join(
            f'<Plane TheZ="0" TheT="{t}" TheC="{c}" '
            f'DeltaT="{offsets[t]:.1f}" DeltaTUnit="s"/>'
            for t in range(frames) for c in range(channels))

    if explicit_tiffdata:
        tiffdata = "".join(
            f'<TiffData FirstZ="0" FirstT="{t}" FirstC="{c}" '
            f'IFD="{t * channels + c}" PlaneCount="1"/>'
            for t in range(frames) for c in range(channels))
    else:
        tiffdata = "<TiffData/>"

    channel_block = "".join(
        f'<Channel ID="Channel:0:{c}" Name="{names[c % len(names)]}" '
        f'SamplesPerPixel="1"/>' for c in range(channels))

    physical = ""
    if um_per_px is not None:
        physical = (f'PhysicalSizeX="{um_per_px}" PhysicalSizeXUnit="um" '
                    f'PhysicalSizeY="{um_per_px}" PhysicalSizeYUnit="um" ')

    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
        '<Image ID="Image:0" Name="synthetic">'
        f'<Pixels ID="Pixels:0" DimensionOrder="XYCZT" Type="{dtype}" '
        f'SizeX="{width}" SizeY="{height}" SizeZ="1" SizeC="{channels}" '
        f'SizeT="{frames}" Interleaved="false" {physical}>'
        f'{channel_block}{tiffdata}{planes}'
        '</Pixels></Image></OME>')


def write_series(path, *, frames: int = 8, channels: int = 3, height: int = 32,
                 width: int = 24, dt_minutes: float = 30.0, seed: int = 0,
                 um_per_px: float | None = 0.65,
                 gap_after: int | None = None, gap_hours: float = 6.0,
                 explicit_tiffdata: bool = False,
                 plane_timestamps: bool = True,
                 data: np.ndarray | None = None) -> np.ndarray:
    """Write a synthetic OME-TIFF and return the array that went into it."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    stack = synthetic_stack(frames, channels, height, width, seed=seed) \
        if data is None else np.asarray(data)
    frames, channels, height, width = stack.shape
    description = ome_xml(frames, channels, height, width,
                          dtype=str(stack.dtype), dt_minutes=dt_minutes,
                          um_per_px=um_per_px, gap_after=gap_after,
                          gap_hours=gap_hours,
                          explicit_tiffdata=explicit_tiffdata,
                          plane_timestamps=plane_timestamps)
    tifffile.imwrite(str(target), stack.reshape(-1, height, width),
                     description=description, metadata=None,
                     photometric="minisblack")
    return stack


def shift_table(shifts, *, one_based: bool = True) -> dict[str, list]:
    """A shifts table in the shape the engines write, for storing as tier A."""
    shifts = np.asarray(shifts, dtype=float)
    start = 1 if one_based else 0
    return {
        "frame": [start + i for i in range(len(shifts))],
        "shift_y_px": [float(v) for v in shifts[:, 0]],
        "shift_x_px": [float(v) for v in shifts[:, 1]],
    }
