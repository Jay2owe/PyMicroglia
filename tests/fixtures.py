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
    "TRACKED_MOVIE_FILES",
    "tracked_movie",
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


# ---------------------------------------------------------------------------
# A tracked movie: what the Motion analysis reads, written to disk.
#
# Ported from ``Motion/analysis/test_column_declarations.py::_movie`` (the
# one synthetic movie every Motion module is checked over) so that the motion
# parity fixture and, from stage 03 of the port, ``pymicroglia.measure`` share
# one builder. Same seed, same shapes, same three cells; the only change is
# that each array becomes a file, because the analysis reads files.
# ---------------------------------------------------------------------------

#: What ``tracked_movie`` writes, keyed by the role the analysis config names.
TRACKED_MOVIE_FILES = {
    "labels": "labels.tif",
    "raw": "raw.tif",
    "unclaimed": "unclaimed.tif",
    "evidence": "evidence.tif",
    "provenance": "provenance.tif",
    "valid_mask": "valid_mask.tif",
    "channel:extra": "extra_channel.tif",
    "channel:extra:shifts": "extra_shifts.csv",
    "objects:scenery": "scenery_objects.tif",
    "side:acquisition": "acquisition.csv",
    "side:genotype": "genotype.csv",
}


def tracked_movie(folder, *, frames: int = 48, height: int = 40,
                  width: int = 40, seed: int = 0) -> dict[str, Path]:
    """Three tracked cells on a small field, long enough for a rhythm fit.

    Deterministic from ``seed``. Returns the written files by role, in the
    words the analysis configuration uses (see ``TRACKED_MOVIE_FILES``).

    * ``labels`` (frames, y, x) uint16: cell 1 drifts on a 24 h sine, cell 2
      shuffles sideways, cell 3 never moves.
    * ``raw`` (frames, y, x) float32: a 24 h sine inside the outlines plus
      uniform noise everywhere.
    * ``unclaimed`` (frames, y, x) uint8: sparse foreground nobody claimed.
    * ``evidence`` (frames, 5, y, x) uint16: five graded tracker channels, the
      fifth a genuine ten-step ramp, because the age columns measure it.
    * ``provenance`` (frames, y, x) uint8: all three bits set on a 2x2 patch.
    * ``valid_mask`` (y, x) uint8: everything but a two-pixel border.
    * ``extra_channel`` (frames, 2, y, x) uint16 with a C axis, so the loader
      is made to index channel 1; it fades, differs between cells and has a
      clipped corner. ``extra_shifts.csv`` shifts it two columns per frame,
      so the last two analysed columns fall off the source and read blank.
    * ``scenery_objects`` (frames, y, x) uint16 label image: one fixed shape
      and one that drifts and touches the top edge.
    * ``acquisition.csv`` keyed on frame_index; ``genotype.csv`` on identity.
    """
    import pandas as pd

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    hours = np.arange(frames) / 2
    labels = np.zeros((frames, height, width), dtype=np.uint16)
    for frame in range(frames):
        drift = int(2 * np.sin(2 * np.pi * hours[frame] / 24))
        labels[frame, 8 + frame % 3:16 + frame % 3, 8 + drift:16 + drift] = 1
        labels[frame, 24:32, 22 + frame % 4:30 + frame % 4] = 2
        labels[frame, 4:9, 30:35] = 3

    noise = np.random.default_rng(seed).random((frames, height, width))
    signal = 100 + 30 * np.sin(2 * np.pi * hours / 24)
    raw = ((labels > 0) * signal[:, None, None] + noise * 5).astype(np.float32)
    unclaimed = ((labels == 0) & (noise > 0.95)).astype(np.uint8)

    full = np.iinfo(np.uint16).max
    evidence = np.zeros((frames, 5, height, width), dtype=np.uint16)
    evidence[:, :, 9:15, 9:15] = full
    ramp = (np.arange(36).reshape(6, 6) % 10) + 1
    evidence[:, 4, 9:15, 9:15] = np.rint(ramp * full / 10).astype(np.uint16)

    provenance = np.zeros((frames, height, width), dtype=np.uint8)
    provenance[:, 8:10, 8:10] = 0b111

    valid = np.ones((height, width), dtype=np.uint8)
    valid[:2, :] = valid[-2:, :] = valid[:, :2] = valid[:, -2:] = 0

    fade = np.linspace(1.0, 0.55, frames)[:, None, None]
    extra = (300.0 + 90.0 * (labels == 1) + 180.0 * (labels == 2)
             + 25.0 * noise) * fade
    extra = np.rint(extra).astype(np.uint16)
    # The shifts table says the source drifted two columns, so analysed column
    # x is read from source column x + 2: the source holds the field two
    # columns to the right, and the last two analysed columns are off it.
    channel = np.zeros((frames, 2, height, width), dtype=np.uint16)
    channel[:, 0] = 50
    channel[:, 1, :, 2:] = extra[:, :, :width - 2]
    channel[:, 1, 0:3, 2:5] = full                      # clipped corner

    scenery = np.zeros((frames, height, width), dtype=np.uint16)
    for frame in range(frames):
        scenery[frame, 18:26, 2:8] = 1
        drift = frame % 5
        scenery[frame, 0:4, 30 + drift:36 + drift] = 2

    written = {}

    def tif(role, array, axes):
        target = folder / TRACKED_MOVIE_FILES[role]
        tifffile.imwrite(str(target), array, metadata={"axes": axes},
                         photometric="minisblack")
        written[role] = target

    tif("labels", labels, "TYX")
    tif("raw", raw, "TYX")
    tif("unclaimed", unclaimed, "TYX")
    tif("evidence", evidence, "TCYX")
    tif("provenance", provenance, "TYX")
    tif("valid_mask", valid, "YX")
    tif("channel:extra", channel, "TCYX")
    tif("objects:scenery", scenery, "TYX")

    shifts = folder / TRACKED_MOVIE_FILES["channel:extra:shifts"]
    pd.DataFrame({"frame": np.arange(1, frames + 1),
                  "shift_y": np.zeros(frames, dtype=int),
                  "shift_x": np.full(frames, -2, dtype=int)}).to_csv(
        shifts, index=False, lineterminator="\n")
    written["channel:extra:shifts"] = shifts

    acquisition = folder / TRACKED_MOVIE_FILES["side:acquisition"]
    pd.DataFrame({
        "frame_index": np.arange(frames),
        "focus": np.round(np.linspace(1.0, 0.5, frames), 6),
        "suspect": (np.arange(frames) == 11).astype(int),
    }).to_csv(acquisition, index=False, lineterminator="\n")
    written["side:acquisition"] = acquisition
    genotype = folder / TRACKED_MOVIE_FILES["side:genotype"]
    pd.DataFrame({"identity": [1, 2, 3], "call": ["wt", "ko", "wt"]}).to_csv(
        genotype, index=False, lineterminator="\n")
    written["side:genotype"] = genotype
    return written
