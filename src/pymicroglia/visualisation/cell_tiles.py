"""Tracked-cell views of original photon frames for the shared grid renderers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import math
from pathlib import Path
from threading import RLock
from typing import Any, Sequence

import numpy as np
import tifffile

from auto_organotypic import series, store
from auto_organotypic.outline.crop import CROP_SCALES
from auto_organotypic.render import outlines as _outlines
from auto_organotypic.render.tile_source import TileSource


class _SharedRaw:
    """Keep one TIFF reader while several movie tiles are open together."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = RLock()
        self.users = 0
        self.context = None
        self.opened = None

    @contextmanager
    def acquire(self):
        with self.lock:
            if self.users == 0:
                self.context = series.open_series(self.path)
                self.opened = self.context.__enter__()
            self.users += 1
        try:
            yield self.opened
        finally:
            with self.lock:
                self.users -= 1
                if self.users == 0:
                    self.context.__exit__(None, None, None)
                    self.context = self.opened = None


@dataclass(frozen=True)
class _CellView:
    original: Any
    centres: np.ndarray
    size: tuple[int, int]
    display_size: tuple[int, int] | None = None
    source_um_per_px: float | None = None
    frame_sizes: np.ndarray | None = None
    clamp_to_frame: bool = False

    @property
    def shape(self):
        frames, channels, _height, _width = self.original.shape
        width, height = self.display_size or self.size
        return frames, channels, height, width

    @property
    def meta(self):
        physical = self.source_um_per_px
        if physical is None:
            physical = getattr(self.original.meta, "um_per_px", None)
        if physical is None:
            return self.original.meta
        width = (self.display_size or self.size)[0]
        return replace(self.original.meta,
                       um_per_px=float(physical) * self.size[0] / width)

    @property
    def dtype(self):
        return np.dtype(np.float32)

    @property
    def display_only(self):
        return self.original.display_only

    def frame(self, time: int, channel: int):
        image = self.original.frame(int(time), int(channel))
        size = (tuple(map(int, self.frame_sizes[int(time)]))
                if self.frame_sizes is not None else self.size)
        crop = _window(image, self.centres[int(time)], size,
                       fill=np.nan, dtype=np.float32,
                       clamp=self.clamp_to_frame)
        return (_resize_photons(crop, self.display_size)
                if self.display_size and self.display_size != size else crop)


def _resize_photons(crop: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Fill a common grid slot while preserving valid edge intensities."""
    from PIL import Image

    if np.isfinite(crop).all():
        return np.asarray(Image.fromarray(crop).resize(
            size, Image.Resampling.BILINEAR), np.float32)
    valid = np.isfinite(crop).astype(np.float32)
    values = np.nan_to_num(crop, nan=0.0)
    numerator = np.asarray(Image.fromarray(values).resize(
        size, Image.Resampling.BILINEAR), np.float32)
    denominator = np.asarray(Image.fromarray(valid).resize(
        size, Image.Resampling.BILINEAR), np.float32)
    out = np.full(numerator.shape, np.nan, np.float32)
    np.divide(numerator, denominator, out=out, where=denominator > .5)
    return out


def _window(image, centre, size, *, fill, dtype, clamp=False):
    width, height = size
    cy, cx = (int(round(value)) for value in centre)
    top, left = cy - height // 2, cx - width // 2
    if clamp:
        top = min(max(top, 0), max(0, image.shape[0] - height))
        left = min(max(left, 0), max(0, image.shape[1] - width))
    out = np.full((height, width), fill, dtype)
    y0, x0 = max(top, 0), max(left, 0)
    y1, x1 = min(top + height, image.shape[0]), min(left + width, image.shape[1])
    if y1 > y0 and x1 > x0:
        out[y0 - top:y1 - top, x0 - left:x1 - left] = image[y0:y1, x0:x1]
    return out


def _labels(value):
    if isinstance(value, (str, Path)):
        path = Path(value)
        try:
            return tifffile.memmap(path, mode="r"), path
        except (ValueError, OSError):
            return tifffile.imread(path), path
    return np.asarray(value), None


def _size_for(reach: tuple[int, int], crop: str) -> tuple[int, int]:
    factor = float(CROP_SCALES[crop])
    side = max(2 * reach[0] + 1, 2 * reach[1] + 1)
    scaled = math.ceil(side * factor)
    if scaled % 2 == 0:
        scaled += 1
    return scaled, scaled


def _smooth_centres(centres: np.ndarray, frames: int) -> np.ndarray:
    """Suppress single-frame centroid errors without shifting motion in time."""
    if frames == 0:
        return centres
    radius = frames // 2
    padded = np.pad(centres, ((radius, radius), (0, 0)), mode="edge")
    median = np.stack([np.median(padded[index:index + frames], axis=0)
                       for index in range(len(centres))])
    weights = np.concatenate((np.arange(1, radius + 2),
                              np.arange(radius, 0, -1))).astype(float)
    weights /= weights.sum()
    return np.stack([
        np.convolve(np.pad(median[:, axis], radius, mode="edge"),
                    weights, mode="valid")
        for axis in range(2)], axis=1)


def _hold_small_movements(centres: np.ndarray, threshold_px: float) -> np.ndarray:
    """Keep a crop still until the candidate moves beyond its pixel deadband."""
    if threshold_px == 0:
        return centres
    held = np.empty_like(centres)
    held[0] = centres[0]
    for index in range(1, len(centres)):
        delta = centres[index] - held[index - 1]
        distance = float(np.hypot(delta[0], delta[1]))
        held[index] = held[index - 1]
        if distance > threshold_px:
            held[index] += delta * ((distance - threshold_px) / distance)
    return held


def _reach_for_bounds(bounds: np.ndarray, centres: np.ndarray,
                      known: np.ndarray) -> list[int]:
    rounded = np.rint(centres[known])
    cell_bounds = bounds[known]
    return [int(np.max(np.abs(cell_bounds[:, :2] - rounded[:, :1]))),
            int(np.max(np.abs(cell_bounds[:, 2:] - rounded[:, 1:])))]


def _mask_centre(yy: np.ndarray, xx: np.ndarray, photons: np.ndarray,
                 method: str) -> tuple[float, float]:
    """Find an observed mask's centre without changing its measured trace."""
    geometric = (float(yy.mean()), float(xx.mean()))
    if method == "mask":
        return geometric
    values = np.asarray(photons[yy, xx], dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return geometric
    # Removing the within-mask floor makes the bright cell body, rather than
    # a uniform camera/background offset, determine the display crop centre.
    weights = np.maximum(values[finite] - np.min(values[finite]), 0.0)
    if not np.any(weights):
        return geometric
    return (float(np.average(yy[finite], weights=weights)),
            float(np.average(xx[finite], weights=weights)))


def cell_tiles(raw, labels, *, source_frame_offset: int = 0,
               frame_interval_h: float | None = None,
               display_raw: str | Path | None = None,
               include_identities: Sequence[int] | None = None,
               crop_basis: str = "largest_cell", crop: str = "tight",
               crop_size_px: tuple[int, int] | None = None,
               identity_prefix: str = "Cell", trace_channel: int = 1,
               missing_centre: str = "hold", outline: bool = False,
               centre_smoothing_frames: int = 0,
               centre_deadband_fraction: float = 0.0,
               centre_method: str = "mask",
               fill_tile: bool = False,
               frame_crop: bool = False,
               um_per_px: float | None = None,
               mask_style: str = "outline", mask_opacity: float = 0.35,
               outline_colour=_outlines.DEFAULT_COLOUR,
               outline_width_px: int = _outlines.DEFAULT_WIDTH_PX,
               outline_opacity: float = _outlines.DEFAULT_OPACITY,
               unavailable_label: str = "CELL NOT OBSERVED",
               ) -> list[TileSource]:
    """One moving, native-pixel tile per positive tracked identity.

    ``fill_tile=True`` enlarges each own-cell still crop to the largest crop's
    grid slot, then adjusts its pixel calibration. The observed mask outline
    is drawn after enlargement so its requested width stays exact.

    ``labels`` is the selected images or videos eligibility view. Its frame
    zero maps to photon frame ``source_frame_offset``. Missing identities use
    the current photon frame. Their crop centres are held or interpolated
    between observed masks; recording-edge centres stay at the nearest known
    position. An absent mask is never inferred or drawn.
    """
    raw_path = Path(raw)
    label_values, label_path = _labels(labels)
    if label_values.ndim != 3:
        raise ValueError("labels must have (T,Y,X) shape")
    offset = int(source_frame_offset)
    basis = str(crop_basis).strip().lower()
    if basis not in ("largest_cell", "own_cell"):
        raise ValueError("crop_basis must be 'largest_cell' or 'own_cell'")
    mode = str(crop).strip().lower()
    if mode not in CROP_SCALES:
        raise ValueError(f"crop must be one of {', '.join(CROP_SCALES)}")
    if missing_centre not in ("hold", "interpolate"):
        raise ValueError("missing_centre must be 'hold' or 'interpolate'")
    if centre_method not in ("mask", "intensity_weighted"):
        raise ValueError("centre_method must be 'mask' or 'intensity_weighted'")
    if (isinstance(centre_smoothing_frames, bool) or
            not isinstance(centre_smoothing_frames, int) or
            centre_smoothing_frames < 0 or
            (centre_smoothing_frames != 0 and centre_smoothing_frames % 2 != 1)):
        raise ValueError("centre_smoothing_frames must be zero or a positive odd integer")
    if (isinstance(centre_deadband_fraction, bool) or
            not np.isfinite(float(centre_deadband_fraction)) or
            not 0 <= float(centre_deadband_fraction) <= 1):
        raise ValueError("centre_deadband_fraction must be between zero and one")
    if not isinstance(outline, bool):
        raise ValueError("outline must be true or false")
    if not isinstance(fill_tile, bool):
        raise ValueError("fill_tile must be true or false")
    if not isinstance(frame_crop, bool):
        raise ValueError("frame_crop must be true or false")
    if um_per_px is not None and (not np.isfinite(float(um_per_px)) or
                                  float(um_per_px) <= 0):
        raise ValueError("um_per_px must be a positive finite number")
    if mask_style not in ("outline", "fill"):
        raise ValueError("mask_style must be 'outline' or 'fill'")
    if not 0 <= float(mask_opacity) <= 1:
        raise ValueError("mask_opacity must be between 0 and 1")
    supports_live_overlay = "frame_overlay" in TileSource.__dataclass_fields__
    if frame_crop and "frame_um_per_px" not in TileSource.__dataclass_fields__:
        raise RuntimeError("frame-specific cell crops need the updated grid renderer")
    if outline and (not supports_live_overlay or
                    not hasattr(_outlines, "paint_mask") or
                    (mask_style == "fill" and
                     not hasattr(_outlines, "paint_mask_fill"))):
        raise RuntimeError("live cell outlines need the updated Auto-Organotypic "
                           "tile renderer")
    if crop_size_px is not None:
        if len(crop_size_px) != 2 or any(int(one) < 1 for one in crop_size_px):
            raise ValueError("crop_size_px must be (positive width, positive height)")
        explicit = tuple(map(int, crop_size_px))
    else:
        explicit = None
    if frame_crop and (basis != "own_cell" or not fill_tile or explicit is not None):
        raise ValueError("frame_crop needs own_cell, fill_tile, and no fixed pixel size")
    selected = (None if include_identities is None else
                {int(identity) for identity in include_identities})
    if selected is not None and (not selected or min(selected) < 1):
        raise ValueError("include_identities must contain positive cell numbers")

    shared = _SharedRaw(raw_path)
    with shared.acquire() as opened:
        total, channels, height, width = opened.shape
        physical_um_per_px = (float(um_per_px) if um_per_px is not None else
                              getattr(opened.meta, "um_per_px", None))
        if label_values.shape[1:] != (height, width):
            raise ValueError("labels and photon frames have different image dimensions")
        if offset < 0 or offset + len(label_values) > total:
            raise ValueError("source_frame_offset puts labels outside the photon recording")
        channel = int(trace_channel) - 1
        if not 0 <= channel < channels:
            raise ValueError(f"trace_channel must be in 1..{channels}")
        centres = {}
        reaches = {}
        bounds = {}
        traces = {}
        observed = {}
        for label_index in range(len(label_values)):
            raw_index = offset + label_index
            frame = np.asarray(label_values[label_index])
            present = [int(one) for one in np.unique(frame) if int(one) > 0]
            if selected is not None:
                present = [one for one in present if one in selected]
            if not present:
                continue
            photons = np.asarray(opened.frame(raw_index, channel))
            for identity in present:
                if identity not in centres:
                    centres[identity] = np.full((total, 2), np.nan, float)
                    reaches[identity] = [0, 0]
                    bounds[identity] = np.full((total, 4), np.nan, float)
                    traces[identity] = np.full(total, np.nan, float)
                    observed[identity] = np.zeros(total, bool)
                yy, xx = np.nonzero(frame == identity)
                cy, cx = _mask_centre(yy, xx, photons, centre_method)
                centres[identity][raw_index] = (cy, cx)
                bounds[identity][raw_index] = (yy.min(), yy.max(),
                                               xx.min(), xx.max())
                rounded_y, rounded_x = round(cy), round(cx)
                reaches[identity][0] = max(reaches[identity][0],
                                           int(np.max(np.abs(yy - rounded_y))))
                reaches[identity][1] = max(reaches[identity][1],
                                           int(np.max(np.abs(xx - rounded_x))))
                traces[identity][raw_index] = float(np.mean(photons[yy, xx]))
                observed[identity][raw_index] = True

    display_path = Path(display_raw) if display_raw is not None else raw_path
    display_shared = (_SharedRaw(display_path)
                      if display_raw is not None else shared)
    if display_raw is not None:
        with display_shared.acquire() as displayed:
            if displayed.shape != (total, channels, height, width):
                raise ValueError("display_raw must match photon frames and channels")

    ids = sorted(centres)
    if selected is not None and set(ids) != selected:
        raise ValueError("include_identities names cells absent from the labels")
    if not ids:
        raise ValueError("the selected label view contains no tracked cells")
    for identity in ids:
        known = np.flatnonzero(observed[identity])
        if missing_centre == "interpolate":
            for axis in range(2):
                centres[identity][:, axis] = np.interp(
                    np.arange(total), known, centres[identity][known, axis])
        else:
            held = centres[identity][known[0]].copy()
            for index in range(total):
                if observed[identity][index]:
                    held = centres[identity][index].copy()
                else:
                    centres[identity][index] = held
        centres[identity] = _smooth_centres(centres[identity],
                                           centre_smoothing_frames)
        reaches[identity] = _reach_for_bounds(
            bounds[identity], centres[identity], known)
    common_reach = (max(one[0] for one in reaches.values()),
                    max(one[1] for one in reaches.values()))
    deadband_px = {}
    for identity in ids:
        reference_size = (explicit if explicit is not None else
                          _size_for(common_reach if basis == "largest_cell"
                                    else tuple(reaches[identity]), mode))
        deadband_px[identity] = (float(centre_deadband_fraction) *
                                 min(reference_size))
        centres[identity] = _hold_small_movements(
            centres[identity], deadband_px[identity])
        reaches[identity] = _reach_for_bounds(
            bounds[identity], centres[identity],
            np.flatnonzero(observed[identity]))
    common_reach = (max(one[0] for one in reaches.values()),
                    max(one[1] for one in reaches.values()))
    common_size = explicit if explicit is not None else _size_for(common_reach, mode)
    frame_sizes: dict[int, np.ndarray] = {}
    if frame_crop:
        for identity in ids:
            sizes = np.tile(_size_for(tuple(reaches[identity]), mode), (total, 1))
            for index in np.flatnonzero(observed[identity]):
                bound = bounds[identity][index]
                cy, cx = np.rint(centres[identity][index])
                reach = (int(np.max(np.abs(bound[:2] - cy))),
                         int(np.max(np.abs(bound[2:] - cx))))
                sizes[index] = _size_for(reach, mode)
            frame_sizes[identity] = sizes
    raw_fingerprint = store.fingerprint(raw_path).as_dict()
    display_fingerprint = (store.fingerprint(display_path).as_dict()
                           if display_raw is not None else raw_fingerprint)
    label_fingerprint = (store.fingerprint(label_path).as_dict()
                         if label_path is not None else
                         {"in_memory": True, "shape": list(label_values.shape),
                          "dtype": str(label_values.dtype)})
    sources = []
    for identity in ids:
        required = reaches[identity]
        if explicit is not None:
            width_px, height_px = explicit
            if (width_px // 2 < required[1] or width_px - width_px // 2 - 1 < required[1]
                    or height_px // 2 < required[0]
                    or height_px - height_px // 2 - 1 < required[0]):
                minimum = (2 * required[1] + 1, 2 * required[0] + 1)
                raise ValueError(f"crop_size_px clips cell {identity}; use at least {minimum}")
            size = explicit
        else:
            size = _size_for(common_reach if basis == "largest_cell"
                             else tuple(required), mode)
        display_size = common_size if fill_tile else size
        sizes_for_cell = frame_sizes.get(identity)
        unavailable = {int(index): "tracked mask absent"
                       for index in range(total) if not observed[identity][index]}
        def opener(centres_for_cell=centres[identity], size_for_cell=size,
                   display_size_for_cell=display_size,
                   frame_sizes_for_cell=sizes_for_cell):
            @contextmanager
            def opened_view():
                with display_shared.acquire() as original:
                    yield _CellView(original, centres_for_cell, size_for_cell,
                                    display_size_for_cell, um_per_px,
                                    frame_sizes_for_cell, frame_crop)
            return opened_view()

        def frame_overlay(rgb, frame, *, cell=identity,
                          centres_for_cell=centres[identity],
                          size_for_cell=size,
                          display_size_for_cell=display_size,
                          frame_sizes_for_cell=sizes_for_cell):
            index = int(frame) - offset
            if not 0 <= index < len(label_values):
                return rgb
            frame_size = (tuple(map(int, frame_sizes_for_cell[int(frame)]))
                          if frame_sizes_for_cell is not None else size_for_cell)
            crop_labels = _window(label_values[index],
                                  centres_for_cell[int(frame)], frame_size,
                                  fill=0, dtype=label_values.dtype,
                                  clamp=frame_crop)
            mask = crop_labels == cell
            if display_size_for_cell != frame_size:
                from PIL import Image
                mask = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(
                    display_size_for_cell, Image.Resampling.NEAREST), bool)
            if mask_style == "fill":
                return _outlines.paint_mask_fill(
                    rgb, mask, colour=outline_colour, opacity=mask_opacity)
            return _outlines.paint_mask(rgb, mask, colour=outline_colour,
                                        width_px=outline_width_px,
                                        opacity=outline_opacity)

        def frame_um_per_px(index, *, sizes_for_cell=sizes_for_cell,
                            display_size_for_cell=display_size):
            if physical_um_per_px is None:
                return None
            return (float(physical_um_per_px) *
                    float(sizes_for_cell[int(index), 0]) / display_size_for_cell[0])

        interval = frame_interval_h
        if interval is None:
            with shared.acquire() as opened:
                from auto_organotypic.render.frames import frame_interval_h as cadence
                interval, _source = cadence(opened, None)
        times = np.arange(total, dtype=float) * float(interval or 1.0)
        provenance = {"identity": identity, "raw": raw_fingerprint,
                      "display_raw": display_fingerprint,
                      "labels": label_fingerprint, "source_frame_offset": offset,
                      "crop_basis": basis if explicit is None else "pixels",
                      "crop": mode if explicit is None else None,
                      "crop_size_px": list(size), "trace_channel": channel + 1,
                      "display_size_px": list(display_size),
                      "tile_fill": bool(fill_tile),
                      "frame_crop": bool(frame_crop),
                      "frame_crop_rule": ("observed mask bounds, tight square"
                                          if frame_crop else None),
                      "source_um_per_px": (float(um_per_px)
                                           if um_per_px is not None else None),
                      "um_per_display_px": (float(um_per_px) * size[0] / display_size[0]
                                            if um_per_px is not None else None),
                      "frame_interval_h": float(interval or 1.0),
                      "missing_centre": missing_centre,
                      "centre_smoothing_frames": centre_smoothing_frames,
                      "centre_deadband_fraction": float(centre_deadband_fraction),
                      "centre_deadband_px": float(deadband_px[identity]),
                      "centre_deadband_reference": "pre-deadband crop short side",
                      "centre_smoothing_method": ("median then triangular mean"
                                                  if centre_smoothing_frames else "none"),
                      "centre_method": centre_method,
                      "centre_weighting": ("raw photons above within-mask minimum"
                                           if centre_method == "intensity_weighted"
                                           else "equal mask-pixel weights"),
                      "outline": bool(outline),
                      "mask_style": mask_style,
                      "mask_opacity": float(mask_opacity),
                      "outline_colour": (list(outline_colour) if not isinstance(outline_colour, str)
                                         else outline_colour),
                      "outline_width_px": int(outline_width_px),
                      "outline_opacity": float(outline_opacity),
                      "trace": "mean original photons inside observed tracked mask; display only"}
        tile_options = ({"frame_overlay": frame_overlay if outline else None,
                         "unavailable_label": unavailable_label}
                        if supports_live_overlay else {})
        sources.append(TileSource(
            key=str(identity), name=f"{identity_prefix} {identity}",
            source_path=display_path, open_series=opener, provenance=provenance,
            trace=(times, traces[identity]), unavailable_frames=unavailable,
            **tile_options))
    return sources
