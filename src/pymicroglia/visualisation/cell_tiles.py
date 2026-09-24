"""Tracked-cell views of original photon frames for the shared grid renderers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
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

    @property
    def shape(self):
        frames, channels, _height, _width = self.original.shape
        width, height = self.size
        return frames, channels, height, width

    @property
    def meta(self):
        return self.original.meta

    @property
    def dtype(self):
        return np.dtype(np.float32)

    @property
    def display_only(self):
        return self.original.display_only

    def frame(self, time: int, channel: int):
        image = self.original.frame(int(time), int(channel))
        return _window(image, self.centres[int(time)], self.size,
                       fill=np.nan, dtype=np.float32)


def _window(image, centre, size, *, fill, dtype):
    width, height = size
    cy, cx = (int(round(value)) for value in centre)
    top, left = cy - height // 2, cx - width // 2
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


def cell_tiles(raw, labels, *, source_frame_offset: int = 0,
               frame_interval_h: float | None = None,
               display_raw: str | Path | None = None,
               include_identities: Sequence[int] | None = None,
               crop_basis: str = "largest_cell", crop: str = "tight",
               crop_size_px: tuple[int, int] | None = None,
               identity_prefix: str = "Cell", trace_channel: int = 1,
               missing_centre: str = "hold", outline: bool = False,
               mask_style: str = "outline", mask_opacity: float = 0.35,
               outline_colour=_outlines.DEFAULT_COLOUR,
               outline_width_px: int = _outlines.DEFAULT_WIDTH_PX,
               outline_opacity: float = _outlines.DEFAULT_OPACITY,
               unavailable_label: str = "CELL NOT OBSERVED",
               ) -> list[TileSource]:
    """One moving, native-pixel tile per positive tracked identity.

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
    if not isinstance(outline, bool):
        raise ValueError("outline must be true or false")
    if mask_style not in ("outline", "fill"):
        raise ValueError("mask_style must be 'outline' or 'fill'")
    if not 0 <= float(mask_opacity) <= 1:
        raise ValueError("mask_opacity must be between 0 and 1")
    supports_live_overlay = "frame_overlay" in TileSource.__dataclass_fields__
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
    selected = (None if include_identities is None else
                {int(identity) for identity in include_identities})
    if selected is not None and (not selected or min(selected) < 1):
        raise ValueError("include_identities must contain positive cell numbers")

    shared = _SharedRaw(raw_path)
    with shared.acquire() as opened:
        total, channels, height, width = opened.shape
        if label_values.shape[1:] != (height, width):
            raise ValueError("labels and photon frames have different image dimensions")
        if offset < 0 or offset + len(label_values) > total:
            raise ValueError("source_frame_offset puts labels outside the photon recording")
        channel = int(trace_channel) - 1
        if not 0 <= channel < channels:
            raise ValueError(f"trace_channel must be in 1..{channels}")
        centres = {}
        reaches = {}
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
                    traces[identity] = np.full(total, np.nan, float)
                    observed[identity] = np.zeros(total, bool)
                yy, xx = np.nonzero(frame == identity)
                cy, cx = float(yy.mean()), float(xx.mean())
                centres[identity][raw_index] = (cy, cx)
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
    common_reach = (max(one[0] for one in reaches.values()),
                    max(one[1] for one in reaches.values()))
    raw_fingerprint = store.fingerprint(raw_path).as_dict()
    display_fingerprint = (store.fingerprint(display_path).as_dict()
                           if display_raw is not None else raw_fingerprint)
    label_fingerprint = (store.fingerprint(label_path).as_dict()
                         if label_path is not None else
                         {"in_memory": True, "shape": list(label_values.shape),
                          "dtype": str(label_values.dtype)})
    sources = []
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
        unavailable = {int(index): "tracked mask absent"
                       for index in range(total) if not observed[identity][index]}
        def opener(centres_for_cell=centres[identity], size_for_cell=size):
            @contextmanager
            def opened_view():
                with display_shared.acquire() as original:
                    yield _CellView(original, centres_for_cell, size_for_cell)
            return opened_view()

        def frame_overlay(rgb, frame, *, cell=identity,
                          centres_for_cell=centres[identity],
                          size_for_cell=size):
            index = int(frame) - offset
            if not 0 <= index < len(label_values):
                return rgb
            crop_labels = _window(label_values[index],
                                  centres_for_cell[int(frame)], size_for_cell,
                                  fill=0, dtype=label_values.dtype)
            mask = crop_labels == cell
            if mask_style == "fill":
                return _outlines.paint_mask_fill(
                    rgb, mask, colour=outline_colour, opacity=mask_opacity)
            return _outlines.paint_mask(rgb, mask, colour=outline_colour,
                                        width_px=outline_width_px,
                                        opacity=outline_opacity)

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
                      "frame_interval_h": float(interval or 1.0),
                      "missing_centre": missing_centre,
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
