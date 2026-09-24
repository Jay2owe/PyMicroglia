"""Tracked-cell views of original photon frames for the shared grid renderers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
import tifffile

from auto_organotypic import series, store
from auto_organotypic.outline.crop import CROP_SCALES
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
        width, height = self.size
        cy, cx = (int(round(value)) for value in self.centres[int(time)])
        top, left = cy - height // 2, cx - width // 2
        out = np.full((height, width), np.nan, np.float32)
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
               crop_basis: str = "largest_cell", crop: str = "tight",
               crop_size_px: tuple[int, int] | None = None,
               identity_prefix: str = "Cell", trace_channel: int = 1
               ) -> list[TileSource]:
    """One moving, native-pixel tile per positive tracked identity.

    ``labels`` is the selected images or videos eligibility view. Its frame
    zero maps to photon frame ``source_frame_offset``. Missing identities use
    the current photon frame at their last known centre and are marked by the
    shared grid. Before first observation their first centre is held.
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
    if crop_size_px is not None:
        if len(crop_size_px) != 2 or any(int(one) < 1 for one in crop_size_px):
            raise ValueError("crop_size_px must be (positive width, positive height)")
        explicit = tuple(map(int, crop_size_px))
    else:
        explicit = None

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

    ids = sorted(centres)
    if not ids:
        raise ValueError("the selected label view contains no tracked cells")
    common_reach = (max(one[0] for one in reaches.values()),
                    max(one[1] for one in reaches.values()))
    raw_fingerprint = store.fingerprint(raw_path).as_dict()
    label_fingerprint = (store.fingerprint(label_path).as_dict()
                         if label_path is not None else
                         {"in_memory": True, "shape": list(label_values.shape),
                          "dtype": str(label_values.dtype)})
    sources = []
    for identity in ids:
        known = np.flatnonzero(observed[identity])
        first = centres[identity][known[0]].copy()
        held = first
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
                with shared.acquire() as original:
                    yield _CellView(original, centres_for_cell, size_for_cell)
            return opened_view()

        interval = frame_interval_h
        if interval is None:
            with shared.acquire() as opened:
                from auto_organotypic.render.frames import frame_interval_h as cadence
                interval, _source = cadence(opened, None)
        times = np.arange(total, dtype=float) * float(interval or 1.0)
        provenance = {"identity": identity, "raw": raw_fingerprint,
                      "labels": label_fingerprint, "source_frame_offset": offset,
                      "crop_basis": basis if explicit is None else "pixels",
                      "crop": mode if explicit is None else None,
                      "crop_size_px": list(size), "trace_channel": channel + 1,
                      "trace": "mean original photons inside observed tracked mask; display only"}
        sources.append(TileSource(
            key=str(identity), name=f"{identity_prefix} {identity}",
            source_path=raw_path, open_series=opener, provenance=provenance,
            trace=(times, traces[identity]), unavailable_frames=unavailable))
    return sources
