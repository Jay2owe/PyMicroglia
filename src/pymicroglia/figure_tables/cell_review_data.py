"""Prepare an offline review of existing identities, without changing them."""
from __future__ import annotations

import base64
from io import BytesIO
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile

from ..measure.cell_signal import analyse as cell_signal
from .cell_review_geometry import crop_size, observation

METHOD_VERSION = "2026-09-28-saved-cell-review-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stack(path: Path) -> np.ndarray:
    try:
        return tifffile.memmap(path, mode="r")
    except (ValueError, OSError):
        return tifffile.imread(path)


def _range(raw: np.ndarray, requested) -> tuple[float, float]:
    if requested is not None:
        if len(requested) != 2:
            raise ValueError("display_range needs a lower and upper value")
        low, high = map(float, requested)
        if not np.isfinite([low, high]).all() or high <= low:
            raise ValueError("display_range must be finite and increasing")
        return low, high
    sample = raw.reshape(-1)[::max(1, raw.size // 1_000_000)]
    sample = sample[np.isfinite(sample)]
    if not len(sample):
        raise ValueError("the raw recording has no finite display pixels")
    low, high = np.percentile(sample, [1, 99.5])
    if high <= low:
        low, high = float(sample.min()), float(sample.max())
    if high <= low:
        high = low + 1
    return float(low), float(high)


def _png(frame: np.ndarray, low: float, high: float, lut: str) -> str:
    normalized = np.nan_to_num((np.asarray(frame, float) - low) / (high - low),
                               nan=0, posinf=1, neginf=0)
    pixels = (np.clip(normalized, 0, 1) * 255).round().astype(np.uint8)
    if lut == "red":
        rgb = np.zeros((*pixels.shape, 3), np.uint8)
        rgb[..., 0] = pixels
        image = Image.fromarray(rgb)
    else:
        image = Image.fromarray(pixels)
    buffer = BytesIO()
    image.save(buffer, format="PNG", compress_level=3)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _json_values(values) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in values]


def prepare(raw, labels, *, title=None, source_frame_offset=0,
            frame_interval_h=.5, display_lut="red", display_range=None) -> dict:
    """Read frozen single-channel TIFF inputs and prepare all four views.

    Offset is the zero-based raw frame containing labels[0]. Source values and
    shapes are never interpolated across absent mask observations. Intensity
    sums use original source pixels, before display contrast mapping.
    """
    raw_path, label_path = Path(raw).resolve(), Path(labels).resolve()
    raw_stack, label_stack = _stack(raw_path), _stack(label_path)
    if raw_stack.ndim != 3 or label_stack.ndim != 3:
        raise ValueError("raw and labels must be single-channel time/y/x TIFF stacks")
    if (not np.issubdtype(raw_stack.dtype, np.number) or
            not np.issubdtype(label_stack.dtype, np.integer)):
        raise ValueError("raw must be numeric and labels must contain integer identities")
    if label_stack.shape[1:] != raw_stack.shape[1:]:
        raise ValueError("raw and label spatial shapes do not match")
    if isinstance(source_frame_offset, bool) or int(source_frame_offset) != source_frame_offset:
        raise ValueError("source_frame_offset must be an integer")
    offset = int(source_frame_offset)
    if offset < 0 or offset + len(label_stack) > len(raw_stack):
        raise ValueError("labels do not fit within the raw recording at source_frame_offset")
    cadence = float(frame_interval_h)
    if not np.isfinite(cadence) or cadence <= 0:
        raise ValueError("frame_interval_h must be finite and positive")
    if display_lut not in {"red", "gray"}:
        raise ValueError("display_lut must be 'red' or 'gray'")
    if label_stack.size == 0 or raw_stack.size == 0:
        raise ValueError("raw and label stacks must not be empty")
    if label_stack.min() < 0 or int(label_stack.max()) > 2 ** 53 - 1:
        raise ValueError("labels must be non-negative integers representable in the browser")
    ids = [int(value) for value in np.unique(label_stack) if value > 0]
    frames, height, width = raw_stack.shape
    cells = {str(identity): {"id": identity, "observations": [None] * frames,
                            "signal_raw": np.full(frames, np.nan),
                            "area": np.full(frames, np.nan)} for identity in ids}
    for label_index, labelled in enumerate(label_stack):
        frame = label_index + offset
        yy, xx = np.nonzero(labelled)
        if not len(yy):
            continue
        values = labelled[yy, xx]
        order = np.argsort(values, kind="stable")
        yy, xx, values = yy[order], xx[order], values[order]
        starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
        stops = np.r_[starts[1:], len(values)]
        for start, stop in zip(starts, stops):
            cell = cells[str(int(values[start]))]
            cell["observations"][frame] = observation(yy[start:stop], xx[start:stop])
    measured = cell_signal(raw_stack, label_stack, source_frame_offset=offset, frame_interval_h=cadence)
    for identity, rows in measured.groupby("identity", sort=False):
        cell = cells[str(int(identity))]
        cell["signal_mean3"] = _json_values(rows.integrated_density_mean3)
        cell["signal_raw"] = _json_values(rows.integrated_density)
        cell["area"] = _json_values(rows.area_px)
        cell["crop_size_px"] = crop_size(cell["observations"])
    low, high = _range(raw_stack, display_range)
    return {"schema_version": 1, "method_version": METHOD_VERSION,
            "title": str(title) if title is not None else raw_path.stem,
            "frame_count": frames, "label_frame_count": len(label_stack),
            "source_frame_offset": offset, "frame_interval_h": cadence,
            "times_h": (np.arange(frames) * cadence).tolist(),
            "source_shape": [height, width], "ids": ids, "cells": cells,
            "display": {"range": [low, high], "lut": display_lut,
                        "filter": None, "resampling": "nearest", "show_motion_trails": False},
            "sources": {"raw": str(raw_path), "labels": str(label_path)},
            "source_sha256": {"raw": sha256(raw_path), "labels": sha256(label_path)},
            "frames": [_png(frame, low, high, display_lut) for frame in raw_stack]}


def measurement_rows(payload: dict):
    """One auditable display-table row per established identity/source frame."""
    for identity in payload["ids"]:
        cell = payload["cells"][str(identity)]
        for frame, time in enumerate(payload["times_h"]):
            yield {"cell_id": identity, "frame": frame, "time_h": time,
                   "mask_present": cell["observations"][frame] is not None,
                   "area_px": cell["area"][frame], "integrated_density": cell["signal_raw"][frame],
                   "integrated_density_mean3": cell["signal_mean3"][frame]}
