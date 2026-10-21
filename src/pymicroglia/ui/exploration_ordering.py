"""Order the displayed barcode through the existing matrix engines, read-only."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from pymicroglia.figure_tables.ordering import (
    DEFAULT_TRACE_ORDER, TRACE_ORDER_ALIASES, TRACE_ORDER_ENGINES,
    trace_comparable, trace_pattern_order,
)
from .exploration_data import (
    PREVIEW_CELLS, _observations, number, overview_indices, recording, table_path,
)


def displayed_matrix(observations, frame_count, metric, time_mean, scale):
    """Mirror the barcode's display mean and colour range; never edit samples."""
    stride, indices = overview_indices(frame_count)
    if time_mean != 1 and stride != 1:
        raise ValueError("Temporal means require exact overview frames.")
    identities = sorted(observations)[:PREVIEW_CELLS]
    values = np.array([[number(observations[i].get(f, {}).get(metric))
                        for f in indices] for i in identities], dtype=float)
    if time_mean != 1:
        source = values
        values = np.full_like(source, np.nan)
        half = time_mean // 2
        for position in range(half, len(indices) - half):
            window = source[:, position - half:position + half + 1]
            valid = np.isfinite(window).all(axis=1)
            values[valid, position] = window[valid].mean(axis=1)
    finite = values[np.isfinite(values)]
    shared_bounds = (finite.min(), finite.max()) if len(finite) else (0.0, 1.0)
    normalized = values.copy()
    for row, trace in enumerate(values):
        present = np.isfinite(trace)
        if not present.any():
            continue
        low, high = (trace[present].min(), trace[present].max()) if scale == "cell" else shared_bounds
        if low == high:
            low, high = low - .5, high + .5
        normalized[row, present] = (trace[present] - low) / (high - low)
    return pd.DataFrame(normalized, index=identities, columns=indices)


@lru_cache(maxsize=32)
def _cached_order(root, movie_json, size, modified, metric, engine, time_mean, scale, subset_json="null"):
    # Size, modification time and alignment settings invalidate the source cache.
    _, observations, frames, metrics = _observations(Path(root), json.loads(movie_json))
    subset = json.loads(subset_json)
    if subset is not None:
        if any(i not in observations for i in subset):
            raise ValueError("Barcode filter identities must belong to this recording")
        observations = {i: observations[i] for i in subset}
    if metric not in {m["key"] for m in metrics}:
        raise ValueError("This ordering metric is not available in the recording.")
    identities = sorted(observations)[:PREVIEW_CELLS]
    comparable = None
    if engine != "identity":
        matrix = displayed_matrix(observations, len(frames), metric, time_mean, scale)
        identities = trace_pattern_order(matrix, method=engine)
        comparable = int(trace_comparable(matrix).sum())
    return {"engine": engine, "identities": [int(i) for i in identities],
            "metric": metric, "time_mean": time_mean, "scale": scale,
            "shown_cells": len(identities), "comparable_cells": comparable}


def load_barcode_order(run, stem, metric, engine=DEFAULT_TRACE_ORDER,
                       time_mean=1, scale="cell", identities=None):
    engine = TRACE_ORDER_ALIASES.get(engine, engine)
    if engine not in {*TRACE_ORDER_ENGINES, "identity"}:
        raise ValueError("Unknown barcode ordering engine.")
    if time_mean not in {1, 3, 7}:
        raise ValueError("Barcode time mean must be 1, 3 or 7 frames.")
    if scale not in {"cell", "recording"}:
        raise ValueError("Barcode scale must be cell or recording.")
    root, _, movie = recording(run, stem)
    path = table_path(root, movie, "cell_frame")
    if path is None:
        raise ValueError("This recording has no saved cell-frame measurements.")
    stat = path.stat()
    try:
        subset = sorted(set(int(i) for i in identities.split(","))) if identities else None
    except (ValueError, AttributeError):
        raise ValueError("Choose valid cell identities for barcode ordering") from None
    return _cached_order(str(root), json.dumps(movie, sort_keys=True),
                         stat.st_size, stat.st_mtime_ns, metric, engine, time_mean, scale, json.dumps(subset))
