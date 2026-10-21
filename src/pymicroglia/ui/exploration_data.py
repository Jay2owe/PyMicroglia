"""Read-only saved observations for coordinated exploration; no analysis runs."""
from __future__ import annotations

import csv
import json
import math
from functools import lru_cache
from pathlib import Path
import shutil
import numpy as np

MAX_TABLE_BYTES = 96 * 1024**2
MAX_ROWS = 500_000
MAX_FRAMES = 100_000
PREVIEW_FRAMES = 500
PREVIEW_CELLS = 128
KEYS = {"identity", "frame_index", "imagej_frame", "source_imagej_frame", "hours", "stem", "condition", "subject"}


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def manifest(run):
    root = Path(run).expanduser().resolve()
    for path in (root / "manifest.json", root / ".auto-organotypic/manifest.json"):
        if path.is_file():
            if path.stat().st_size > 8 * 1024**2:
                raise ValueError("The run manifest exceeds the interactive metadata budget.")
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data.get("movies"), list) or not data["movies"]:
                raise ValueError("This folder has no saved measurement recordings.")
            return root, data
    raise ValueError("Choose a saved measurement run containing a manifest.")


def recording(run, stem=None):
    root, data = manifest(run)
    movie = next((m for m in data["movies"] if stem is None or m["stem"] == stem), None)
    if movie is None:
        raise ValueError("The selected recording is not declared by this run.")
    return root, data, movie


def table_path(root, movie, name):
    if name in {'lifecycle_events', 'lifecycle_cells'}:
        from .exploration_lifecycle import attached_table
        attached = attached_table(root, movie, name)
        if attached is not None:
            return attached
    record = movie.get("tables", {}).get(name, {})
    if isinstance(record, str):
        record = {"path": record}
    folder = "tracker" if record.get("origin") == "tracker" else "measure"
    stored = record.get("path", name + ".csv")
    candidates = [root / stored, root / folder / movie["stem"] / stored,
                  root / folder / str(record.get("folder", movie["stem"])) / stored,
                  root / "measure" / movie["stem"] / (name + ".csv")]
    for path in candidates:
        path = path.resolve()
        if not path.is_relative_to(root):
            raise ValueError("A saved table points outside the selected run.")
        if path.is_file():
            return path
    return None


@lru_cache(maxsize=2)
def _read_csv(path, size, modified):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        rows = []
        for row in csv.DictReader(stream):
            if len(rows) == MAX_ROWS:
                raise ValueError("This table exceeds the interactive row budget; use a smaller saved run for exploration.")
            rows.append(row)
        return rows


def read_rows(path):
    if path is None:
        return []
    stat = path.stat()
    if stat.st_size > MAX_TABLE_BYTES:
        raise ValueError("This saved table exceeds the 96 MiB interactive budget.")
    if shutil.disk_usage(path.parent).free < stat.st_size * 4 + 64 * 1024**2:
        raise ValueError("There is insufficient local space to read this saved table safely.")
    return _read_csv(str(path), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=1)
def column_metadata():
    from pymicroglia.measure import modules  # populate the existing declarations
    modules.load()
    from pymicroglia.measure.declare import declared_columns
    return declared_columns()


def _observations_uncached(root, movie):
    path = table_path(root, movie, "cell_frame")
    if path is None:
        raise ValueError(f"{movie['stem']}: no saved cell-frame measurements were found.")
    rows = read_rows(path)
    if not rows:
        raise ValueError("This recording has no observed cells.")
    if not {"identity", "frame_index", "hours"}.issubset(rows[0]):
        raise ValueError("Saved observations must include identity, frame_index and hours.")
    offset = int(movie.get("spec", {}).get("source_frame_offset", 0))
    by_cell, times = {}, {}
    for row in rows:
        identity, frame, hour = number(row.get("identity")), number(row.get("frame_index")), number(row.get("hours"))
        if identity is None or frame is None or identity <= 0 or frame < 0 or frame != int(frame) or identity != int(identity):
            raise ValueError("Saved observations contain an invalid identity or frame index.")
        identity, frame = int(identity), int(frame)
        if frame >= MAX_FRAMES:
            raise ValueError("This recording exceeds the 100,000-frame interactive budget.")
        source = number(row.get("source_imagej_frame"))
        if source is not None and source != frame + offset + 1:
            raise ValueError("Saved frame indices disagree with the declared source offset.")
        if frame in by_cell.setdefault(identity, {}):
            raise ValueError("Saved cell-frame observations contain a duplicate identity/frame pair.")
        if frame in times and times[frame] != hour:
            raise ValueError("Cells disagree about the time of a saved frame.")
        times[frame] = hour
        by_cell[identity][frame] = row
    count = max(times) + 1
    summary = movie.get("summary", {})
    count = max(count, int(summary.get("frames", summary.get("n_frames", count))))
    if count > MAX_FRAMES:
        raise ValueError("This recording exceeds the interactive frame budget.")
    # Only extend time through missing observations when a regular saved axis supports it.
    known = sorted((f, h) for f, h in times.items() if h is not None)
    origin = step = None
    if len(known) >= 2:
        f0, h0 = known[0]; f1, h1 = known[-1]
        step = (h1 - h0) / (f1 - f0)
        origin = h0 - step * f0
        if step <= 0 or any(abs(h - (origin + step * f)) > 1e-6 for f, h in known):
            origin = step = None
    frames = [{"frame_index": f, "source_frame_index": f + offset,
               "hours": times.get(f, None if step is None else origin + step * f)} for f in range(count)]
    keys = [k for k in rows[0] if k not in KEYS and any(number(r.get(k)) is not None for r in rows[:1000])]
    meta = column_metadata()
    metrics = [{"key": k, "label": meta[k].label if k in meta else k.replace("_", " ").capitalize(),
                "unit": meta[k].unit if k in meta else "", "role": meta[k].role if k in meta else "measurement"} for k in keys]
    return path, by_cell, frames, metrics


@lru_cache(maxsize=2)
def _cached_observations(root, description, size, modified):
    return _observations_uncached(Path(root), json.loads(description))


def _observations(root, movie):
    path = table_path(root, movie, "cell_frame")
    if path is None:
        raise ValueError(f"{movie['stem']}: no saved cell-frame measurements were found.")
    stat = path.stat()
    # Every cache is invalidated by source bytes/mtime and the declared alignment.
    if stat.st_size > MAX_TABLE_BYTES:
        raise ValueError("This saved table exceeds the 96 MiB interactive budget.")
    return _cached_observations(str(root), json.dumps(movie, sort_keys=True), stat.st_size, stat.st_mtime_ns)


def _cell(identity, observations, indices, metrics, movie):
    condition = movie.get("condition", {})
    group = condition.get("condition") if isinstance(condition, dict) else condition
    spec = movie.get("spec", {})
    rows = [observations.get(i) for i in indices]
    observed = [row is not None and str(row.get('mask_present','true')).lower() not in {'false','0','no'} for row in rows]
    present = [(i, row) for i, row in enumerate(rows) if observed[i]]
    values = {}
    for metric in metrics:
        key = metric["key"]
        column = [None] * len(rows)
        for i, row in present:
            column[i] = number(row.get(key))
        values[key] = column
    return {"identity": identity, "group": group or spec.get("condition") or "Unassigned",
            "subject": movie.get("subject") or spec.get("subject"),
              "observed": observed, "values": values}


def load_exploration(run, stem=None, overview=True):
    root, saved, movie = recording(run, stem)
    path, observations, frames, metrics = _observations(root, movie)
    stride, indices = overview_indices(len(frames))
    identities = sorted(observations)
    from .exploration_results import saved_results
    from pymicroglia.figure_tables.ordering import trace_order_catalogue
    from pymicroglia.visualisation.panels import colour
    from .measurement_catalogue import measurement_catalogue
    settings = saved.get("settings", {})
    from .measurement_geometry import display_options
    microns = number(settings.get("microns_per_pixel"))
    return {"version": 1, "run": str(root), "stem": movie["stem"],
            "recordings": [{"stem": m["stem"], "condition": m.get("spec", {}).get("condition"), "subject": m.get("subject")} for m in saved["movies"]],
            "frames": frames, "sample_indices": indices, "metrics": metrics,
            "measurement_catalogue": measurement_catalogue(metrics),
              "measurement_display": display_options(movie),
            "identities": identities, "cells": [_cell(i, observations[i], indices,
                metrics if overview else [m for m in metrics if m["key"] in {"centroid_x", "centroid_y"}], movie)
                for i in identities[:PREVIEW_CELLS]],
            "total_cells": len(identities), "source_table": str(path),
            "barcode_ordering": trace_order_catalogue(),
            "trace_style": {"rhythmic": colour("circadian_red"), "other": colour("nan_text")},
            "preview": {"stride": stride, "shown_cells": min(len(identities), PREVIEW_CELLS), "total_cells": len(identities),
                        **({"pending": True} if not overview else {}),
                        "message": f"Recording overview: {min(len(identities), PREVIEW_CELLS)} / {len(identities)} cells; every {stride} frame(s). Selected-cell trace uses exact saved frames."},
            "results": saved_results(root, movie), "windows": movie.get("windows", []),
            "units": {"length": "µm" if microns else "px", "interval": f"{settings.get('frame_interval_min', 'saved')} min"},
            "centroid_pixels": not bool(microns), "centroid_pixel_scale": 1 / microns if microns else 1}


def load_overview(run, stem):
    """The complete existing overview, loaded after current-frame media."""
    root, _, movie = recording(run, stem)
    _, observations, frames, metrics = _observations(root, movie)
    stride, indices = overview_indices(len(frames))
    identities = sorted(observations)
    return {"run": str(root), "stem": movie["stem"],
            "cells": [_cell(i, observations[i], indices, metrics, movie) for i in identities[:PREVIEW_CELLS]]}


def overview_indices(frame_count):
    """One sampling axis shared by overview drawing and its row ordering."""
    stride = max(1, math.ceil(frame_count / PREVIEW_FRAMES))
    indices = list(range(0, frame_count, stride))
    if indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)
    return stride, indices


def load_cell(run, stem, identity):
    root, _, movie = recording(run, stem)
    _, observations, frames, metrics = _observations(root, movie)
    if identity not in observations:
        raise ValueError("This identity has no saved observations in this recording.")
    return {"stem": stem, "frames": frames, "sample_indices": list(range(len(frames))),
            "cell": _cell(identity, observations[identity], range(len(frames)), metrics, movie)}


def load_frame_values(run, stem, frame):
    root, _, movie = recording(run, stem)
    _, observations, frames, metrics = _observations(root, movie)
    if not 0 <= frame < len(frames):
        raise ValueError("Frame is outside the saved recording.")
    return {"frame": frames[frame], "cells": [_cell(i, obs, [frame], metrics, movie) for i, obs in observations.items() if frame in obs]}


def load_trace_grid(run, stem, metric, offset=0, limit=128, metrics=None):
    """Every identity's exact measured samples, paged without overview limits.

    Sparse frame indices preserve gaps without allocating a dense matrix of
    every identity against every frame. Requested measurements share one read.
    """
    root, _, movie = recording(run, stem)
    path, observations, frames, definitions = _observations(root, movie)
    available = {m["key"] for m in definitions}
    # Keep the single-metric API and its values field for existing consumers.
    requested = list(dict.fromkeys([metric, *(metrics or [])]))
    if any(key not in available for key in requested):
        raise ValueError("The trace metric is not available in this recording.")
    if len(requested) > 32:
        raise ValueError("Choose at most 32 measurements per trace request.")
    if offset < 0 or not 1 <= limit <= 128:
        raise ValueError("Trace pages require a nonnegative offset and a limit of 1 to 128 cells.")
    identities = sorted(observations)
    cells = []
    for identity in identities[offset:offset + limit]:
        indices = sorted(observations[identity])
        values = {key: [number(observations[identity][f].get(key)) for f in indices]
                  for key in requested}
        cells.append({"identity": identity, "indices": indices,
                      "values": values[metric], "values_by_metric": values})
    next_offset = offset + len(cells)
    return {"stem": stem, "metric": metric, "metrics": requested, "cells": cells,
            "total_cells": len(identities), "source_table": str(path),
            "next_offset": next_offset if next_offset < len(identities) else None,
            "exact": True}
