"""Bounded, read-only adapters for the native tissue maps and track timelines."""
from __future__ import annotations

import base64
from functools import lru_cache
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image

from .exploration_data import _observations, number, read_rows, recording, table_path
from .exploration_images import declared_source, signature

MAX_WORK_BYTES = 512 * 1024**2


@lru_cache(maxsize=2)
def _labels(path, size, modified):
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        shape = series.shape
        if len(shape) != 3 or series.axes[-2:] != "YX":
            raise ValueError("Tissue tectonics needs saved frame-by-row-by-column labels.")
        # Native occupancy preparation holds several float arrays at its peak.
        if np.prod(shape) * (series.dtype.itemsize + 32) > MAX_WORK_BYTES:
            raise ValueError("This mask movie exceeds the 512 MiB interactive tissue budget; use the tissue tectonics pipeline for this recording.")
        labels = series.asarray()
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError("Tissue tectonics needs nonnegative integer cell labels.")
    labels.setflags(write=False)
    return labels


def _defaults():
    from pymicroglia.visualisation.figures import load
    return {option.name: option.default for option in load()["tissue_tectonics"].options}


@lru_cache(maxsize=24)
def _map(path, size, modified, layer, hours_json, frame_json, evidence_json, events_json, period_limits_json, identities_json="null"):
    from pymicroglia.figure_tables import territory_values as values
    from pymicroglia.figure_tables import tissue
    from pymicroglia.figure_tables.metric_maps import cell_metric
    from matplotlib import colormaps

    labels = _labels(path, size, modified)
    identities = json.loads(identities_json)
    if identities is not None:
        # A display-only array for the native reducers; the immutable source is untouched.
        labels = np.where(np.isin(labels, identities), labels, 0)
    hours = np.asarray(json.loads(hours_json), dtype=float)
    if len(hours) != len(labels) or not np.isfinite(hours).all() or np.any(np.diff(hours) <= 0):
        raise ValueError("Saved measurement times must align with every mask frame.")
    defaults = _defaults()
    assignment = values.metric_assignment(defaults["map_assignment"])
    span = max(float(hours[-1] - hours[0]), np.finfo(float).eps)
    message = ""
    if layer == "first_coverage":
        array = values.first_coverage_time(labels, hours)
        limits, caption = (0, span), "First covered (h from recording start)"
    elif layer == "cumulative_occupancy":
        array = values.cumulative_occupancy(labels, hours)[-1]
        array[~np.any(labels > 0, axis=0)] = np.nan
        limits, caption = (0, span), "Cumulative occupied time (h)"
    elif layer == "unique_cells":
        array = values.distinct_owner_count(labels).astype(float)
        array[array == 0] = np.nan
        high = max(1, float(np.nanmax(array))) if np.isfinite(array).any() else 1
        limits, caption = (1, high + (high == 1)), "Distinct cells covering each pixel"
    elif layer in {"speed", "significant_period"}:
        if layer == "speed":
            frame = pd.DataFrame(json.loads(frame_json))
            series = (tissue._speed_values(frame, defaults["map_summary"])
                      if "speed" in frame and np.isfinite(pd.to_numeric(frame["speed"], errors="coerce")).any()
                      else pd.Series(dtype=float))
            caption = "Median cell speed (saved measurement units)"
        else:
            evidence = json.loads(evidence_json)
            series = pd.Series({r["identity"]: r["period_hours"] for r in evidence
                                if r.get("status") == "rhythmic" and number(r.get("period_hours")) is not None}, dtype=float)
            caption = "Significant cell period (h); saved tests only"
        array, _, _, limits = cell_metric(labels, series, metric=layer, assignment=assignment,
                                          range_mode="full" if layer == "significant_period" else defaults["map_range"],
                                          limits=tuple(json.loads(period_limits_json)) if layer == "significant_period" else None,
                                          allow_empty=True)
        caption += "; overlap: " + values.MAP_ASSIGNMENT_LABELS[assignment]
        if not np.isfinite(array).any():
            message = "No saved significant cell periods for this measurement." if layer == "significant_period" else "No saved cell speeds."
        if layer == "significant_period" and any(r.get("status") == "rhythmic" and r.get("period_underdetermined") for r in json.loads(evidence_json)):
            caption += "; includes short-record estimates"
    else:
        events = pd.DataFrame(json.loads(events_json))
        if events.empty:
            array = np.full(labels.shape[1:], np.nan)
            message = "No saved tracker contact-separation events for this recording."
        else:
            array, _ = values.split_event_areas(labels, events)
            array = array.astype(float)
            array[array == 0] = np.nan
            if not np.isfinite(array).any():
                message = "No recorded contact-separation events."
        high = max(1, float(np.nanmax(array))) if np.isfinite(array).any() else 1
        limits, caption = (1, high + (high == 1)), "Tracker contact separations per pixel"
    cmap = tissue._panel_luts(list(defaults["map_luts"]))[layer] or "viridis"
    low, high = map(float, limits)
    normalized = np.nan_to_num(np.clip((array - low) / max(high - low, np.finfo(float).eps), 0, 1))
    rgba = colormaps[cmap](normalized, bytes=True)
    rgba[:, :, 3] = np.isfinite(array).astype(np.uint8) * 255
    stream = BytesIO()
    Image.fromarray(rgba).save(stream, format="PNG")
    return {"image": "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode(),
            "width": labels.shape[2], "height": labels.shape[1], "limits": [low, high],
            "layer": layer, "label": caption, "message": message, "lut": cmap,
            "legend": colormaps[cmap](np.linspace(0, 1, 128), bytes=True)[:, :3].tolist(),
            "scope": "Whole saved recording" if identities is None else f"Filtered cohort: {len(identities)} cells; whole recording", "source": path,
            "producer": "pymicroglia.figure_tables.tissue / territory_values / metric_maps"}


def load_tissue_map(run, stem, layer="first_coverage", metric=None, *, identities=None, analysis_id=None):
    from pymicroglia.figure_tables.tissue import PANEL_ORDER, _frame_hours
    from .exploration_results import saved_results

    if layer not in PANEL_ORDER:
        raise ValueError("Unknown tissue tectonics layer: " + layer)
    root, _, movie = recording(run, stem)
    path, observations, frames, _ = _observations(root, movie)
    if identities is not None:
        try:
            identities = sorted(set(int(i) for i in identities.split(","))) if isinstance(identities, str) and identities else list(identities) if not isinstance(identities, str) else []
        except (TypeError, ValueError):
            raise ValueError("Choose valid saved identities for the tissue map") from None
        if any(isinstance(i, bool) or not isinstance(i, int) or i not in observations for i in identities):
            raise ValueError("Tissue map identities must belong to this recording")
    # Fingerprint the inputs used by each native reducer; no new rhythm fit runs.
    rows = [row for cell in observations.values() for row in cell.values()] if layer == "speed" else []
    if layer == "significant_period" and analysis_id is not None:
        from .exploration_rhythms import load_attached_rhythms
        evidence = load_attached_rhythms(root, movie, analysis_id) or {}
    else:
        evidence = saved_results(root, movie)["rhythms"] if layer == "significant_period" else {}
    metric = metric or next(iter(evidence.get("metrics", [])), None)
    tested = [] if evidence.get("preview") else [row for row in evidence.get("rows", []) if row.get("metric") == metric]
    if identities is not None:
        tested = [row for row in tested if row["identity"] in identities]
    defaults, resolved = _defaults(), evidence.get("resolved_options", {})
    period_limits = [resolved.get(key, defaults[key]) for key in ("period_min_hours", "period_max_hours")]
    event_path = table_path(root, movie, "history_merge_split_events") if layer == "splitting_events" else None
    events = read_rows(event_path) if event_path is not None else []
    mask_signature = signature(declared_source(root, movie, "labels"))
    frame_path = table_path(root, movie, "frame_summary")
    hours = (_frame_hours(pd.DataFrame(read_rows(frame_path)), len(_labels(*mask_signature))).tolist()
             if frame_path else [frame["hours"] for frame in frames])
    result = _map(*mask_signature, layer, json.dumps(hours), json.dumps(rows),
                  json.dumps(tested), json.dumps(events), json.dumps(period_limits), json.dumps(identities))
    return {**result, "stem": stem, "metric": metric, "measurement_source": str(path)}


def load_track_timeline(run, stem, order="start"):
    from pymicroglia.figure_tables.lifecycle import event_timelines
    from .exploration_history import timeline_explanations

    root, _, movie = recording(run, stem)
    path, observations, frames, _ = _observations(root, movie)
    saved_path = table_path(root, movie, "lifecycle_cells")
    saved = {int(r["identity"]): r for r in read_rows(saved_path)} if saved_path else {}
    explanations, legend, history_sources, history_message = timeline_explanations(root, movie, observations, frames)
    cells = []
    for identity, rows in observations.items():
        indices = sorted(rows)
        start, end = indices[0], indices[-1]
        segments = [[start, start]]
        for frame in indices[1:]:
            if frame == segments[-1][1] + 1:
                segments[-1][1] = frame
            else:
                segments.append([frame, frame])
        native = saved.get(identity, {})
        cells.append({"identity": identity, "start_hours": frames[start]["hours"],
                      "end_hours": frames[end]["hours"], "start_frame": start, "end_frame": end,
                      "start_event": native.get("start_event") or "first_observed",
                      "end_event": native.get("end_event") or "last_observed",
                      "start_status": native.get("start_status"), "end_status": native.get("end_status"),
                      "segments": segments, "observations": len(indices), **explanations[identity]})
    # Native ordering is shared with the exported lifecycle figure. Unknown
    # endpoints remain observation endpoints rather than inferred birth/death.
    prepared = event_timelines(cells, order=order, lineage=False)
    lanes = dict(prepared["data"][["identity", "lane"]].drop_duplicates().itertuples(index=False, name=None))
    return {"stem": stem, "cells": sorted(cells, key=lambda c: lanes[c["identity"]]),
            "source": str(saved_path or path), "saved_events": bool(saved_path),
            "gap_legend": legend, "history_sources": history_sources, "history_message": history_message,
            "meaning": "Solid bars: observed masks. Hatched gaps: missing observations coloured by recorded tracker reason; point or click for details. Orange gaps have no recorded explanation. Crosses mark recorded silent endings away from the field edge. Tracker assessments are not confirmed biological events.",
            "producer": "pymicroglia.figure_tables.lifecycle.event_timelines"}
