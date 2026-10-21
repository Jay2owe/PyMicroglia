"""Bounded, fit-free chart data from a selected saved measurement run."""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path


# These are measured columns. The workbench never infers a rhythm from them.
PAGE_METRICS = {
    "rhythms": ("integrated_density", "corrected_integrated_density", "signal_mean", "dff"),
    "tissue": ("domain_occupancy", "footprint_px", "area_px", "local_density"),
    "cell_explorer": ("integrated_density", "area_px", "speed", "ramification_index"),
    "shape": ("area_px", "ramification_index", "branch_count", "turnover_index"),
    "movement": ("speed", "step_px", "displacement_from_first_px", "cumulative_path_px"),
    "states": ("turnover_index", "ramification_index", "area_px", "speed"),
    "relationships": ("nearest_neighbour_px", "neighbour_cells_present", "local_density", "domain_occupancy"),
    "interventions": ("integrated_density", "area_px", "speed", "turnover_index"),
    "methods": ("observed_px", "inferred_px", "inferred_fraction", "unresolved_px"),
}
METRICS = tuple(dict.fromkeys(metric for group in PAGE_METRICS.values() for metric in group))
MAX_CELLS = 32
MAX_POINTS = 450
MAX_SOURCE_BYTES = 256 * 1024 * 1024


def _number(value: str | None) -> float | None:
    try:
        number = float(value) if value not in (None, "") else float("nan")
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _within(root: Path, candidate: Path) -> Path | None:
    path = candidate.resolve()
    return path if path.is_file() and root in path.parents else None


def _table_paths(root: Path, manifest: dict, name: str) -> list[Path]:
    pooled = [root / "pooled" / f"{name}.csv", root / "pooled" / "tables" / f"{name}.csv",
              root / f"{name}.csv"]
    for candidate in pooled:
        found = _within(root, candidate)
        if found:
            return [found]
    found = []
    for movie in manifest.get("movies", [])[:32]:
        stem = str(movie.get("stem") or "")
        if not stem or Path(stem).name != stem:
            continue
        candidates = [root / "measure" / stem / f"{name}.csv",
                      root / stem / "tables" / f"{name}.csv"]
        for candidate in candidates:
            path = _within(root, candidate)
            if path:
                found.append(path)
                break
    return found


def _source_budget(paths: list[Path]) -> str | None:
    size = sum(path.stat().st_size for path in paths)
    if size > MAX_SOURCE_BYTES:
        return f"Trace tables exceed the {MAX_SOURCE_BYTES // (1024 * 1024)} MB automatic preview limit; saved figures remain available."
    if size > 64 * 1024 * 1024 and shutil.disk_usage(paths[0].anchor).free < size * 2:
        return "There is not enough free local space to load these trace tables automatically."
    return None


def _rhythms(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                if len(rows) >= 3000:
                    return rows
                rows.append({"stem": row.get("stem") or path.parent.name,
                             "identity": str(row.get("identity") or ""),
                             "metric": row.get("metric") or "",
                             "status": row.get("rhythm_status") or "unknown",
                             "period_hours": _number(row.get("best_period_hours") or row.get("period_hours"))})
    return rows


def load_quicklook(run: str | Path) -> dict:
    """Read only the named run's measurement CSVs; never fit or write figures."""
    root = Path(run).expanduser().resolve(strict=True)
    manifest_path = root / "manifest.json"
    if not root.is_dir() or not manifest_path.is_file():
        raise ValueError("Choose a saved PyMicroglia measurement run with manifest.json")
    if manifest_path.stat().st_size > 5 * 1024 * 1024:
        raise ValueError("The measurement manifest is too large for a quick preview")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("movies"), list):
        raise ValueError("This is not a PyMicroglia measurement run")
    paths = _table_paths(root, manifest, "cell_frame")
    result = {"run": str(root), "pages": PAGE_METRICS, "metrics": [], "traces": [],
              "rhythms": [], "total_cells": 0, "displayed_cells": 0,
              "sources": [str(path.relative_to(root)) for path in paths], "message": ""}
    if not paths:
        result["message"] = "No measured cell traces were found in this run. Saved figures may still be available."
        return result
    rhythm_paths = _table_paths(root, manifest, "rhythms")
    budget = _source_budget(paths + rhythm_paths)
    if budget:
        result["message"] = budget
        return result
    traces: dict[tuple[str, str], dict] = {}
    all_cells: set[tuple[str, str]] = set()
    available: set[str] = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or not {"identity", "hours"}.issubset(reader.fieldnames):
                continue
            present = set(reader.fieldnames) & set(METRICS)
            available.update(present)
            for row in reader:
                stem = str(row.get("stem") or path.parent.name)
                identity = str(row.get("identity") or "")
                if not identity:
                    continue
                key = (stem, identity)
                all_cells.add(key)
                if key not in traces:
                    if len(traces) >= MAX_CELLS:
                        continue
                    traces[key] = {"stem": stem, "identity": identity, "count": 0,
                                   "stride": 1, "points": []}
                trace = traces[key]
                index = trace["count"]
                trace["count"] += 1
                if index % trace["stride"]:
                    continue
                hour = _number(row.get("hours"))
                if hour is None:
                    continue
                trace["points"].append((index, hour, {metric: _number(row.get(metric))
                                                      for metric in present}))
                if len(trace["points"]) > MAX_POINTS:
                    trace["stride"] *= 2
                    trace["points"] = [point for point in trace["points"]
                                       if point[0] % trace["stride"] == 0]
    result["metrics"] = sorted(available)
    result["total_cells"] = len(all_cells)
    result["displayed_cells"] = len(traces)
    result["traces"] = [{"stem": item["stem"], "identity": item["identity"],
                         "hours": [point[1] for point in item["points"]],
                         "values": {metric: [point[2].get(metric) for point in item["points"]]
                                    for metric in available}}
                        for item in traces.values()]
    result["rhythms"] = _rhythms(rhythm_paths)
    if not available:
        result["message"] = "The saved cell table has no metrics supported by the quick charts."
    return result
