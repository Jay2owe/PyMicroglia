"""Preserve independent observations before same-time or shifted matching."""
from __future__ import annotations
from pymicroglia._sources import source_file

from importlib.metadata import version as library_version
from pathlib import Path

import numpy as np
import pandas as pd

from pymicroglia.pipelines._contracts import ArtifactRef, Settings, StepResult, content_id
from pymicroglia.pipelines.relationships.options import LAG_CONVENTION
from pymicroglia.pipelines._screening import _write_json, file_hash, read_verified_tables, write_table


KEYS = ["source_run", "movie", "identity"]
PAIR_KEYS = KEYS + ["pair_id", "reference", "target"]
TRACE_COLUMNS = KEYS + ["measurement", "trace_id", "observation_id", "source_position", "frame_index",
    "hours", "time_kind", "raw_value", "raw_kind", "processed_value", "within_range", "raw_valid", "processed_valid"]
MATCH_COLUMNS = ["reference_observation", "target_observation", "reference_hours", "target_hours",
                 "reference_value", "target_value", "matching_error_hours"]
SUPPORT_COLUMNS = PAIR_KEYS + ["question", "requested", "status", "reason", "lag_hours", "paired_observations",
    "overlap_span_hours", "reference_start_hours", "reference_end_hours", "target_start_hours", "target_end_hours",
    "ambiguous_observations", "gap_count", "largest_gap_hours"]


def _kind(value):
    if np.isnan(value): return "missing"
    if np.isposinf(value): return "positive_infinity"
    if np.isneginf(value): return "negative_infinity"
    return "finite"


def lag_values(question):
    if not question["enabled"]: return []
    low, high = question["range_hours"]
    count = int(round((high - low) / question["resolution_hours"]))
    return np.linspace(low, high, count + 1).tolist()


def match_observations(reference, target, lag_hours, support):
    """Shift independent clocks, then match valid values without reusing a point.

    Exact matching allows only floating-point subtraction roundoff. Nearest-unique
    matching rejects equidistant candidates and all collisions, so input row order
    cannot decide which observation enters the analysis. Gaps are never compressed.
    """
    left = reference.loc[reference.within_range & reference.processed_valid].sort_values("hours")
    right = target.loc[target.within_range & target.processed_valid].sort_values("hours")
    empty = pd.DataFrame(columns=MATCH_COLUMNS)
    if left.empty or right.empty:
        return empty, {"ambiguous_observations": 0}
    lt, rt = left.hours.to_numpy(float), right.hours.to_numpy(float)
    if np.any(np.diff(lt) <= 0) or np.any(np.diff(rt) <= 0):
        raise ValueError("Prepared observation clocks must be unique and increasing")
    desired = lt - float(lag_hours)
    scale = max(1., float(np.max(np.abs(np.r_[lt, rt, desired]))))
    roundoff = np.finfo(float).eps * scale * 8
    tolerance = 0. if support["matching"] == "exact" else support["matching_tolerance_hours"]
    proposed, ambiguous = {}, set()
    for i, value in enumerate(desired):
        insertion = int(np.searchsorted(rt, value))
        candidates = [j for j in (insertion - 1, insertion) if 0 <= j < len(rt)]
        distances = [abs(rt[j] - value) for j in candidates]
        closest = min(distances)
        if closest > tolerance + roundoff: continue
        best = [j for j, distance in zip(candidates, distances) if abs(distance - closest) <= roundoff]
        if len(best) != 1:
            ambiguous.add(i)
        else:
            proposed[i] = best[0]
    counts = {}
    for j in proposed.values(): counts[j] = counts.get(j, 0) + 1
    ambiguous.update(i for i, j in proposed.items() if counts[j] > 1)
    records = []
    for i, j in proposed.items():
        if i in ambiguous: continue
        a, b = left.iloc[i], right.iloc[j]
        records.append({"reference_observation": a.observation_id, "target_observation": b.observation_id,
            "reference_hours": a.hours, "target_hours": b.hours,
            "reference_value": a.processed_value, "target_value": b.processed_value,
            "matching_error_hours": float(b.hours - (a.hours - lag_hours))})
    return pd.DataFrame(records, columns=MATCH_COLUMNS), {"ambiguous_observations": len(ambiguous)}


def paired_support(matches, diagnostics, settings):
    n = len(matches)
    lt, rt = matches.reference_hours.to_numpy(float), matches.target_hours.to_numpy(float)
    span = float(lt[-1] - lt[0]) if n else 0.
    gaps = np.maximum(np.diff(lt), np.diff(rt)) if n > 1 else np.array([])
    enough = n >= settings["min_observations"] and span >= settings["min_span_hours"]
    return {"status": "eligible" if enough else "insufficient",
        "reason": "Prepared observations; statistical-method eligibility is evaluated separately" if enough else
                  "Too few matched observations or too little matched time span",
        "paired_observations": n, "overlap_span_hours": span,
        "reference_start_hours": float(lt[0]) if n else None, "reference_end_hours": float(lt[-1]) if n else None,
        "target_start_hours": float(rt[0]) if n else None, "target_end_hours": float(rt[-1]) if n else None,
        "ambiguous_observations": diagnostics["ambiguous_observations"],
        "gap_count": int(np.sum(gaps > settings["max_gap_hours"])),
        "largest_gap_hours": float(np.max(gaps)) if len(gaps) else None}


def _prepare_trace(resolved, cell, measurement, source):
    request = resolved.request
    key = {**cell.as_dict(), "measurement": measurement.column}
    trace_id = content_id({**key, "table": measurement.table})
    chosen = source.loc[source.stem.eq(cell.movie) & source.identity.eq(cell.identity)].copy()
    details = {**key, "trace_id": trace_id, "table": measurement.table, "unit": measurement.unit,
               "label": measurement.label, "summary": measurement.summary, "representation": request.representation}
    if "hours" not in measurement.grain and "frame_index" not in measurement.grain:
        value = float(chosen[measurement.column].iloc[0]) if len(chosen) else np.nan
        scalar = {**key, "trace_id": trace_id, "table": measurement.table, "value": value, "value_kind": _kind(value)}
        metadata = {**details, "kind": "scalar", "status": "available" if np.isfinite(value) else "missing",
                    "reason": "Original saved whole-recording summary", "valid_processed": int(np.isfinite(value)),
                    "processed_span_hours": None}
        return pd.DataFrame(columns=TRACE_COLUMNS), metadata, scalar, {**details, "action": "saved-scalar"}
    chosen = chosen.sort_values("hours", kind="stable")
    hours = chosen.hours.to_numpy(float)
    raw = chosen[measurement.column].to_numpy(float)
    finite_time, finite_raw = np.isfinite(hours), np.isfinite(raw)
    selected = finite_time.copy()
    if request.time_range_hours is not None:
        start, end = request.time_range_hours
        selected &= (hours >= start) & (hours < end)
    recorded = hours[finite_time]
    recorded_start = float(np.min(recorded)) if len(recorded) else None
    recorded_end = float(np.max(recorded)) if len(recorded) else None
    start, end = request.time_range_hours or (recorded_start, recorded_end)
    covered = None
    if start is not None and end > start:
        covered = (max(0., min(end, recorded_end) - max(start, recorded_start)) / (end - start)
                   if recorded_start is not None else 0.)
    processed = np.full(len(raw), np.nan)
    status, reason = "prepared", "Original observations retained"
    processing = {**details, "action": request.representation, "settings": resolved.detrending.as_dict()}
    selected_hours = hours[selected]
    if len(np.unique(selected_hours)) != len(selected_hours):
        status, reason = "invalid", "Duplicate observation times within the selected cell/measurement"
    elif not np.any(selected & finite_raw):
        status, reason = "missing", "No finite observed values in the requested time range"
    elif request.representation == "raw":
        processed[selected & finite_raw] = raw[selected & finite_raw]
    else:
        import pymicroglia.workbench as circadian
        try:
            values = raw[selected].copy()
            values[~np.isfinite(values)] = np.nan
            result = circadian.detrend_trace(selected_hours, values, resolved.detrending.as_dict())
        except circadian.cw.WorkbenchInputError as error:
            status, reason = "processing-unavailable", str(error)
        else:
            processing["workbench_result"] = result
            clock = np.asarray(result["processed_trace"]["hours"], dtype=float)
            output = np.asarray(result["values"], dtype=float)
            if (len(clock) != len(selected_hours) or len(output) != len(selected_hours)
                    or not np.allclose(clock, selected_hours, rtol=0., atol=1e-9)):
                status, reason = "processing-unavailable", "Workbench processing changed the original observation clock"
            else:
                # The input observation mask is authoritative, even if a backend
                # supplies values for display at a missing position.
                output[~np.isfinite(values)] = np.nan
                processed[selected] = output
                processing["original_missing_mask_reapplied"] = True
    records = []
    for position, (source_index, row) in enumerate(chosen.iterrows()):
        row_key = {name: row[name] for name in measurement.grain}
        records.append({**key, "trace_id": trace_id, "observation_id": content_id({"trace": trace_id, "row": row_key}),
            "source_position": int(source_index), "frame_index": row.get("frame_index"),
            "hours": hours[position], "time_kind": _kind(hours[position]),
            "raw_value": raw[position], "raw_kind": _kind(raw[position]), "processed_value": processed[position],
            "within_range": bool(selected[position]), "raw_valid": bool(finite_raw[position]),
            "processed_valid": bool(np.isfinite(processed[position]))})
    observed = hours[np.isfinite(processed)]
    metadata = {**details, "kind": "trace", "status": status, "reason": reason,
        "source_observations": len(hours), "requested_observations": int(np.sum(selected)),
        "invalid_time_observations": int(np.sum(~finite_time)),
        "valid_raw": int(np.sum(selected & finite_raw)), "valid_processed": len(observed),
        "requested_start_hours": start, "requested_end_hours": end,
        "recorded_start_hours": recorded_start, "recorded_end_hours": recorded_end,
        "recorded_fraction_of_requested_span": covered,
        "valid_fraction_of_recorded_positions": float(np.mean(finite_raw[selected])) if np.any(selected) else None,
        "processed_span_hours": float(observed[-1] - observed[0]) if len(observed) else 0.,
        "gap_count": int(np.sum(np.diff(observed) > request.support["max_gap_hours"]))}
    processing.update(status=status, reason=reason)
    return pd.DataFrame(records, columns=TRACE_COLUMNS), metadata, None, processing


def identity(context):
    from pymicroglia.pipelines._screening import producer_identity
    read_verified_tables(context.table_paths, context.request.inputs.table_hashes)
    engine = producer_identity() if context.request.request.representation == "detrended" else None
    return content_id({"request": context.request.scientific_id, "processing_engine": engine,
        "libraries": {name: library_version(name) for name in ("numpy", "pandas")},
        "code": {name: file_hash(source_file(name)) for name in
                 ("relationship_inputs.py", "relationship_options.py", "screening.py", "contracts.py")}})


def produce(context):
    resolved, request = context.request, context.request.request
    tables = read_verified_tables(context.table_paths, resolved.inputs.table_hashes)
    frames, metadata, scalars, processing, by_trace, by_meta = [], [], [], [], {}, {}
    for cell in resolved.inputs.cells:
        for measurement in resolved.measurements:
            frame, info, scalar, details = _prepare_trace(resolved, cell, measurement, tables[measurement.table])
            frames.append(frame)
            metadata.append(info)
            processing.append(details)
            if scalar is not None: scalars.append(scalar)
            key = (cell.movie, cell.identity, measurement.column)
            by_trace[key], by_meta[key] = frame, info
    inventory, matches_all, support_rows, lag_support = [], [], [], []
    sample_map = {item.movie: item for item in resolved.inputs.samples}
    for cell in resolved.inputs.cells:
        for pair in request.pairs:
            key = {**cell.as_dict(), "pair_id": pair.record_id, **pair.as_dict()}
            sample = sample_map[cell.movie]
            inventory.append({**key, "sample": sample.sample, "sample_confirmed": sample.confirmed})
            left, right = [by_trace[(cell.movie, cell.identity, name)] for name in (pair.reference, pair.target)]
            infos = [by_meta[(cell.movie, cell.identity, name)] for name in (pair.reference, pair.target)]
            failures = [info for info in infos if info["status"] in {"invalid", "processing-unavailable", "missing"}]
            problem = {"status": next((info["status"] for info in failures
                        if info["status"] != "missing"), "insufficient"),
                       "reason": "; ".join(f"{info['measurement']}: {info['reason']}" for info in failures)} if failures else {}
            matched, diagnostics = match_observations(left, right, 0., request.support)
            matches_all.extend({**key, **row} for row in matched.to_dict("records"))
            base_support = {**paired_support(matched, diagnostics, request.support), **problem}
            support_rows.append({**key, **base_support, "question": "within_cell", "lag_hours": 0.,
                "requested": request.within_cell["enabled"], **({"status": "disabled", "reason": "Within-cell association was not requested"}
                    if not request.within_cell["enabled"] else {})})
            lag_rows = []
            for shift in lag_values(request.lag):
                matched, diagnostics = match_observations(left, right, shift, request.support)
                lag_rows.append({**key, **paired_support(matched, diagnostics, request.support), **problem,
                                 "question": "lag", "requested": True, "lag_hours": shift})
            lag_support.extend(lag_rows)
            best = max(lag_rows, key=lambda row: (row["status"] == "eligible", row["paired_observations"]), default=None)
            support_rows.append({**key, **(best or base_support), "question": "lag", "lag_hours": None,
                "requested": request.lag["enabled"], **({"status": "disabled", "reason": "Lag analysis was not requested"}
                    if not request.lag["enabled"] else {"reason": "At least one lag has prepared support; statistical-method eligibility remains separate"}
                    if best and best["status"] == "eligible" else {})})
            usable = all((info["kind"] == "scalar" and info["status"] == "available") or
                (info["kind"] == "trace" and info["valid_processed"] >= request.support["min_observations"]
                 and info["processed_span_hours"] >= request.support["min_span_hours"]) for info in infos)
            support_rows.append({**key, "question": "between_cells", "requested": request.between_cells["enabled"],
                "status": "disabled" if not request.between_cells["enabled"] else "eligible" if usable else "insufficient",
                "reason": "Between-cell analysis was not requested" if not request.between_cells["enabled"] else
                          "Independent summary support; simultaneous overlap is not required" if usable else
                          problem.get("reason", "One or both measurement summaries lack their required observations"),
                "lag_hours": None, "paired_observations": None, "overlap_span_hours": None})
    output_tables = {
        "inventory": pd.DataFrame(inventory, columns=PAIR_KEYS + ["sample", "sample_confirmed"]),
        "traces": pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=TRACE_COLUMNS),
        "trace_inventory": pd.DataFrame(metadata, columns=list(dict.fromkeys(
            KEYS + ["measurement", "trace_id", "kind", "status", "reason"] + [key for row in metadata for key in row]))),
        "scalars": pd.DataFrame(scalars, columns=KEYS + ["measurement", "trace_id", "table", "value", "value_kind"]),
        "same_time_pairs": pd.DataFrame(matches_all, columns=PAIR_KEYS + MATCH_COLUMNS),
        "support": pd.DataFrame(support_rows, columns=SUPPORT_COLUMNS),
        "lag_support": pd.DataFrame(lag_support, columns=SUPPORT_COLUMNS)}
    context.output.mkdir(parents=True)
    refs = []
    for name, frame in output_tables.items():
        path = context.output / (name + ".json")
        path = write_table(path, frame)
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id, columns=tuple(frame.columns)))
    documents = {"processing": processing, "provenance": {"schema_version": 1,
        "scientific_id": context.scientific_id, "design_id": context.saved("relationship-design").outcome.scientific_id,
        "resolved_request": resolved.as_dict(), "lag_convention": LAG_CONVENTION,
        "window_boundary": "start inclusive, end exclusive; omitted range retains all recorded observations",
        "matching": "Shift clocks before valid-value matching; ties and target collisions are excluded",
        "support": "Prepared observation support does not certify a statistical method's assumptions",
        "scientific_tests_performed": False}}
    for name, document in documents.items():
        path = context.output / (name + ".json")
        _write_json(path, document)
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id))
    return StepResult(context.step.name, context.scientific_id, "completed",
        "Saved complete independent traces, full pair inventory and actual-clock matching support", tuple(refs))
