"""Complete rhythm screens, immutable artifacts and saved display selections.

The scientific gateway supplies estimates and raw test evidence. This producer
defines the requested population and correction families before any selection.
Read-back validates saved files and never calls the scientific engine.
"""

from __future__ import annotations
from pymicroglia._results import read_document
from pymicroglia._sources import source_file

import hashlib
import io
import json
import math
import os
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Mapping
from uuid import uuid4

import numpy as np
import pandas as pd

from pymicroglia.pipelines._contracts import ArtifactRef, CellKey, CellMeasurementKey, SelectionRecord, Settings, StepResult, content_id, plain
from pymicroglia.pipelines.rhythm.discovery import ResolvedRhythmRequest

SCHEMA_VERSION = 1
KEY_COLUMNS = ("source_run", "movie", "identity", "measurement")
RESULT_COLUMNS = (*KEY_COLUMNS, "source_table", "unit", "sample", "sample_confirmed",
                  "observed_subject", "status", "test_status", "reason", "method",
                  "significance_method", "p_value", "q_value", "significant", "family_id",
                  "family_requested", "family_usable", "family_tests", "estimate_status",
                  "estimate_reason", "period_hours", "period_available", "cycles_observed",
                  "period_underdetermined", "period_at_search_edge", "observations",
                  "span_hours", "input_rows", "invalid_observations", "input_eligible", "nonconsecutive_steps",
                  "estimate_result", "significance_result", "display_id", "candidate_id", "applied_recipe", "settings_profile_id")
TRACE_COLUMNS = (*KEY_COLUMNS, "source_table", "input_row", "frame_index", "hours",
                 "value", "time_kind", "value_kind", "eligible_input", "rejection_reason", "filtered_value", "filter_status")
FAMILY_COLUMNS = ("family_id", "scope", "measurement", "movie", "members", "requested",
                  "usable", "valid_tests", "correction_denominator", "correction", "alpha",
                  "missing_probability_policy", "backend", "backend_version")
DISPLAY_COLUMNS = (*KEY_COLUMNS, "display_id", "method", "native_series", "native_result",
                   "processed_trace", "processed_trace_provenance", "processed_trace_status",
                   "processed_trace_reason", "processing_source_method", "waveform_status",
                   "waveform_reason", "used_observation_mask_status", "phase_status")


def file_hash(path: str | Path) -> str:
    from .._results import document
    from ._tables import fingerprint
    path=Path(path)
    return fingerprint(document(path) if path.suffix=='.json' else path)


def _json_value(value):
    """Use JSON null for missing science; raw non-finite kinds have their own mask."""
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported saved scientific value: {type(value).__name__}")


def _write_json(path: Path, value) -> None:
    from .._results import write_document
    from auto_organotypic import store
    target=write_document(path,_json_value(value))
    store.claim('pipeline-document',store.fingerprint(target),{'document':target.name},path=target,display_only=False,method_version='2')
    return target


def write_table(path: Path, frame: pd.DataFrame) -> None:
    from ._tables import write
    return write(path,frame)


def read_table(path: str | Path) -> pd.DataFrame:
    from ._tables import read
    return read(path)


def read_verified_tables(paths: Mapping[str, str | Path],
                         expected: Mapping[str, str]) -> dict[str, pd.DataFrame]:
    """Hash and parse the same bytes, avoiding a check/read race."""
    result = {}
    for name, fingerprint in expected.items():
        if name not in paths:
            raise ValueError(f"missing required input table {name!r}")
        data = Path(paths[name]).read_bytes()
        if hashlib.sha256(data).hexdigest() != fingerprint:
            raise ValueError(f"input fingerprint changed: {name}")
        result[name] = pd.read_csv(io.BytesIO(data), dtype={"stem": str, "subject": str})
    return result


def producer_identity() -> dict:
    """Include editable Workbench code, not just its distribution version."""
    import pymicroglia.workbench as circadian

    root = Path(__file__).resolve().parents[1]
    own = {str(p.relative_to(root)): file_hash(p) for p in
           (Path(__file__), source_file("circadian.py"), source_file("contracts.py"),
            source_file("rhythm_discovery.py"))}
    own.update({str(p.relative_to(root)): file_hash(p) for p in (root / "_workbench").glob("*.py")})
    backend_root = Path(circadian.cw.__file__).resolve().parent
    backend = {p.relative_to(backend_root).as_posix(): file_hash(p)
               for p in sorted(backend_root.rglob("*.py"))}
    return {"schema_version": SCHEMA_VERSION, "producer_sources": own,
            "workbench_version": circadian.WORKBENCH_VERSION,
            "workbench_code": content_id(backend),
            "libraries": {name: version(name) for name in ("numpy", "pandas", "scipy")}}


def screen_identity(resolved: ResolvedRhythmRequest) -> str:
    """No display names, pagination, render settings or output path in this ID."""
    return content_id({"producer": producer_identity(), "inputs": resolved.screen_inputs,
                       "measurements": resolved.test_measurements,
                       "analysis": resolved.analysis_options, "params": resolved.rhythm_params,
                       "correction_scope": resolved.request.correction_scope,
                       "methods": resolved.period_methods, "measurement_recipes": resolved.measurement_recipes,
                       "settings_profile": resolved.profile_provenance})


def _key(row) -> CellMeasurementKey:
    return CellMeasurementKey(CellKey(row["source_run"], row["movie"], int(row["identity"])),
                              row["measurement"])


def _families(resolved) -> list[dict]:
    scope = resolved.request.correction_scope
    buckets = {}
    if scope == "all":
        buckets[(None, None)] = []
    elif scope == "measurement":
        buckets = {(metric.column, None): [] for metric in resolved.test_measurements}
    for pair in resolved.expected_pairs:
        key = (None, None) if scope == "all" else (
            pair.measurement, pair.cell.movie if scope == "movie_measurement" else None)
        buckets.setdefault(key, []).append(pair)
    rows = []
    for (metric, movie), members in buckets.items():
        options = resolved.measurement_recipes[metric]["analysis_options"] if metric in resolved.measurement_recipes else resolved.analysis_options
        correction, alpha = options["multiple_testing"], options["rhythmic_alpha"]
        identity = {"scope": scope, "members": plain(members),
                    "correction": correction, "alpha": alpha}
        rows.append({"family_id": content_id(identity), "scope": scope,
                     "measurement": metric, "movie": movie, "members": plain(members),
                     "requested": len(members), "usable": 0, "valid_tests": 0,
                     "correction_denominator": 0, "correction": correction, "alpha": alpha,
                     "missing_probability_policy": "Workbench excludes non-finite probabilities; "
                     "requested members remain in the inventory",
                     "backend": "circadian_workbench.statistics.adjust_pvalues",
                     "backend_version": resolved.workbench_version})
    return rows


def _number_kind(value) -> str:
    if pd.isna(value):
        return "missing"
    number = float(value)
    return "finite" if math.isfinite(number) else "positive_infinity" if number > 0 else "negative_infinity"


def _selections(results: pd.DataFrame, measurements, scope: str,
                scientific_id: str) -> tuple[SelectionRecord, ...]:
    selections = []
    union = set()
    for metric in measurements:
        for status in ("significant", "not-significant", "untestable"):
            rows = results[results.measurement.eq(metric) & results.status.eq(status)]
            members = tuple(_key(row) for row in rows.to_dict("records"))
            selections.append(SelectionRecord(f"{status}:{metric}", scientific_id,
                Settings({"measurement": metric, "status": status,
                          "source": "rhythm_results", "correction_scope": scope}),
                members))
            if status == "significant":
                union.update(member.cell for member in members)
    selections.append(SelectionRecord("any-significant", scientific_id,
        Settings({"rule": "union of measurement-specific significant cells",
                  "cell_level_significance_test": False}), tuple(sorted(union))))
    return tuple(selections)


@dataclass(frozen=True)
class ScreenData:
    results: pd.DataFrame
    traces: pd.DataFrame
    families: pd.DataFrame
    display_inputs: pd.DataFrame
    outcome: StepResult


def screen(resolved: ResolvedRhythmRequest, table_paths: Mapping[str, str | Path],
           output: str | Path) -> ScreenData:
    """Evaluate a resolved request from verified input files into a new directory."""
    import pymicroglia.workbench as circadian

    if resolved.workbench_version != circadian.WORKBENCH_VERSION or \
            resolved.period_methods.as_dict() != circadian.PERIOD_METHODS:
        raise ValueError("Workbench changed since request resolution; resolve the request again")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"screen output already exists: {output}")
    tables = read_verified_tables(table_paths, resolved.screen_inputs.table_hashes)
    scientific_id = screen_identity(resolved)
    family_rows = _families(resolved)
    pair_family = {content_id(member): f["family_id"]
                   for f in family_rows for member in f["members"]}
    rows, traces, displays = [], [], []
    samples = {s.movie: s for s in resolved.inputs.samples}
    metrics = {m.column: m for m in resolved.test_measurements}
    for pair in resolved.expected_pairs:
        cell, metric = pair.cell, metrics[pair.measurement]
        recipe = resolved.measurement_recipes.get(metric.column)
        params = recipe["rhythm_params"] if recipe else resolved.rhythm_params.as_dict()
        options = recipe["analysis_options"] if recipe else resolved.analysis_options
        frame = tables[metric.table]
        observed = frame[frame.stem.eq(cell.movie) & frame.identity.eq(cell.identity)].sort_values("hours", kind="stable")
        key = {**cell.as_dict(), "measurement": metric.column}
        values = pd.to_numeric(observed[metric.column], errors="raise").to_numpy(float)
        hours = pd.to_numeric(observed.hours, errors="raise").to_numpy(float)
        finite = np.isfinite(hours) & np.isfinite(values)
        duplicate_time = bool(pd.Series(hours[np.isfinite(hours)]).duplicated().any())
        filtered, filter_failure, filter_status = values.copy(), "", "not requested"
        if recipe and not duplicate_time and len(observed):
            try:
                processing = circadian.filter_rhythm_trace(hours.tolist(), values.tolist(), recipe["filtering"])
                filtered = np.asarray(processing["values"], float)
                filter_status = "ok"
            except (ValueError, RuntimeError, circadian.cw.WorkbenchError) as error:
                filtered = np.full(values.shape, np.nan)
                filter_failure, filter_status = str(error), "failed"
        input_eligible = (not duplicate_time and finite.sum() >= options["min_observations"]
                          and not np.allclose(values[finite], values[finite][0]))
        for offset, (index, row) in enumerate(observed.iterrows()):
            reason = "duplicate_time" if duplicate_time else (
                "invalid_time" if not np.isfinite(hours[offset]) else
                "missing_or_nonfinite_value" if not np.isfinite(values[offset]) else "")
            traces.append({**key, "source_table": metric.table, "input_row": int(index),
                           "frame_index": row.get("frame_index"), "hours": hours[offset],
                           "value": values[offset], "time_kind": _number_kind(hours[offset]),
                           "value_kind": _number_kind(values[offset]),
                           "eligible_input": bool(finite[offset] and not duplicate_time),
                           "rejection_reason": reason, "filtered_value": filtered[offset], "filter_status": filter_status})
        if duplicate_time:
            evaluated = {"test_status": "not_tested", "reason": "duplicate_time",
                         "estimate_status": "not_tested", "observations": int(finite.sum())}
        elif filter_failure:
            evaluated = {"test_status": "not_tested", "reason": "filter_failed: " + filter_failure,
                         "estimate_status": "not_tested", "estimate_reason": filter_failure, "observations": 0}
        elif len(observed):
            long = pd.DataFrame({"trace": 0, "hours": hours, "value": filtered})
            result = circadian.estimate_grouped_rhythms(
                long, group_columns=["trace"], value_column="value", params=params,
                method=options["fit_method"], significance_method=options["significance_method"],
                detrend=options["detrend"], detrend_window_hours=options["detrend_window_hours"],
                min_observations=options["min_observations"], min_cycles=options["min_cycles"],
                correction="none", capture_details=True)
            evaluated = result.iloc[0].to_dict()
            for field in ("trace", "metric", "q_value", "significant", "rhythm_status", "family_tests"):
                evaluated.pop(field, None)
        else:
            evaluated = {"test_status": "not_tested", "reason": "no_observations",
                         "estimate_status": "not_tested", "observations": 0}
        estimate = evaluated.get("estimate_result", {})
        processing_source = estimate
        if not estimate.get("display_processed_trace") and evaluated.get("significance_result", {}).get("display_processed_trace"):
            processing_source = evaluated["significance_result"]
        native_series = estimate.get("native_series", {})
        native = estimate.get("native_result", {})
        display_id = content_id({"scientific_id": scientific_id, "pair": pair})
        # Native payload/series are retained verbatim (JSON null for missing data).
        # An unexported estimator trace or curve is never reconstructed locally.
        displays.append({**key, "display_id": display_id, "method": options["fit_method"],
                         "native_series": native_series, "native_result": native,
                         "processed_trace": processing_source.get("display_processed_trace"),
                         "processed_trace_provenance": processing_source.get("display_processing_run_record"),
                         "processed_trace_status": "diagnostic" if processing_source.get("display_processed_trace") else "unavailable",
                         "processing_source_method": processing_source.get("method") if processing_source.get("display_processed_trace") else None,
                         "processed_trace_reason": processing_source.get("display_processing_reason",
                            "Period action does not export its processed trace; original observations are preserved"),
                         "waveform_status": "native-series" if native_series else "unavailable",
                         "waveform_reason": "" if native_series else "Estimator exports no native fitted series",
                         "used_observation_mask_status": "finite-input mask saved; backend internal mask not exported",
                         "phase_status": "available" if pd.notna(evaluated.get("phase_hours")) else "unavailable"})
        sample = samples[cell.movie]
        consecutive = observed.sort_values("hours").get("frame_index", pd.Series(dtype=float)).diff().dropna()
        base = {column: None for column in RESULT_COLUMNS}
        rows.append({**base, **evaluated, **key, "source_table": metric.table, "unit": metric.unit,
                     "sample": sample.sample, "sample_confirmed": sample.confirmed,
                     "observed_subject": sample.observed_subject, "family_id": pair_family[pair.record_id],
                     "method": options["fit_method"], "significance_method": options["significance_method"],
                     "input_rows": len(observed), "invalid_observations": int((~finite).sum()),
                     "input_eligible": bool(input_eligible),
                     "nonconsecutive_steps": int(consecutive.ne(1).sum()), "display_id": display_id,
                     "estimate_result": estimate, "significance_result": evaluated.get("significance_result", {}),
                     "candidate_id": recipe["candidate_id"] if recipe else None, "applied_recipe": recipe,
                     "settings_profile_id": resolved.profile_provenance.get("profile_id"),
                     "period_available": bool(evaluated.get("period_available", False)),
                     "period_underdetermined": bool(evaluated.get("period_underdetermined", True)),
                     "period_at_search_edge": bool(evaluated.get("period_at_search_edge", False))})
    results = pd.DataFrame(rows) if rows else pd.DataFrame(columns=RESULT_COLUMNS)
    for family in family_rows:
        indices = results.index[results.family_id.eq(family["family_id"])]
        raw = pd.to_numeric(results.loc[indices, "p_value"], errors="coerce").to_numpy(float)
        valid = results.loc[indices, "test_status"].eq("ok").to_numpy() & np.isfinite(raw) & (raw >= 0) & (raw <= 1)
        raw[~valid] = np.nan
        adjusted = circadian.adjust_pvalues(raw, family["correction"])
        family["valid_tests"] = family["correction_denominator"] = int(valid.sum())
        family["usable"] = int(results.loc[indices, "input_eligible"].sum())
        results.loc[indices, "p_value"] = raw
        results.loc[indices, "q_value"] = adjusted
        results.loc[indices, "significant"] = valid & (adjusted < family["alpha"])
        results.loc[indices, "status"] = np.where(~valid, "untestable",
            np.where(adjusted < family["alpha"], "significant", "not-significant"))
        for column, value in (("family_requested", family["requested"]),
                              ("family_usable", family["usable"]), ("family_tests", family["valid_tests"])):
            results.loc[indices, column] = value
    selections = _selections(results, [m.column for m in resolved.test_measurements],
                             resolved.request.correction_scope, scientific_id)
    frames = {"rhythm_results": results, "trace_inputs": pd.DataFrame(traces, columns=TRACE_COLUMNS),
              "correction_families": pd.DataFrame(family_rows, columns=FAMILY_COLUMNS),
              "display_inputs": pd.DataFrame(displays, columns=DISPLAY_COLUMNS)}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}-{uuid4().hex}.partial")
    staging.mkdir()
    refs = []
    for name, frame in frames.items():
        filename = name + ".csv"
        write_table(staging / filename, frame)
        refs.append(ArtifactRef(name, filename, file_hash(staging / filename), scientific_id,
                                KEY_COLUMNS if name in {"rhythm_results", "display_inputs"} else (),
                                tuple(frame.columns)))
    provenance = {"resolved_request": resolved.as_dict(), "producer": producer_identity(),
                  "missing_scientific_values": "JSON null; raw observation non-finite kinds are explicit",
                  "selection_note": "any-significant is a display union, not a cell-level significance test"}
    _write_json(staging / "provenance.json", provenance)
    refs.append(ArtifactRef("provenance", "provenance.json", file_hash(staging / "provenance.json"), scientific_id))
    outcome = StepResult("rhythm-screen", scientific_id, "completed",
        f"Screened {len(results)} requested cell-measurement pairs; "
        f"{sum(results.status.eq('significant'))} significant", tuple(refs), selections,
        Settings({"schema_version": SCHEMA_VERSION, "producer": provenance["producer"]}))
    _write_json(staging / "result.json", outcome.as_dict())
    # An interrupted writer leaves only a .partial directory, never reusable output.
    read_screen(staging, expected_id=scientific_id)
    # The sync client can briefly hold the completed directory. The store's
    # atomic promotion retries those sharing failures without rewriting data.
    from auto_organotypic.io import replace_with_retry
    if output.exists():
        raise FileExistsError(output)
    replace_with_retry(staging, output)
    return read_screen(output, expected_id=scientific_id)


def read_screen(output: str | Path, *, expected_id: str | None = None) -> ScreenData:
    """Validate every required artifact and reconstruct selections without fitting."""
    output = Path(output)
    from .._results import read_document
    data = read_document(output / "result.json")
    scientific_id = data["scientific_id"]
    if expected_id is not None and scientific_id != expected_id:
        raise ValueError("saved screen scientific identity does not match")
    if data["status"] != "completed" or data["provenance"].get("schema_version") != SCHEMA_VERSION:
        raise ValueError("saved screen is not a completed supported schema")
    refs = tuple(ArtifactRef(**{**a, "grain": tuple(a["grain"]), "columns": tuple(a["columns"])})
                 for a in data["artifacts"])
    required = {"rhythm_results", "trace_inputs", "correction_families", "display_inputs", "provenance"}
    if {r.name for r in refs} != required or len(refs) != len(required):
        raise ValueError("saved screen is missing required artifacts")
    for ref in refs:
        path = (output / ref.path).resolve()
        if path.suffix=='.json':
            from .._results import document
            path=document(path).resolve()
        if not path.is_relative_to(output.resolve()) or ref.scientific_id != scientific_id:
            raise ValueError("invalid saved artifact identity/path")
        if not path.is_file() or file_hash(path) != ref.sha256:
            raise ValueError(f"saved artifact missing or changed: {ref.name}")
    selections = []
    for raw in data["selections"]:
        if raw["scientific_id"] != scientific_id:
            raise ValueError("saved selection belongs to a different scientific result")
        members = tuple(CellMeasurementKey(CellKey(**m["cell"]), m["measurement"])
                        if "cell" in m else CellKey(**m) for m in (entry.get('key',entry) for entry in raw["members"]))
        selections.append(SelectionRecord(raw["name"], scientific_id, Settings(raw["rule"]), members))
    frames = {ref.name: read_table(output / ref.path) for ref in refs if ref.name != "provenance"}
    for ref in refs:
        if ref.name in frames and tuple(frames[ref.name].columns) != ref.columns:
            raise ValueError(f"saved table schema changed: {ref.name}")
    provenance_ref = next(ref for ref in refs if ref.name == "provenance")
    provenance = read_document(output / provenance_ref.path)
    request = provenance["resolved_request"]
    expected_selections = _selections(frames["rhythm_results"],
        [m["column"] for m in request["test_measurements"]],
        request["request"]["correction_scope"], scientific_id)
    if tuple(selections) != expected_selections:
        raise ValueError("saved selections disagree with the complete corrected results")
    for column, kind in (("value", "value_kind"), ("hours", "time_kind")):
        for label, number in (("positive_infinity", np.inf), ("negative_infinity", -np.inf)):
            frames["trace_inputs"].loc[frames["trace_inputs"][kind].eq(label), column] = number
    outcome = StepResult(data["step"], scientific_id, data["status"], data["reason"], refs,
                         tuple(selections), Settings(data["provenance"]))
    return ScreenData(frames["rhythm_results"], frames["trace_inputs"],
                      frames["correction_families"], frames["display_inputs"], outcome)
