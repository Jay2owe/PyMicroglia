"""Saved screening evidence into public, gap-preserving within-cell timing."""
from pymicroglia._results import read_document
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from pymicroglia.pipelines._contracts import ArtifactRef, SelectionRecord, Settings, StepResult, content_id
from pymicroglia.pipelines._screening import _write_json, file_hash, read_screen, write_table

KEYS = ("source_run", "movie", "identity")


def validate_options(declared, measurements):
    if not isinstance(declared, dict) or set(declared) - {"phase_options", "component_bands", "trace_representation"}:
        raise ValueError("timing accepts phase_options, component_bands and trace_representation")
    result = {"phase_options": {}, "component_bands": {}, "trace_representation": "filtered", **declared}
    if result["trace_representation"] not in {"raw", "filtered"}:
        raise ValueError("timing.trace_representation must be raw or filtered saved observations")
    if not isinstance(result["phase_options"], dict): raise ValueError("timing.phase_options must be an object")
    bands = result["component_bands"]
    if not isinstance(bands, dict) or set(bands) - set(measurements):
        raise ValueError("timing.component_bands must name tested measurements")
    for measurement, limits in bands.items():
        if not isinstance(limits, list) or len(limits) != 2 or not all(isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) for v in limits) or not 0 < limits[0] < limits[1]:
            raise ValueError(f"timing.component_bands.{measurement} requires increasing positive hours")
    return result


def version():
    import pymicroglia.workbench as circadian
    return content_id({"producer": file_hash(__file__), "gateway": file_hash(circadian.__file__),
                       "workbench": circadian.rhythm_environment()})


def identity(context):
    return content_id({"producer": version(), "screen": context.saved("rhythm-screen").outcome.scientific_id,
        "pairs": context.request.request.pairs, "settings": context.request.request.timing})


def produce(context):
    import pymicroglia.workbench as circadian
    resolved = context.request
    settings = validate_options(resolved.request.timing.as_dict(), [m.column for m in resolved.test_measurements])
    options = dict(settings["phase_options"])
    options.setdefault("min_observations", max(24, resolved.analysis_options["min_observations"]))
    options.setdefault("min_cycles", max(3, math.ceil(resolved.analysis_options["min_cycles"])))
    saved = context.saved("rhythm-screen")
    screen = read_screen(saved.root)
    lookup = {tuple(row[k] for k in (*KEYS, "measurement")): row for row in screen.results.to_dict("records")}
    traces = {key: group for key, group in screen.traces.groupby([*KEYS, "measurement"], sort=False)}
    rows, segments, points, details, selections = [], [], [], [], []
    for pair in resolved.request.pairs:
        pair_info = {"pair_id": content_id(pair), "reference": pair.reference, "target": pair.target}
        members = {"eligible": [], "ineligible": []}
        for cell in resolved.inputs.cells:
            key = tuple(getattr(cell, k) for k in KEYS)
            evidence, payloads = [], []
            for measurement in (pair.reference, pair.target):
                record = lookup.get((*key, measurement), {"status": "untestable", "reason": "Missing saved partner"})
                evidence.append(record)
                trace = traces.get((*key, measurement), pd.DataFrame(columns=screen.traces.columns))
                # Invalid hours never acquire invented coordinates. Duplicated
                # actual hours are retained for the public engine's refusal.
                trace = trace[np.isfinite(pd.to_numeric(trace.hours, errors="coerce"))].sort_values("hours", kind="stable")
                value_column = "filtered_value" if settings["trace_representation"] == "filtered" else "value"
                payloads.append({"hours": trace.hours.tolist(), "values": trace[value_column].tolist(),
                    "detected": record["status"] == "significant",
                    "period_supported": bool(record.get("period_available")) and not bool(record.get("period_underdetermined")) and not bool(record.get("period_at_search_edge")),
                    "estimate": record.get("estimate_result") or {}, "significance": record.get("significance_result") or {},
                    **({"component_period_band": settings["component_bands"][measurement]} if measurement in settings["component_bands"] else {})})
            try:
                result = circadian.rhythm_pair_timing(*payloads, settings=options)
            except circadian.cw.WorkbenchInputError as error:
                # Malformed scientific options are a failed requested analysis,
                # while unsupported observation grids are retained data cases.
                if any(np.any(np.diff(np.asarray(p["hours"])) <= 0) for p in payloads):
                    result = {"status": "ineligible", "reason": str(error), "segments": [], "timecourse": []}
                else: raise
            info = {**pair_info, **cell.as_dict(), "sample": evidence[0].get("sample"),
                    "sample_confirmed": evidence[0].get("sample_confirmed", False)}
            row = {**info, "status": result["status"], "reason": result["reason"],
                **{name: result.get(name) for name in ("period_hours", "offset_hours", "offset_interval_hours", "descriptive_relation", "temporal_status", "summary_segment", "phase_definition", "sign_convention", "common_observations", "jointly_finite_observations")},
                "both_significant": all(item["status"] == "significant" for item in evidence),
                "reference_status": evidence[0]["status"], "target_status": evidence[1]["status"],
                "reference_family": evidence[0].get("family_id"), "target_family": evidence[1].get("family_id"),
                "reference_period": evidence[0].get("period_hours"), "target_period": evidence[1].get("period_hours"),
                "reference_method": evidence[0].get("method"), "target_method": evidence[1].get("method"),
                "reference_test_method": evidence[0].get("significance_method"), "target_test_method": evidence[1].get("significance_method"),
                "reference_p": evidence[0].get("p_value"), "target_p": evidence[1].get("p_value"),
                "reference_q": evidence[0].get("q_value"), "target_q": evidence[1].get("q_value")}
            rows.append(row)
            members[result["status"]].append(cell)
            segments.extend({**info, **{k: v for k, v in segment.items() if k != "native"}} for segment in result.get("segments", []))
            points.extend({**info, **point} for point in result.get("timecourse", []))
            details.append({**info, "result": result})
        selections.extend(SelectionRecord(f"timing-{status}:{pair_info['pair_id']}", context.scientific_id,
            Settings({"pair": pair.as_dict(), "status": status, "screen_id": saved.outcome.scientific_id}), tuple(cells))
            for status, cells in members.items())
    context.output.mkdir(parents=True, exist_ok=True)
    refs = []
    for name, records in (("pairs", rows), ("segments", segments), ("timecourse", points)):
        path = context.output / (name + ".json")
        path = write_table(path, pd.DataFrame(records))
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id))
    _write_json(context.output / "engine_details.json", details)
    _write_json(context.output / "provenance.json", {"schema_version": 1, "screen_id": saved.outcome.scientific_id,
        "source_screen_provenance": read_document(saved.artifact("provenance")),
        "settings": settings, "effective_phase_options": options, "workbench_version": circadian.WORKBENCH_VERSION,
        "source_analysis_recomputed": False, "timing_analysis": "Public Workbench native period comparability and observed band-limited analytic phase",
        "trace_policy": "Saved raw or filtered observations; the explicitly defined analytic band supplies phase preprocessing. No fitted or normalized display curves are analysed",
        "inference_policy": "Conditional descriptive timing and interval containment after rhythm selection; no secondary p-values or selected-subset correction",
        "requested_pairs": [pair.as_dict() for pair in resolved.request.pairs], "requested_cells": len(resolved.inputs.cells),
        "producer": version()})
    refs.extend(ArtifactRef(name, name + ".json", file_hash(context.output / (name + ".json")), context.scientific_id)
                for name in ("engine_details", "provenance"))
    return StepResult(context.step.name, context.scientific_id, "completed",
        "Saved within-cell timing decisions and supported signed observations for every requested pair", tuple(refs), tuple(selections))
