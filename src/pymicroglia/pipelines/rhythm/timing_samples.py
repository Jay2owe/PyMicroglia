"""Uncertainty-aware saved timing summaries with complete biological units."""
from pymicroglia._results import read_document
from pymicroglia._sources import source_file
import json
import math
from pathlib import Path

import pandas as pd

from pymicroglia.pipelines._contracts import ArtifactRef, SelectionRecord, Settings, StepResult, content_id
from pymicroglia.pipelines._screening import _write_json, file_hash, read_screen, read_table, write_table
from pymicroglia.pipelines.rhythm.agreement import independence

KEYS = ("source_run", "movie", "identity")
EVIDENCE = ("status", "reason", "period_hours", "offset_hours", "offset_interval_hours", "phase_definition",
            "sign_convention", "period_evidence", "temporal_status")


def validate_options(declared):
    if not isinstance(declared, dict) or set(declared) - {"summary_options", "period_strata", "interval_correction"}:
        raise ValueError("timing_summary accepts summary_options, period_strata and interval_correction")
    result = {"summary_options": {}, "period_strata": [{"name": "all", "period_hours": None}],
              "interval_correction": "bonferroni", **declared}
    if result["interval_correction"] not in {"none", "bonferroni"}:
        raise ValueError("timing_summary.interval_correction must be none or bonferroni")
    if not isinstance(result["summary_options"], dict): raise ValueError("timing_summary.summary_options must be an object")
    strata, names = result["period_strata"], set()
    if not isinstance(strata, list): raise ValueError("period_strata must be a list")
    intervals = []
    for row in strata:
        if not isinstance(row, dict) or set(row) != {"name", "period_hours"}:
            raise ValueError("Each period stratum requires name and period_hours")
        name, bounds = row["name"], row["period_hours"]
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("Period stratum names must be nonempty and unique")
        names.add(name)
        if bounds is None:
            if len(strata) != 1: raise ValueError("An unbounded period stratum must stand alone")
            continue
        if not isinstance(bounds, list) or len(bounds) != 2 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in bounds) or not 0 < bounds[0] < bounds[1]:
            raise ValueError("Stratum period_hours requires increasing positive bounds")
        if any(max(bounds[0], previous[0]) < min(bounds[1], previous[1]) for previous in intervals):
            raise ValueError("Period strata must not overlap")
        intervals.append(bounds)
    return result


def version():
    import pymicroglia.workbench as circadian
    return content_id({"producer": file_hash(__file__), "independence": file_hash(source_file("rhythm_agreement.py")),
                       "gateway": file_hash(circadian.__file__), "workbench": circadian.rhythm_environment()})


def identity(context):
    return content_id({"producer": version(), "timing": context.saved("within-cell-timing").outcome.scientific_id,
        "screen": context.saved("rhythm-screen").outcome.scientific_id, "settings": context.request.request.timing_summary})


def _in_stratum(result, bounds):
    if bounds is None: return True
    evidence = result.get("period_evidence", {})
    if set(evidence) != {"reference", "target"}: return False
    for name in ("reference", "target"):
        interval = evidence[name].get("interval_hours")
        if not isinstance(interval, list) or len(interval) != 2 or not all(isinstance(v, (float, int)) and math.isfinite(v) for v in interval): return False
        if not bounds[0] <= interval[0] <= interval[1] < bounds[1]: return False
    return True


def produce(context):
    import pymicroglia.workbench as circadian
    settings = validate_options(context.request.request.timing_summary.as_dict())
    original_options = settings["summary_options"]
    requested_confidence = original_options.get("confidence", .95)
    if isinstance(requested_confidence, bool) or not isinstance(requested_confidence, (float, int)) or not 0 < requested_confidence < 1:
        raise ValueError("timing_summary confidence must be a finite fraction strictly between zero and one")
    count = len(context.request.request.pairs) * len(settings["period_strata"])
    confidence = 1 - (1 - requested_confidence) / max(1, count) if settings["interval_correction"] == "bonferroni" else requested_confidence
    options = {**original_options, "confidence": confidence}
    source = context.saved("within-cell-timing")
    screen_source = context.saved("rhythm-screen")
    screen = read_screen(screen_source.root)
    source_rows = read_table(source.artifact("pairs"))
    native = read_document(source.artifact("engine_details"))
    native_lookup = {(entry["pair_id"], *(entry[k] for k in KEYS)): entry["result"] for entry in native}
    assignments = {a.movie: a for a in context.request.inputs.samples}
    cell_lookup = {content_id(cell): cell for cell in context.request.inputs.cells}
    rows, units, members, details, selections = [], [], [], [], []
    for pair in context.request.request.pairs:
        pair_id = content_id(pair)
        originals = source_rows[source_rows.pair_id.eq(pair_id)] if not source_rows.empty else source_rows
        independent, reasons, dependencies = independence(screen, pair, assignments)
        for stratum in settings["period_strata"]:
            info = {"pair_id": pair_id, "reference": pair.reference, "target": pair.target,
                    "stratum": stratum["name"], "stratum_id": content_id(stratum), "period_stratum_hours": stratum["period_hours"]}
            records, annotations, unit_info = [], {}, {}
            for row in originals.to_dict("records"):
                cell_key = {k: row[k] for k in KEYS}
                identifier = content_id(cell_key)
                original = native_lookup[(pair_id, *(row[k] for k in KEYS))]
                # Preserve all scientific eligibility evidence; source artifacts
                # retain the full native timecourses without duplicating them in
                # every summary action's input/run record.
                evidence = {key: original.get(key) for key in EVIDENCE}
                inside = _in_stratum(original, stratum["period_hours"])
                if not inside:
                    evidence.update(status="ineligible", reason="Native period uncertainty does not fit wholly inside the declared [lower, upper) stratum")
                assignment = assignments[row["movie"]]
                token = {"sample": assignment.sample} if assignment.confirmed else {"movie": assignment.movie}
                unit = content_id(token)
                unit_info[unit] = {"unit_label": assignment.sample if assignment.confirmed else assignment.movie,
                    "unit_type": "biological sample" if assignment.confirmed else "recording", "sample_confirmed": assignment.confirmed}
                records.append({"id": identifier, "unit": unit, "reference": pair.reference, "target": pair.target, "timing": evidence})
                annotations[identifier] = {**row, "source_timing_eligible": original.get("status") == "eligible",
                    "in_declared_stratum": inside, "unit": unit}
            completed = circadian.rhythm_timing_summary(records, independent_units=independent, settings=options)
            population = completed["population"]
            row = {**info, **population, "both_significant_cells": sum(bool(a["both_significant"]) for a in annotations.values()),
                "source_timing_eligible_cells": sum(a["source_timing_eligible"] for a in annotations.values()),
                "independence_reason": "; ".join(reasons), "interval_family": "timing-summary-regions"}
            if reasons and row["status"] == "unavailable": row["reason"] = "; ".join(reasons)
            rows.append(row)
            for unit in completed["units"]:
                unit_rows = [a for a in annotations.values() if a["unit"] == unit["unit"]]
                units.append({**info, **unit_info[unit["unit"]], **unit,
                    "both_significant_cells": sum(bool(a["both_significant"]) for a in unit_rows),
                    "source_timing_eligible_cells": sum(a["source_timing_eligible"] for a in unit_rows)})
            selected = []
            for member in completed["members"]:
                members.append({**annotations[member["id"]], **info, **member})
                if member["across_unit_comparable"]: selected.append(cell_lookup[member["id"]])
            selections.append(SelectionRecord(f"timing-comparable:{pair_id}:{info['stratum_id']}", context.scientific_id,
                Settings({"pair": pair.as_dict(), "stratum": stratum, "timing_id": source.outcome.scientific_id}), tuple(selected)))
            details.append({**info, "result": completed, "decision_dependencies": dependencies})
    context.output.mkdir(parents=True, exist_ok=True)
    frames = {"summary": pd.DataFrame(rows), "units": pd.DataFrame(units), "members": pd.DataFrame(members),
        "families": pd.DataFrame([{"family": "timing-summary-regions", "requested": count,
            "interval_correction": settings["interval_correction"], "requested_confidence": requested_confidence,
            "effective_summary_confidence": confidence, "coordinates_per_summary": 2,
            "coordinate_allocation": "Workbench Bonferroni allocation within each requested summary"}])}
    refs = []
    for name, frame in frames.items():
        path = context.output / (name + ".json")
        path = write_table(path, frame)
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id))
    _write_json(context.output / "engine_details.json", details)
    _write_json(context.output / "provenance.json", {"schema_version": 1, "timing_id": source.outcome.scientific_id,
        "screen_id": screen_source.outcome.scientific_id,
        "source_timing_provenance": read_document(source.artifact("provenance")),
        "settings": settings, "effective_summary_options": options, "workbench_version": circadian.WORKBENCH_VERSION,
        "source_analysis_recomputed": False, "producer": version(),
        "stratum_policy": "Both native component uncertainty intervals must fit wholly in [lower, upper); outside or unavailable cells remain in every requested stratum ledger",
        "inference_policy": "One equally weighted observed sample vector; conditional sampling uncertainty is separate from propagated input phase intervals. Shared correction dependencies disable independent inference",
        "payload_policy": "Public summary receives the complete comparison population and all native eligibility/period/offset evidence; original timing artifacts retain timecourses and full run records"})
    refs.extend(ArtifactRef(name, name + ".json", file_hash(context.output / (name + ".json")), context.scientific_id)
                for name in ("engine_details", "provenance"))
    return StepResult(context.step.name, context.scientific_id, "completed",
        "Saved comparable cell/sample timing summaries with original temporal outcomes and full exclusions", tuple(refs), tuple(selections))
