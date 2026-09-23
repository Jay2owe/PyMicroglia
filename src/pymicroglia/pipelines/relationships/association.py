"""Save same-time association evidence for the complete declared cell population."""
from __future__ import annotations
from pymicroglia._sources import source_file

from importlib.metadata import version as library_version
from pathlib import Path

import numpy as np
import pandas as pd

from pymicroglia.pipelines._contracts import ArtifactRef, SelectionRecord, Settings, StepResult, content_id
from pymicroglia.pipelines.relationships.inputs import KEYS, PAIR_KEYS
from pymicroglia.pipelines._screening import _write_json, file_hash, read_table, write_table


RESULT_COLUMNS = PAIR_KEYS + ["sample", "sample_confirmed", "question", "representation", "statistic",
    "evidence_method", "effect", "effect_population", "full_overlap_effect", "effect_interval", "interval_status",
    "p_value", "q_value", "significant", "status", "reason", "support_status", "paired_observations", "overlap_span_hours",
    "tested_observations", "tested_start_hours", "tested_end_hours", "minimum_attainable_p",
    "family_id", "family_requested", "family_tested", "correction", "alpha"]
FAMILY_COLUMNS = ["family_id", "question", "scope", "pair_id", "movie", "correction", "alpha", "requested",
                  "tested", "unavailable", "members", "missing_probability_policy"]


def implementation_version():
    return content_id({"code": {str(path.name): file_hash(path) for path in
        (Path(__file__), source_file('relationship_statistics.py'), source_file('circadian.py'))},
        "libraries": {name: library_version(name) for name in ("numpy", "scipy", "pandas")}})


def correct_families(rows, request, question):
    """Freeze requested membership before correction; missing tests retain a slot."""
    import pymicroglia.workbench as circadian
    options = request.inference
    scope, correction, alpha = options.get("correction_scope", "all"), options.get("multiple_testing", "none"), options.get("alpha")
    buckets = {(): []} if scope == "all" else {(pair.record_id,): [] for pair in request.pairs} if scope == "pair" else {}
    for row in rows:
        key = () if scope == "all" else (row["pair_id"],) if scope == "pair" else (row["pair_id"], row["movie"])
        buckets.setdefault(key, []).append(row)
    families = []
    for key, members in buckets.items():
        definitions = [{name: row[name] for name in PAIR_KEYS} for row in members]
        family_id = content_id({"question": question, "scope": scope, "correction": correction, "alpha": alpha, "members": definitions})
        valid = [row["p_value"] is not None and np.isfinite(row["p_value"]) for row in members]
        adjusted = (circadian.adjust_pvalues([row["p_value"] if usable else 1. for row, usable in zip(members, valid)], correction)
                    if any(valid) else [None] * len(members))
        for row, usable, q in zip(members, valid, adjusted):
            row.update(family_id=family_id, family_requested=len(members), family_tested=sum(valid), correction=correction,
                       alpha=alpha, q_value=float(q) if usable else None, significant=bool(usable and q <= alpha))
            if usable:
                row["status"] = ("positive-association" if row["effect"] > 0 else "negative-association") if row["significant"] else "no-detected-association"
                row["reason"] = "Supported signed association in the tested central window" if row["significant"] else \
                                "The declared corrected dependence test did not detect an association"
        families.append({"family_id": family_id, "question": question, "scope": scope,
            "pair_id": key[0] if key else None, "movie": key[1] if len(key) > 1 else None,
            "correction": correction, "alpha": alpha, "requested": len(members), "tested": sum(valid),
            "unavailable": len(members)-sum(valid), "members": definitions,
            "missing_probability_policy": "Retain the requested slot as p=1 for correction; restore its unavailable result afterwards"})
    return pd.DataFrame(families, columns=FAMILY_COLUMNS)


def _selection_members(rows, statuses):
    return tuple(Settings({name: row[name] for name in PAIR_KEYS}) for row in rows if row["status"] in statuses)


def produce(context):
    import pymicroglia.measure.relationship_statistics as statistics
    resolved, request = context.request, context.request.request
    question = request.within_cell
    statistics.validate_question(question)
    saved = context.saved("paired-inputs")
    inventory = read_table(saved.artifact("inventory"))
    traces = read_table(saved.artifact("traces"))
    matched = read_table(saved.artifact("same_time_pairs"))
    support = read_table(saved.artifact("support"))
    support = support.loc[support.question.eq("within_cell")]
    lookup = {tuple(row[name] for name in PAIR_KEYS): row for row in support.to_dict("records")}
    rows, details = [], []
    for key in inventory.to_dict("records"):
        token = tuple(key[name] for name in PAIR_KEYS)
        supported = lookup[token]
        pair_rows = matched
        cell_traces = traces
        for name in KEYS:
            pair_rows = pair_rows.loc[pair_rows[name].eq(key[name])]
            cell_traces = cell_traces.loc[cell_traces[name].eq(key[name])]
        pair_rows = pair_rows.loc[pair_rows.pair_id.eq(key["pair_id"])]
        row = {**key, "question": "within_cell", "representation": request.representation,
            "statistic": question.get("statistic"), "evidence_method": question.get("evidence", {}).get("method"),
            "effect": None, "effect_population": "Full saved overlap; descriptive coefficient",
            "full_overlap_effect": None, "effect_interval": None, "interval_status": "not-provided-by-method",
            "p_value": None, "q_value": None, "significant": False,
            "status": "disabled" if not question["enabled"] else "untestable", "reason": supported["reason"],
            "support_status": supported["status"], "paired_observations": supported["paired_observations"],
            "overlap_span_hours": supported["overlap_span_hours"],
            "tested_observations": None, "tested_start_hours": None, "tested_end_hours": None, "minimum_attainable_p": None}
        result = {"status": row["status"], "reason": row["reason"]}
        if question["enabled"]:
            value, diagnostic = statistics.coefficient(pair_rows.reference_value, pair_rows.target_value, question["statistic"])
            row.update(full_overlap_effect=value, effect=value)
            if supported["status"] == "eligible" and value is not None:
                left, right = [cell_traces.loc[cell_traces.measurement.eq(key[name])] for name in ("reference", "target")]
                result = statistics.same_time_evidence(left, right, question=question, support=request.support)
                row.update(status=result["status"], reason=result["reason"], p_value=result.get("p_value"))
                for name in ("effect", "effect_population", "tested_observations", "tested_start_hours", "tested_end_hours", "minimum_attainable_p"):
                    if name in result: row[name] = result[name]
            elif value is None and supported["status"] == "eligible":
                row["reason"] = diagnostic
        rows.append(row)
        result = {**result, "status": row["status"], "reason": row["reason"]}
        details.append({**key, "result": result, "full_overlap_effect": row["full_overlap_effect"],
                        "prepared_support": supported})
    families = correct_families(rows, request, "within_cell")
    selections = []
    categories = {"association-positive": {"positive-association"}, "association-negative": {"negative-association"},
        "association-supported": {"positive-association", "negative-association"},
        "association-not-detected": {"no-detected-association"}, "association-untestable": {"untestable"},
        "association-descriptive": {"descriptive"}}
    for name, statuses in categories.items():
        selections.append(SelectionRecord(name, context.scientific_id,
            Settings({"question": "within_cell", "statuses": sorted(statuses), "families": families.family_id.tolist(),
                      "union_is_new_test": False}), _selection_members(rows, statuses)))
    context.output.mkdir(parents=True)
    refs = []
    for name, frame in (("results", pd.DataFrame(rows, columns=RESULT_COLUMNS)), ("families", families)):
        path = context.output / (name + ".json")
        path = write_table(path, frame)
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id, columns=tuple(frame.columns)))
    documents = {"engine_details": details, "provenance": {"schema_version": 1, "scientific_id": context.scientific_id,
        "prepared_input_id": saved.outcome.scientific_id, "question": question.as_dict(), "inference": request.inference.as_dict(),
        "implementation": implementation_version(), "scipy_version": library_version("scipy"),
        "test_reference": statistics.REFERENCE, "effect_intervals": "The shift test supplies no coefficient confidence interval",
        "interpretation": "Dependence evidence conditional on declared stationarity; no rhythm, phase, stability or causal claim",
        "preprocessing_repeated": False, "period_fit_performed": False,
        "effect_population": "Inferential coefficients use the fixed central test window; full-overlap descriptions are saved separately",
        "branch_status": "enabled" if question["enabled"] else "disabled"}}
    for name, document in documents.items():
        path = context.output / (name + ".json")
        _write_json(path, document)
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id))
    return StepResult(context.step.name, context.scientific_id, "completed",
        "Saved every declared same-time association outcome and its complete correction family", tuple(refs), tuple(selections))
