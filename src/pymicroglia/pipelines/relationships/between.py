"""Separate measured cell summaries from inference across biological samples."""
from __future__ import annotations
from pymicroglia._sources import source_file

from importlib.metadata import version as library_version
from pathlib import Path

import numpy as np
import pandas as pd

from pymicroglia.pipelines._contracts import ArtifactRef, SelectionRecord, Settings, StepResult, content_id
from pymicroglia.pipelines.relationships.inputs import KEYS, PAIR_KEYS
from pymicroglia.pipelines._screening import _json_value, _write_json, file_hash, read_table, write_table


SCALAR_COLUMNS = KEYS + ["measurement", "table", "label", "unit", "kind", "summary", "time_range_hours", "summary_id",
    "representation", "value", "status", "reason", "observations", "span_hours", "recorded_fraction", "valid_fraction"]
PAIR_COLUMNS = PAIR_KEYS + ["sample", "sample_confirmed", "reference_value", "target_value", "reference_summary_id",
                          "target_summary_id", "status", "reason", "eligible"]
UNIT_COLUMNS = ["source_run", "pair_id", "reference", "target", "experimental_unit", "unit_id", "sample", "confirmed",
                "reference_value", "target_value", "cells", "movies", "members"]
RESULT_COLUMNS = ["source_run", "pair_id", "reference", "target", "question", "estimand", "representation", "statistic",
    "experimental_unit", "aggregation", "evidence_method", "effect", "cell_effect", "effect_interval", "interval_status",
    "p_value", "q_value", "significant", "status", "reason", "cells_requested", "cells_eligible", "recordings_requested",
    "recordings_eligible", "confirmed_samples", "unconfirmed_cells", "experimental_units", "members", "family_id",
    "family_requested", "family_tested", "correction", "alpha"]
GROUP_COLUMNS = ["source_run", "pair_id", "reference", "target", "level", "group_id", "confirmed", "cells", "movies",
                 "effect", "status", "reason", "members"]
FAMILY_COLUMNS = ["family_id", "question", "scope", "pair_id", "requested", "tested", "unavailable", "members",
                  "correction", "alpha", "missing_probability_policy"]


def implementation_version():
    return content_id({"code": {path.name: file_hash(path) for path in (Path(__file__),
        source_file('relationship_population_statistics.py'), source_file('relationship_statistics.py'),
        source_file('circadian.py'))},
        "libraries": {name: library_version(name) for name in ("numpy", "pandas", "scipy")}})


def correct_aggregate_families(rows, request, question):
    """Aggregate hypotheses are pair-level questions, with their own families."""
    import pymicroglia.workbench as circadian
    scope = request.inference.get("correction_scope", "all")
    if scope == "movie_pair" and any(row["p_value"] is not None for row in rows):
        raise ValueError("Aggregate biological-sample inference requires all or pair correction_scope")
    correction, alpha = request.inference.get("multiple_testing", "none"), request.inference.get("alpha")
    buckets = {None: rows} if scope == "all" else {pair.record_id: [r for r in rows if r["pair_id"] == pair.record_id] for pair in request.pairs}
    families = []
    for pair_id, members in buckets.items():
        keys = [{name: row[name] for name in ("source_run", "pair_id", "question")} for row in members]
        fid = content_id({"question": question, "scope": scope, "members": keys, "correction": correction, "alpha": alpha})
        valid = [r["p_value"] is not None and np.isfinite(r["p_value"]) for r in members]
        adjusted = circadian.adjust_pvalues([r["p_value"] if ok else 1. for r,ok in zip(members,valid)], correction) if any(valid) else [None]*len(members)
        for row, usable, q in zip(members, valid, adjusted):
            significant = bool(usable and q <= alpha)
            row.update(family_id=fid, family_requested=len(members), family_tested=sum(valid), q_value=float(q) if usable else None,
                significant=significant, correction=correction, alpha=alpha)
            if usable:
                row.update(status=("positive-association" if row["effect"] > 0 else "negative-association") if significant else "no-detected-association",
                    reason="Supported association between the declared independent units" if significant else "The corrected test did not detect an association")
        families.append({"family_id": fid, "question": question, "scope": scope, "pair_id": pair_id,
            "requested": len(members), "tested": sum(valid), "unavailable": len(members)-sum(valid), "members": keys,
            "correction": correction, "alpha": alpha,
            "missing_probability_policy": "Keep every requested pair as a correction slot; missing p enters as one then returns to unavailable"})
    return pd.DataFrame(families, columns=FAMILY_COLUMNS)


def _scalars(resolved, tables):
    request = resolved.request
    traces, original = tables["traces"], tables["scalars"]
    rows = []
    for info in tables["trace_inventory"].to_dict("records"):
        fields = KEYS+["measurement"]
        data = original if info["kind"] == "scalar" else traces
        for name in fields: data = data.loc[data[name].eq(info[name])]
        if info["kind"] == "scalar":
            value = data.value.iloc[0] if len(data) else None
        else:
            data = data.loc[data.within_range & data.processed_valid]
            value = getattr(data.processed_value, info["summary"])() if len(data) and info.get("summary") else None
        finite = value is not None and np.isfinite(value)
        row = {**{name: info.get(name) for name in KEYS+["measurement", "table", "label", "unit", "kind", "summary", "representation"]},
            "time_range_hours": request.time_range_hours, "value": float(value) if finite else None,
            "status": "available" if finite else "unavailable", "reason": info["reason"],
            "observations": info["valid_processed"], "span_hours": info["processed_span_hours"],
            "recorded_fraction": info.get("recorded_fraction_of_requested_span"), "valid_fraction": info.get("valid_fraction_of_recorded_positions")}
        row["summary_id"] = content_id({"input": resolved.scientific_id, **_json_value(row)})
        rows.append(row)
    return pd.DataFrame(rows, columns=SCALAR_COLUMNS)


def _members(frame):
    return frame[PAIR_KEYS].to_dict("records")


def produce(context):
    import pymicroglia.measure.relationship_population_statistics as statistics
    from pymicroglia.measure.relationship_statistics import coefficient
    resolved, request = context.request, context.request.request
    question = request.between_cells
    statistics.validate_between(question)
    if question["enabled"] and question["evidence"]["method"] != "none" and request.inference["correction_scope"] == "movie_pair":
        raise ValueError("between_cells inference requires all or pair correction_scope")
    saved = context.saved("paired-inputs")
    tables = {name: read_table(saved.artifact(name)) for name in ("inventory", "trace_inventory", "traces", "scalars", "support")}
    # A disabled scalar question must not manufacture summaries of trace-only requests.
    scalar_frame = _scalars(resolved, tables) if question["enabled"] else pd.DataFrame(columns=SCALAR_COLUMNS)
    scalar_lookup = {tuple(row[name] for name in KEYS+["measurement"]): row for row in scalar_frame.to_dict("records")}
    support = tables["support"].loc[tables["support"].question.eq("between_cells")]
    supported = {tuple(row[name] for name in PAIR_KEYS): row for row in support.to_dict("records")}
    paired = []
    for key in tables["inventory"].to_dict("records"):
        info = supported[tuple(key[name] for name in PAIR_KEYS)]
        a,b = [scalar_lookup.get(tuple(key[name] for name in KEYS)+(key[side],), {}) for side in ("reference", "target")]
        paired.append({**key, "reference_value": a.get("value"), "target_value": b.get("value"),
            "reference_summary_id": a.get("summary_id"), "target_summary_id": b.get("summary_id"),
            "status": info["status"], "reason": info["reason"], "eligible": bool(info["status"] == "eligible" and
                a.get("status") == b.get("status") == "available")})
    paired_frame = pd.DataFrame(paired, columns=PAIR_COLUMNS)
    rows, units, groups, details = [], [], [], []
    for pair in request.pairs:
        all_cells = paired_frame.loc[paired_frame.pair_id.eq(pair.record_id)]
        eligible = all_cells.loc[all_cells.eligible]
        key = {"source_run": resolved.inputs.source_run, "pair_id": pair.record_id, **pair.as_dict()}
        cell_effect, cell_reason = coefficient(eligible.reference_value, eligible.target_value, question.get("statistic", "pearson"))
        local_units = []
        if question.get("experimental_unit") == "cell":
            for cell in eligible.to_dict("records"):
                local_units.append({**key, "experimental_unit": "cell", "unit_id": content_id({name:cell[name] for name in KEYS}),
                    "sample": cell["sample"], "confirmed": cell["sample_confirmed"], "reference_value": cell["reference_value"],
                    "target_value": cell["target_value"], "cells": 1, "movies": [cell["movie"]],
                    "members": [{name: cell[name] for name in PAIR_KEYS}]})
        elif question["enabled"]:
            for sample, cells in eligible.loc[eligible.sample_confirmed].groupby("sample", sort=True):
                local_units.append({**key, "experimental_unit": "biological_sample", "unit_id": content_id({"source_run": key["source_run"], "sample": sample}),
                    "sample": sample, "confirmed": True,
                    "reference_value": float(getattr(cells.reference_value,question["aggregation"])()),
                    "target_value": float(getattr(cells.target_value,question["aggregation"])()),
                    "cells": len(cells), "movies": sorted(cells.movie.unique()), "members": _members(cells)})
        units.extend(local_units)
        for level, grouper, population in (("recording", "movie", eligible),
                ("biological_sample", "sample", eligible.loc[eligible.sample_confirmed])):
            for group_id, cells in population.groupby(grouper, sort=True):
                effect, reason = coefficient(cells.reference_value, cells.target_value, question.get("statistic", "pearson"))
                groups.append({**key, "level": level, "group_id": group_id, "confirmed": level == "biological_sample",
                    "cells": len(cells), "movies": sorted(cells.movie.unique()), "effect": effect,
                    "status": "descriptive" if effect is not None else "untestable", "reason": reason, "members": _members(cells)})
        unconfirmed = int((~eligible.sample_confirmed).sum())
        row = {**key, "question": "between_cells", "representation": request.representation, "statistic": question.get("statistic"),
            "experimental_unit": question.get("experimental_unit"), "aggregation": question.get("aggregation"),
            "estimand": "Association between biological-sample aggregates of cell summaries" if question.get("experimental_unit") == "biological_sample" else "Descriptive association between individual cell summaries",
            "evidence_method": question.get("evidence",{}).get("method"), "effect": None, "cell_effect": cell_effect,
            "effect_interval": None, "interval_status": "not-requested", "p_value": None, "q_value": None, "significant": False,
            "status": "disabled", "reason": "Between-cell association was not requested", "cells_requested": len(all_cells),
            "cells_eligible": len(eligible), "recordings_requested": all_cells.movie.nunique(), "recordings_eligible": eligible.movie.nunique(),
            "confirmed_samples": eligible.loc[eligible.sample_confirmed, "sample"].nunique(), "unconfirmed_cells": unconfirmed,
            "experimental_units": len(local_units), "members": _members(eligible)}
        native = {}
        if question["enabled"]:
            if question["evidence"]["method"] != "none" and (question["experimental_unit"] != "biological_sample" or unconfirmed):
                effect, _ = coefficient([u["reference_value"] for u in local_units], [u["target_value"] for u in local_units], question["statistic"])
                native = {"effect": effect, "status": "untestable", "reason": "Biological inference requires confirmed sample mapping for every eligible cell and biological_sample units"}
            else:
                native = statistics.sample_association([u["reference_value"] for u in local_units], [u["target_value"] for u in local_units], question)
            # Native "reference" is a method citation, never a measured column.
            row.update({name: value for name,value in native.items() if name in RESULT_COLUMNS and name not in PAIR_KEYS})
        rows.append(row); details.append({**key, "result": native, "unit_ids": [u["unit_id"] for u in local_units]})
    families = correct_aggregate_families(rows, request, "between_cells")
    selections = [SelectionRecord(name, context.scientific_id, Settings({"question": "between_cells", "status": status}),
        tuple(Settings({field: row[field] for field in ("source_run", "pair_id", "reference", "target")}) for row in rows if row["status"] == status))
        for name,status in (("between-positive", "positive-association"), ("between-negative", "negative-association"),
            ("between-not-detected", "no-detected-association"), ("between-untestable", "untestable"), ("between-descriptive", "descriptive"))]
    context.output.mkdir(parents=True)
    refs = []
    frames = {"cell_scalars": scalar_frame, "paired_scalars": paired_frame, "units": pd.DataFrame(units, columns=UNIT_COLUMNS),
        "results": pd.DataFrame(rows, columns=RESULT_COLUMNS), "group_descriptions": pd.DataFrame(groups, columns=GROUP_COLUMNS), "families": families}
    for name,frame in frames.items():
        path=context.output/(name+".json"); path = write_table(path,frame)
        refs.append(ArtifactRef(name,path.name,file_hash(path),context.scientific_id,columns=tuple(frame.columns)))
    for name,value in {"engine_details":details,"provenance":{"schema_version":1,"scientific_id":context.scientific_id,
        "prepared_input_id":saved.outcome.scientific_id,"question":question.as_dict(),"inference":request.inference.as_dict(),
        "implementation":implementation_version(),"scipy_version":library_version("scipy"),"time_range_hours":request.time_range_hours,
        "summary_population":"One declared scalar per eligible cell; no pooled frames",
        "sample_aggregation":"Each cell contributes once to its confirmed sample; each sample contributes once to inference",
        "missing_mapping_policy":"Withhold biological inference if any eligible cell lacks confirmed sample mapping",
        "test_reference":statistics.PERMUTATION_REFERENCE,"interval_reference":statistics.BOOTSTRAP_REFERENCE,
        "period_fit_performed":False,"preprocessing_repeated":False,"within_cell_tests_repeated":False}}.items():
        path=context.output/(name+".json"); _write_json(path,value)
        refs.append(ArtifactRef(name,path.name,file_hash(path),context.scientific_id))
    return StepResult(context.step.name,context.scientific_id,"completed",
        "Saved independent cell summaries and separately labelled biological-sample association evidence",tuple(refs),tuple(selections))
