"""Persist complete lag profiles, search evidence and separately resolved delays."""
from __future__ import annotations
from pymicroglia._sources import source_file

from importlib.metadata import version as library_version
from pathlib import Path

import numpy as np
import pandas as pd

from pymicroglia.pipelines._contracts import ArtifactRef, SelectionRecord, Settings, StepResult, content_id
from pymicroglia.pipelines.relationships.association import correct_families
from pymicroglia.pipelines.relationships.inputs import KEYS, PAIR_KEYS, lag_values, match_observations
from pymicroglia.pipelines._screening import _write_json, file_hash, read_table, write_table


RESULT_COLUMNS = PAIR_KEYS + ["sample", "sample_confirmed", "question", "representation", "statistic", "evidence_method",
    "effect", "association_sign", "peak_effects", "effect_population", "search_statistic", "p_value", "q_value", "significant", "status", "reason",
    "tested_observations", "tested_start_hours", "tested_end_hours", "minimum_attainable_p", "best_lag_hours",
    "empirical_peak_lags_hours", "delay_hours", "delay_interval_hours", "candidate_lags_hours", "resolution_status",
    "resolution_reason", "delay_supported", "uncertainty_method", "uncertainty_scope", "family_id", "family_requested",
    "family_tested", "correction", "alpha"]
PROFILE_COLUMNS = PAIR_KEYS + ["lag_hours", "effect", "status", "reason", "paired_observations", "overlap_span_hours",
    "gap_count", "largest_gap_hours", "ambiguous_observations", "tested_effect", "tested_observations",
    "coefficient_lower", "coefficient_upper", "interval_kind", "compatible_delay"]


def implementation_version():
    return content_id({"code": {path.name: file_hash(path) for path in (Path(__file__),
        source_file("relationship_association.py"), source_file("relationship_inputs.py"),
        source_file('relationship_lag_statistics.py'), source_file('relationship_statistics.py'),
        source_file('circadian.py'))},
        "libraries": {name: library_version(name) for name in ("numpy", "pandas", "scipy", "arch")}})


def produce(context):
    import pymicroglia.measure.relationship_lag_statistics as statistics
    from pymicroglia.measure.relationship_statistics import coefficient, TIE_TOLERANCE
    from pymicroglia.pipelines.relationships.options import LAG_CONVENTION
    request = context.request.request
    question = request.lag
    statistics.validate_lag_question(question)
    saved = context.saved("paired-inputs")
    inventory, traces, support = [read_table(saved.artifact(name)) for name in ("inventory", "traces", "lag_support")]
    lags = lag_values(question) if question["enabled"] else []
    rows, profiles, details = [], [], []
    for key in inventory.to_dict("records"):
        cell_traces, local_support = traces, support
        for name in KEYS:
            cell_traces = cell_traces.loc[cell_traces[name].eq(key[name])]
            local_support = local_support.loc[local_support[name].eq(key[name])]
        local_support = local_support.loc[local_support.pair_id.eq(key["pair_id"])]
        left, right = [cell_traces.loc[cell_traces.measurement.eq(key[name])] for name in ("reference", "target")]
        local_profiles = []
        for lag in lags:
            prepared = local_support.loc[np.isclose(local_support.lag_hours, lag, rtol=0., atol=1e-9)].iloc[0].to_dict()
            pairs, _ = match_observations(left, right, lag, request.support)
            value, reason = coefficient(pairs.reference_value, pairs.target_value, question["statistic"])
            local_profiles.append({**key, **{name: prepared.get(name) for name in PROFILE_COLUMNS if name in prepared},
                "lag_hours": float(lag), "effect": value,
                "status": "descriptive" if prepared["status"] == "eligible" and value is not None else "untestable",
                "reason": reason if prepared["status"] == "eligible" else prepared["reason"],
                "tested_effect": None, "tested_observations": None, "coefficient_lower": None, "coefficient_upper": None,
                "interval_kind": None, "compatible_delay": None})
        result = statistics.evaluate(left, right, question, request.support, lags) if question["enabled"] else {
            "status": "disabled", "reason": "Lag association was not requested", "p_value": None,
            "resolution_status": "disabled", "resolution_reason": "Lag association was not requested"}
        row = {**key, "question": "lag", "representation": request.representation, "statistic": question.get("statistic"),
            "evidence_method": question.get("evidence", {}).get("method"), "effect": None,
            "effect_population": "Fixed common reference window for the complete inferential search",
            "search_statistic": None, "p_value": None, "q_value": None, "significant": False, "best_lag_hours": None,
            "delay_hours": None, "delay_interval_hours": None, "candidate_lags_hours": [], "empirical_peak_lags_hours": [],
            "delay_supported": False, "uncertainty_method": question.get("peak_resolution", {}).get("method", "none"),
            "uncertainty_scope": result.get("uncertainty", {}).get("coverage_scope")}
        # Native method provenance also has a "reference" (the paper URL).
        # Measurement and cell keys always come from the prepared inventory.
        row.update({name: result[name] for name in RESULT_COLUMNS if name in result and name not in PAIR_KEYS})
        if result.get("profile") is not None:
            uncertainty = result.get("uncertainty", {})
            bounds = uncertainty.get("coefficient_bounds")
            for index, profile in enumerate(local_profiles):
                profile.update(tested_effect=result["profile"][index], tested_observations=result["tested_observations"])
                if bounds is not None:
                    profile.update(coefficient_lower=bounds[0][index], coefficient_upper=bounds[1][index],
                        interval_kind=uncertainty["coefficient_interval_kind"], compatible_delay=profile["lag_hours"] in result["candidate_lags_hours"])
        elif row["status"] == "descriptive":
            valid = [item for item in local_profiles if item["status"] == "descriptive"]
            if valid:
                maximum = max(abs(item["effect"]) for item in valid)
                peaks = [item for item in valid if abs(item["effect"]) >= maximum-TIE_TOLERANCE]
                row.update(effect=peaks[0]["effect"], search_statistic=maximum,
                    empirical_peak_lags_hours=[item["lag_hours"] for item in peaks],
                    effect_population="Descriptive per-lag overlaps; support can differ across the grid")
            else: row.update(status="untestable", reason="No declared lag has sufficient finite, nonconstant paired observations")
        peaks = row["empirical_peak_lags_hours"]
        if len(peaks) == 1: row["best_lag_hours"] = peaks[0]
        values = result.get("profile", [item["effect"] for item in local_profiles])
        effects = [value for lag, value in zip(lags, values) if lag in peaks and value is not None]
        signs = set(np.sign(effects))
        row.update(peak_effects=effects, association_sign="mixed" if len(signs) > 1 else
            "positive" if signs == {1} else "negative" if signs == {-1} else "undefined")
        rows.append(row); profiles.extend(local_profiles)
        details.append({**key, "result": result})
    families = correct_families(rows, request, "lag")
    for row in rows:
        row["delay_supported"] = bool(row["significant"] and row.get("resolution_status") == "resolved")
        if not row["delay_supported"]: row["delay_hours"] = None
        if row["significant"]: row["reason"] = "Supported association after the complete declared lag search and family correction"
        if row["association_sign"] == "mixed":
            row["effect"] = None
            if row["significant"]: row["status"] = "supported-association"
    selections = []
    predicates = {"lag-association-supported": lambda r: r["significant"],
        "lag-delay-supported": lambda r: r["delay_supported"],
        "lag-delay-unresolved": lambda r: r["significant"] and not r["delay_supported"],
        "lag-not-detected": lambda r: r["status"] == "no-detected-association",
        "lag-untestable": lambda r: r["status"] == "untestable", "lag-descriptive": lambda r: r["status"] == "descriptive"}
    for name, predicate in predicates.items():
        selections.append(SelectionRecord(name, context.scientific_id,
            Settings({"question": "lag", "families": families.family_id.tolist(), "complete_search": True}),
            tuple(Settings({field: row[field] for field in PAIR_KEYS}) for row in rows if predicate(row))))
    context.output.mkdir(parents=True)
    refs = []
    for name, frame in (("results", pd.DataFrame(rows, columns=RESULT_COLUMNS)),
                        ("profiles", pd.DataFrame(profiles, columns=PROFILE_COLUMNS)), ("families", families)):
        path = context.output/(name+".json"); path = write_table(path, frame)
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id, columns=tuple(frame.columns)))
    for name, value in {"engine_details": details, "provenance": {
        "schema_version": 1, "scientific_id": context.scientific_id, "prepared_input_id": saved.outcome.scientific_id,
        "question": question.as_dict(), "inference": request.inference.as_dict(), "implementation": implementation_version(),
        "lag_convention": LAG_CONVENTION, "test_reference": statistics.REFERENCE,
        "uncertainty_reference": statistics.BOOTSTRAP_REFERENCE, "preprocessing_repeated": False, "period_fit_performed": False,
        "interpretation": "Association and grid-delay uncertainty under explicit assumptions; no rhythm, phase, stability or causal claim",
        "descriptive_profile": "Original per-lag matches with separately recorded count and duration",
        "tested_profile": "Identical original reference observation window at every declared lag and every surrogate",
        "uncertainty": "Approximate simultaneous within-curve confidence set; no selection-adjusted coverage across cells",
        "branch_status": "enabled" if question["enabled"] else "disabled"}}.items():
        path = context.output/(name+".json"); _write_json(path, value)
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id))
    return StepResult(context.step.name, context.scientific_id, "completed",
        "Saved complete lag profiles, corrected search evidence and separate delay uncertainty", tuple(refs), tuple(selections))
