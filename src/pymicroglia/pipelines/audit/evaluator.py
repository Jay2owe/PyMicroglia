"""Complete recipe evaluation shared by real traces and known-truth cases."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pymicroglia.pipelines._contracts import ArtifactRef, Settings, StepResult, content_id
from pymicroglia.pipelines._screening import file_hash, read_table, read_verified_tables, write_table


KEYS = ["case_id", "profile_id", "measurement", "movie", "identity", "family_block"]
RESULT_COLUMNS = [*KEYS, "candidate_id", "family_id", "preprocessing_status", "preprocessing_reason",
    "test_status", "reason", "estimate_status", "estimate_reason", "period_hours", "period_available",
    "period_underdetermined", "p_value", "q_value", "significant", "status", "observations",
    "input_observations", "estimate_result", "significance_result", "components", "metadata"]
TRACE_COLUMNS = [*KEYS, "candidate_id", "hours", "raw", "filtered", "filter_result",
                 "processed_trace", "processing_provenance", "processing_reason", "processing_source_method", "native_series",
                 "native_result", "waveform_status"]
FAMILY_COLUMNS = ["candidate_id", "family_id", "family_block", "scope", "measurement", "movie",
                  "members", "requested", "valid_tests", "correction", "alpha"]


@dataclass(frozen=True)
class Evaluation:
    results: pd.DataFrame
    traces: pd.DataFrame
    families: pd.DataFrame


def safe_estimate(hours, values, params, method, detrend, *, capture_details=False):
    """Preserve an individual failure without deleting the other scientific branch."""
    import pymicroglia.workbench as circadian

    try:
        return circadian.estimate_one(hours, values, params, method, detrend=detrend,
                                       **({"capture_details": True} if capture_details else {}))
    except (ValueError, RuntimeError, circadian.cw.WorkbenchError) as error:
        return {"method": method, "method_label": circadian.PERIOD_METHODS[method]["label"],
                "status": "failed", "message": str(error), "diagnostics": {"reason": str(error)},
                "components": [], "period_hours": np.nan, "p_value": np.nan,
                "workbench_version": circadian.WORKBENCH_VERSION}


def real_cases(resolved, table_paths):
    """Read original values for the frozen coverage-selected recording profiles."""
    tables = read_verified_tables(table_paths, resolved.inputs.table_hashes)
    cases = []
    for profile in resolved.profiles:
        metadata = profile["metadata"]
        cell, measurement = metadata["cell"], metadata["measurement"]
        table = tables[metadata["table"]]
        frame = table[table.stem.eq(cell["movie"]) & table.identity.eq(cell["identity"])].sort_values("hours")
        if frame.hours.to_list() != profile["hours"]:
            raise ValueError("Original recording times changed after profile resolution")
        values = [float(v) if pd.notna(v) and np.isfinite(v) else None for v in frame[measurement]]
        if [v is None for v in values] != profile["missing"]:
            raise ValueError("Original missing mask changed after profile resolution")
        cases.append(Settings({"case_id": content_id({"profile": profile["id"], "values": values}),
            "profile_id": profile["id"], "measurement": measurement, "movie": cell["movie"],
            "identity": cell["identity"], "family_block": "real", "hours": profile["hours"],
            "values": values, "metadata": metadata}))
    return tuple(cases)


def _families(candidate, cases):
    buckets = {}
    for case in cases:
        scope = candidate.correction_scope
        key = (case["family_block"], None if scope == "all" else case["measurement"],
               case["movie"] if scope == "movie_measurement" else None)
        buckets.setdefault(key, []).append(case["case_id"])
    families = []
    for (block, measurement, movie), ids in sorted(buckets.items(), key=lambda item: str(item[0])):
        record = {"candidate_id": candidate.candidate_id, "family_block": block,
            "scope": candidate.correction_scope, "measurement": measurement, "movie": movie,
            "members": sorted(ids), "requested": len(ids), "valid_tests": 0,
            "correction": candidate.analysis_options["multiple_testing"],
            "alpha": candidate.analysis_options["rhythmic_alpha"]}
        record["family_id"] = content_id({k: v for k, v in record.items() if k != "valid_tests"})
        families.append(record)
    return families


def evaluate_candidate(candidate, cases) -> Evaluation:
    """Run every declared case and correct within its complete candidate family.

    Known truth is metadata only here. It never changes fitting, test eligibility
    or family membership. Reading or drawing these outputs does not call this function.
    """
    import pymicroglia.workbench as circadian

    cases = tuple(cases)
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Candidate evaluation requires unique case ids")
    family_rows = _families(candidate, cases)
    family_for = {key: family["family_id"] for family in family_rows for key in family["members"]}
    params, options = candidate.rhythm_params.as_dict(), candidate.analysis_options
    results, traces = [], []
    for case in cases:
        base = {name: case[name] for name in KEYS}
        base.update(candidate_id=candidate.candidate_id, family_id=family_for[case["case_id"]],
                    metadata=case["metadata"])
        hours, raw = np.asarray(case["hours"], float), np.asarray(case["values"], float)
        if hours.ndim != 1 or hours.shape != raw.shape:
            raise ValueError("Every case requires matching one-dimensional hours and values")
        processing, failure = {}, ""
        filtered = np.full(raw.shape, np.nan)
        try:
            processing = circadian.filter_rhythm_trace(hours.tolist(), raw.tolist(), candidate.filtering)
            filtered = np.asarray(processing["values"], float)
        except (ValueError, RuntimeError, circadian.cw.WorkbenchError) as error:
            failure = str(error)
        row = {name: None for name in RESULT_COLUMNS}
        row.update(base, preprocessing_status="failed" if failure else "ok", preprocessing_reason=failure,
                   input_observations=int(np.isfinite(raw).sum()), observations=int(np.isfinite(filtered).sum()),
                   test_status="not_tested", estimate_status="not_tested", reason=failure or "no_observations",
                   estimate_reason=failure, period_available=False, period_underdetermined=True,
                   estimate_result={}, significance_result={}, components=[])
        if not failure and len(hours):
            evaluated = circadian.estimate_grouped_rhythms(
                pd.DataFrame({"case": 0, "hours": hours, "value": filtered}),
                group_columns=["case"], value_column="value", params=params,
                method=options["fit_method"], significance_method=options["significance_method"],
                detrend=options["detrend"], detrend_window_hours=options["detrend_window_hours"],
                min_observations=options["min_observations"], min_cycles=options["min_cycles"],
                correction="none", capture_details=True).iloc[0].to_dict()
            for field in ("case", "metric", "q_value", "significant", "rhythm_status", "family_tests"):
                evaluated.pop(field, None)
            row.update(evaluated)
        estimate = row["estimate_result"]
        processing_source = estimate
        if not estimate.get("display_processed_trace") and row["significance_result"].get("display_processed_trace"):
            processing_source = row["significance_result"]
        diagnostic = processing_source.get("display_processed_trace")
        traces.append({**{name: case[name] for name in KEYS}, "candidate_id": candidate.candidate_id,
            "hours": hours.tolist(), "raw": case["values"], "filtered": filtered.tolist(),
            "filter_result": processing, "processed_trace": diagnostic,
            "processing_provenance": processing_source.get("display_processing_run_record"),
            "processing_reason": processing_source.get("display_processing_reason", failure or "No diagnostic available"),
            "processing_source_method": processing_source.get("method") if diagnostic else None,
            "native_series": estimate.get("native_series", {}), "native_result": estimate.get("native_result", {}),
            "waveform_status": "native-series" if estimate.get("native_series") else "unavailable"})
        results.append(row)
    frame = pd.DataFrame(results) if results else pd.DataFrame(columns=RESULT_COLUMNS)
    for family in family_rows:
        indices = frame.index[frame.family_id.eq(family["family_id"])]
        raw_p = pd.to_numeric(frame.loc[indices, "p_value"], errors="coerce").to_numpy(float)
        valid = frame.loc[indices, "test_status"].eq("ok").to_numpy() & np.isfinite(raw_p) & (raw_p >= 0) & (raw_p <= 1)
        raw_p[~valid] = np.nan
        adjusted = circadian.adjust_pvalues(raw_p, family["correction"])
        significant = valid & (adjusted < family["alpha"])
        frame.loc[indices, "p_value"] = raw_p
        frame.loc[indices, "q_value"] = adjusted
        frame.loc[indices, "significant"] = significant
        frame.loc[indices, "status"] = np.where(~valid, "untestable", np.where(significant, "significant", "not-significant"))
        family["valid_tests"] = int(valid.sum())
    return Evaluation(frame, pd.DataFrame(traces, columns=TRACE_COLUMNS),
                       pd.DataFrame(family_rows, columns=FAMILY_COLUMNS))


def save_evaluations(context, evaluations):
    """Save typed evidence once for subsequent scoring and rendering."""
    context.output.mkdir(parents=True)
    refs = []
    for name, columns in (("results", RESULT_COLUMNS), ("traces", TRACE_COLUMNS), ("families", FAMILY_COLUMNS)):
        frame = pd.concat([getattr(value, name) for value in evaluations], ignore_index=True) if evaluations else pd.DataFrame(columns=columns)
        path = context.output / f"{name}.json"
        path = write_table(path, frame)
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id,
                                columns=tuple(frame.columns)))
    return StepResult(context.step.name, context.scientific_id, "completed",
        "Saved complete candidate evidence and correction families", tuple(refs),
        provenance=Settings({"source": "original inputs", "candidate_count": len(evaluations),
                             "descriptive_cosinor": False}))


def read_evaluation(saved):
    """Read verified saved artifacts without numerical analysis."""
    return Evaluation(*(read_table(saved.artifact(name)) for name in ("results", "traces", "families")))


def evaluate_real(context):
    cases = real_cases(context.request, context.table_paths)
    return save_evaluations(context, [evaluate_candidate(candidate, cases) for candidate in context.request.candidates])
