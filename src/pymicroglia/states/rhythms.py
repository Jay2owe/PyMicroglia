"""Rhythms in probabilities and original traces, solely through the gateway."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext

import numpy as np
import pandas as pd

from pymicroglia import workbench
from pymicroglia.measure.modules.rhythms import DEFAULTS
from pymicroglia.states.dynamics import adjacent_edges, persistence_sample, state_trace_columns
from pymicroglia.states.features import CELL, META


def resolved_settings(stems, source, options):
    unknown = set(options.rhythms) - set(workbench.CIRCADIAN_ANALYSIS_OPTIONS)
    if unknown:
        raise ValueError(f"Unknown rhythm options: {sorted(unknown)}")
    controls = {**workbench.CIRCADIAN_ANALYSIS_OPTION_DEFAULTS, **options.rhythms}
    inherited = {}
    for movie in source.get("measurement_runs", []):
        for module in movie.get("modules", []):
            if module.get("module") == "rhythms" and module.get("status") == "done":
                inherited[movie["stem"]] = module.get("parameters", {})
    return {stem: workbench.resolve_analysis_options({**DEFAULTS, **inherited.get(stem, {})}, controls.get)
            for stem in stems}


def trace_table(assignments, snapshots, changes, settings, options, probabilities_only=False):
    originals = pd.concat([snapshots, changes], axis=1)
    pieces, audit = [], []
    for stem, group in assignments.groupby("stem", sort=True):
        probability_names = state_trace_columns(group)
        for column in probability_names:
            part = group[META].copy()
            part["trace_kind"], part["trace_key"] = ("state_probability" if column.startswith("state_probability_") else "state_indicator"), column
            part["value"] = group[column].to_numpy(float)
            pieces.append(part)
        if probabilities_only:
            continue
        selected = options.rhythm_metrics if options.rhythm_metrics is not None else settings[stem]["params"]["metrics"]
        names = [c for c in originals if "*" in selected or c in selected or c.split("|")[2] in selected]
        if options.rhythm_metrics is not None:
            unmatched = [c for c in selected if c != "*" and c not in names and not any(n.split("|")[2] == c for n in names)]
            if unmatched:
                raise ValueError(f"Unknown original rhythm measurements: {unmatched}")
        audit.append({"stem": stem, "original_trace_features": names, "requested_original_metrics": list(selected)})
        for column in names:
            part = group[META].copy()
            part["trace_kind"], part["trace_key"] = "original_measurement", column
            part["value"] = originals.loc[group.index, column].to_numpy(float)
            pieces.append(part)
    return (pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=[*META, "trace_kind", "trace_key", "value"])), audit


def _fit_trace_batch(task):
    """Fit complete traces without correcting across a partial cohort."""
    frame, config, max_gap_hours = task
    result = workbench.estimate_grouped_rhythms(
            frame, group_columns=[*CELL, "trace_kind", "trace_key"], value_column="value",
            params=config["params"], method=config["method"], significance_method=config["significance_method"],
            detrend=config["detrend"], detrend_window_hours=config["detrend_window_hours"],
            min_observations=config["min_observations"], correction="none", min_cycles=config["min_cycles"])
    if result.empty:
        return result
    result["applied_settings_json"] = json.dumps(config, sort_keys=True)
    result["correction"] = config["multiple_testing"]
    result["minimum_cycles"] = config["min_cycles"]
    coverage = []
    for key, group in frame.groupby([*CELL, "trace_kind", "trace_key"], sort=True):
        group = group.sort_values("frame_index")
        edges, delta = adjacent_edges(group, max_gap_hours)
        finite = np.isfinite(group.value.to_numpy(float))
        valid = edges & finite[:-1] & finite[1:]
        coverage.append({**dict(zip([*CELL, "trace_kind", "trace_key"], key)),
                         "contiguous_observed_hours": float(delta[valid].sum()),
                         "missing_observations": int((~finite).sum())})
    return result.merge(pd.DataFrame(coverage), on=[*CELL, "trace_kind", "trace_key"], validate="one_to_one")


def fit_traces(traces, settings, max_gap_hours=None, *, executor=None):
    """Correct complete cohort families after all independent worker fits return."""
    if executor is None:
        tasks = ((frame, settings[stem], max_gap_hours) for stem, frame in traces.groupby("stem", sort=True))
        fitted = map(_fit_trace_batch, tasks)
    else:
        # Never split a cell's trace across workers or adjust p-values within
        # a worker. Ordered map also preserves the serial result row order.
        tasks = ((frame, settings[stem], max_gap_hours)
                 for (stem, _), frame in traces.groupby(CELL, sort=True))
        fitted = executor.map(_fit_trace_batch, tasks)
    pieces = [result for result in fitted if not result.empty]
    if not pieces:
        return pd.DataFrame()
    result = pd.concat(pieces, ignore_index=True)
    for _, indices in result.groupby("trace_kind").groups.items():
        family = result.loc[indices]
        # Each row retains its requested correction, applied to the same family.
        raw = pd.to_numeric(family.p_value, errors="coerce").to_numpy(float)
        raw[~family.test_status.eq("ok").to_numpy()] = np.nan
        for correction in family.correction.unique():
            adjusted = workbench.adjust_pvalues(raw, correction)
            mask = family.correction.eq(correction).to_numpy()
            result.loc[np.asarray(indices)[mask], "q_value"] = adjusted[mask]
        result.loc[indices, "family_tests"] = int(np.isfinite(raw).sum())
    tested = result.test_status.eq("ok") & np.isfinite(result.q_value)
    result["significant"] = tested & (result.q_value < result.alpha)
    result["rhythm_status"] = np.select([~tested, result.significant], ["not tested", "rhythmic"], default="not rhythmic")
    result["observed_duration_cycles"] = result.contiguous_observed_hours / result.period_hours
    result["supported_period_for_grouping"] = (result.period_available & result.significant &
        ~result.period_underdetermined & ~result.period_at_search_edge &
        result.observed_duration_cycles.ge(result.minimum_cycles))
    # This is a conservative use restriction, not a new rhythm estimator/test.
    return result


def analyse_rhythms(assignments, snapshots, changes, source, options, progress=None):
    if not options.rhythm_enabled:
        return {"results": pd.DataFrame(), "traces": pd.DataFrame(), "null_results": pd.DataFrame(),
                "null_summary": pd.DataFrame(), "report": {"enabled": False}}
    settings = resolved_settings(assignments.stem.unique(), source, options)
    traces, audit = trace_table(assignments, snapshots, changes, settings, options)
    simulations = []
    rng = np.random.default_rng(options.seed)
    has_probability_variation = any(assignments[c].nunique() > 1 for c in state_trace_columns(assignments))
    workers = min(options.rhythm_workers, len(assignments[CELL].drop_duplicates()))
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else nullcontext(None)
    with pool as executor:
        if progress:
            progress(f"Fitting observed rhythm traces with {workers} worker process(es).")
        results = fit_traces(traces, settings, options.max_gap_hours, executor=executor)
        if progress:
            progress(f"Observed rhythm fits complete: {len(results)} traces.")
        if has_probability_variation:
            for replicate in range(options.persistence_surrogates):
                # Generate simulations serially with the original shared RNG;
                # scheduling the independent fits cannot change the null draws.
                simulated = persistence_sample(assignments, rng, options.max_gap_hours)
                null_traces, _ = trace_table(simulated, snapshots, changes, settings, options, probabilities_only=True)
                fitted = fit_traces(null_traces, settings, options.max_gap_hours, executor=executor)
                fitted["replicate"] = replicate
                simulations.append(fitted)
                if progress:
                    progress(f"Persistence simulations complete: {replicate + 1}/{options.persistence_surrogates}.")
    null_results = pd.concat(simulations, ignore_index=True) if simulations else pd.DataFrame()
    null_summary = pd.DataFrame()
    if not null_results.empty:
        null_summary = null_results.groupby([*CELL, "trace_key"]).agg(
            simulations=("replicate", "size"), tested=("test_status", lambda v: int(v.eq("ok").sum())),
            significant_simulations=("significant", "sum"), supported_period_simulations=("supported_period_for_grouping", "sum")).reset_index()
        null_summary["significant_fraction_of_tested"] = null_summary.significant_simulations / null_summary.tested.replace(0, np.nan)
    results = attach_persistence_diagnostic(results, null_summary)
    return {"results": results, "traces": traces, "null_results": null_results, "null_summary": null_summary,
            "report": {"enabled": True, "worker_processes": workers,
                       "settings_by_movie": settings, "original_trace_selection": audit,
                       "workbench_version": workbench.WORKBENCH_VERSION,
                       "available_period_methods": list(workbench.PERIOD_METHODS),
                       "correction_families": ["all cell/state probabilities" if any(c.startswith("state_probability_") for c in assignments) else "all cell/state hard indicators", "all cell/original measurement traces"],
                       "null": {"requested_simulations": options.persistence_surrogates, "completed_simulations": len(simulations),
                                "seed": options.seed, "model": "time-homogeneous first-order Markov switching with within-state state-trace resampling",
                                "scope": "diagnostic for first-order persistence; not a calibrated rhythm p-value or a control for all slow drift",
                                "skip_reason": "no variable state traces" if not has_probability_variation else ""}}}


def attach_persistence_diagnostic(results, null_summary):
    """Conservative use restriction; does not create a new rhythm p-value."""
    results = results.copy()
    results["period_meets_test_and_coverage"] = results.supported_period_for_grouping
    if null_summary.empty:
        results["persistence_null_tested"] = 0
        results["persistence_null_significant_fraction"] = np.nan
    else:
        summary = null_summary[[*CELL, "trace_key", "tested", "significant_fraction_of_tested"]].rename(
            columns={"tested": "persistence_null_tested", "significant_fraction_of_tested": "persistence_null_significant_fraction"})
        results = results.merge(summary, on=[*CELL, "trace_key"], how="left", validate="one_to_one")
        results["persistence_null_tested"] = results.persistence_null_tested.fillna(0).astype(int)
    sensitive = results.persistence_null_significant_fraction.ge(results.alpha)
    results["persistence_diagnostic"] = np.select(
        [results.persistence_null_tested.eq(0), sensitive],
        ["not_assessed", "selected_test_sensitive_to_persistence"],
        default="no_excess_detected_in_simulations")
    results["supported_period_for_grouping"] &= ~sensitive
    return results
