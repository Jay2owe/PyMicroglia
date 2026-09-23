"""One gateway from cell traces into Circadian Workbench.

Circadian Workbench owns the scientific implementations.  This module only
translates the analysis package's ``hours, values`` arrays into the timestamped
frame its core accepts and translates the returned rows back into plain dicts.
Keeping that seam here means future circadian analyses can be added without
copying another implementation into ``analysis.modules``.

Only this module imports Circadian Workbench. Private adapters preserve the
recording clock, option inheritance, and result table contract.
"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Sequence

import numpy as np
import pandas as pd

import circadian_workbench as cw

WORKBENCH_VERSION = cw.__version__

# New uniform entrance: exactly Workbench's arguments, defaults, units, caller
# and completed-result/figure contract. No Motion preset or history lookup.
trace = cw.trace
describe = cw.describe

def rhythm_environment() -> dict:
    """Public catalogue provenance, including same-version editable engine identity."""
    return deepcopy(cw.call("period_methods").run_record["environment"])

def argument_group(action: str = "compare_periods") -> dict[str, dict[str, Any]]:
    """Detached, authoritative scientific settings accepted by an action."""
    description = cw.describe(action)
    if not description["ok"]:
        raise ValueError(description["error"])
    references = {key for item in description["params"]
                  for key in item.get("references", [])}
    return {row["key"]: deepcopy(row)
            for row in description.get("config_arguments", [])
            if row["key"] in references}

# Circadian Workbench is the authority for baseline removal. Discover its
# choices through the public configuration contract; omit only alternate
# spellings of an algorithm already present under its canonical name.
_WORKBENCH_CONFIG_SCHEMA = argument_group()
_DETREND_SPEC = _WORKBENCH_CONFIG_SCHEMA["period_detrend"]
_DETREND_ALLOWED = [*_DETREND_SPEC["allowed"], *_DETREND_SPEC["aliases"]]
_PREFERRED_DETREND_ORDER = (
    "none", "running_mean", "moving_average", "moving_median",
    "running_median", "median", "linear", "robust_linear", "robust",
    "huber", "first_difference", "first-difference", "difference", "diff",
    "lowess", "loess", "savitzky_golay", "savitzky-golay", "savgol",
    "cubic", "bicubic", "poly6", "polynomial", "poly3", "degree6",
    "kernel", "baseline", "amp_baseline", "amp&baseline",
    "asymmetric_least_squares", "asymmetric-least-squares", "asls", "als",
    "frequency",
)
DETREND_METHODS: tuple[str, ...] = tuple(
    [name for name in _PREFERRED_DETREND_ORDER if name in _DETREND_ALLOWED]
    + sorted(set(_DETREND_ALLOWED) - set(_PREFERRED_DETREND_ORDER))
)

# Explicit compatibility values for existing measurement workflows, not a
# second default registry for the uniform trace entrance above.
DETREND_DEFAULTS: dict[str, Any] = {
    "detrend": "robust_linear",
    "detrend_window_hours": 24.0,
    "detrend_polynomial_degree": 3,
    "detrend_min_valid_fraction": 0.5,
    "detrend_bandwidth_hours": None,
    "detrend_low_cut_hours": 45.0,
    "detrend_high_cut_hours": 4.0,
    "detrend_filter_order": 2,
    "detrend_lowess_fraction": None,
    "detrend_lowess_iterations": 3,
    "detrend_asls_smoothness": 1_000_000.0,
    "detrend_asls_asymmetry": 0.01,
    "detrend_asls_iterations": 10,
}

# Shared by the main rhythms module and any other module that estimates a
# period. Keeping this beside the adapter prevents a coupling analysis or a new
# figure from silently reverting to a different search or estimator.
PERIOD_ANALYSIS_DEFAULTS: dict[str, Any] = {
    "period_search_hours": [2.0, 48.0],
    "period_methods": ["lomb", "chi_square", "f"],
    "period_estimation_method": "lomb",
    "primary_rhythm_test": "lomb",
    "fixed_period_hours": 24.0,
    "workbench_config": {},
    "jtk_periods": [4.0, 6.0, 8.0, 12.0, 16.0, 20.0, 24.0,
                    28.0, 32.0, 36.0, 48.0],
    "ejtk_permutations": 1000,
    "jtk_correction": "none",
    "jtk_seed": 20260901,
    "min_observations": 24,
    "min_cycles_for_confident_period": 3.0,
    "rhythmic_alpha": 0.05,
}

#: One option contract for every figure that performs a fresh period or rhythm
#: analysis. The broad search and significance defaults are intentionally
#: visible on every such figure; method and detrending choices inherit the
#: run's ``rhythms`` module setting when unset.
CIRCADIAN_ANALYSIS_OPTION_DEFAULTS: dict[str, Any] = {
    "fit_method": None,
    "significance_method": None,
    "period_config": {},
    "period_min_hours": 2.0,
    "period_max_hours": 48.0,
    "rhythmic_alpha": 0.05,
    "multiple_testing": "bh",
    "min_observations": 24,
    "min_cycles": 3.0,
    **{name: None for name in DETREND_DEFAULTS},
}
CIRCADIAN_ANALYSIS_OPTIONS: tuple[str, ...] = tuple(
    CIRCADIAN_ANALYSIS_OPTION_DEFAULTS
)

PERIOD_METHODS: dict[str, dict[str, Any]] = {
    str(entry["key"]): dict(entry)
    for entry in cw.call("period_methods").data["methods"]
}
SIGNIFICANCE_METHODS: tuple[str, ...] = tuple(
    key for key, entry in PERIOD_METHODS.items() if entry["gives_significance"]
)
NORMALIZATION_METHODS: dict[str, dict[str, Any]] = {
    str(entry["key"]): dict(entry)
    for entry in cw.normalization_methods().data["methods"]
}

def sample_interval_minutes(hours: Sequence[float]) -> float:
    """Median positive sampling interval of one trace, in minutes."""
    ordered = np.sort(np.unique(np.asarray(hours, dtype=float)))
    differences = np.diff(ordered)
    positive = differences[np.isfinite(differences) & (differences > 0)]
    if positive.size == 0:
        raise ValueError("a rhythm trace needs at least two distinct times")
    return float(np.median(positive) * 60.0)

def selected_frame(hours: Sequence[float], values: Sequence[float]) -> cw.TraceData:
    """A cell trace in Circadian Workbench's public input shape."""
    return cw.TraceData(hours, values, name="cell trace")

def available_normalization_methods() -> list[dict[str, Any]]:
    """The installed workbench's normalisation keys, formulas and aliases."""

    return [dict(entry) for entry in NORMALIZATION_METHODS.values()]

def generate_benchmark_cases(design, profiles, *, partition, seed):
    """Generate known-truth cases through Workbench, retaining its run record.

    The pipeline owns candidate selection and confirmation access. This gateway
    adds no local waveform, noise, truth-eligibility or resampling formula.
    """
    if not hasattr(cw, "benchmark_cases"):
        raise RuntimeError("Installed Circadian Workbench lacks public benchmark_cases support")
    completed = cw.benchmark_cases(design, profiles, partition=partition, seed=seed)
    result = deepcopy(completed.data)
    result["workbench_run_record"] = deepcopy(completed.run_record)
    return result

def filter_rhythm_trace(hours, values, filtering):
    """Filter through Workbench while retaining original positions and its run record."""
    settings = {key: value for key, value in dict(filtering).items()
                if key in {"method", "window_hours", "max_gap_hours", "min_observations"}}
    completed = cw.filter_trace(hours, values, filtering=settings)
    return {**deepcopy(completed.data), "workbench_run_record": deepcopy(completed.run_record)}

def benchmark_score_interval(units, *, confidence, bounds):
    """Uncertainty for declared independent simulation units, computed by Workbench."""
    completed = cw.bounded_mean_interval(units, confidence=confidence, bounds=bounds)
    return {**deepcopy(completed.data), "workbench_run_record": deepcopy(completed.run_record)}

def rhythm_timing_summary(records, *, independent_units=False, settings=None):
    """Keep comparable cell offsets and independent biological samples separate."""
    completed = cw.rhythm_timing_summary(records, independent_units=independent_units, settings=settings)
    return {**deepcopy(completed.data), "workbench_version": WORKBENCH_VERSION,
            "workbench_run_record": deepcopy(completed.run_record)}

def rhythm_pair_timing(reference, target, *, settings=None):
    """Use public Workbench period comparability and observed relative timing."""
    completed = cw.rhythm_pair_timing(reference, target, settings=settings)
    return {**deepcopy(completed.data), "workbench_version": WORKBENCH_VERSION,
            "workbench_run_record": deepcopy(completed.run_record)}

def rhythm_window_comparison(reference, target, *, settings=None):
    """Direct native component comparisons across independently analysed windows."""
    completed = cw.rhythm_window_comparison(reference, target, settings=settings)
    return {**deepcopy(completed.data), "workbench_version": WORKBENCH_VERSION,
            "workbench_run_record": deepcopy(completed.run_record)}

def rhythm_detection_agreement(records, *, independent_units=False, confidence=.95,
                               sample_method="none", permutations=9999, seed=20260910):
    """Public binary-detection statistics; never fit, threshold or re-correct traces."""
    completed = cw.detection_agreement(records, independent_units=independent_units,
        confidence=confidence, sample_method=sample_method, permutations=permutations, seed=seed)
    return {**deepcopy(completed.data), "workbench_version": WORKBENCH_VERSION,
            "workbench_run_record": deepcopy(completed.run_record)}

def omit_rhythm_intervals(hours, values, intervals):
    """Mask declared closed time intervals through Workbench, preserving original rows."""
    completed = cw.filter_trace(hours, values, filtering={"method": "none", "omit_intervals_hours": intervals})
    return {**deepcopy(completed.data), "workbench_run_record": deepcopy(completed.run_record)}

def adjust_pvalues(values: Sequence[float], correction: str) -> np.ndarray:
    """Correct one p-value family through Workbench's public statistics API."""
    raw = np.asarray(values, dtype=float)
    adjusted = cw.statistics.adjust_pvalues(raw, correction)
    return np.asarray(adjusted, dtype=float)

def resolve_significance_method(method: str, significance_method: str | None = None) -> str:
    """Name the statistical test separately when the period fitter has none."""
    if method not in PERIOD_METHODS:
        raise ValueError(f"unknown period method {method!r}")
    selected = significance_method or (method if method in SIGNIFICANCE_METHODS else "lomb")
    if selected not in SIGNIFICANCE_METHODS:
        raise ValueError(
            f"{selected!r} cannot test rhythmicity; choose " + ", ".join(SIGNIFICANCE_METHODS))
    return selected

# Public statistical modules, owned and tested by Circadian Workbench.
statistics = cw.statistics
group_contrasts = cw.group_contrasts
association = cw.association
sample_contrasts = cw.sample_contrasts
spatial_permutation = cw.spatial_permutation
segmented_regression = cw.segmented_regression
population, call = cw.population, cw.call
PUBLIC_API_VERSION = cw.PUBLIC_API_VERSION
from ._workbench.options import (
    resolve_analysis_options, scientific_options, detrend_settings, workbench_config,
)
from ._workbench.processing import (
    _processed_trace_in_input_time, normalize_trace, detrend_trace, scale_detrended,
)
from ._workbench.descriptive import (
    daily_measures, daily_profile_metrics, cosinor, descriptive_cosinor_fitted_values, phase_summary, _number_or_nan, _daily_summary, nonparametric, fft_nlls_fitted_values,
)
from ._workbench.estimates import (
    estimate_trace, estimate_one, lomb_periodogram, estimate_grouped_rhythms,
)

phases, channels = cw.phases, cw.channels


def surrogate(values, model, generator):
    """Construct the unchanged seeded rhythm null in Circadian Workbench."""
    return cw.null_models.surrogate(values, model, generator)


def cycle_profile(phases, values, positions):
    return cw.display_values.cycle_profile(phases,values,positions)


def peak_fraction(profile, positions):
    return cw.display_values.peak_fraction(profile,positions)


def cycle_fraction(timing, period):
    return cw.display_values.cycle_fraction(timing,period)


def circular_range(hours, period):
    return cw.display_values.circular_range(hours,period)


def peak_aligned_summary(traces,peaks,*,period_hours,bins=24):
    return cw.display_values.peak_aligned_summary(traces,peaks,period_hours=period_hours,bins=bins)

contact_statistics = cw.contact_statistics


def weighted_cycle_vector(hours, weights, *, period_hours):
    return cw.display_values.weighted_cycle_vector(hours, weights, period_hours=period_hours)
