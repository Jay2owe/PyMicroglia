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
from contextlib import contextmanager
from contextvars import ContextVar
import json
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .baseline_defaults import DEFAULT_DETREND_METHOD, FFT_COMPONENT_DETRENDS

import circadian_workbench as cw
_test_components = cw.component_significance.test_components

WORKBENCH_VERSION = cw.__version__

# New uniform entrance: exactly Workbench's arguments, defaults, units, caller
# and completed-result/figure contract. No Motion preset or history lookup.
trace = cw.trace
describe = cw.describe


def test_simulation_recording(hours, values, responses, outline_areas, candidates, **options):
    """Recording-wide optical rhythm test through Workbench's public module."""
    engine=getattr(cw,'optical_rhythm',None)
    if engine is None:
        raise RuntimeError('Update Circadian Workbench to a build with optical_rhythm support')
    return engine.test_recording(hours,values,responses,outline_areas,candidates,**options)


def cell_analysis_catalogue():
    """Workbench UI and core plots that accept a single measured cell."""
    grid = getattr(cw, 'analysis_grid', None)
    return {} if grid is None else grid.catalogue()


def fft_component_values(hours, estimate):
    function = getattr(cw.display_values, 'fft_nlls_component_values', None)
    if function is None:
        raise RuntimeError('Update Circadian Workbench to enable native component displays')
    return function(hours, estimate)


def fft_component_test_significant(test, alpha):
    return cw.display_values.fft_component_test_significant(test, alpha)


def fft_component_test_label(test, period):
    return cw.display_values.fft_component_test_label(test, period)


def test_fft_component_view(definition, **settings):
    return _cell_grid_api().test_fft_component_view(definition, **settings)


def style_fft_component_view(definition, tests, **settings):
    return _cell_grid_api().style_fft_component_view(definition, tests, **settings)


def fft_component_lanes(definition):
    return _cell_grid_api().fft_component_lanes(definition)


def _cell_grid_api():
    grid = getattr(cw, 'analysis_grid', None)
    if grid is None:
        raise RuntimeError('Update Circadian Workbench to a build with analysis_grid support to use Cell analysis grid')
    return grid


def validate_cell_analysis(analysis, **options):
    return _cell_grid_api().validate(analysis, **options)


def analyze_cell_view(hours, values, **options):
    return _cell_grid_api().analyze(hours, values, **options)


def analyze_cell_views(hours, values, **options):
    return _shared_cell_grid_api().analyze_many(hours, values, **options)


def cell_grid_shape(count, rows=None, columns=None):
    return _shared_cell_grid_api().grid_shape(count, rows, columns)


def create_cell_grid(canvas, **options):
    return _shared_cell_grid_api().create_layout(canvas, **options)


def create_cell_grid_lanes(canvas, **options):
    function = getattr(_cell_grid_api(), 'create_lane_layout', None)
    if not callable(function):
        raise RuntimeError('Update Circadian Workbench to enable stacked cell-grid lanes')
    return function(canvas, **options)


def overlay_cell_views(panels, **options):
    function = getattr(_cell_grid_api(), 'overlay', None)
    if not callable(function):
        raise RuntimeError('Update Circadian Workbench to enable native cell-grid overlays')
    return function(panels, **options)


def _shared_cell_grid_api():
    grid = _cell_grid_api()
    if not all(callable(getattr(grid, key, None)) for key in ('analyze_many', 'grid_shape', 'create_layout')):
        raise RuntimeError('Update Circadian Workbench to a build with shared cell-grid support')
    return grid


def cell_view_data(definition):
    return _cell_grid_api().figure_data(definition)


def empty_cell_view(title, reason):
    return _cell_grid_api().empty(title, reason)


def draw_cell_views(panels, **options):
    return _cell_grid_api().draw(panels, **options)

def rhythm_environment() -> dict:
    """Public catalogue provenance, including same-version editable engine identity."""
    return deepcopy(cw.call("period_methods").run_record["environment"])


_ANALYSIS_CAPTURE = ContextVar("pymicroglia_analysis_capture", default=None)


def verify_rhythm_environment(expected):
    """Verify a sweep's captured software identity before retaining its results."""
    cw.run_records.verify_environment(expected)


@contextmanager
def analysis_session():
    """Share the existing provenance capture across a batch of trace calls.

    Workbench's nested calls inherit one software identity. Check that identity
    again at the boundary so source edits during a run cannot be attached as
    if they were produced by one unchanged environment.
    """
    existing = _ANALYSIS_CAPTURE.get()
    if existing is not None:
        # The owning outer sweep performs the fresh environment verification.
        yield existing
        return
    with cw.run_records.capture_inputs() as capture:
        token = _ANALYSIS_CAPTURE.set(capture)
        try:
            yield capture
        finally:
            _ANALYSIS_CAPTURE.reset(token)
    verify_rhythm_environment(capture.environment)

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

# Default for fresh microglial rhythm analyses and measured-cell displays.
# Explicit saved recipes continue to supply their own detrending choice.
# Explicit compatibility values for existing measurement workflows, not a
# second default registry for the uniform trace entrance above.
DETREND_DEFAULTS: dict[str, Any] = {
    "detrend": DEFAULT_DETREND_METHOD,
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

def filter_methods():
    """Input-filter controls are declared by the scientific engine."""
    return deepcopy(cw.filter_methods())


def validate_trace_filter(filtering):
    """Canonical recipe shared by audit variants and main analysis rows."""
    if (not isinstance(filtering, dict) or not isinstance(filtering.get('method'), str)
            or filtering['method'] not in filter_methods()):
        raise ValueError('Choose an available input filter.')
    method = filtering['method']
    allowed = {'method'} if method == 'none' else {'method', 'window_points'}
    if set(filtering) - allowed:
        raise ValueError('Unknown input-filter settings.')
    if method == 'none':
        return {'method': 'none'}
    points = filtering.get('window_points')
    if isinstance(points, bool) or not isinstance(points, int) or not 3 <= points <= 101 or points % 2 == 0:
        raise ValueError('Filter window must be an odd sample count from 3 to 101.')
    return {'method': method, 'window_points': points}


def audit_input_filters(filtering=None, *, median_window_points=None):
    """Always retain unfiltered input; add only explicitly requested checks."""
    if median_window_points is not None:
        validate_trace_filter({'method': 'median', 'window_points': median_window_points})
    if filtering is None:
        filtering = [{'method': 'none'}]
        if median_window_points is not None:
            filtering.append({'method': 'median', 'window_points': median_window_points})
    if not isinstance(filtering, list):
        raise ValueError('Audit input filters must be a list of filter settings.')
    output = [{'method': 'none'}]
    for recipe in filtering:
        recipe = validate_trace_filter(recipe)
        if recipe not in output:
            output.append(recipe)
    return output


def audit_filter_key(recipe):
    return 'raw' if recipe['method'] == 'none' else f"{recipe['method']}_{recipe['window_points']}"


def audit_filter_label(recipe):
    if recipe['method'] == 'none':
        return 'None (unfiltered)'
    return f"{filter_methods()[recipe['method']]['label']} ({recipe['window_points']} samples)"


def filter_rhythm_trace(hours, values, filtering):
    """Filter through Workbench while retaining original positions and its run record."""
    filtering = dict(filtering)
    points = filtering.pop('window_points', None)
    if points is not None:
        if isinstance(points, bool) or not isinstance(points, int) or points < 3 or points % 2 == 0:
            raise ValueError('window_points must be an odd sample count of at least 3')
        interval = sample_interval_minutes(hours) / 60.0
        filtering.update(window_hours=points * interval, max_gap_hours=1.5 * interval,
                         min_observations=1)
    settings = {key: value for key, value in dict(filtering).items()
                if key in {"method", "window_hours", "max_gap_hours", "min_observations"}}
    completed = cw.filter_trace(hours, values, filtering=settings)
    return {**deepcopy(completed.data), "workbench_run_record": deepcopy(completed.run_record)}


def restore_rhythm_fit(hours, values, fitted, params):
    """Restore the native analysis baseline through Workbench's display API."""
    options = detrend_settings(params)
    settings = {key.removeprefix('detrend_'): value for key, value in options.items()}
    settings['method'] = settings.pop('detrend')
    restore = getattr(cw.display_values, 'restore_baseline', None)
    if restore is None:
        raise ValueError('Update Circadian Workbench to enable saved model baseline restoration')
    return restore(hours, values, fitted, **settings)


def test_fft_components(hours, detrended_values, periods_hours, **settings):
    """Use Workbench's conditional, block-surrogate component test."""
    return _test_components(
        hours, detrended_values, periods_hours, **settings
    )

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
cosine_fit_batch = cw.batch_periods.cosine_fit_batch


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
