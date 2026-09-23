"""Input and result translations; science is called through workbench."""
from __future__ import annotations
from copy import deepcopy
import json
from typing import Any, Sequence
import numpy as np
import pandas as pd
from .. import workbench as wb

def daily_measures(hours: Sequence[float], values: Sequence[float], params: dict) -> dict:
    """Daily onset, offset and active duration from Workbench's public action."""
    config = wb.workbench_config(params, hours)
    return dict(wb.cw.trace(hours, values).run('daily_measures', config=config).data)

def daily_profile_metrics(hours: Sequence[float], values: Sequence[float], params: dict) -> dict:
    """Workbench's M10/L5 contrast, timing and interdaily stability only.

    The public nonparametric action works directly on the measured trace. It
    does not fit a cosine, search a period, or invoke the full analysis action.
    These are fixed 24-hour summaries, independent of period-fit settings.
    """
    bin_hours = float(params.get('bin_hours', 1.0))
    if not np.isfinite(bin_hours) or bin_hours <= 0 or (not np.isclose(24 / bin_hours, round(24 / bin_hours))):
        raise ValueError('daily-profile bin_hours must divide a 24-hour day')
    config = wb.workbench_config(params, hours)
    config['bin_minutes'] = int(round(bin_hours * 60))
    return dict(wb.cw.trace(hours, values).run('nonparametric', config=config).data)

def cosinor(hours: Sequence[float], values: Sequence[float], period_hours: float, reference_level: float | None=None) -> dict[str, Any]:
    """Fit one fixed-period cosine through Circadian Workbench's public API."""
    if len(hours) < 4:
        return {}
    result = dict(wb.cw.trace(hours, values, name='cell trace').cosinor(period_hours=float(period_hours)).data)
    mesor = result.get('mesor')
    amplitude = result.get('amplitude')
    if amplitude is None:
        return {}
    denominator = reference_level if reference_level is not None else mesor
    relative = float(amplitude) / abs(float(denominator)) if denominator is not None and np.isfinite(denominator) and (denominator != 0) else np.nan
    return {'cosinor_mesor': wb._number_or_nan(mesor), 'cosinor_amplitude': wb._number_or_nan(amplitude), 'cosinor_relative_amplitude': relative, 'cosinor_peak_hour': wb._number_or_nan(result.get('acrophase_hours')), 'cosinor_phase_convention': 'positive_peak_time', 'cosinor_r_squared': wb._number_or_nan(result.get('r_squared')), 'cosinor_p_value': wb._number_or_nan(result.get('p_value')), 'cosinor_relative_amplitude_error': np.nan, 'workbench_version': wb.WORKBENCH_VERSION}

def descriptive_cosinor_fitted_values(hours: Sequence[float], values: Sequence[float], period_hours: float) -> np.ndarray:
    """Evaluate an explicitly requested fixed-period Workbench cosinor fit."""
    time = np.asarray(hours, dtype=float)
    measured = np.asarray(values, dtype=float)
    fitted = np.full(time.shape, np.nan, dtype=float)
    usable = np.isfinite(time) & np.isfinite(measured)
    if usable.sum() < 4 or not np.isfinite(period_hours) or period_hours <= 0:
        return fitted
    result = dict(wb.cw.trace(time[usable], measured[usable], name='display trace').cosinor(period_hours=float(period_hours)).data)
    mesor = wb._number_or_nan(result.get('mesor'))
    amplitude = wb._number_or_nan(result.get('amplitude'))
    acrophase = wb._number_or_nan(result.get('acrophase_hours'))
    if not np.isfinite([mesor, amplitude, acrophase]).all():
        return fitted
    return wb.cw.display_values.cosinor_values(time,mesor=mesor,amplitude=amplitude,
        acrophase_hours=acrophase,period_hours=float(period_hours))

def phase_summary(peak_hours: Sequence[float], period_hours: float) -> dict[str, Any]:
    """Summarize phase concentration through Workbench's public phase caller."""
    finite = np.asarray(peak_hours, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 3:
        return {}
    result = dict(wb.cw.phases(finite.tolist(), period_hours=float(period_hours)).summary(label='cells').data)
    if result.get('status') != 'ok':
        return {}
    return {'cells': int(result['n']), 'vector_length': wb._number_or_nan(result.get('resultant_length')), 'mean_peak_hour': wb._number_or_nan(result.get('mean_hours')), 'rayleigh_p_value': wb._number_or_nan(result.get('p_value')), 'workbench_version': wb.WORKBENCH_VERSION}

def _number_or_nan(value: Any) -> float:
    """One optional Workbench scalar in the project's numeric table shape."""
    return float(value) if value is not None else np.nan

def _daily_summary(days: Sequence[dict[str, Any]], key: str, *, mean: bool=False) -> float:
    """Summarize a Workbench daily field while preserving missing results."""
    values = np.asarray([row.get(key) if row.get(key) is not None else np.nan for row in days], dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return np.nan
    return float(np.mean(finite) if mean else np.median(finite))

def nonparametric(hours: Sequence[float], values: Sequence[float], params: dict) -> dict[str, Any]:
    """Return the nonparametric and active-phase block from Workbench actions."""
    bin_hours = float(params['bin_hours'])
    bins_per_day = 24.0 / bin_hours if bin_hours > 0 else 0.0
    if bin_hours <= 0 or abs(bins_per_day - round(bins_per_day)) > 1e-09:
        raise ValueError(f'rhythms bin_hours must divide a 24-hour day, and {bin_hours!r} does not')
    if float(params.get('low_window_hours', 5.0)) != 5.0 or float(params.get('high_window_hours', 10.0)) != 10.0:
        raise ValueError('Circadian Workbench defines the nonparametric windows as L5 and M10; low_window_hours must be 5 and high_window_hours must be 10')
    trace = wb.cw.trace(hours, values, name='cell trace')
    config = wb.workbench_config(params, hours)
    config['bin_minutes'] = int(round(bin_hours * 60.0))
    metrics = dict(trace.run('nonparametric', config=config).data)
    daily = dict(trace.run('daily_measures', config=config).data)
    days = list(daily.get('days', []))
    hourly_config = {**config, 'bin_minutes': 60}
    hourly = dict(trace.run('time_series', config=hourly_config).data)
    hourly_values = hourly.get('activity', [])
    hours_binned = sum((value is not None for value in hourly_values))
    elapsed = np.asarray(hours, dtype=float)
    span = float(np.nanmax(elapsed) - np.nanmin(elapsed)) if len(elapsed) else np.nan
    days_covered = span / 24.0
    onset = wb._daily_summary(days, 'onset_hours')
    offset = wb._daily_summary(days, 'offset_hours')
    duration = wb._daily_summary(days, 'alpha_hours', mean=True)
    return {'m10': wb._number_or_nan(metrics.get('m10_mean')), 'm10_onset_hour': wb._number_or_nan(metrics.get('m10_start_hours')), 'l5': wb._number_or_nan(metrics.get('l5_mean')), 'l5_onset_hour': wb._number_or_nan(metrics.get('l5_start_hours')), 'relative_amplitude': wb._number_or_nan(metrics.get('relative_amplitude')), 'onset_hour': onset, 'offset_hour': offset, 'active_duration_hours': duration, 'interdaily_stability': wb._number_or_nan(metrics.get('interdaily_stability')), 'intradaily_variability': wb._number_or_nan(metrics.get('intradaily_variability')), 'days_covered': days_covered, 'hours_binned': int(hours_binned), 'stability_underdetermined': days_covered < float(params['min_days_for_stability']), 'onset_found': bool(np.isfinite(onset)), 'workbench_version': wb.WORKBENCH_VERSION}

def fft_nlls_fitted_values(hours: Sequence[float], estimate: dict) -> np.ndarray:
    """Evaluate the returned multi-component model; never refit a substitute cosine."""
    diagnostics = estimate.get('diagnostics') or {}
    components = estimate.get('components') or []
    if estimate.get('method') != 'fft_nlls' or diagnostics.get('status') != 'ok' or (not components) or (not diagnostics.get('phase_zero_timestamp')):
        return np.full(len(hours), np.nan)
    origin = (pd.Timestamp(diagnostics['phase_zero_timestamp']) - pd.Timestamp('2000-01-01')).total_seconds() / 3600
    local = np.asarray(hours, float) - origin
    fitted = np.full(local.shape, float(diagnostics['mesor']))
    for component in components:
        fitted += float(component['amplitude']) * np.cos(2 * np.pi * (local - float(component['phase_hours'])) / float(component['period_hours']))
    return fitted
