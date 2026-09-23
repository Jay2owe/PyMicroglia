"""Input and result translations; science is called through workbench."""
from __future__ import annotations
from copy import deepcopy
import json
from typing import Any, Sequence
import numpy as np
import pandas as pd
from .. import workbench as wb

def resolve_analysis_options(params: dict, option: Any) -> dict[str, Any]:
    """Resolve the shared Circadian Workbench controls for one figure.

    ``option`` is normally ``FigureContext.option``. Keeping this resolution
    beside the gateway prevents individual plots from assigning different
    meanings or inheritance rules to the same controls.
    """
    inherited = dict(params)

    def choose(name: str, fallback: Any) -> Any:
        value = option(name)
        return fallback if value is None else value
    method = str(choose('fit_method', inherited.get('period_estimation_method', inherited.get('primary_rhythm_test', 'lomb'))))
    significance_method = wb.resolve_significance_method(method, choose('significance_method', inherited.get('primary_rhythm_test', 'lomb')))
    low = float(choose('period_min_hours', inherited['period_search_hours'][0]))
    high = float(choose('period_max_hours', inherited['period_search_hours'][1]))
    alpha = float(choose('rhythmic_alpha', inherited.get('rhythmic_alpha', 0.05)))
    min_observations = int(choose('min_observations', inherited.get('min_observations', 24)))
    min_cycles = float(choose('min_cycles', inherited.get('min_cycles_for_confident_period', 3.0)))
    correction = str(choose('multiple_testing', inherited.get('multiple_testing', 'bh')))
    if not 0 < low < high:
        raise ValueError('period limits must satisfy 0 < minimum < maximum')
    if not 0 < alpha < 1:
        raise ValueError('rhythmic_alpha must be between 0 and 1')
    if min_observations < 6:
        raise ValueError('min_observations must be at least 6')
    if not np.isfinite(min_cycles) or min_cycles <= 0:
        raise ValueError('min_cycles must be finite and positive')
    if correction not in {'bh', 'bonferroni', 'sidak', 'none'}:
        raise ValueError('multiple_testing must be bh, bonferroni, sidak or none')
    detrend_args = {'method': option('detrend'), 'window_hours': option('detrend_window_hours'), 'polynomial_degree': option('detrend_polynomial_degree'), 'min_valid_fraction': option('detrend_min_valid_fraction'), 'bandwidth_hours': option('detrend_bandwidth_hours'), 'low_cut_hours': option('detrend_low_cut_hours'), 'high_cut_hours': option('detrend_high_cut_hours'), 'filter_order': option('detrend_filter_order'), 'lowess_fraction': option('detrend_lowess_fraction'), 'lowess_iterations': option('detrend_lowess_iterations'), 'asls_smoothness': option('detrend_asls_smoothness'), 'asls_asymmetry': option('detrend_asls_asymmetry'), 'asls_iterations': option('detrend_asls_iterations')}
    detrending = wb.detrend_settings(inherited, **detrend_args)
    extra = option('period_config')
    if not isinstance(extra, dict):
        raise ValueError('period_config must be a JSON object')
    methods = list(inherited.get('period_methods', []))
    for required in (method, significance_method):
        if required not in methods:
            methods.append(required)
    resolved_params = {**inherited, **detrending, 'period_search_hours': [low, high], 'period_estimation_method': method, 'primary_rhythm_test': significance_method, 'period_methods': methods, 'rhythmic_alpha': alpha, 'multiple_testing': correction, 'min_observations': min_observations, 'min_cycles_for_confident_period': min_cycles, 'workbench_config': {**dict(inherited.get('workbench_config') or {}), **extra}}
    return {'params': resolved_params, 'method': method, 'significance_method': significance_method, 'period_min_hours': low, 'period_max_hours': high, 'rhythmic_alpha': alpha, 'multiple_testing': correction, 'min_observations': min_observations, 'min_cycles': min_cycles, **detrending}

def scientific_options() -> dict[str, dict[str, Any]]:
    """Workbench meanings with Motion spelling/role bindings, not new defaults.

    Family correction and data/cycle sufficiency are Motion workflow controls;
    they are deliberately not presented as Workbench estimator arguments.
    """
    config = wb.argument_group()
    mapping = {'fit_method': 'period_method', 'period_methods': 'period_methods', 'significance_method': 'period_method', 'secondary_significance_method': 'period_method', 'period_min_hours': 'period_min_hours', 'period_max_hours': 'period_max_hours', 'rhythmic_alpha': 'periodogram_alpha', **{name: 'period_' + name for name in wb.DETREND_DEFAULTS}, 'detrend_methods': 'period_detrend'}
    entries = {name: {**deepcopy(config[key]), 'reference': 'config.' + key} for name, key in mapping.items()}
    for name in ('significance_method', 'secondary_significance_method'):
        entries[name]['allowed'] = list(wb.SIGNIFICANCE_METHODS)
        entries[name]['role'] = 'Significance test, selected separately from period estimation.'
    entries['rhythmic_alpha']['references'] = ['config.periodogram_alpha', 'config.jtk_alpha']
    entries['rhythmic_alpha']['role'] = 'Explicit threshold also supplied to rank tests and family correction.'
    entries['detrend_methods']['type'] = 'list'
    entries['period_config'] = deepcopy(next((item for item in wb.cw.describe('compare_periods')['params'] if item['name'] == 'config')))
    return entries

def detrend_settings(params: dict | None=None, *, method: str | None=None, window_hours: float | None=None, polynomial_degree: int | None=None, min_valid_fraction: float | None=None, bandwidth_hours: float | None=None, low_cut_hours: float | None=None, high_cut_hours: float | None=None, filter_order: int | None=None, lowess_fraction: float | None=None, lowess_iterations: int | None=None, asls_smoothness: float | None=None, asls_asymmetry: float | None=None, asls_iterations: int | None=None) -> dict[str, Any]:
    """Resolve and validate the shared Circadian Workbench detrending controls."""
    supplied = dict(params or {})
    requested = str(method if method is not None else supplied.get('detrend', wb.DETREND_DEFAULTS['detrend']))
    if requested not in wb.DETREND_METHODS:
        raise ValueError(f'unknown detrend method {requested!r}; available: ' + ', '.join(wb.DETREND_METHODS))
    window = float(window_hours if window_hours is not None else supplied.get('detrend_window_hours', wb.DETREND_DEFAULTS['detrend_window_hours']))
    if not np.isfinite(window) or window <= 0:
        raise ValueError('detrend_window_hours must be a finite positive number')
    degree = int(polynomial_degree if polynomial_degree is not None else supplied.get('detrend_polynomial_degree', wb.DETREND_DEFAULTS['detrend_polynomial_degree']))
    if degree < 0:
        raise ValueError('detrend_polynomial_degree must be at least zero')
    valid_fraction = float(min_valid_fraction if min_valid_fraction is not None else supplied.get('detrend_min_valid_fraction', wb.DETREND_DEFAULTS['detrend_min_valid_fraction']))
    if not np.isfinite(valid_fraction) or not 0 <= valid_fraction <= 1:
        raise ValueError('detrend_min_valid_fraction must be between zero and one')
    bandwidth = bandwidth_hours if bandwidth_hours is not None else supplied.get('detrend_bandwidth_hours')
    bandwidth = None if bandwidth in (None, '') else float(bandwidth)
    if bandwidth is not None and (not np.isfinite(bandwidth) or bandwidth <= 0):
        raise ValueError('detrend_bandwidth_hours must be a finite positive number')
    low_cut = float(low_cut_hours if low_cut_hours is not None else supplied.get('detrend_low_cut_hours', wb.DETREND_DEFAULTS['detrend_low_cut_hours']))
    high_cut = float(high_cut_hours if high_cut_hours is not None else supplied.get('detrend_high_cut_hours', wb.DETREND_DEFAULTS['detrend_high_cut_hours']))
    if not np.isfinite(low_cut) or low_cut <= 0:
        raise ValueError('detrend_low_cut_hours must be a finite positive number')
    if not np.isfinite(high_cut) or high_cut <= 0:
        raise ValueError('detrend_high_cut_hours must be a finite positive number')
    order = int(filter_order if filter_order is not None else supplied.get('detrend_filter_order', wb.DETREND_DEFAULTS['detrend_filter_order']))
    if order < 1:
        raise ValueError('detrend_filter_order must be at least one')
    lowess_share = lowess_fraction if lowess_fraction is not None else supplied.get('detrend_lowess_fraction')
    lowess_share = None if lowess_share in (None, '') else float(lowess_share)
    if lowess_share is not None and (not np.isfinite(lowess_share) or not 0 < lowess_share <= 1):
        raise ValueError('detrend_lowess_fraction must be between zero and one')
    lowess_passes = int(lowess_iterations if lowess_iterations is not None else supplied.get('detrend_lowess_iterations', wb.DETREND_DEFAULTS['detrend_lowess_iterations']))
    if not 0 <= lowess_passes <= 20:
        raise ValueError('detrend_lowess_iterations must be between zero and 20')
    smoothness = float(asls_smoothness if asls_smoothness is not None else supplied.get('detrend_asls_smoothness', wb.DETREND_DEFAULTS['detrend_asls_smoothness']))
    if not np.isfinite(smoothness) or smoothness <= 0:
        raise ValueError('detrend_asls_smoothness must be finite and positive')
    asymmetry = float(asls_asymmetry if asls_asymmetry is not None else supplied.get('detrend_asls_asymmetry', wb.DETREND_DEFAULTS['detrend_asls_asymmetry']))
    if not np.isfinite(asymmetry) or not 0 < asymmetry < 0.5:
        raise ValueError('detrend_asls_asymmetry must be between zero and 0.5')
    asls_passes = int(asls_iterations if asls_iterations is not None else supplied.get('detrend_asls_iterations', wb.DETREND_DEFAULTS['detrend_asls_iterations']))
    if not 1 <= asls_passes <= 100:
        raise ValueError('detrend_asls_iterations must be between one and 100')
    return {'detrend': requested, 'detrend_window_hours': window, 'detrend_polynomial_degree': degree, 'detrend_min_valid_fraction': valid_fraction, 'detrend_bandwidth_hours': bandwidth, 'detrend_low_cut_hours': low_cut, 'detrend_high_cut_hours': high_cut, 'detrend_filter_order': order, 'detrend_lowess_fraction': lowess_share, 'detrend_lowess_iterations': lowess_passes, 'detrend_asls_smoothness': smoothness, 'detrend_asls_asymmetry': asymmetry, 'detrend_asls_iterations': asls_passes}

def workbench_config(params: dict, hours: Sequence[float], *, detrend: str | None=None, detrend_window_hours: float | None=None, methods: Sequence[str] | None=None) -> dict[str, Any]:
    """Map the rhythm-module settings onto Circadian Workbench's config."""
    low, high = params['period_search_hours']
    chosen = list(methods if methods is not None else params['period_methods'])
    primary = str(params['primary_rhythm_test'])
    estimator = str(params.get('period_estimation_method', primary))
    unknown = [name for name in chosen if name not in wb.PERIOD_METHODS]
    if unknown:
        raise ValueError(f'unknown Circadian Workbench period method {unknown[0]!r}; available: ' + ', '.join(wb.PERIOD_METHODS))
    if methods is None:
        if primary not in chosen:
            raise ValueError(f'primary_rhythm_test {primary!r} is not in period_methods')
        if primary not in wb.SIGNIFICANCE_METHODS:
            raise ValueError(f'primary_rhythm_test {primary!r} cannot call a trace rhythmic; choose one of ' + ', '.join(wb.SIGNIFICANCE_METHODS))
        if estimator not in chosen:
            raise ValueError(f'period_estimation_method {estimator!r} is not in period_methods')
    detrending = wb.detrend_settings(params, method=detrend, window_hours=detrend_window_hours)
    overrides = dict(params.get('workbench_config') or {})
    overrides.update({'period_hours': float(params['fixed_period_hours']), 'period_min_hours': float(low), 'period_max_hours': float(high), 'bin_minutes': max(1, int(round(wb.sample_interval_minutes(hours)))), 'periodogram_alpha': float(params['rhythmic_alpha']), 'jtk_alpha': float(params['rhythmic_alpha']), 'period_detrend': detrending['detrend'], 'period_detrend_window_hours': detrending['detrend_window_hours'], 'period_detrend_polynomial_degree': detrending['detrend_polynomial_degree'], 'period_detrend_min_valid_fraction': detrending['detrend_min_valid_fraction'], 'period_detrend_bandwidth_hours': detrending['detrend_bandwidth_hours'], 'period_detrend_low_cut_hours': detrending['detrend_low_cut_hours'], 'period_detrend_high_cut_hours': detrending['detrend_high_cut_hours'], 'period_detrend_filter_order': detrending['detrend_filter_order'], 'period_detrend_lowess_fraction': detrending['detrend_lowess_fraction'], 'period_detrend_lowess_iterations': detrending['detrend_lowess_iterations'], 'period_detrend_asls_smoothness': detrending['detrend_asls_smoothness'], 'period_detrend_asls_asymmetry': detrending['detrend_asls_asymmetry'], 'period_detrend_asls_iterations': detrending['detrend_asls_iterations'], 'period_method': estimator if estimator in chosen else chosen[0], 'period_methods': chosen, 'jtk_periods': list(params['jtk_periods']), 'ejtk_permutations': int(params['ejtk_permutations']), 'jtk_correction': str(params['jtk_correction']), 'jtk_seed': int(params['jtk_seed']), 'onset_off_hours': float(params.get('onset_off_hours', 6.0)), 'onset_on_hours': float(params.get('onset_on_hours', 6.0)), 'onset_threshold_percent': float(params.get('onset_threshold_percent', 80.0))})
    return dict(wb.cw.call('normalize_config', config=overrides).data)
