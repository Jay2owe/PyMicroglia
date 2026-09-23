"""Period-method comparisons through the shared Workbench gateway."""
from __future__ import annotations
from typing import Any
import json
import numpy as np
import pandas as pd
from .. import workbench
from ..measure.modules.rhythms import DEFAULTS as RHYTHM_DEFAULTS
from ..pipelines.audit.evaluator import safe_estimate
from .radial import _selected_identities
from .prepared import PreparedViews

DEFAULT_DETRENDS = ('linear', 'robust_linear', 'first_difference', 'poly3', 'poly6', 'baseline', 'amp_baseline')

DEFAULT_METHODS = ('lomb', 'ejtk', 'fft_nlls', 'mesa')

DETREND_LABELS = {'none': 'No detrending', 'linear': 'Linear', 'robust_linear': 'Robust linear', 'first_difference': 'First difference', 'poly3': 'Third-degree polynomial', 'cubic': 'Third-degree polynomial', 'bicubic': 'Third-degree polynomial', 'poly6': 'Sixth-degree polynomial', 'degree6': 'Sixth-degree polynomial', 'baseline': 'Kernel baseline', 'kernel': 'Kernel baseline', 'amp_baseline': 'Amplitude and baseline', 'amp&baseline': 'Amplitude and baseline', 'running_mean': 'Running mean', 'polynomial': 'Polynomial', 'frequency': 'Frequency filter'}

CLAIM = 'For each selected cell trace, the main period estimate and any candidate secondary components can be judged against input filtering, detrending choice, significance, and recording-length sufficiency.'

def _safe_estimate(hours: np.ndarray, values: np.ndarray, params: dict[str, Any], method: str, detrend: str) -> dict[str, Any]:
    return safe_estimate(hours, values, params, method, detrend)

def _finite(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return np.nan
    return numeric if np.isfinite(numeric) else np.nan

def _array(value: Any, length: int) -> np.ndarray:
    if value is None:
        return np.full(length, np.nan, dtype=float)
    result = np.asarray(value, dtype=float)
    return result if result.shape == (length,) else np.full(length, np.nan, dtype=float)

def analyse_traces(cell_frame: pd.DataFrame, *, identities: list[int], metrics: list[str], detrends: list[str], estimators: list[str], median_window_points: int, params: dict[str, Any], fallback_significance_method: str, correction: str, alpha: float, min_observations: int, min_cycles: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the complete requested grid and return every auditable table."""
    trace_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    evidence_cache: dict[str, dict[str, Any]] = {}
    for identity in identities:
        selected = cell_frame.loc[cell_frame['identity'].eq(identity)].sort_values('hours', kind='mergesort')
        for metric in metrics:
            trace = selected[['hours', metric]]
            hours = trace['hours'].to_numpy(float)
            raw = trace[metric].to_numpy(float)
            if not np.isfinite(hours).all() or np.any(np.diff(hours) <= 0):
                raise ValueError('Audit traces require finite, unique, increasing observation times')
            trace_id = f'{identity}|{metric}'
            if not len(hours):
                continue
            for hour, value in zip(hours, raw):
                trace_rows.append({'trace_id': trace_id, 'identity': identity, 'metric': metric, 'display_panel': 'raw_reference', 'hours': hour, 'raw_value': value, 'analysis_input_value': value, 'baseline_value': np.nan, 'plotted_value': value, 'fitted_value': np.nan, 'preprocessor': 'raw_reference', 'median_window_points': median_window_points, 'detrend': 'raw_reference', 'estimator': 'raw_reference', 'fit_kind': 'none'})
            interval = workbench.sample_interval_minutes(hours) / 60.0
            filter_result = workbench.filter_rhythm_trace(hours, raw, {'method': 'median', 'window_hours': median_window_points * interval, 'max_gap_hours': 1.5 * interval, 'min_observations': 1})
            filtered_inputs = {'raw': raw.copy(), 'median': np.asarray(filter_result['values'], float)}
            span = float(hours.max() - hours.min()) if len(hours) > 1 else 0.0
            for detrend in detrends:
                for preprocessor, analysis_input in filtered_inputs.items():
                    observation_count = int(np.isfinite(analysis_input).sum())
                    observed_hours = hours[np.isfinite(analysis_input)]
                    span = float(observed_hours[-1] - observed_hours[0]) if len(observed_hours) else 0.0
                    try:
                        detrended = workbench.detrend_trace(hours, analysis_input, params, method=detrend) if len(hours) >= 2 else {}
                        residual = _array(detrended.get('values'), len(hours))
                        baseline = _array(detrended.get('baseline'), len(hours))
                        detrend_message = ''
                    except (ValueError, RuntimeError) as error:
                        residual = np.full(len(hours), np.nan)
                        baseline = np.full(len(hours), np.nan)
                        detrend_message = str(error)
                    for hour, raw_value, input_value, base, value in zip(hours, raw, analysis_input, baseline, residual):
                        trace_rows.append({'trace_id': trace_id, 'identity': identity, 'metric': metric, 'display_panel': 'detrended', 'hours': hour, 'raw_value': raw_value, 'analysis_input_value': input_value, 'baseline_value': base, 'plotted_value': value, 'fitted_value': np.nan, 'preprocessor': preprocessor, 'median_window_points': median_window_points, 'filter_window_hours': filter_result['settings']['window_hours'] if preprocessor == 'median' else None, 'filter_max_gap_hours': filter_result['settings']['max_gap_hours'] if preprocessor == 'median' else None, 'filter_run_record_json': json.dumps(filter_result['workbench_run_record'], allow_nan=False) if preprocessor == 'median' else '', 'detrend': detrend, 'estimator': 'none', 'fit_kind': 'none'})
                    estimate_cache: dict[str, dict[str, Any]] = {}
                    comparison_definition = ''
                    comparison_record = ''
                    can_fit = observation_count >= min_observations and (not detrend_message)
                    if can_fit:
                        try:
                            compared = workbench.estimate_trace(hours, analysis_input, params, methods=estimators, detrend=detrend)
                            details = {entry['method']: entry for entry in compared['comparison']['estimates']}
                            comparison_record = json.dumps(compared['run_record'], allow_nan=False)
                            estimate_cache = {row['method']: {**row, 'diagnostics': details[row['method']].get('diagnostics', {}), 'components': details[row['method']].get('components', []), 'workbench_run_record_json': comparison_record} for row in compared['rows']}
                            comparison_definition = json.dumps(compared['result'].plot().definition.as_dict(), allow_nan=False)
                        except (ValueError, RuntimeError):
                            estimate_cache = {}
                    for estimator in estimators:
                        estimate = estimate_cache.get(estimator) or (_safe_estimate(hours, analysis_input, params, estimator, detrend) if can_fit else {'method': estimator, 'method_label': workbench.PERIOD_METHODS[estimator]['label'], 'status': 'not_tested' if observation_count < min_observations else 'failed', 'message': 'too_few_observations' if observation_count < min_observations else detrend_message, 'diagnostics': {}, 'components': [], 'period_hours': np.nan, 'p_value': np.nan, 'workbench_version': workbench.WORKBENCH_VERSION})
                        estimate_cache[estimator] = estimate
                        significance_method = fallback_significance_method
                        evidence_id = json.dumps([identity, metric, preprocessor, detrend, significance_method], separators=(',', ':'))
                        if evidence_id not in evidence_cache:
                            evidence = estimate if significance_method == estimator or not can_fit else estimate_cache.get(significance_method) or _safe_estimate(hours, analysis_input, params, significance_method, detrend)
                            evidence_cache[evidence_id] = {'evidence_id': evidence_id, 'trace_id': trace_id, 'identity': identity, 'metric': metric, 'preprocessor': preprocessor, 'detrend': detrend, 'significance_method': significance_method, 'significance_method_label': workbench.PERIOD_METHODS[significance_method]['label'], 'status': str(evidence.get('status') or 'failed'), 'period_hours': _finite(evidence.get('period_hours')), 'p_value': _finite(evidence.get('p_value')), 'message': str(evidence.get('message') or ''), 'workbench_run_record_json': evidence.get('workbench_run_record_json', '')}
                        period = _finite(estimate.get('period_hours'))
                        if params.get('descriptive_cosinor', False):
                            fitted = workbench.descriptive_cosinor_fitted_values(hours, residual, period)
                            fit_kind = 'descriptive Workbench cosine at estimated period'
                        else:
                            fitted = np.full(len(hours), np.nan)
                            fit_kind = 'native fitted series unavailable; components retained'
                        for hour, raw_value, input_value, base, value, fit in zip(hours, raw, analysis_input, baseline, residual, fitted):
                            trace_rows.append({'trace_id': trace_id, 'identity': identity, 'metric': metric, 'display_panel': 'fit', 'hours': hour, 'raw_value': raw_value, 'analysis_input_value': input_value, 'baseline_value': base, 'plotted_value': value, 'fitted_value': fit, 'preprocessor': preprocessor, 'median_window_points': median_window_points, 'detrend': detrend, 'estimator': estimator, 'fit_kind': fit_kind})
                        diagnostics = dict(estimate.get('diagnostics') or {})
                        returned_components = list(estimate.get('components') or [])
                        for component_index, component in enumerate(returned_components, start=1):
                            component_period = _finite(component.get('period_hours'))
                            selected_component = bool(component.get('selected', False))
                            if not selected_component and np.isfinite(period) and np.isfinite(component_period):
                                selected_component = bool(np.isclose(component_period, period, rtol=1e-06, atol=1e-06))
                            component_rows.append({'trace_id': trace_id, 'identity': identity, 'metric': metric, 'preprocessor': preprocessor, 'detrend': detrend, 'estimator': estimator, 'component_index': component_index, 'period_hours': component_period, 'period_error_hours': _finite(component.get('period_error_hours')), 'amplitude': _finite(component.get('amplitude')), 'amplitude_error': _finite(component.get('amplitude_error')), 'phase_hours': _finite(component.get('phase_hours')), 'phase_error_hours': _finite(component.get('phase_error_hours')), 'relative_amplitude_error': _finite(component.get('rae')), 'selected': selected_component, 'component_significance': 'not tested'})
                        diagnostic_periods = np.asarray(diagnostics.get('periods_hours') if diagnostics.get('periods_hours') is not None else [], dtype=float)
                        diagnostic_power = np.asarray(diagnostics.get('power') if diagnostics.get('power') is not None else [], dtype=float)
                        if diagnostic_periods.size == diagnostic_power.size and diagnostic_periods.size:
                            maximum = float(np.nanmax(diagnostic_power))
                            normalised = diagnostic_power / maximum if np.isfinite(maximum) and maximum > 0 else np.zeros_like(diagnostic_power)
                            for spectrum_period, power, normalised_power in zip(diagnostic_periods, diagnostic_power, normalised):
                                spectrum_rows.append({'trace_id': trace_id, 'identity': identity, 'metric': metric, 'preprocessor': preprocessor, 'detrend': detrend, 'estimator': estimator, 'period_hours': spectrum_period, 'power': power, 'normalised_power': normalised_power})
                        cycles = span / period if np.isfinite(period) and period > 0 else np.nan
                        low, high = map(float, params['period_search_hours'])
                        edge_tolerance = max(0.1, 0.005 * (high - low))
                        at_edge = bool(np.isfinite(period) and (period <= low + edge_tolerance or period >= high - edge_tolerance))
                        filter_delta = analysis_input - raw
                        result_rows.append({'evidence_id': evidence_id, 'trace_id': trace_id, 'identity': identity, 'metric': metric, 'observations': observation_count, 'span_hours': span, 'preprocessor': preprocessor, 'median_window_points': median_window_points, 'filter_changed_points': int(np.count_nonzero(~np.isclose(analysis_input, raw, equal_nan=True))), 'filter_max_abs_change': float(np.nanmax(np.abs(filter_delta))), 'detrend': detrend, 'applied_detrend': str(estimate.get('detrend') or detrend), 'detrend_window_hours': float(params['detrend_window_hours']), 'estimator': estimator, 'estimator_label': workbench.PERIOD_METHODS[estimator]['label'], 'estimate_status': str(estimate.get('status') or 'failed'), 'estimated_period_hours': period, 'period_error_hours': _finite(estimate.get('period_error_hours')), 'estimator_p_value': _finite(estimate.get('p_value')), 'estimator_goodness_of_fit': _finite(estimate.get('goodness_of_fit')), 'significance_method': significance_method, 'significance_method_label': workbench.PERIOD_METHODS[significance_method]['label'], 'cycles_observed': cycles, 'min_cycles': min_cycles, 'period_underdetermined': bool(not np.isfinite(cycles) or cycles < min_cycles), 'period_at_search_edge': at_edge, 'period_search_min_hours': low, 'period_search_max_hours': high, 'fit_kind': fit_kind, 'component_count': len(returned_components), 'component_significance': 'not tested', 'mesa_peaks_in_search_band': diagnostics.get('peaks_in_search_band'), 'diagnostic_warnings_json': json.dumps(list(diagnostics.get('warnings') or [])), 'diagnostic_stop_reason': str(diagnostics.get('stopped_because') or ''), 'workbench_version': workbench.WORKBENCH_VERSION, 'workbench_run_record_json': estimate.get('workbench_run_record_json', ''), 'workbench_comparison_json': comparison_definition, 'message': str(estimate.get('message') or '')})
    evidence = pd.DataFrame(evidence_cache.values())
    if not evidence.empty:
        evidence['q_value'] = workbench.adjust_pvalues(evidence['p_value'].to_numpy(float), correction)
        tested = evidence['status'].eq('ok') & evidence['q_value'].notna()
        evidence['significant'] = tested & evidence['q_value'].lt(alpha)
        evidence['rhythm_status'] = np.where(tested, np.where(evidence['significant'], 'significant', 'not significant'), 'not tested')
        evidence['correction_method'] = correction
        evidence['alpha'] = alpha
        evidence_by_id = evidence.set_index('evidence_id')
        for row in result_rows:
            test = evidence_by_id.loc[row['evidence_id']]
            row.update({'significance_status': test['status'], 'significance_period_hours': test['period_hours'], 'p_value': test['p_value'], 'q_value': test['q_value'], 'correction_method': correction, 'alpha': alpha, 'significant': bool(test['significant']), 'rhythm_status': test['rhythm_status']})
    trace_columns = ['trace_id', 'identity', 'metric', 'display_panel', 'hours', 'raw_value', 'analysis_input_value', 'baseline_value', 'plotted_value', 'fitted_value', 'preprocessor', 'median_window_points', 'detrend', 'estimator', 'fit_kind', 'filter_window_hours', 'filter_max_gap_hours', 'filter_run_record_json']
    component_columns = ['trace_id', 'identity', 'metric', 'preprocessor', 'detrend', 'estimator', 'component_index', 'period_hours', 'period_error_hours', 'amplitude', 'amplitude_error', 'phase_hours', 'phase_error_hours', 'relative_amplitude_error', 'selected', 'component_significance']
    spectrum_columns = ['trace_id', 'identity', 'metric', 'preprocessor', 'detrend', 'estimator', 'period_hours', 'power', 'normalised_power']
    return (pd.DataFrame(trace_rows, columns=trace_columns), pd.DataFrame(result_rows), pd.DataFrame(component_rows, columns=component_columns), pd.DataFrame(spectrum_rows, columns=spectrum_columns), evidence)

def statistics_table(results: pd.DataFrame) -> pd.DataFrame:
    """Normalize every displayed trace-level test for the figure bundle."""
    statistics = results.copy()
    statistics['test_name'] = statistics['significance_method_label']
    statistics['estimate_name'] = 'estimated period'
    statistics['estimate'] = statistics['estimated_period_hours']
    statistics['estimate_units'] = 'h'
    statistics['effect_size_name'] = 'estimator goodness of fit'
    statistics['effect_size'] = statistics['estimator_goodness_of_fit']
    statistics['algorithm_id'] = 'circadian-workbench:' + statistics['significance_method'].astype(str)
    statistics['inputs_json'] = statistics.apply(lambda row: json.dumps({'trace_id': row['trace_id'], 'identity': int(row['identity']), 'metric': row['metric'], 'observations': int(row['observations']), 'span_hours': float(row['span_hours']), 'preprocessor': row['preprocessor']}), axis=1)
    statistics['parameters_json'] = statistics.apply(lambda row: json.dumps({'detrend': row['detrend'], 'detrend_window_hours': float(row['detrend_window_hours']), 'estimator': row['estimator'], 'significance_method': row['significance_method'], 'period_search_hours': [float(row['period_search_min_hours']), float(row['period_search_max_hours'])], 'alpha': float(row['alpha']), 'correction': row['correction_method'], 'minimum_cycles': float(row['min_cycles']), 'median_window_points': int(row['median_window_points'])}), axis=1)
    statistics['expected_json'] = statistics.apply(lambda row: json.dumps({'p_value': _nullable(row['p_value']), 'q_value': _nullable(row['q_value']), 'significant': bool(row['significant']), 'period_hours': _nullable(row['estimate'])}), axis=1)
    statistics['display_json'] = statistics.apply(lambda row: json.dumps({'trace_id': row['trace_id'], 'preprocessor': row['preprocessor'], 'detrend': row['detrend'], 'estimator': row['estimator']}), axis=1)
    statistics['tolerances_json'] = json.dumps({'rtol': 1e-09, 'atol': 1e-12})
    return statistics

def _nullable(value: Any) -> float | None:
    numeric = _finite(value)
    return float(numeric) if np.isfinite(numeric) else None

def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value

def prepare(source,options):
    cell_frame = source.table('cell_frame.csv')
    metrics = [str(value) for value in options.get('metrics')]
    if not metrics:
        raise ValueError('--metrics needs at least one measured column')
    missing_metrics = [name for name in metrics if name not in cell_frame]
    if missing_metrics:
        raise ValueError('--metrics names columns not in cell_frame.csv: ' + ', '.join(missing_metrics))
    nonnumeric = [name for name in metrics if cell_frame[name].dtype.kind not in 'fi']
    if nonnumeric:
        raise ValueError('--metrics must be numeric: ' + ', '.join(nonnumeric))
    identity = options.get('identity')
    if identity is None:
        identities = _selected_identities(cell_frame, options.get('cells'))
    else:
        identities = [int(identity)]
        available = set((int(value) for value in cell_frame['identity'].unique()))
        if identities[0] not in available:
            raise ValueError(f'identity {identities[0]} is not in cell_frame.csv')
    empty_traces = [f'cell {identity}, {metric}' for identity in identities for metric in metrics if cell_frame.loc[cell_frame['identity'].eq(identity), ['hours', metric]].dropna().empty]
    if empty_traces:
        raise ValueError('selected cell/measurement traces contain no finite observations: ' + '; '.join(empty_traces))
    median_points = int(options.get('median_window_points'))
    if median_points < 3 or median_points % 2 == 0:
        raise ValueError('--median-window-points must be an odd integer of at least 3')
    inherited = {**RHYTHM_DEFAULTS, **source.module_params('rhythms')}
    try:
        resolved = workbench.resolve_analysis_options(inherited, options.get)
    except ValueError as error:
        raise ValueError(str(error)) from None
    params = dict(resolved['params'])
    detrends = [str(value) for value in options.get('detrend_methods')]
    primary_detrend = str(resolved['detrend'])
    detrends = list(dict.fromkeys([primary_detrend, *detrends]))
    unknown_detrends = [name for name in detrends if name not in workbench.DETREND_METHODS]
    if unknown_detrends:
        raise ValueError('--detrend-methods includes unknown choices: ' + ', '.join(unknown_detrends) + '. Available: ' + ', '.join(workbench.DETREND_METHODS))
    estimators = [str(value) for value in options.get('period_methods')]
    primary_estimator = str(resolved['method'])
    estimators = list(dict.fromkeys([primary_estimator, *estimators]))
    unknown_estimators = [name for name in estimators if name not in workbench.PERIOD_METHODS]
    if unknown_estimators:
        raise ValueError('--period-methods includes unknown choices: ' + ', '.join(unknown_estimators) + '. Available: ' + ', '.join(workbench.PERIOD_METHODS))
    trace_data, results, components, spectra, evidence = analyse_traces(cell_frame, identities=identities, metrics=metrics, detrends=detrends, estimators=estimators, median_window_points=median_points, params=params, fallback_significance_method=str(resolved['significance_method']), correction=str(resolved['multiple_testing']), alpha=float(resolved['rhythmic_alpha']), min_observations=int(resolved['min_observations']), min_cycles=float(resolved['min_cycles']))
    statistics = statistics_table(results)
    trace_keys = [{'trace_id': f'{identity}|{metric}', 'identity': int(identity), 'metric': metric} for identity in identities for metric in metrics]
    n_columns = 1 + len(estimators) + int('fft_nlls' in estimators) + int('mesa' in estimators)
    nominal_width = max(13.8, 2.0 + 3.75 * n_columns)
    comparison_inches = max(2.0, 0.4 * (len(estimators) + 2))
    nominal_height = 4.2 + len(trace_keys) * (2.0 + (2.45 + comparison_inches) * len(detrends))
    config=dict(trace_keys=trace_keys,detrends=detrends,estimators=estimators,resolved=resolved,median_points=median_points,
        detrend_labels={name:DETREND_LABELS.get(name,name.replace('_',' ')) for name in detrends},
        estimator_labels={name:workbench.PERIOD_METHODS[name]['label'] for name in estimators},
        nominal_width=nominal_width,nominal_height=nominal_height,hour_ticks=options.get('hour_ticks'))
    return PreparedViews({'method_grid':dict(table=trace_data,results=results,components=components,spectra=spectra,config=config)},
        auxiliary={'results.csv':results,'components.csv':components,'spectra.csv':spectra,'evidence_tests.csv':evidence,'statistics.csv':statistics},
        wording=dict(title=f'Period-method audit: {len(identities)} cells and {len(metrics)} measurements',
        subtitle=f'{len(detrends)} detrending settings; {len(estimators)} estimators; raw and {median_points}-point median-filtered input.',
        footnote=f"Circadian Workbench {workbench.WORKBENCH_VERSION}; significance test {resolved['significance_method']}; correction {resolved['multiple_testing']}; alpha {resolved['rhythmic_alpha']}. Period sufficiency requires {resolved['min_cycles']} observed cycles. Returned components and spectral peaks are not individually significance-tested.")),statistics
