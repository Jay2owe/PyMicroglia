"""Input and result translations; science is called through workbench."""
from __future__ import annotations
from copy import deepcopy
import json
from typing import Any, Sequence
import numpy as np
import pandas as pd
from .. import workbench as wb

def estimate_trace(hours: Sequence[float], values: Sequence[float], params: dict, *, methods: Sequence[str] | None=None, detrend: str | None=None, detrend_window_hours: float | None=None) -> dict[str, Any]:
    """Run the requested Circadian Workbench period methods on one cell trace."""
    trace = wb.cw.trace(hours, values, name='cell trace')
    config = wb.workbench_config(params, hours, detrend=detrend, detrend_window_hours=detrend_window_hours, methods=methods)
    completed = trace.compare_periods(list(methods) if methods else None, settings=config)
    comparison = completed.data
    comparison = {key: value for key, value in comparison.items() if key not in {'recording', 'methods_paragraph'}}
    rows = []
    primary = str(params['primary_rhythm_test'])
    for returned in comparison['table']:
        row = dict(returned)
        method_name = str(row['method'])
        row.pop('label', None)
        tests_significance = bool(wb.PERIOD_METHODS[method_name]['gives_significance'])
        significant = row.get('significant') if tests_significance else None
        status = str(row.get('status') or 'failed')
        if status != 'ok' or significant is None:
            rhythm_status = 'unknown'
            rhythmic = None
        else:
            rhythmic = bool(significant)
            rhythm_status = 'rhythmic' if rhythmic else 'arrhythmic'
        row.update({'method_label': wb.PERIOD_METHODS[method_name]['label'], 'is_significance_test': tests_significance, 'alpha': (float(config['jtk_alpha']) if method_name in ('jtk', 'ejtk') else float(config['periodogram_alpha'])) if tests_significance else None, 'rhythmic': rhythmic, 'rhythm_status': rhythm_status, 'primary': method_name == primary, 'workbench_version': wb.WORKBENCH_VERSION})
        rows.append(row)
    return {'config': config, 'rows': rows, 'comparison': comparison, 'result': completed, 'run_record': completed.run_record}

def estimate_one(hours: Sequence[float], values: Sequence[float], params: dict, method: str, *, detrend: str | None=None, detrend_window_hours: float | None=None, capture_details: bool=False) -> dict[str, Any]:
    """One method through the same public workbench estimator used above."""
    trace = wb.cw.trace(hours, values, name='cell trace')
    config = wb.workbench_config(params, hours, detrend=detrend, detrend_window_hours=detrend_window_hours, methods=[method])
    completed = trace.period(method, settings=config)
    payload = completed.data
    returned = dict(payload['row'])
    returned.pop('label', None)
    tests_significance = bool(wb.PERIOD_METHODS[method]['gives_significance'])
    significant = returned.get('significant') if tests_significance else None
    status = str(returned.get('status') or 'failed')
    rhythmic = bool(significant) if status == 'ok' and significant is not None else None
    result = {**returned, 'components': payload['estimate'].get('components', []), 'diagnostics': payload['estimate'].get('diagnostics', {}), 'method_label': wb.PERIOD_METHODS[method]['label'], 'is_significance_test': tests_significance, 'alpha': (float(config['jtk_alpha']) if method in ('jtk', 'ejtk') else float(config['periodogram_alpha'])) if tests_significance else None, 'rhythmic': rhythmic, 'rhythm_status': 'unknown' if rhythmic is None else 'rhythmic' if rhythmic else 'arrhythmic', 'primary': method == str(params['primary_rhythm_test']), 'workbench_version': wb.WORKBENCH_VERSION, 'workbench_run_record_json': json.dumps(completed.run_record, allow_nan=False)}
    if capture_details:
        result['native_result'] = deepcopy(payload)
        result['native_series'] = {name: series.as_dict() for name, series in completed.series.items()}
        result['requested_settings'] = deepcopy(config)
        result['effective_settings'] = deepcopy(completed.run_record.get('configurations', []))
        result['phase_reference'] = returned.get('phase_reference')
        result['phase_units'] = returned.get('phase_units')
        result['display_processed_trace'] = None
        result['display_processing_reason'] = 'method handles its own detrending internally'
        if not wb.PERIOD_METHODS[method].get('forced_detrending'):
            try:
                processing = trace.detrend(method=config['period_detrend'], settings=config, **{name: config['period_detrend_' + name] for name in ('window_hours', 'polynomial_degree', 'min_valid_fraction', 'bandwidth_hours', 'low_cut_hours', 'high_cut_hours', 'filter_order', 'lowess_fraction', 'lowess_iterations', 'asls_smoothness', 'asls_asymmetry', 'asls_iterations')})
                result['display_processed_trace'] = wb._processed_trace_in_input_time(trace, processing)
                result['display_processing_run_record'] = deepcopy(processing.run_record)
                result['display_processing_reason'] = 'Separate Workbench detrending diagnostic with the applied configuration; not an exported internal estimator frame; public clock expressed in input recording hours'
            except (wb.cw.WorkbenchError, ValueError) as error:
                result['display_processing_reason'] = str(error)
    return result

def lomb_periodogram(hours: Sequence[float], values: Sequence[float], params: dict, *, detrend: str | None=None, detrend_window_hours: float | None=None) -> dict[str, Any]:
    """One Lomb-Scargle result through Workbench's public single-method route."""
    time = np.asarray(hours, dtype=float)
    observed = np.asarray(values, dtype=float)
    keep = np.isfinite(time) & np.isfinite(observed)
    time, observed = (time[keep], observed[keep])
    order = np.argsort(time, kind='mergesort')
    time, observed = (time[order], observed[order])
    if len(time) < 6 or np.allclose(observed, observed[0]):
        return {'method': 'lomb', 'status': 'failed', 'p_value': np.nan, 'period_hours': np.nan, 'significant': None, 'rhythm_status': 'unknown', 'workbench_version': wb.WORKBENCH_VERSION}
    estimate = wb.estimate_one(time, observed, params, 'lomb', detrend=detrend, detrend_window_hours=detrend_window_hours)
    return {**estimate, 'peak_power': estimate.get('goodness_of_fit', np.nan), 'threshold': np.nan}

def estimate_grouped_rhythms(frame: pd.DataFrame, *, group_columns: Sequence[str], value_column: str, params: dict, time_column: str='hours', method: str='lomb', significance_method: str | None=None, detrend: str | None=None, detrend_window_hours: float | None=None, min_observations: int=24, correction: str='bh', min_cycles: float=3.0, capture_details: bool=False) -> pd.DataFrame:
    """Broad-period rhythm results for any collection of grouped traces.

    This is deliberately independent of radial occupancy.  A caller supplies
    the grouping columns and measurement column, so the same tested matrix can
    represent cells by radial band, genes by condition, or any other family of
    traces.  Period selection and raw significance come from Circadian
    Workbench; its batch correction supplies the final per-family verdict.
    """
    missing = [name for name in [*group_columns, time_column, value_column] if name not in frame.columns]
    if missing:
        raise KeyError('grouped rhythm input is missing ' + ', '.join(missing))
    significance_method = wb.resolve_significance_method(method, significance_method)
    params = {**params, 'primary_rhythm_test': significance_method}
    if int(min_observations) < 6:
        raise ValueError('min_observations must be at least 6')
    if float(min_cycles) <= 0:
        raise ValueError('min_cycles must be positive')
    if correction not in {'bh', 'bonferroni', 'sidak', 'none'}:
        raise ValueError('multiple-testing correction must be bh, bonferroni, sidak or none')
    detrending = wb.detrend_settings(params, method=detrend, window_hours=detrend_window_hours)
    low, high = map(float, params['period_search_hours'])
    edge_tolerance = max(0.1, 0.005 * (high - low))
    rows: list[dict[str, Any]] = []
    grouping = list(group_columns)
    grouper: Any = grouping[0] if len(grouping) == 1 else grouping
    for key, group in frame.groupby(grouper, sort=True, dropna=False):
        keys = (key,) if len(grouping) == 1 else tuple(key)
        identity = dict(zip(grouping, keys))
        usable = group[[time_column, value_column]].dropna()
        usable = usable[np.isfinite(usable[time_column].to_numpy(float)) & np.isfinite(usable[value_column].to_numpy(float))]
        usable = usable.sort_values(time_column, kind='mergesort')
        time = usable[time_column].to_numpy(float)
        values = usable[value_column].to_numpy(float)
        span = float(time.max() - time.min()) if len(time) else 0.0
        base = {**identity, 'metric': value_column, 'observations': int(len(usable)), 'span_hours': span, 'input_mean': float(np.mean(values)) if len(values) else np.nan, 'input_sd': float(np.std(values)) if len(values) else np.nan, 'method': method, 'significance_method': significance_method, 'estimate_status': 'not_tested', 'estimate_reason': '', 'cycles_observed': np.nan, 'period_underdetermined': True, 'period_at_search_edge': False, 'estimator_p_value': np.nan, 'correction': correction, 'alpha': float(params['rhythmic_alpha']), 'period_search_min_hours': low, 'period_search_max_hours': high, **detrending, 'workbench_version': wb.WORKBENCH_VERSION}
        if capture_details:
            base.update(estimate_result={}, significance_result={})
        if len(usable) < int(min_observations):
            rows.append({**base, 'test_status': 'not_tested', 'reason': 'too_few_observations', 'period_hours': np.nan, 'p_value': np.nan})
            continue
        if np.allclose(values, values[0]):
            rows.append({**base, 'test_status': 'not_tested', 'reason': 'no_occupancy_variation', 'period_hours': np.nan, 'p_value': np.nan})
            continue
        try:
            estimate = wb.estimate_one(time, values, params, method, detrend=detrending['detrend'], detrend_window_hours=detrending['detrend_window_hours'], **{'capture_details': True} if capture_details else {})
        except (ValueError, RuntimeError, wb.cw.WorkbenchError) as error:
            estimate = {'status': 'failed', 'diagnostics': {'reason': str(error)}}
        period = estimate.get('period_hours')
        period = float(period) if period is not None and np.isfinite(period) else np.nan
        if method == significance_method:
            evidence = estimate
        else:
            try:
                evidence = wb.estimate_one(time, values, params, significance_method, detrend=detrending['detrend'], detrend_window_hours=detrending['detrend_window_hours'], **{'capture_details': True} if capture_details else {})
            except (ValueError, RuntimeError, wb.cw.WorkbenchError) as error:
                evidence = {'status': 'failed', 'diagnostics': {'reason': str(error)}}
        p_value = evidence.get('p_value')
        p_value = float(p_value) if p_value is not None and np.isfinite(p_value) else np.nan
        status = str(evidence.get('status') or 'failed')
        if status != 'ok' or not 0 <= p_value <= 1:
            p_value = np.nan
        evidence_reason = str(evidence.get('diagnostics', {}).get('reason') or 'workbench_failed')
        cycles = span / period if np.isfinite(period) and period > 0 else np.nan
        rows.append({**base, **({'estimate_result': deepcopy(estimate), 'significance_result': deepcopy(evidence)} if capture_details else {}), 'test_status': 'ok' if status == 'ok' and np.isfinite(p_value) else 'not_tested', 'reason': '' if status == 'ok' and np.isfinite(p_value) else evidence_reason, 'period_hours': period, 'estimate_status': str(estimate.get('status') or 'failed'), 'estimate_reason': estimate.get('diagnostics', {}).get('reason', ''), 'components': estimate.get('components', []), 'estimate_diagnostics': estimate.get('diagnostics', {}), 'workbench_run_record_json': estimate.get('workbench_run_record_json', ''), 'significance_run_record_json': evidence.get('workbench_run_record_json', ''), 'estimator_p_value': estimate.get('p_value'), 'significance_period_hours': evidence.get('period_hours'), 'period_difference_hours': period - float(evidence['period_hours']) if evidence.get('period_hours') is not None else np.nan, **{key: estimate.get(key) for key in ('period_error_hours', 'phase_hours', 'phase_error_hours', 'amplitude', 'amplitude_error', 'rae', 'goodness_of_fit')}, 'p_value': p_value, 'periodogram_significant': evidence.get('significant'), 'peak_power': evidence.get('peak_power', evidence.get('goodness_of_fit', np.nan)), 'threshold': evidence.get('threshold', np.nan), 'detrend': estimate.get('detrend', detrending['detrend']), 'cycles_observed': cycles, 'period_underdetermined': bool(not np.isfinite(cycles) or cycles < float(min_cycles)), 'period_at_search_edge': bool(np.isfinite(period) and (period <= low + edge_tolerance or period >= high - edge_tolerance))})
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    raw = pd.to_numeric(result['p_value'], errors='coerce').to_numpy(float)
    adjusted = wb.adjust_pvalues(raw, correction)
    result['q_value'] = adjusted
    tested = result['test_status'].eq('ok') & np.isfinite(adjusted)
    significant = tested & (adjusted < float(params['rhythmic_alpha']))
    result['significant'] = significant
    result['rhythm_status'] = np.select([~tested, significant], ['not tested', 'rhythmic'], default='not rhythmic')
    result['family_tests'] = int(tested.sum())
    result['period_available'] = result['estimate_status'].eq('ok') & np.isfinite(result['period_hours'])
    return result
