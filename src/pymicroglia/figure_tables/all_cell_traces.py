"""Prepare the full measured cell inventory and separate display and inference."""
from __future__ import annotations
import inspect,json,math
import numpy as np
import pandas as pd
from .. import workbench
from ..measure.modules.rhythms import DEFAULTS as RHYTHM_DEFAULTS
from ..visualisation.labels import describe
from .radial import _selected_identities
from .prepared import PreparedViews
from .fft_component_grid import median_then_mean, test_cell_components


def parse_hours_per_tick(value):
    step=float(value)
    if not math.isfinite(step) or step<=0:
        raise ValueError('hour_ticks must be finite and positive')
    return step

def grid_shape(count: int, rows: int | None, columns: int | None) -> tuple[int, int]:
    """Square by default; one fixed dimension computes the other."""
    for name, value in (('grid_rows', rows), ('grid_columns', columns)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(name + ' must be a positive integer')
    if columns is None and rows is None:
        columns = math.ceil(math.sqrt(count))
    if columns is None:
        columns = math.ceil(count / rows)
    if rows is None:
        rows = math.ceil(count / columns)
    if rows * columns < count:
        raise ValueError(f'grid_rows * grid_columns must hold all {count} selected cells')
    return (rows, columns)

def trace_data(frame: pd.DataFrame, metrics: list[str], identities: list[int], resolved: dict, view: str, normalization: str, normalization_config: dict, period_testing: bool=True, *, fft_component_test: bool=False, recording: str | None=None, component_surrogates: int=199, component_block_hours: float=4.0, component_seed: int=20260923) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit raw values; normalize independently for drawing, preserving gaps."""
    if view not in {'raw', 'detrended'}:
        raise ValueError('trace_view must be raw or detrended')
    if normalization not in workbench.NORMALIZATION_METHODS:
        raise ValueError('trace_normalization must be one of ' + ', '.join(workbench.NORMALIZATION_METHODS))
    if not isinstance(period_testing, bool):
        raise ValueError('period_testing must be true or false')
    allowed = set(inspect.signature(workbench.normalize_trace).parameters) - {'hours', 'values', 'method', 'detrended'}
    if not isinstance(normalization_config, dict) or set(normalization_config) - allowed:
        raise ValueError('trace_normalization_config contains unknown Workbench normalization arguments')
    if resolved['method'] not in workbench.PERIOD_METHODS:
        raise ValueError('unknown fit_method: ' + resolved['method'])
    if not fft_component_test and resolved['significance_method'] not in workbench.SIGNIFICANCE_METHODS:
        raise ValueError('significance_method must provide significance')
    if fft_component_test and (resolved['method'] != 'fft_nlls' or resolved['multiple_testing'] != 'none' or resolved['detrend'] != 'robust_linear'):
        raise ValueError('FFT component testing requires fft_nlls, robust_linear and no multiple-testing correction')
    if fft_component_test and (component_surrogates < 1 or component_block_hours <= 0):
        raise ValueError('component_surrogates and component_block_hours must be positive')
    params = resolved['params']
    points, tests = ([], [])
    for identity in identities:
        cell = frame.loc[frame.identity.eq(identity)].sort_values('frame_index', kind='stable')
        for metric in metrics:
            hours = cell.hours.to_numpy(float)
            raw = pd.to_numeric(cell[metric], errors='coerce').to_numpy(float)
            filtered = median_then_mean(raw) if fft_component_test else raw.copy()
            if not np.array_equal(np.isfinite(raw), np.isfinite(filtered)):
                raise ValueError('Three-frame filters changed measured frame availability')
            usable = np.isfinite(hours) & np.isfinite(filtered)
            processed = np.full(len(raw), np.nan)
            display = np.full(len(raw), np.nan)
            reason = ''
            display_note = ''
            if usable.sum() >= 2:
                if view == 'detrended':
                    prepared = workbench.detrend_trace(hours[usable], filtered[usable], params)
                    processed[usable] = np.asarray(prepared['values'], float)
                else:
                    processed[usable] = filtered[usable]
                if normalization == 'none':
                    display[usable] = processed[usable]
                elif normalization == 'minmax' and np.ptp(processed[usable]) == 0:
                    display[usable] = 0.0
                    display_note = 'flat trace: zero display fallback, not a Workbench normalization'
                else:
                    scaled = workbench.normalize_trace(hours[usable], processed[usable], normalization, detrended=view == 'detrended', **normalization_config)
                    display[usable] = np.asarray(scaled['values'], float)
            else:
                reason = 'fewer than two observations'
                if usable.sum() == 1:
                    display[usable] = raw[usable] if normalization == 'none' else 0.0
                    display_note = 'single raw observation shown as a dot; detrending and normalization unavailable'
            for position, (hour, original, prepared, value) in enumerate(zip(hours, raw, processed, display)):
                points.append({'identity': identity, 'metric': metric, 'frame_index': cell.frame_index.iloc[position], 'hours': hour, 'raw_value': original, 'filtered_value': filtered[position], 'processed_value': prepared, 'value': value})
            entry = {'identity': identity, 'metric': metric, 'observations': int(usable.sum()), 'estimation_method': resolved['method'], 'significance_method': 'fft_component' if fft_component_test else resolved['significance_method'], 'detrend': params['detrend'], 'trace_view': view, 'normalization': normalization, 'workbench_version': workbench.WORKBENCH_VERSION, 'applied_settings_json': json.dumps(params, default=str), 'estimate_status': 'not_tested', 'significance_status': 'not_tested', 'period_hours': np.nan, 'p_value': np.nan, 'q_value': np.nan, 'correction': resolved['multiple_testing'], 'alpha': resolved['rhythmic_alpha'], 'rhythm_status': 'unknown', 'reason': reason, 'display_note': display_note, 'period_testing': period_testing, 'cycles_observed': np.nan, 'supported_period': False, 'estimate_components_json': '[]', 'estimate_diagnostics_json': '{}', 'component_results_json': '[]', 'fft_selected_period_hours': np.nan, 'period_source_component': np.nan, 'alternate_fft_component': False, 'component_test_surrogates': component_surrogates if fft_component_test else np.nan, 'component_test_block_hours': component_block_hours if fft_component_test else np.nan}
            if not period_testing:
                entry['reason'] = 'period testing disabled'
            elif usable.sum() >= resolved['min_observations']:
                try:
                    estimate = workbench.estimate_one(hours[usable], filtered[usable], params, resolved['method'])
                    if fft_component_test:
                        components = estimate.get('components') or []
                        if view == 'detrended':
                            detrended = processed[usable]
                        else:
                            detrended = np.asarray(workbench.detrend_trace(hours[usable], filtered[usable], params)['values'], float)
                        results, selected = test_cell_components(
                            hours[usable], detrended, cell.frame_index.to_numpy(int)[usable],
                            components, recording=recording, identity=int(identity),
                            seed=component_seed, surrogates=component_surrogates,
                            block_hours=component_block_hours, alpha=resolved['rhythmic_alpha'],
                            period_min_hours=resolved['period_min_hours'],
                            period_max_hours=resolved['period_max_hours'],
                        )
                        tested_components = [row for row in results if row['status'] == 'ok'
                                             and row['empirical_p_uncorrected'] is not None
                                             and resolved['period_min_hours'] <= row['period_hours'] <= resolved['period_max_hours']]
                        best_test = min(tested_components, key=lambda row: row['empirical_p_uncorrected']) if tested_components else None
                        source = selected or best_test
                        entry.update(
                            period_hours=source['period_hours'] if source else np.nan,
                            fft_selected_period_hours=estimate.get('period_hours'),
                            estimate_status=estimate.get('status', 'failed'),
                            significance_status='ok' if tested_components else 'untestable',
                            p_value=source['empirical_p_uncorrected'] if source else np.nan,
                            rhythm_status='rhythmic' if selected else 'not rhythmic' if tested_components else 'unknown',
                            period_source_component=source['component'] if source else np.nan,
                            alternate_fft_component=bool(selected and not selected['selected_fft_component']),
                            component_results_json=json.dumps(results, default=str),
                            workbench_run_record_json=estimate.get('workbench_run_record_json', ''),
                            estimate_components_json=json.dumps(components, default=str),
                            estimate_diagnostics_json=json.dumps(estimate.get('diagnostics') or {}, default=str),
                        )
                    else:
                        significance = estimate if resolved['method'] == resolved['significance_method'] else workbench.estimate_one(hours[usable], filtered[usable], params, resolved['significance_method'])
                        entry.update(period_hours=estimate.get('period_hours'), estimate_status=estimate.get('status', 'failed'), significance_status=significance.get('status', 'failed'), p_value=significance.get('p_value'), workbench_run_record_json=estimate.get('workbench_run_record_json', ''), significance_run_record_json=significance.get('workbench_run_record_json', ''), estimate_components_json=json.dumps(estimate.get('components') or [], default=str), estimate_diagnostics_json=json.dumps(estimate.get('diagnostics') or {}, default=str))
                    period = entry['period_hours']
                    span = float(hours[usable].max() - hours[usable].min())
                    entry['cycles_observed'] = span / float(period) if period is not None and np.isfinite(period) and (period > 0) else np.nan
                    entry['supported_period'] = entry['estimate_status'] == 'ok' and np.isfinite(entry['cycles_observed']) and (entry['cycles_observed'] >= resolved['min_cycles'])
                except ValueError as error:
                    entry['reason'] = str(error)
            elif not reason:
                entry['reason'] = 'below min_observations'
            tests.append(entry)
    evidence = pd.DataFrame(tests)
    if not fft_component_test:
        for metric, indices in evidence.groupby('metric', sort=False).groups.items():
            evidence.loc[indices, 'q_value'] = workbench.adjust_pvalues(evidence.loc[indices, 'p_value'].to_numpy(float), resolved['multiple_testing'])
        tested = evidence.significance_status.eq('ok') & np.isfinite(evidence.q_value)
        evidence.loc[tested, 'rhythm_status'] = np.where(evidence.loc[tested, 'q_value'] < resolved['rhythmic_alpha'], 'rhythmic', 'not rhythmic')
    return (pd.DataFrame(points), evidence)

def _fft_nlls_display_fit(segment: pd.DataFrame, result: pd.Series) -> np.ndarray:
    """Put the native multi-component fit on the existing affine display scale."""
    empty = np.full(len(segment), np.nan)
    if result.get('trace_view') != 'detrended':
        return empty
    try:
        estimate = {'method': 'fft_nlls', 'diagnostics': json.loads(result.get('estimate_diagnostics_json') or '{}'), 'components': json.loads(result.get('estimate_components_json') or '[]')}
    except (TypeError, ValueError, json.JSONDecodeError):
        return empty
    native = workbench.fft_nlls_fitted_values(segment.hours.to_numpy(float), estimate)
    processed = segment.processed_value.to_numpy(float)
    displayed = segment.value.to_numpy(float)
    usable = np.isfinite(processed) & np.isfinite(displayed)
    if usable.sum() < 2 or np.ptp(processed[usable]) <= 0:
        return empty
    design = np.column_stack([processed[usable], np.ones(usable.sum())])
    slope, intercept = np.linalg.lstsq(design, displayed[usable], rcond=None)[0]
    reproduced = slope * processed[usable] + intercept
    if not np.allclose(reproduced, displayed[usable], rtol=1e-8, atol=1e-10):
        return empty
    time = segment.hours.to_numpy(float)
    native[(time < np.nanmin(time[usable])) | (time > np.nanmax(time[usable]))] = np.nan
    native[~usable] = np.nan
    return slope * native + intercept


def add_period_display(points: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    """Style the display from saved period evidence without changing the test."""
    styled = points.copy()
    styled['descriptive_fit'] = np.nan
    styled['fit_kind'] = 'none'
    styled['period_test_significant'] = False
    for (identity, metric), indices in styled.groupby(['identity', 'metric'], sort=False).groups.items():
        matched = evidence.loc[evidence.identity.eq(identity) & evidence.metric.eq(metric)]
        if len(matched) != 1:
            continue
        result = matched.iloc[0]
        period = pd.to_numeric(result.get('period_hours'), errors='coerce')
        p_value = pd.to_numeric(result.get('p_value'), errors='coerce')
        alpha = pd.to_numeric(result.get('alpha'), errors='coerce')
        styled.loc[indices, 'period_test_significant'] = bool(result.get('rhythm_status') == 'rhythmic' and result.get('significance_status') == 'ok' and np.isfinite(p_value) and np.isfinite(alpha) and (p_value < alpha))
        if not np.isfinite(period) or period <= 0:
            continue
        segment = styled.loc[indices]
        if result.get('estimation_method') == 'fft_nlls':
            fitted = _fft_nlls_display_fit(segment, result)
            styled.loc[indices, 'descriptive_fit'] = fitted
            if np.isfinite(fitted).any():
                styled.loc[indices, 'fit_kind'] = 'actual summed FFT-NLLS fit'
        else:
            styled.loc[indices, 'descriptive_fit'] = workbench.descriptive_cosinor_fitted_values(segment.hours.to_numpy(float), segment.value.to_numpy(float), float(period))
            styled.loc[indices, 'fit_kind'] = 'descriptive single cosinor'
    return styled

def prepare(source, options):
    frame = source.table('cell_frame.csv')
    metrics = list(options.get('metrics'))
    if not metrics or len(metrics) != len(set(metrics)) or any((m not in frame for m in metrics)):
        raise ValueError('metrics must be distinct, nonempty columns of cell_frame.csv')
    if any((not pd.api.types.is_numeric_dtype(frame[m]) for m in metrics)):
        raise ValueError('metrics must be numeric columns')
    present = sorted((int(x) for x in frame.identity.dropna().unique()))
    requested = options.get('cells')
    identities = present if str(requested).lower() == 'all' else _selected_identities(frame, requested)
    if len(identities) != len(set(identities)) or not identities:
        raise ValueError('cells must name distinct measured cell numbers')
    if not identities:
        raise ValueError('cell_frame.csv contains no cells')
    rows, columns = grid_shape(len(identities), options.get('grid_rows'), options.get('grid_columns'))
    fft_component_test = options.get('fft_component_test')
    if fft_component_test is None:
        fft_component_test = False
    if not isinstance(fft_component_test, bool):
        raise ValueError('fft_component_test must be true or false')
    if fft_component_test and options.get('significance_method') is not None:
        raise ValueError('Set fft_component_test=false to select a periodogram significance method')
    resolved = workbench.resolve_analysis_options({**RHYTHM_DEFAULTS, **source.module_params('rhythms')}, options.get)
    if fft_component_test:
        resolved['significance_method'] = 'fft_component'
        resolved['params'] = {**resolved['params'], 'primary_rhythm_test': 'fft_component', 'period_methods': ['fft_nlls']}
    view, normal = (options.get('trace_view'), options.get('trace_normalization'))
    config = options.get('trace_normalization_config')
    period_testing = options.get('period_testing')
    if not isinstance(period_testing, bool):
        raise ValueError('period_testing must be true or false')
    hour_ticks = parse_hours_per_tick(options.get('hour_ticks'))
    points, tests = trace_data(
        frame, metrics, identities, resolved, view, normal, config,
        period_testing=period_testing, fft_component_test=fft_component_test,
        recording=getattr(source, 'stem', None),
        component_surrogates=options.get('component_surrogates') or 199,
        component_block_hours=options.get('component_block_hours') or 4.0,
        component_seed=options.get('component_seed') or 20260923,
    )
    reference_style = period_testing and len(metrics) == 1 and (resolved['multiple_testing'] == 'none')
    if reference_style:
        points = add_period_display(points, tests)
    finite = frame.hours.to_numpy(float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        raise ValueError('cell_frame.csv has no finite recording time')
    native_fit = resolved['method'] == 'fft_nlls'
    fit_label = 'Actual summed FFT-NLLS fit' if native_fit and view == 'detrended' else '' if native_fit else 'Descriptive cosinor fit'
    fit_claim = 'Dotted curves are the actual summed FFT-NLLS models.' if native_fit and view == 'detrended' else 'No model curve is substituted because the exact FFT-NLLS model is on the detrended scale.' if native_fit else 'Dotted fits describe the displayed trace at each tested period.'
    title = ('Native-frame cell traces | median3 → mean3 → robust linear → FFT-NLLS component test'
             if fft_component_test else
             f"Native-frame cell traces | {str(resolved['significance_method']).replace('_', ' ').title()} period test: uncorrected p"
             if resolved['multiple_testing'] == 'none' else 'All measured cell traces')
    claim = ('Each cell retains its original observed clock; a three-frame median and then mean are fitted by FFT-NLLS after robust-linear detrending. Red means an uncorrected conditional component test p < 0.05. Dotted curves are the actual summed FFT-NLLS models.'
             if fft_component_test else
             f'Each cell is measured at its original time points; display traces are normalized independently, and measured values before display scaling are period-tested with uncorrected p-values. {fit_claim}'
             if resolved['multiple_testing'] == 'none' and period_testing else
             'Every selected cell retains its original clock; each displayed measurement is normalized independently.')
    settings = {'identities': identities, 'rows': rows, 'columns': columns, 'metrics': metrics, 'labels': [describe(name).label for name in metrics], 'unit': 'Self-normalized' if normal != 'none' else 'Original units', 'xlim': [float(finite.min()), float(finite.max())], 'title': title, 'trace_view': view, 'trace_normalization': normal, 'trace_normalization_config': config, 'period_testing': period_testing, 'hour_ticks': hour_ticks, 'period_display_style': reference_style, 'multiple_testing': resolved['multiple_testing'], 'rhythmic_alpha': resolved['rhythmic_alpha'], 'fit_method': resolved['method'], 'fit_label': fit_label, 'significance_method': resolved['significance_method'], 'detrend': resolved['detrend'], 'fft_component_test': fft_component_test, 'component_surrogates': options.get('component_surrogates'), 'component_block_hours': options.get('component_block_hours'), 'component_seed': options.get('component_seed'), 'workbench_version': workbench.WORKBENCH_VERSION, 'claim': claim, 'output_name': 'all-cell-trace-grid'}
    annotations = tests.copy() if settings['period_display_style'] else tests
    if settings['period_display_style'] and resolved['multiple_testing'] == 'none':
        annotations['q_value'] = np.nan
    auxiliary = {'statistics.csv': tests, 'settings.csv': pd.DataFrame([{'settings_json':json.dumps(settings)}])}
    if fft_component_test:
        component_rows = []
        for result in tests.itertuples():
            for component in json.loads(result.component_results_json):
                component_rows.append({'identity': result.identity, 'metric': result.metric,
                                       'recording': getattr(source, 'stem', None), **component})
        auxiliary['components.csv'] = pd.DataFrame(component_rows)
    return PreparedViews({'traces': {'table': points, 'settings': settings, 'annotations': annotations}},
        auxiliary=auxiliary,
        wording={'title':settings['title'], 'claim':settings['claim']}), tests
