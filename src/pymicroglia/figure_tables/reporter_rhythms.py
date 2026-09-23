"""Prepare independent cell rhythms and the original continuum display order."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .. import workbench
from ..measure.modules.rhythms import DEFAULTS as RHYTHM_DEFAULTS
from ..visualisation.labels import describe
from . import ordering as matrix_ordering
from .rhythm_processing import detrended_traces
from .prepared import PreparedViews

ORDER_ALIASES = {'principal_component': 'pattern_rank', 'pattern': 'pattern_rank', 'spectral': 'spectral_rank', 'onset': 'displayed_onset_hours', 'status': 'rhythm_rank', 'period': 'period_hours', 'significance': 'q_value', 'identity': 'identity'}

ORDER_LABELS = {'principal_component': 'principal-component gradient', 'pattern': 'principal-component gradient', 'spectral': 'spectral continuum', 'onset': 'displayed onset', 'status': 'rhythm status', 'period': 'estimated period', 'significance': 'adjusted significance', 'identity': 'cell identity'}

def _resolved_order(tokens: list[str]) -> list[str]:
    return matrix_ordering.resolve_order(tokens, aliases=ORDER_ALIASES, append=('period_hours',))

def _order_label(tokens: list[str]) -> str:
    return matrix_ordering.order_label(tokens, ORDER_LABELS)

def _period_edges(search: list[float], bins: int, requested: list[float] | None) -> np.ndarray:
    low, high = map(float, search)
    if requested:
        edges = np.asarray(requested, dtype=float)
    else:
        if int(bins) < 2:
            raise ValueError('--bins must be at least 2')
        edges = np.linspace(low, high, int(bins) + 1)
    if edges.ndim != 1 or len(edges) < 3 or (not np.isfinite(edges).all()) or (not np.all(np.diff(edges) > 0)):
        raise ValueError('--period-bins needs at least three increasing finite boundaries')
    if edges[0] > low or edges[-1] < high:
        raise ValueError('--period-bins must cover the complete configured period search')
    return edges

def _fit_cells(frame: pd.DataFrame, metric: str, resolved: dict, secondary_method: str | None) -> pd.DataFrame:
    """Fit periods and one or two corrected rhythm-test families."""
    params = resolved['params']
    method = resolved['method']
    primary_method = resolved['significance_method']
    fit_args = dict(frame=frame[['identity', 'hours', metric]], group_columns=['identity'], value_column=metric, params=params, method=method, detrend=resolved['detrend'], detrend_window_hours=resolved['detrend_window_hours'], min_observations=resolved['min_observations'], correction=resolved['multiple_testing'], min_cycles=resolved['min_cycles'])
    fits = workbench.estimate_grouped_rhythms(**fit_args, significance_method=primary_method)
    if fits.empty:
        return fits
    fits['period_estimation_method'] = method
    fits['primary_significance_method'] = primary_method
    fits['primary_test_status'] = fits['test_status']
    fits['primary_p_value'] = fits['p_value']
    fits['primary_q_value'] = fits['q_value']
    fits['primary_significant'] = fits['significant']
    secondary = None if secondary_method in (None, '') else str(secondary_method)
    if secondary is None:
        fits['secondary_significance_method'] = None
        fits['verdict_rule'] = f'{primary_method} only'
    else:
        try:
            secondary = workbench.resolve_significance_method(method, secondary)
        except ValueError as error:
            raise ValueError(str(error)) from None
        if secondary == primary_method:
            raise ValueError('--secondary-significance-method must differ from --significance-method')
        second = workbench.estimate_grouped_rhythms(**fit_args, significance_method=secondary)[['identity', 'test_status', 'p_value', 'q_value', 'significant', 'significance_period_hours']].rename(columns={'test_status': 'secondary_test_status', 'p_value': 'secondary_p_value', 'q_value': 'secondary_q_value', 'significant': 'secondary_significant', 'significance_period_hours': 'secondary_significance_period_hours'})
        fits = fits.merge(second, on='identity', how='left', validate='one_to_one')
        both_tested = fits['primary_test_status'].eq('ok') & fits['secondary_test_status'].eq('ok')
        fits['significant'] = both_tested & fits['primary_significant'].fillna(False).astype(bool) & fits['secondary_significant'].fillna(False).astype(bool)
        fits['test_status'] = np.where(both_tested, 'ok', 'not_tested')
        fits['rhythm_status'] = np.select([~both_tested, fits['significant']], ['not tested', 'rhythmic'], default='not rhythmic')
        fits['secondary_significance_method'] = secondary
        fits['verdict_rule'] = f'{primary_method} and {secondary}'
    fits['rhythmic'] = fits['significant']
    return fits

def _statistics_table(fits: pd.DataFrame) -> pd.DataFrame:
    """One auditable row per cell and significance test."""
    shared = ['identity', 'metric', 'observations', 'span_hours', 'period_estimation_method', 'period_hours', 'period_error_hours', 'estimate_status', 'cycles_observed', 'period_underdetermined', 'period_at_search_edge', 'correction', 'alpha', 'family_tests', 'period_search_min_hours', 'period_search_max_hours', 'detrend', 'detrend_window_hours', 'detrend_polynomial_degree', 'detrend_min_valid_fraction', 'detrend_bandwidth_hours', 'detrend_low_cut_hours', 'detrend_high_cut_hours', 'detrend_filter_order', 'workbench_version', 'analysis_parameters_json', 'verdict_rule', 'row_order_method', 'row_order_keys_json']
    shared = [column for column in shared if column in fits]

    def rows_for(role: str, prefix: str, period_column: str) -> pd.DataFrame:
        table = fits[shared].copy()
        table['test_role'] = role
        table['significance_method'] = fits[f'{prefix}_significance_method']
        table['test_status'] = fits[f'{prefix}_test_status']
        table['p_value'] = fits[f'{prefix}_p_value']
        table['q_value'] = fits[f'{prefix}_q_value']
        table['test_significant'] = fits[f'{prefix}_significant']
        table['significance_period_hours'] = fits[period_column]
        table['final_cell_significant'] = fits['significant']
        table['final_rhythm_status'] = fits['rhythm_status']
        return table
    tables = [rows_for('primary', 'primary', 'significance_period_hours')]
    if 'secondary_test_status' in fits and fits['secondary_significance_method'].notna().any():
        tables.append(rows_for('secondary', 'secondary', 'secondary_significance_period_hours'))
    result = pd.concat(tables, ignore_index=True)
    leading = ['identity', 'metric', 'test_role', 'significance_method', 'test_status', 'p_value', 'q_value', 'test_significant', 'final_cell_significant', 'final_rhythm_status', 'period_estimation_method', 'period_hours', 'significance_period_hours']
    return result[[*leading, *[column for column in result if column not in leading]]]

def prepare(source,options):
    metric = str(options.get('metrics'))
    bins = int(options.get('bins'))
    period_bins = options.get('period_bins')
    raster_lut = options.get('trace_luts')
    hour_ticks = options.get('hour_ticks')
    order_requested = list(options.get('order'))
    secondary_requested = options.get('secondary_significance_method')
    panels = ('raster','period_peak_matrix','period_histogram')
    frame = source.table('cell_frame.csv')
    if metric not in frame:
        numeric = [column for column in frame.select_dtypes(include=np.number).columns if column not in {'identity', 'frame_index', 'imagej_frame', 'hours'}]
        raise ValueError(f"--metrics {metric} is absent from cell_frame.csv. Numeric measurements available: {', '.join(numeric)}")
    if not pd.api.types.is_numeric_dtype(frame[metric]):
        raise ValueError(f'--metrics {metric} must be numeric')
    inherited = {**RHYTHM_DEFAULTS, **source.module_params('rhythms')}
    try:
        resolved = workbench.resolve_analysis_options(inherited, options.get)
    except ValueError as error:
        raise ValueError(str(error)) from None
    params = resolved['params']
    search = [resolved['period_min_hours'], resolved['period_max_hours']]
    period_edges = _period_edges(search, bins, period_bins)
    first_eight_hour_boundary = np.ceil(search[0] / 8.0) * 8.0
    period_peak_edges = np.unique(np.concatenate(([search[0]], np.arange(first_eight_hour_boundary, search[1], 8.0), [search[1]])))
    fits = _fit_cells(frame, metric, resolved, secondary_requested)
    if fits.empty:
        raise ValueError('no cell traces remain for rhythm testing')
    fits['analysis_parameters_json'] = json.dumps(params, sort_keys=True, default=str)
    primary_method = resolved['significance_method']
    secondary_method = str(fits['secondary_significance_method'].dropna().iloc[0]) if fits['secondary_significance_method'].notna().any() else None
    estimator_method = resolved['method']
    primary_label = workbench.PERIOD_METHODS[primary_method]['label']
    secondary_label = workbench.PERIOD_METHODS[secondary_method]['label'] if secondary_method else None
    estimator_label = workbench.PERIOD_METHODS[estimator_method]['label']
    significant = fits['significant'].fillna(False).astype(bool)
    period_available = fits['period_available'].fillna(False).astype(bool)
    histogram_mask = significant & period_available
    phase_available = pd.to_numeric(fits['phase_hours'], errors='coerce').notna()
    period_peak_mask = histogram_mask & phase_available
    fits['rhythm_rank'] = np.select([significant, fits['test_status'].eq('ok')], [0, 1], default=2)
    traces = detrended_traces(frame, fits['identity'], metric, method=resolved['detrend'], window_hours=resolved['detrend_window_hours'], detrend_options=params)
    unordered_matrix = traces.pivot(index='identity', columns='hours', values='detrended_z')
    resolved_order = _resolved_order(order_requested)
    trace_comparable = matrix_ordering.trace_comparable(unordered_matrix)
    fits['trace_comparable'] = fits['identity'].map(trace_comparable).eq(True)
    resolved_order_columns = {column.removeprefix('-') for column in resolved_order}
    uses_continuum = bool({'pattern_rank', 'spectral_rank'} & resolved_order_columns)
    sort_order = resolved_order if uses_continuum else ['-trace_comparable', *resolved_order]
    verdict_masks = (significant, ~significant & fits['test_status'].eq('ok'), ~fits['test_status'].eq('ok'))
    rank_methods = {'pattern_rank': 'principal_component_gradient', 'spectral_rank': 'spectral_continuum'}
    for rank_column, ordering_method in rank_methods.items():
        if rank_column not in resolved_order_columns:
            continue
        ranks: dict[object, int] = {}
        for verdict_mask in verdict_masks:
            identities = fits.loc[verdict_mask, 'identity'].tolist()
            ordered = matrix_ordering.trace_pattern_order(unordered_matrix.reindex(identities), method=ordering_method)
            ranks.update({identity: rank for rank, identity in enumerate(ordered)})
        fits[rank_column] = fits['identity'].map(ranks)
    if 'displayed_onset_hours' in resolved_order_columns:
        displayed_onsets = matrix_ordering.displayed_onsets(unordered_matrix)
        fits['displayed_onset_hours'] = fits['identity'].map(displayed_onsets)
    row_order_method = _order_label(order_requested)
    fits['row_order_method'] = row_order_method
    fits['row_order_keys_json'] = json.dumps(sort_order)
    significant_ids = matrix_ordering.ordered_identities(fits[significant], sort_order)
    nonsignificant_ids = matrix_ordering.ordered_identities(fits[~significant & fits['test_status'].eq('ok')], sort_order)
    untested_ids = matrix_ordering.ordered_identities(fits[~fits['test_status'].eq('ok')], sort_order)
    order = significant_ids + nonsignificant_ids + untested_ids
    label = describe(metric).label
    cells_tested = int(fits['test_status'].eq('ok').sum())
    significant_count = int(significant.sum())
    histogram_count = int(histogram_mask.sum())
    not_tested_count = int(len(fits) - cells_tested)
    significant_periods = pd.to_numeric(fits.loc[histogram_mask, 'period_hours'], errors='coerce')
    median_period = float(significant_periods.median()) if histogram_count else np.nan
    edge_count = int((histogram_mask & fits['period_at_search_edge'].fillna(False)).sum())
    underdetermined_count = int((histogram_mask & fits['period_underdetermined'].fillna(True)).sum())
    span_hours = float(source.summary['hours_covered'])
    provenance = ''
    correction = resolved['multiple_testing']
    correction_label = {'bh': 'Benjamini-Hochberg', 'bonferroni': 'Bonferroni', 'sidak': 'Sidak'}.get(correction, correction)
    correction_text = 'unadjusted p' if correction == 'none' else f'{correction_label}-adjusted q'
    detrend_label = str(resolved['detrend']).replace('_', ' ')
    verdict_text = f"{primary_label} and {secondary_label} must both have {correction_text} < {resolved['rhythmic_alpha']:g}" if secondary_label else f"{primary_label} {correction_text} < {resolved['rhythmic_alpha']:g}"
    whole_trace_ordered = uses_continuum
    onset_ordered = 'displayed_onset_hours' in resolved_order_columns
    footnote = f"{estimator_label} estimated period over {search[0]:g}-{search[1]:g} h after {detrend_label} detrending; {verdict_text} defines significant. Periods supported by fewer than {resolved['min_cycles']:g} observed cycles are flagged but not removed from the significant-period distribution." + (f'\n{row_order_method.capitalize()} ordering uses a centred five-sample smoothed copy; the displayed values and times remain unchanged. It does not compare phase or imply synchrony.' if whole_trace_ordered else '') + ('\nDisplayed onset is the first three-sample blue-to-red transition in absolute recording hours; it is not a shared phase.' if onset_ordered else '') + ("\nThe peak-position matrix divides each fitted peak by that cell's own detected period. It is descriptive and does not establish a shared phase, synchrony or a common tissue clock." if 'period_peak_matrix' in panels else '') + (f'\n{provenance}' if provenance else '')
    matrix = unordered_matrix.reindex(order)
    hours_axis = matrix.columns.to_numpy(float)
    traces_table = matrix.reset_index().melt(id_vars='identity', var_name='hours', value_name='detrended_z').dropna(subset=['detrended_z'])
    traces_table['row'] = traces_table['identity'].map({identity: row for row, identity in enumerate(order)})
    trace_columns = ['identity', 'period_hours', 'p_value', 'q_value', 'significant', 'rhythm_status', 'period_underdetermined', 'period_at_search_edge', 'period_estimation_method', 'primary_significance_method', 'secondary_significance_method', 'workbench_version', 'row_order_method', 'row_order_keys_json', 'trace_comparable']
    for column in ('pattern_rank', 'spectral_rank', 'displayed_onset_hours'):
        if column in fits:
            trace_columns.append(column)
    traces_table = traces_table.merge(fits[trace_columns], on='identity', how='left')
    keep=period_peak_mask.to_numpy(bool)
    fractions=workbench.cycle_fraction(fits.loc[keep,'phase_hours'].to_numpy(float),fits.loc[keep,'period_hours'].to_numpy(float))
    phase_edges=np.linspace(0.,1.,bins+1)
    counts,_,_=np.histogram2d(fits.loc[keep,'period_hours'].to_numpy(float),fractions,bins=(period_peak_edges,phase_edges))
    r,c=np.indices(counts.shape)
    peaks=pd.DataFrame(dict(period_start_hour=period_peak_edges[r.ravel()],period_end_hour=period_peak_edges[r.ravel()+1],phase_start_fraction=phase_edges[c.ravel()],phase_end_fraction=phase_edges[c.ravel()+1],cells=counts.ravel().astype(int)))
    counts_period,_=np.histogram(significant_periods.dropna().to_numpy(float),bins=period_edges)
    hist=pd.DataFrame(dict(period_start_hour=period_edges[:-1],period_end_hour=period_edges[1:],significant_cell_count=counts_period))
    total=int(counts_period.sum());hist['significant_cell_frequency']=counts_period/total if total else 0.
    stats=_statistics_table(fits)
    return PreparedViews({'raster':dict(table=traces_table,values=matrix.to_numpy(float),hours=hours_axis,order=order,blocks=[len(significant_ids),len(nonsignificant_ids),len(untested_ids)],label=label),
        'period_peak_matrix':dict(table=peaks,counts=counts,period_edges=period_peak_edges,phase_edges=phase_edges),
        'period_histogram':dict(table=hist)},auxiliary={'rhythm_fits.csv':fits,'statistics.csv':stats},wording=dict(title=f'{label}: independently estimated cell rhythms',footnote=footnote)),stats
