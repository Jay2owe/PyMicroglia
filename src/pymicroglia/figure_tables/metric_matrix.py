"""Prepare broad-period fits through the shared Circadian Workbench gateway."""
from __future__ import annotations
import textwrap
import numpy as np
import pandas as pd
from .. import workbench
from ..measure.modules.rhythms import DEFAULTS as RHYTHM_DEFAULTS
from ..visualisation.labels import semantic_label
from . import ordering as matrix_ordering
from .prepared import PreparedViews
from .rhythm_processing import detrended_z

ORDER_ALIASES = {'period': 'median_rhythmic_period_hours', 'significance': 'best_q_value', 'rhythmic_fraction': 'rhythmic_fraction', 'rhythmic_count': 'rhythmic_metrics', 'identity': 'identity'}

METHOD_LABELS = {'lomb': 'Lomb–Scargle periodogram', 'chi_square': 'Enright chi-square periodogram', 'f': 'F periodogram', 'jtk': 'JTK_CYCLE', 'ejtk': 'empirical JTK_CYCLE'}

def _resolved_order(requested: list[str]) -> list[str]:
    return matrix_ordering.resolve_order(requested, aliases=ORDER_ALIASES)

def _test_metrics(long: pd.DataFrame, *, metrics: list[str], params: dict, method: str, detrend: str, detrend_window_hours: float, min_observations: int, correction: str, correction_scope: str, min_cycles: float, significance_method: str | None=None) -> pd.DataFrame:
    """Test one trace per cell and metric under the requested correction family."""
    common_args = {'value_column': 'value', 'params': params, 'method': method, 'significance_method': significance_method, 'detrend': detrend, 'detrend_window_hours': detrend_window_hours, 'min_observations': min_observations, 'correction': correction, 'min_cycles': min_cycles}
    if correction_scope == 'matrix':
        fits = workbench.estimate_grouped_rhythms(long, group_columns=['identity', 'measurement'], **common_args)
        fits = fits.drop(columns='metric').rename(columns={'measurement': 'metric'})
        fits['correction_family'] = 'all selected cells and metrics'
        return fits
    if correction_scope != 'metric':
        raise ValueError('--correction-scope must be matrix or metric')
    pieces = []
    for metric in metrics:
        selected = long[long['measurement'].eq(metric)].copy()
        fit = workbench.estimate_grouped_rhythms(selected, group_columns=['identity'], **common_args)
        fit['metric'] = metric
        fit['correction_family'] = f'cells within {metric}'
        pieces.append(fit)
    return pd.concat(pieces, ignore_index=True)

def prepare(source, options):
    frame = source.table('cell_frame.csv')
    metrics = [str(metric) for metric in options.get('metrics')]
    if not metrics:
        raise ValueError('--metrics needs at least one cell-frame measurement')
    if len(set(metrics)) != len(metrics):
        raise ValueError('--metrics contains the same measurement more than once')
    unknown = [metric for metric in metrics if metric not in frame]
    if unknown:
        numeric = [column for column in frame.select_dtypes(include=np.number).columns if column not in {'identity', 'frame_index', 'imagej_frame', 'hours'}]
        raise ValueError(f"--metrics names {', '.join(unknown)}, which cell_frame.csv does not contain. Numeric measurements available: {', '.join(numeric)}")
    nonnumeric = [metric for metric in metrics if not pd.api.types.is_numeric_dtype(frame[metric])]
    if nonnumeric:
        raise ValueError('--metrics must be numeric: ' + ', '.join(nonnumeric))
    try:
        labels = [semantic_label(metric) for metric in metrics]
    except ValueError as error:
        raise ValueError(str(error)) from None
    label_wrap = int(options.get('column_label_wrap'))
    if label_wrap < 0:
        raise ValueError('--column-label-wrap must be 0 or a positive number')
    if label_wrap:
        labels = [textwrap.fill(label, width=max(8, label_wrap), break_long_words=False, break_on_hyphens=False) for label in labels]
    inherited = {**RHYTHM_DEFAULTS, **source.module_params('rhythms')}
    resolved = workbench.resolve_analysis_options(inherited, options.get)
    params = resolved['params']
    period_min = resolved['period_min_hours']
    period_max = resolved['period_max_hours']
    alpha = resolved['rhythmic_alpha']
    min_cycles = resolved['min_cycles']
    min_observations = resolved['min_observations']
    method = resolved['method']
    significance_method = resolved['significance_method']
    detrend = resolved['detrend']
    correction = resolved['multiple_testing']
    correction_scope = str(options.get('correction_scope'))
    long = frame[['identity', 'hours', *metrics]].melt(id_vars=['identity', 'hours'], value_vars=metrics, var_name='measurement', value_name='value')
    fits = _test_metrics(long, metrics=metrics, params=params, method=method, significance_method=significance_method, detrend=detrend, detrend_window_hours=resolved['detrend_window_hours'], min_observations=min_observations, correction=correction, correction_scope=correction_scope, min_cycles=min_cycles)
    if fits.empty:
        raise ValueError('none of the selected cell-metric traces could be assessed')
    summary = fits.groupby('identity', sort=True).agg(tested_metrics=('test_status', lambda values: int(values.eq('ok').sum())), rhythmic_metrics=('significant', 'sum'), best_q_value=('q_value', 'min')).reset_index()
    significant_periods = fits[fits['significant']].groupby('identity')['period_hours'].median().rename('median_rhythmic_period_hours')
    summary = summary.merge(significant_periods, on='identity', how='left')
    summary['rhythmic_fraction'] = np.divide(summary['rhythmic_metrics'], summary['tested_metrics'], out=np.zeros(len(summary), dtype=float), where=summary['tested_metrics'].to_numpy() > 0)
    order_requested = list(options.get('order'))
    order = matrix_ordering.ordered_identities(summary, _resolved_order(order_requested))
    summary['display_row'] = summary['identity'].map({identity: row + 1 for row, identity in enumerate(order)})
    matrix_index = pd.MultiIndex.from_product([order, metrics], names=['identity', 'metric'])
    indexed = fits.set_index(['identity', 'metric']).reindex(matrix_index)
    periods = indexed['period_hours'].where(indexed['rhythm_status'].eq('rhythmic')).to_numpy(float).reshape(len(order), len(metrics))
    statuses = indexed['rhythm_status'].fillna('not tested').to_numpy(object).reshape(len(order), len(metrics))
    unavailable = indexed['rhythm_status'].eq('rhythmic') & ~indexed['period_available'].fillna(False)
    statuses[unavailable.to_numpy().reshape(statuses.shape)] = 'period unavailable'
    uncertain = indexed.get('period_underdetermined', pd.Series(False, index=indexed.index)).astype('boolean').fillna(False).to_numpy(dtype=bool).reshape(len(order), len(metrics))
    fits['correction_scope'] = correction_scope
    fits['display_row'] = fits['identity'].map({identity: row + 1 for row, identity in enumerate(order)})
    fits['display_column'] = fits['metric'].map({metric: column + 1 for column, metric in enumerate(metrics)})
    statistics_columns = ['identity', 'metric', 'method', 'significance_method', 'observations', 'span_hours', 'estimate_status', 'estimate_reason', 'period_available', 'estimator_p_value', 'significance_period_hours', 'period_difference_hours', 'period_error_hours', 'phase_hours', 'phase_error_hours', 'amplitude', 'amplitude_error', 'rae', 'goodness_of_fit', 'period_hours', 'p_value', 'q_value', 'correction', 'correction_scope', 'correction_family', 'alpha', 'significant', 'rhythm_status', 'test_status', 'reason', 'periodogram_significant', 'peak_power', 'threshold', 'detrend', 'detrend_window_hours', 'cycles_observed', 'period_underdetermined', 'period_at_search_edge', 'period_search_min_hours', 'period_search_max_hours', 'family_tests', 'workbench_version']
    statistics = fits[[column for column in statistics_columns if column in fits]].copy()
    return PreparedViews({'matrix':dict(table=fits,periods=periods,statuses=statuses,uncertain=uncertain,order=order,labels=labels,low=period_min,high=period_max)},auxiliary={'statistics.csv':statistics,'cell_order.csv':summary},wording={'title':'Rhythms across selected cell measurements','footnote':f"Estimator {method}; significance test {significance_method}; correction {correction} over {correction_scope}; Workbench {workbench.WORKBENCH_VERSION}. Outlines mark insufficient observed cycles; hatching marks a significant trace without a period estimate."}),statistics
