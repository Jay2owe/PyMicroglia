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

VALUE_COLUMN = 'mean_relative_radial_occupancy'

ORDER_ALIASES = {'status': 'rhythm_rank', 'period': 'period_hours', 'significance': 'q_value', 'occupancy': 'mean_occupancy', 'identity': 'identity'}

def _resolved_order(requested: list[str]) -> list[str]:
    return matrix_ordering.resolve_order(requested, aliases=ORDER_ALIASES)

def _relative_occupancy_trace(sholl: pd.DataFrame, *, support_threshold: float, length_per_pixel: float) -> tuple[pd.DataFrame, int]:
    """One normalised profile area per cell-frame.

    All retained rings must have enough observable annulus in that frame. This
    keeps a ring disappearing at the image boundary from masquerading as an
    occupancy change.
    """
    ring_count = int(sholl['ring'].nunique())
    if ring_count < 1:
        raise ValueError('no radial rings remain after filtering')
    inner_px = sholl['radius_inner'].to_numpy(float) / float(length_per_pixel)
    outer_px = sholl['radius_outer'].to_numpy(float) / float(length_per_pixel)
    expected = np.pi * (outer_px ** 2 - inner_px ** 2)
    prepared = sholl.copy()
    prepared['annulus_support_fraction'] = np.minimum(prepared['annulus_px'].to_numpy(float) / np.maximum(expected, 1e-12), 1.0)
    prepared['ring_usable'] = prepared['annulus_support_fraction'].ge(float(support_threshold)) & prepared['occupancy'].notna()

    def reduce_frame(group: pd.DataFrame) -> pd.Series:
        usable = group['ring_usable'].to_numpy(bool)
        complete = len(group) == ring_count and int(usable.sum()) == ring_count
        return pd.Series({'hours': float(group['hours'].iloc[0]), 'valid_rings': int(usable.sum()), 'total_rings': ring_count, VALUE_COLUMN: float(group.loc[usable, 'occupancy'].mean()) if complete else np.nan})
    traces = prepared.groupby(['identity', 'frame_index'], sort=True, as_index=False).apply(reduce_frame, include_groups=False).reset_index(drop=True)
    return (traces, ring_count)

def _add_display_values(traces: pd.DataFrame, *, display: str, detrend: str, window_hours: float, detrend_options: dict | None=None) -> pd.DataFrame:
    result = traces.copy()
    if display == 'raw':
        result['display_value'] = result[VALUE_COLUMN]
        return result
    if display != 'detrended':
        raise ValueError('--display must be raw or detrended')
    result['display_value'] = np.nan
    for _, group in result.groupby('identity', sort=True):
        usable = group[['hours', VALUE_COLUMN]].dropna().sort_values('hours')
        if len(usable) < 2:
            continue
        result.loc[usable.index, 'display_value'] = detrended_z(usable['hours'], usable[VALUE_COLUMN], method=detrend, window_hours=window_hours, detrend_options=detrend_options)
    return result

def prepare(source, options):
    sholl = source.table('sholl.csv')
    if 'scaling' not in sholl:
        raise ValueError('this sholl.csv predates labelled radial scalings; re-run the sholl module')
    scaling = str(options.get('scaling'))
    if scaling not in {'cell', 'global'}:
        raise ValueError('--scaling must be cell or global')
    sholl = sholl[sholl['scaling'].eq(scaling)].copy()
    if sholl.empty:
        raise ValueError(f'this run has no {scaling!r} radial profile')
    requested_rings = options.get('ring_count', options.get('rings'))
    if requested_rings is not None:
        if int(requested_rings) < 1:
            raise ValueError('--rings needs a positive number of inner rings')
        sholl = sholl[sholl['ring'] < int(requested_rings)].copy()
    support_threshold = float(options.get('annulus_support'))
    if not 0.0 < support_threshold <= 1.0:
        raise ValueError('--annulus-support must be above 0 and at most 1')
    traces, ring_count = _relative_occupancy_trace(sholl, support_threshold=support_threshold, length_per_pixel=float(source.scale.microns_per_pixel) if source.scale.calibrated else 1.0)
    inherited = {**RHYTHM_DEFAULTS, **source.module_params('rhythms')}
    resolved = workbench.resolve_analysis_options(inherited, options.get)
    params = resolved['params']
    period_min = resolved['period_min_hours']
    period_max = resolved['period_max_hours']
    alpha = resolved['rhythmic_alpha']
    min_cycles = resolved['min_cycles']
    min_observations = resolved['min_observations']
    detrend = resolved['detrend']
    method = resolved['method']
    significance_method = resolved['significance_method']
    fits = workbench.estimate_grouped_rhythms(traces, group_columns=['identity'], value_column=VALUE_COLUMN, params=params, method=method, significance_method=significance_method, detrend=detrend, detrend_window_hours=resolved['detrend_window_hours'], min_observations=min_observations, correction=resolved['multiple_testing'], min_cycles=min_cycles)
    if fits.empty:
        raise ValueError('no cell traces remain after radial-profile reduction')
    observed = traces.groupby('identity', sort=True).agg(observed_frames=(VALUE_COLUMN, 'count'), mean_occupancy=(VALUE_COLUMN, 'mean')).reset_index()
    cell_summary = fits.merge(observed, on='identity', how='left', validate='one_to_one')
    underdetermined = cell_summary.get('period_underdetermined', pd.Series(True, index=cell_summary.index)).astype('boolean').fillna(True).astype(bool)
    cell_summary['rhythm_rank'] = np.select([cell_summary['rhythm_status'].eq('rhythmic') & ~underdetermined, cell_summary['rhythm_status'].eq('rhythmic'), cell_summary['rhythm_status'].eq('not rhythmic')], [0, 1, 2], default=3)
    order_requested = list(options.get('order'))
    order = matrix_ordering.ordered_identities(cell_summary, _resolved_order(order_requested))
    cell_summary['display_row'] = cell_summary['identity'].map({identity: row + 1 for row, identity in enumerate(order)})
    display = str(options.get('display'))
    traces = _add_display_values(traces, display=display, detrend=detrend, window_hours=float(params['detrend_window_hours']), detrend_options=params)
    hours = np.sort(traces['hours'].dropna().unique().astype(float))
    trace_matrix = traces.pivot(index='identity', columns='hours', values='display_value').reindex(index=order, columns=hours)
    status_rows = cell_summary.set_index('identity').reindex(order)
    periods = status_rows['period_hours'].where(status_rows['rhythm_status'].eq('rhythmic')).to_numpy(float)[:, None]
    statuses = status_rows['rhythm_status'].fillna('not tested').to_numpy(object)[:, None]
    unavailable = status_rows['rhythm_status'].eq('rhythmic') & ~status_rows['period_available'].fillna(False)
    statuses[unavailable.to_numpy(), 0] = 'period unavailable'
    uncertain = status_rows.get('period_underdetermined', pd.Series(False, index=status_rows.index)).astype('boolean').fillna(False).to_numpy(dtype=bool)[:, None]
    statistics_columns = ['identity', 'metric', 'method', 'significance_method', 'observations', 'span_hours', 'estimate_status', 'estimate_reason', 'period_available', 'estimator_p_value', 'significance_period_hours', 'period_difference_hours', 'period_error_hours', 'phase_hours', 'phase_error_hours', 'amplitude', 'amplitude_error', 'rae', 'goodness_of_fit', 'period_hours', 'p_value', 'q_value', 'correction', 'alpha', 'significant', 'rhythm_status', 'test_status', 'reason', 'periodogram_significant', 'peak_power', 'threshold', 'detrend', 'detrend_window_hours', 'cycles_observed', 'period_underdetermined', 'period_at_search_edge', 'period_search_min_hours', 'period_search_max_hours', 'family_tests', 'workbench_version']
    statistics = fits[[column for column in statistics_columns if column in fits]].copy()
    figure_data = traces.merge(cell_summary[['identity', 'display_row', 'rhythm_status', 'period_hours', 'q_value', 'significant', 'period_underdetermined']], on='identity', how='left', validate='many_to_one')
    values=trace_matrix.to_numpy(float)
    finite=np.abs(values[np.isfinite(values)])
    limit=max(float(np.percentile(finite,98)) if finite.size else 1.,1e-9)
    return PreparedViews({'matrix':dict(table=figure_data,values=values,hours=hours,periods=periods,statuses=statuses,uncertain=uncertain,order=order,labels=['Estimated period'],low=period_min,high=period_max,display=display,limit=limit)},auxiliary={'statistics.csv':statistics,'cell_order.csv':cell_summary},wording={'title':'Relative radial occupancy and its rhythm evidence','footnote':f"Estimator {method}; significance test {significance_method}; Workbench {workbench.WORKBENCH_VERSION}. Each trace averages {ring_count} radial bands; every band must retain at least {support_threshold:.0%} annulus support."}),statistics
