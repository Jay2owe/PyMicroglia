"""Prepare repeatability within a cell using its saved detected period."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .. import workbench
from ..measure.modules.rhythms import DEFAULTS as RHYTHM_DEFAULTS
from ..visualisation.labels import describe
from .rhythm_processing import detrended_z
from .prepared import PreparedViews

METHOD_LABELS = {'lomb': 'Lomb-Scargle periodogram', 'chi_square': 'Enright-Sokolove periodogram', 'f': 'F periodogram', 'jtk': 'JTK_CYCLE', 'ejtk': 'empirical JTK_CYCLE'}

def _as_bool(values: pd.Series) -> pd.Series:
    """Read old CSV booleans as strictly as current boolean columns."""
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin({'true', '1', 'yes'})

def _period_class(period: pd.Series, params: dict) -> pd.Series:
    low, high = map(float, params.get('circadian_band_hours', [20.0, 28.0]))
    return pd.Series(np.select([period < low, period <= high, np.isfinite(period)], ['ultradian', 'circadian-like', 'infradian'], default='unknown'), index=period.index)

def _primary_rows(fits: pd.DataFrame, params: dict) -> pd.DataFrame:
    """One selected period and one rhythm-test result per cell, including legacy runs."""
    source = fits.copy()
    modern = 'best_period_hours' in source
    primary = str(params.get('primary_rhythm_test', 'lomb'))
    estimator = str(params.get('period_estimation_method', primary))
    primary_label = workbench.PERIOD_METHODS.get(primary, {'label': METHOD_LABELS.get(primary, primary)})['label']
    if modern:
        period = pd.to_numeric(source['best_period_hours'], errors='coerce')
        p_value = pd.to_numeric(source.get('best_p_value'), errors='coerce')
        alpha = pd.to_numeric(source.get('best_alpha', pd.Series(params['rhythmic_alpha'], index=source.index)), errors='coerce')
        rhythmic = _as_bool(source['rhythmic'])
        status = source.get('rhythm_status', pd.Series(np.where(rhythmic, 'rhythmic', 'arrhythmic'), index=source.index))
        estimator_label = source.get('best_method_label', pd.Series(workbench.PERIOD_METHODS.get(estimator, {'label': METHOD_LABELS.get(estimator, estimator)})['label'], index=source.index))
        at_edge = _as_bool(source.get('best_period_at_search_edge', pd.Series(False, index=source.index)))
        period_class = source.get('best_period_class', _period_class(period, params))
        cycles = pd.to_numeric(source.get('cycles_covered_at_best_period', source.get('span_hours') / period), errors='coerce')
    else:
        if primary != 'lomb' or estimator != 'lomb':
            raise ValueError('This older run only stored the Lomb-Scargle period and verdict; rerun rhythms before selecting another estimator or rhythm test.')
        period = pd.to_numeric(source.get('lombscargle_period_hours'), errors='coerce')
        p_value = pd.to_numeric(source.get('lombscargle_false_alarm'), errors='coerce')
        alpha = pd.Series(float(params.get('rhythmic_alpha', 0.05)), index=source.index)
        rhythmic = _as_bool(source.get('rhythmic_lombscargle', pd.Series(False, index=source.index)))
        status = pd.Series(np.where(rhythmic, 'rhythmic', 'arrhythmic'), index=source.index)
        estimator_label = pd.Series(METHOD_LABELS['lomb'], index=source.index)
        low, high = map(float, params['period_search_hours'])
        tolerance = max(0.1, 0.005 * (high - low))
        at_edge = (period <= low + tolerance) | (period >= high - tolerance)
        period_class = _period_class(period, params)
        cycles = pd.to_numeric(source.get('span_hours'), errors='coerce') / period
    return pd.DataFrame({'identity': pd.to_numeric(source['identity'], errors='coerce').astype('Int64'), 'metric': source['metric'].astype(str), 'primary_method': primary, 'primary_method_label': primary_label, 'period_estimation_method': estimator, 'period_estimation_method_label': estimator_label, 'rhythm_status': status, 'rhythmic': rhythmic, 'detected_period_hours': period, 'period_class': period_class, 'p_value': p_value, 'alpha': alpha, 'period_at_search_edge': at_edge, 'cycles_in_recording': cycles, 'detrend': source.get('detrend', pd.Series('unknown', index=source.index)), 'workbench_version': source.get('workbench_version', pd.Series('legacy run', index=source.index))})

def _fallback_traces(frame: pd.DataFrame, metric: str, detrend: str, detrend_window_hours: float, detrend_options: dict | None=None) -> pd.DataFrame:
    """Recreate a pre-table run through the same Circadian Workbench detrend."""
    if metric not in frame:
        raise ValueError(f'--metrics {metric} is not in cell_frame.csv')
    rows = []
    for identity, group in frame[['identity', 'hours', metric]].dropna().groupby('identity', sort=True):
        group = group.sort_values('hours')
        hours = group['hours'].to_numpy(float)
        values = group[metric].to_numpy(float)
        scaled = detrended_z(hours, values, method=detrend, window_hours=detrend_window_hours, detrend_options=detrend_options)
        rows.extend(({'identity': int(identity), 'metric': metric, 'hours': float(hour), 'detrended_z': float(value), 'trace_source': 'legacy trace reconstructed for display'} for hour, value in zip(hours, scaled)))
    return pd.DataFrame(rows)

def _selected_identities(data: pd.DataFrame, requested: str) -> list[int]:
    text = str(requested).strip().lower()
    if text in {'', 'all'}:
        return [int(value) for value in data['identity']]
    try:
        count = int(text)
    except ValueError:
        wanted = [int(value) for value in [part.strip() for part in str(requested).split(',') if part.strip()]]
        missing = sorted(set(wanted) - set(data['identity'].astype(int)))
        if missing:
            raise ValueError('--cells includes identities that are not eligible for this comparison: ' + ', '.join(map(str, missing)))
        return wanted
    if count < 1:
        raise ValueError('--cells needs a positive count, comma-separated identities, or all')
    return [int(value) for value in data.sort_values(['minimum_cycle_coverage', 'frames_cycle_1'], ascending=False)['identity'].head(count)]

def prepare(source,options):
    metric = options.get('metrics')
    params = {**RHYTHM_DEFAULTS, **source.module_params('rhythms')}
    fitted = source.table('rhythms.csv')
    fitted = fitted[fitted['metric'] == metric].copy()
    if fitted.empty:
        available = ', '.join(sorted(set(source.table('rhythms.csv')['metric'])))
        raise ValueError(f'--metrics {metric} was not rhythm-tested; available: {available}')
    tests = _primary_rows(fitted, params)
    stored_traces = source.table('rhythm_traces.csv',optional=True)
    if stored_traces is not None:
        traces = stored_traces[stored_traces['metric'] == metric].copy()
        if traces.empty:
            raise ValueError(f'rhythm_traces.csv has no rows for --metrics {metric}')
        traces['trace_source'] = 'Circadian Workbench detrended trace'
    else:
        frame = source.table('cell_frame.csv',optional=True)
        if frame is None:
            raise ValueError('This run has neither rhythm_traces.csv nor cell_frame.csv; rerun rhythms.')
        detrends = tests['detrend'].dropna().astype(str)
        detrend = detrends.mode().iloc[0] if not detrends.empty else str(params['detrend'])
        traces = _fallback_traces(frame, metric, detrend, float(params['detrend_window_hours']), detrend_options=params)
    profile_bins = int(options.get('profile_bins'))
    min_coverage = float(options.get('min_coverage'))
    if profile_bins < 4:
        raise ValueError('--profile-bins needs at least 4 positions')
    if not 0 < min_coverage <= 1:
        raise ValueError('--min-coverage must be greater than 0 and no more than 1')
    positions = (np.arange(profile_bins, dtype=float) + 0.5) / profile_bins
    aligned_positions = (np.arange(profile_bins, dtype=float) - profile_bins // 2) / profile_bins
    eligibility = tests.copy()
    eligibility['included_in_plot'] = False
    eligibility['exclusion_reason'] = 'not significantly rhythmic by the primary test'
    invalid_period = ~np.isfinite(eligibility['detected_period_hours']) | (eligibility['detected_period_hours'] <= 0)
    eligibility.loc[invalid_period, 'exclusion_reason'] = 'no valid detected period'
    eligibility.loc[eligibility['rhythmic'] & eligibility['period_at_search_edge'], 'exclusion_reason'] = 'detected period touches the search boundary'
    initial = eligibility[eligibility['rhythmic'] & ~eligibility['period_at_search_edge'] & ~invalid_period]
    cell_rows: list[dict] = []
    profile_rows: list[dict] = []
    for test in initial.itertuples(index=False):
        group = traces[traces['identity'] == test.identity].dropna(subset=['hours', 'detrended_z']).sort_values('hours')
        if group.empty:
            eligibility.loc[eligibility['identity'] == test.identity, 'exclusion_reason'] = 'no detrended trace'
            continue
        hours = group['hours'].to_numpy(float)
        values = group['detrended_z'].to_numpy(float)
        period = float(test.detected_period_hours)
        relative = hours - float(hours.min())
        intervals = np.diff(np.unique(hours))
        intervals = intervals[intervals > 0]
        interval = float(np.median(intervals)) if len(intervals) else np.nan
        expected = period / interval if np.isfinite(interval) and interval > 0 else np.nan
        profiles = []
        counts = []
        coverages = []
        for cycle in (0, 1):
            keep = (relative >= cycle * period) & (relative < (cycle + 1) * period)
            counts.append(int(np.count_nonzero(keep)))
            coverage = min(counts[-1] / expected, 1.0) if np.isfinite(expected) else 0.0
            coverages.append(float(coverage))
            if counts[-1] < 4 or coverage < min_coverage:
                profiles.append(None)
                continue
            phase = (relative[keep] - cycle * period) / period
            profiles.append(workbench.cycle_profile(phase, values[keep], positions))
        if profiles[0] is None or profiles[1] is None:
            eligibility.loc[eligibility['identity'] == test.identity, 'exclusion_reason'] = f'fewer than two cycles with {min_coverage:.0%} measurement coverage'
            continue
        peak_1 = workbench.peak_fraction(profiles[0], positions)
        peak_2 = workbench.peak_fraction(profiles[1], positions)
        shift = abs((peak_2 - peak_1 + 0.5) % 1.0 - 0.5)
        signed_shift = (peak_2 - peak_1 + 0.5) % 1.0 - 0.5
        correlation = float(np.corrcoef(profiles[0], profiles[1])[0, 1])
        peak_index_1 = int(np.argmin(np.abs(positions - peak_1)))
        alignment = profile_bins // 2 - peak_index_1
        aligned_profiles = [np.roll(profile, alignment) for profile in profiles]
        cell_rows.append({'identity': int(test.identity), 'metric': metric, 'primary_method': test.primary_method, 'primary_method_label': test.primary_method_label, 'period_estimation_method': test.period_estimation_method, 'period_estimation_method_label': test.period_estimation_method_label, 'detected_period_hours': period, 'period_class': test.period_class, 'p_value': test.p_value, 'alpha': test.alpha, 'cycle_1_peak_percent': peak_1 * 100.0, 'cycle_2_peak_percent': peak_2 * 100.0, 'signed_peak_shift_percent': signed_shift * 100.0, 'absolute_peak_shift_percent': shift * 100.0, 'waveform_correlation': correlation, 'frames_cycle_1': counts[0], 'frames_cycle_2': counts[1], 'cycle_1_coverage': coverages[0], 'cycle_2_coverage': coverages[1], 'minimum_cycle_coverage': min(coverages), 'cycle_origin_hour': float(hours.min())})
        for cycle, profile in enumerate(aligned_profiles, start=1):
            profile_rows.extend(({'identity': int(test.identity), 'metric': metric, 'detected_period_hours': period, 'cycle': cycle, 'position_from_cycle_1_peak_percent': float(position * 100.0), 'detrended_signal_sd': float(value)} for position, value in zip(aligned_positions, profile)))
        mask = eligibility['identity'] == test.identity
        eligibility.loc[mask, 'included_in_plot'] = True
        eligibility.loc[mask, 'exclusion_reason'] = ''
    cells = pd.DataFrame(cell_rows)
    profiles = pd.DataFrame(profile_rows)
    if cells.empty:
        raise ValueError('No cells passed the primary rhythm test and supplied two sufficiently observed detected cycles.')
    selected = _selected_identities(cells, options.get('cells'))
    cells = cells[cells['identity'].isin(selected)].copy()
    profiles = profiles[profiles['identity'].isin(selected)].copy()
    eligibility['selected_by_cells_option'] = eligibility['identity'].isin(selected)
    population = profiles.groupby(['metric', 'cycle', 'position_from_cycle_1_peak_percent'], as_index=False).agg(median_signal_sd=('detrended_signal_sd', 'median'), lower_quartile_signal_sd=('detrended_signal_sd', lambda values: values.quantile(0.25)), upper_quartile_signal_sd=('detrended_signal_sd', lambda values: values.quantile(0.75)), cells=('identity', 'nunique'))
    statistics = eligibility[['identity', 'metric', 'primary_method', 'primary_method_label', 'period_estimation_method', 'period_estimation_method_label', 'rhythm_status', 'rhythmic', 'detected_period_hours', 'period_class', 'p_value', 'alpha', 'period_at_search_edge', 'cycles_in_recording', 'included_in_plot', 'selected_by_cells_option', 'exclusion_reason', 'detrend', 'workbench_version']].copy()
    count=int(options['bins'])
    if count<1:raise ValueError('bins must be positive')
    histogram,edges=np.histogram(cells.absolute_peak_shift_percent,bins=np.linspace(0.,50.,count+1))
    return PreparedViews({'profiles':dict(table=population),'peak_agreement':dict(table=cells),
        'peak_shift':dict(table=pd.DataFrame({'left':edges[:-1],'right':edges[1:],'cells':histogram}),median=float(cells.absolute_peak_shift_percent.median()))},
        auxiliary={'cell_repeatability.csv':cells,'cycle_profiles.csv':profiles,'statistics.csv':statistics},
        wording=dict(title='Repeatability within each cell at its own detected period',footnote='Each cell compares its own first two sufficiently observed cycles. Alignment describes waveform repeatability; it does not establish comparable timing, synchrony or a shared period between cells.')),statistics
