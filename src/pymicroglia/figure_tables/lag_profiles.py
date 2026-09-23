"""Prepare the saved lag profiles and descriptive peak locations."""
import pandas as pd
from .prepared import PreparedViews

def prepare(source,options):
    data = source.table('lag_profiles.csv')
    requested = options.get('metrics')
    pairs = []
    for token in requested:
        pieces = token.split(':')
        if len(pieces) != 2 or not all(pieces):
            raise ValueError('--metrics must contain a:b metric-pair tokens')
        pairs.append(tuple(pieces))
    wanted = pd.MultiIndex.from_tuples(pairs)
    pair_index = pd.MultiIndex.from_frame(data[['metric_a', 'metric_b']])
    data = data[pair_index.isin(wanted)].copy()
    if data.empty:
        available = sorted({f'{a}:{b}' for a, b in zip(pair_index.get_level_values(0), pair_index.get_level_values(1))})
        raise ValueError(f"requested --metrics pair was not measured; available: {', '.join(available)}")
    maximum = float(options.get('max_lag'))
    data = data[data['lag_hours'].abs() <= maximum]
    grouped = data.groupby(['metric_a', 'metric_b', 'lag_frames', 'lag_hours'], as_index=False).agg(mean_correlation=('correlation', 'mean'), lo=('correlation', lambda values: float(values.quantile(0.25))), hi=('correlation', lambda values: float(values.quantile(0.75))), cells=('identity', 'nunique'), surrogate_mean=('surrogate_mean', 'mean'), surrogate_lo=('surrogate_lo', 'mean'), surrogate_hi=('surrogate_hi', 'mean'), pairs=('pairs', 'sum'))
    figure_data = grouped[['metric_a', 'metric_b', 'lag_frames', 'lag_hours', 'mean_correlation', 'lo', 'hi', 'cells', 'surrogate_mean', 'surrogate_lo', 'surrogate_hi']]
    peaks = []
    for keys, group in data.groupby(['identity', 'metric_a', 'metric_b'], sort=True):
        valid = group.dropna(subset=['correlation'])
        if valid.empty:
            continue
        row = valid.loc[valid['correlation'].abs().idxmax()]
        peaks.append({'identity': int(keys[0]), 'metric_a': keys[1], 'metric_b': keys[2], 'peak_lag_hours': float(row['lag_hours']), 'peak_correlation': float(row['correlation']), 'pairs': int(row['pairs'])})
    peak_data = pd.DataFrame(peaks)
    if grouped.empty:raise ValueError('No saved lag estimates within max_lag')
    first=grouped.loc[grouped.metric_a.eq(pairs[0][0]) & grouped.metric_b.eq(pairs[0][1])]
    return PreparedViews({'profile':dict(table=figure_data,rows=first),
        'per_cell':dict(table=peak_data),'pairs':dict(table=figure_data,rows=grouped)},
        auxiliary={'per_cell_peaks.csv':peak_data},wording=dict(
            subtitle=f'Negative lag means the first named series leads; {data.identity.nunique()} identities, capped at {maximum:g} h.',
            footnote='Correlations and surrogate ranges are read from the saved lag search. Peak selection is descriptive; it does not establish a shared rhythm or causal timing.')),None
