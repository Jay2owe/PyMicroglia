"""Prepare recurrence display distances using the saved measurement settings."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from ..measure.modules.recurrence import DEFAULTS as RECURRENCE_DEFAULTS,state_matrix
from .radial import _selected_identities
from .prepared import PreparedViews

def prepare(source,options):
    rates = source.table('recurrence.csv')
    quantified = source.table('recurrence_quantification.csv')
    frame = source.table('cell_frame.csv')
    settings = {**RECURRENCE_DEFAULTS, **source.module_params('recurrence')}
    if 'detrend' in quantified and quantified['detrend'].notna().any():
        settings['detrend'] = quantified['detrend'].dropna().mode().iloc[0]
    if 'detrend_window_hours' in quantified and quantified['detrend_window_hours'].notna().any():
        settings['detrend_window_hours'] = float(quantified['detrend_window_hours'].dropna().mode().iloc[0])
    metrics = [metric for metric in settings['metrics'] if metric in frame]
    if not metrics:
        raise ValueError('none of the recurrence state vector is in cell_frame.csv')
    quantile = float(settings['threshold_quantile'])
    complete = frame.dropna(subset=metrics)
    measured = set(quantified['identity'].astype(int))
    cells = [identity for identity in _selected_identities(complete, options.get('cells')) if identity in measured]
    thresholds = quantified.set_index('identity')['recurrence_threshold_distance']
    walls = {}
    for identity in cells:
        group = complete[complete['identity'] == identity][['frame_index', 'hours', *metrics]].sort_values('frame_index')
        hours = group['hours'].to_numpy(float)
        values = state_matrix(hours, group[metrics].to_numpy(float), detrend=str(settings['detrend']), window_hours=float(settings['detrend_window_hours']))
        finite_state = np.isfinite(values).all(axis=1)
        walls[int(identity)] = (cdist(values[finite_state], values[finite_state]), hours[finite_state], float(thresholds[identity]))
    drawn = rates['identity'].isin(cells)
    figure_data = rates.loc[drawn, ['identity', 'lag_frames', 'lag_hours', 'recurrence_rate', 'recurrence_surrogate_mean', 'recurrence_surrogate_lo', 'recurrence_surrogate_hi', 'recurrence_pairs']].copy().rename(columns={'lag_frames': 'time_separation_frames', 'lag_hours': 'time_separation_hours', 'recurrence_rate': 'same_state_pair_fraction', 'recurrence_surrogate_mean': 'matched_noise_mean', 'recurrence_surrogate_lo': 'matched_noise_lower_95', 'recurrence_surrogate_hi': 'matched_noise_upper_95', 'recurrence_pairs': 'observation_pairs_compared'})
    quantification = quantified[quantified['identity'].isin(cells)].drop(columns=[c for c in ('stem', 'condition', 'subject') if c in quantified.columns]).rename(columns={'recurrence_threshold_distance': 'same_state_distance_cutoff', 'determinism': 'repeated_sequence_fraction', 'laminarity': 'stationary_run_fraction', 'determinism_surrogate': 'matched_noise_repeated_sequence_fraction', 'laminarity_surrogate': 'matched_noise_stationary_run_fraction', 'recurrence_observations': 'observations', 'recurrence_metrics_used': 'measurements_used'})
    if not walls:raise ValueError('No measured cells have complete recurrence states')
    aggregate=figure_data.groupby('time_separation_hours',as_index=False).agg(same_state_pair_fraction=('same_state_pair_fraction','median'),matched_noise_mean=('matched_noise_mean','mean'),matched_noise_lower_95=('matched_noise_lower_95','mean'),matched_noise_upper_95=('matched_noise_upper_95','mean'))
    pixels=[]
    for identity,(distance,times,threshold) in walls.items():
        row,column=np.indices(distance.shape)
        pixels.append(pd.DataFrame(dict(identity=identity,first_hours=times[column.ravel()],second_hours=times[row.ravel()],state_distance=distance.ravel(),threshold=threshold)))
    return PreparedViews({'wall':dict(table=pd.concat(pixels,ignore_index=True),walls=walls),
        'rate':dict(table=figure_data,aggregate=aggregate),
        'quantified':dict(table=quantification.sort_values('repeated_sequence_fraction'))},
        auxiliary={'recurrence_quantification.csv':quantification},
        wording=dict(title='Returns to a previously measured cell state',footnote='Distances use the saved recurrence settings and thresholds. The diagonal is excluded from recurrence rates. Noise bands retain the saved surrogate comparison.')),None
