"""Shared measured-cell selection, native clocks and declared preprocessing."""
from __future__ import annotations

from copy import deepcopy
import numpy as np
import pandas as pd

from .. import workbench
from .radial import _selected_identities


def identities(frame, requested='all'):
    selected = (sorted(int(x) for x in frame.identity.dropna().unique())
                if str(requested).lower() == 'all' else _selected_identities(frame, requested))
    if not selected or len(set(selected)) != len(selected):
        raise ValueError('cells must name distinct measured cell numbers')
    return selected


def recording_clock(frame):
    required = {'identity', 'hours', 'frame_index'}
    if required - set(frame):
        raise ValueError('cell_frame.csv needs identity, hours and frame_index')
    clock = frame[['frame_index', 'hours']].drop_duplicates().sort_values('frame_index', kind='stable')
    if clock.frame_index.duplicated().any() or clock.hours.duplicated().any():
        raise ValueError('Native frame indices must identify one unique recording time')
    if not np.isfinite(clock.hours.to_numpy(float)).all() or not np.all(np.diff(clock.hours) > 0):
        raise ValueError('Recording hours must be finite and increasing')
    return clock


def native_cell(frame, identity, clock):
    cell = frame.loc[frame.identity.eq(identity)]
    if cell.frame_index.duplicated().any():
        raise ValueError(f'Cell {identity}: duplicate frame observations')
    return clock.merge(cell.drop(columns='hours'), on='frame_index', how='left', validate='one_to_one')


def preprocessing_contract():
    return {
        'mean_frames': {'name': 'mean_frames', 'type': 'int', 'default': 1, 'units': 'frames',
                        'description': 'Centered mean of an odd number of consecutive native frames; missing observations stay missing.'},
        'detrend': {'name': 'detrend', 'type': 'str', 'default': 'none', 'units': '-',
                    'choices': list(workbench.DETREND_METHODS), 'description': 'Workbench detrending applied after the frame mean.'},
        'detrend_options': {'name': 'detrend_options', 'type': 'mapping', 'default': {}, 'units': '-',
                            'description': 'Keyword options of the public Workbench trace.detrend method.'},
    }


def recipe(value):
    if value is None:
        return None
    contract = preprocessing_contract()
    if not isinstance(value, dict) or set(value) - set(contract):
        raise ValueError('preprocessing accepts ' + ', '.join(contract))
    resolved = {key: deepcopy(row['default']) for key, row in contract.items()}
    resolved.update(deepcopy(value))
    count = resolved['mean_frames']
    if isinstance(count, bool) or not isinstance(count, int) or count < 1 or count > 101 or count % 2 == 0:
        raise ValueError('mean_frames must be an odd integer from 1 to 101')
    if resolved['detrend'] not in workbench.DETREND_METHODS:
        raise ValueError('Choose detrend from ' + ', '.join(workbench.DETREND_METHODS))
    from inspect import signature
    accepted = set(signature(workbench.cw.trace([0., 1.], [1., 2.]).detrend).parameters) - {'method'}
    if not isinstance(resolved['detrend_options'], dict) or set(resolved['detrend_options']) - accepted:
        raise ValueError('Unknown Workbench detrend_options')
    return resolved


def frame_mean(values, frames, count=3):
    """Strict centered means; never bridge a missing frame or missing value."""
    from ..measure.signal import centered_frame_mean
    return centered_frame_mean(np.asarray(values, float), np.asarray(frames, int), window=count)


def process(hours, values, frames, resolved, *, name='cell trace'):
    raw = np.asarray(values, float)
    if resolved is None:
        return raw.copy(), raw.copy(), ()
    mean = frame_mean(raw, frames, resolved['mean_frames'])
    if resolved['detrend'] == 'none':
        return mean, mean.copy(), ()
    if len(hours) < 2:
        raise workbench.cw.WorkbenchInputError('At least two native observations are needed for detrending')
    interval = workbench.sample_interval_minutes(hours)
    completed = workbench.cw.trace(hours, mean, name=name, settings={'bin_minutes': interval}).detrend(
        resolved['detrend'], **resolved['detrend_options'])
    processed = np.asarray(completed.trace_data.values, float)
    if processed.shape != raw.shape:
        raise ValueError('Workbench preprocessing changed the number of native observations')
    return mean, processed, (completed.run_record,)
