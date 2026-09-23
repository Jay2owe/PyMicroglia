"""Input and result translations; science is called through workbench."""
from __future__ import annotations
from copy import deepcopy
import json
from typing import Any, Sequence
import numpy as np
import pandas as pd
from .. import workbench as wb

def _processed_trace_in_input_time(caller, completed):
    """Express a public processed clock in the caller's recording coordinates."""
    import pandas as pd
    processed = deepcopy(completed.data['processed_trace'])
    original = caller.trace_data.to_recording().frame['timestamp']
    origin = original.iloc[0] - pd.to_timedelta(float(caller.trace_data.hours[0]), unit='h')
    clock = pd.Timestamp(processed['origin']) + pd.to_timedelta(processed['hours'], unit='h')
    processed['hours'] = ((clock - origin) / pd.Timedelta(hours=1)).tolist()
    processed['origin'] = origin.isoformat()
    return processed

def normalize_trace(hours: Sequence[float], values: Sequence[float], method: str='minmax', *, target_min: float=-1.0, target_max: float=1.0, reference_value: float | None=None, reference_start_hours: float | None=None, reference_end_hours: float | None=None, reference_statistic: str='mean', standard_deviation_ddof: int=0, detrended: bool=False, envelope_floor_fraction: float=0.1) -> dict[str, Any]:
    """Normalise one motion trace without duplicating the shared arithmetic.

    ``method`` is a key or alias from :func:`available_normalization_methods`.
    ``target_min`` and ``target_max`` define min–max output bounds. Reference
    methods take either ``reference_value`` or both inclusive reference-window
    bounds; ``reference_statistic`` selects their mean or median.
    ``standard_deviation_ddof`` is 0 for a population z-score and 1 for a
    sample z-score. ``detrended`` guards invalid mean-ratio scaling, while
    ``envelope_floor_fraction`` controls when a decayed fitted envelope becomes
    too small to divide by safely.
    """
    caller = wb.cw.trace(hours, values, name='motion trace')
    original_hours = np.asarray(caller.trace_data.hours, dtype=float)
    offset = float(original_hours[0])
    result = caller.normalize(method=method, target_min=target_min, target_max=target_max, reference_value=reference_value, reference_start_hours=None if reference_start_hours is None else reference_start_hours - offset, reference_end_hours=None if reference_end_hours is None else reference_end_hours - offset, reference_statistic=reference_statistic, standard_deviation_ddof=standard_deviation_ddof, detrended=detrended, envelope_floor_fraction=envelope_floor_fraction)
    original_clock = caller.trace_data.to_recording().frame['timestamp']
    processed_clock = result.trace_data.to_recording().frame['timestamp']
    if len(original_clock) != len(processed_clock) or not np.allclose((processed_clock.to_numpy() - original_clock.to_numpy()) / np.timedelta64(1, 'h'), 0.0, rtol=0.0, atol=1e-09):
        raise ValueError('Workbench normalization changed the supplied observation clock')
    data = deepcopy(dict(result.data))
    data['workbench_processed_trace'] = deepcopy(data['processed_trace'])
    data['processed_trace'] = wb._processed_trace_in_input_time(caller, result)
    data['processed_trace']['hours'] = original_hours.tolist()
    data['workbench_relative_hours'] = data['hours']
    data['hours'] = original_hours.tolist()
    data['time_reference'] = 'caller-supplied recording hours; public processed clock verified'
    if data.get('reference_window') is not None:
        data['workbench_reference_window'] = dict(data['reference_window'])
        data['reference_window'].update(start_hours=reference_start_hours, end_hours=reference_end_hours)
    data['workbench_run_record'] = deepcopy(result.run_record)
    return {**data, 'source': 'circadian_workbench.analysis.normalize_profile'}

def detrend_trace(hours: Sequence[float], values: Sequence[float], params: dict, *, method: str | None=None, window_hours: float | None=None) -> dict[str, Any]:
    """Detrend through Circadian Workbench and keep its audit metadata."""
    detrending = wb.detrend_settings(params, method=method, window_hours=window_hours)
    caller = wb.cw.trace(hours, values)
    completed = caller.detrend(method=detrending['detrend'], window_hours=detrending['detrend_window_hours'], polynomial_degree=detrending['detrend_polynomial_degree'], min_valid_fraction=detrending['detrend_min_valid_fraction'], bandwidth_hours=detrending['detrend_bandwidth_hours'], low_cut_hours=detrending['detrend_low_cut_hours'], high_cut_hours=detrending['detrend_high_cut_hours'], filter_order=detrending['detrend_filter_order'], lowess_fraction=detrending['detrend_lowess_fraction'], lowess_iterations=detrending['detrend_lowess_iterations'], asls_smoothness=detrending['detrend_asls_smoothness'], asls_asymmetry=detrending['detrend_asls_asymmetry'], asls_iterations=detrending['detrend_asls_iterations'])
    data = deepcopy(completed.data)
    data['workbench_processed_trace'] = deepcopy(data['processed_trace'])
    data['processed_trace'] = wb._processed_trace_in_input_time(caller, completed)
    data['time_reference'] = 'caller-supplied recording hours; native processed clock retained separately'
    action_only = {'recording', 'hours', 'raw', 'detrended', 'smooth_window_hours', 'exclude', 'fit'}
    return {**{key: value for key, value in data.items() if key not in action_only}, 'workbench_run_record': completed.run_record}

def scale_detrended(detrended: Sequence[float], raw: Sequence[float]) -> np.ndarray:
    """Detrended values in units of their own spread, with flat traces kept flat."""
    return wb.cw.display_values.scale_detrended(detrended,raw)
