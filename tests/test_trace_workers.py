"""Parallel traces preserve ordering, full-family tests and actual progress."""
import numpy as np
import pandas as pd
import pytest

from pymicroglia import workbench
from pymicroglia.figure_tables.all_cell_traces import trace_data
from pymicroglia.measure.modules.rhythms import DEFAULTS


@pytest.mark.parametrize('identities,preprocessing', [((19, 26, 41), None), ((19,), None),
                                                    ((19, 26, 41), {'mean_frames': 3})])
def test_processes_preserve_global_correction_and_cell_order(identities, preprocessing):
    resolved = workbench.resolve_analysis_options(DEFAULTS, {
        **workbench.CIRCADIAN_ANALYSIS_OPTION_DEFAULTS,
        'fit_method': 'lomb', 'significance_method': 'lomb',
        'period_min_hours': 2, 'period_max_hours': 8,
        'min_observations': 6, 'multiple_testing': 'bh'}.get)
    hours = np.arange(32.) / 2
    frame = pd.concat([pd.DataFrame({'identity': identity, 'frame_index': np.arange(32),
                                    'hours': hours, 'area_px': 100 + np.sin(hours * identity),
                                    'signal_mean': 50 + np.cos(hours / 2 + identity)})
                       for identity in identities], ignore_index=True)
    if preprocessing is not None:
        # The newly enabled preprocessing path must retain clock gaps and an
        # entirely unavailable measurement, including its unknown result.
        frame = frame.loc[~(frame.identity.eq(identities[0]) & frame.frame_index.isin([5, 6]))].copy()
        frame.loc[frame.identity.eq(identities[-1]), 'signal_mean'] = np.nan
        frame.loc[frame.frame_index.eq(12), 'area_px'] = np.nan
    events = []
    with workbench.analysis_session():
        serial = trace_data(frame, ['area_px', 'signal_mean'], list(identities), resolved,
                            'raw', 'none', {}, preprocessing=preprocessing)
        parallel = trace_data(frame, ['area_px', 'signal_mean'], list(identities), resolved,
                              'raw', 'none', {}, workers=2, progress=events.append, preprocessing=preprocessing)
    pd.testing.assert_frame_equal(serial[0], parallel[0], check_exact=True)
    pd.testing.assert_frame_equal(serial[1], parallel[1], check_exact=True)
    assert events[0]['stage'] == 'starting_workers'
    assert events[-1]['completed_cells'] == len(identities)
    assert events[-1]['completed'] == events[-1]['total'] == 2 * len(identities)
    assert [event['completed'] for event in events] == sorted(event['completed'] for event in events)
    assert max(len(event['active_cells']) for event in events) <= 2


def test_native_components_preserve_each_measurement_and_its_phase():
    resolved = workbench.resolve_analysis_options(DEFAULTS, {
        **workbench.CIRCADIAN_ANALYSIS_OPTION_DEFAULTS,
        'fit_method': 'fft_nlls', 'multiple_testing': 'none'}.get)
    hours = np.arange(145.) / 2
    frame = pd.concat([pd.DataFrame({
        'identity': identity, 'frame_index': np.arange(len(hours)), 'hours': hours,
        'signal_mean': 100 + 20 * np.cos(2 * np.pi * (hours - phase) / 24),
        'area_px': 200 + 30 * np.cos(2 * np.pi * (hours - phase - 5) / 24)})
        for identity, phase in ((19, 6), (26, 13))], ignore_index=True)
    events = []
    settings = dict(fft_component_test=True, component_surrogates=9, recording='test-metrics')
    with workbench.analysis_session():
        serial = trace_data(frame, ['area_px', 'signal_mean'], [19, 26], resolved,
                            'raw', 'none', {}, **settings)
        parallel = trace_data(frame, ['area_px', 'signal_mean'], [19, 26], resolved,
                              'raw', 'none', {}, workers=2, progress=events.append, **settings)
    pd.testing.assert_frame_equal(serial[0], parallel[0], check_exact=True)
    columns = [column for column in serial[1] if 'run_record' not in column]
    pd.testing.assert_frame_equal(serial[1][columns], parallel[1][columns], check_exact=True)
    assert events[-1]['completed'] == 4
    assert events[-1]['completed_cells'] == 2
    assert list(zip(parallel[1].identity, parallel[1].metric)) == [
        (19, 'area_px'), (19, 'signal_mean'), (26, 'area_px'), (26, 'signal_mean')]
