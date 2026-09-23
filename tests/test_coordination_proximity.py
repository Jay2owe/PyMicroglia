"""Moving-distance comparisons concern the same pair and preserve window support."""
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
import pymicroglia.measure.relationship_statistics as native
import pymicroglia.pipelines.coordination.proximity as proximity, pymicroglia.pipelines.coordination.simultaneous as simultaneous, pymicroglia.pipelines.coordination.inputs as inputs
from pymicroglia.pipelines.coordination.options import resolve_request, run_request
from pymicroglia.pipelines.coordination.temporal_inputs import ADJUSTMENT_MEANING
from pymicroglia.pipelines._screening import file_hash, read_table
from tests.test_coordination_options import request, resolve
from tests.test_coordination_simultaneous import fixture, prepared

def question(formal=False, **changes):
    return {'enabled': True, 'statistic': 'spearman', 'window_hours': 8.0, 'step_hours': 4.0, 'coordination': 'simultaneous', 'min_windows': 3, 'min_distance_range': 0.1, 'evidence': {'method': 'truncated_time_shift', 'radius_hours': 96.0, 'stationary_series': 'distance', 'stationarity_justification': 'Controlled stationary original position and window-summary software processes'} if formal else {'method': 'none'}, **changes}

def proximity_fixture(positions=None, values=None, q=None, **changes):
    n = 120 if values is None else len(values[0])
    distance = np.linspace(2.0, 40.0, n) if positions is None else np.asarray(positions)
    if values is None:
        x = np.tile([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 0.0], 15)
        z = np.tile(np.array([1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0, 0.0]) * np.sqrt(7.0), 15)
        rho = np.repeat(np.linspace(0.95, -0.95, 15), 8)
        values = [x, rho * x + np.sqrt(1 - rho ** 2) * z]
    _, tables, _ = fixture(values=values)
    frame = tables['cell_frame']
    frame.loc[frame.identity.eq(1), 'centroid_x'] = 0.0
    frame.loc[frame.identity.eq(2), 'centroid_x'] = distance
    q = q or question()
    sim = {'enabled': True, 'statistic': 'pearson', 'evidence': {'method': 'none'}}
    questions = {'simultaneous': sim, 'proximity': q}
    if q['coordination'] == 'delay':
        questions = {'delay': {'enabled': True, 'statistic': 'pearson', 'range_hours': [-1.0, 1.0], 'resolution_hours': 0.5, 'evidence': {'method': 'none'}, 'peak_resolution': {}}, 'proximity': q}
    req = request(target_measurements=['custom_signal'], questions=questions, **{'inference': {'alpha': 0.05, 'multiple_testing': 'bonferroni', 'correction_scope': 'question'}} if q['evidence']['method'] != 'none' else {}, **changes)
    resolved = resolve(tables, req)
    source = prepared(resolved, tables)
    temporal, _ = simultaneous.analyse(resolved, source, 'temporal')
    temporal['provenance'] = {'adjustment_interpretation': ADJUSTMENT_MEANING}
    return (resolved, tables, source, temporal)

def test_approach_history_uses_actual_positions_and_fixed_overlapping_windows():
    resolved, tables, source, temporal = proximity_fixture()
    output = proximity.analyse(resolved, source, temporal, 'science')
    windows = output['windows']
    result = output['pair_effects'].iloc[0]
    assert len(windows) == 13 and windows.window_index.tolist() == list(range(13))
    assert windows.start_hours.iloc[0] == 50.0 and windows.end_hours.iloc[0] == 58.0
    assert windows.joint_observations.eq(16).all() and windows.complete_original_support.all()
    assert not windows.window_is_independent_replicate.any() and windows.p_value.isna().all()
    distance = tables['cell_frame'].query('identity == 2').centroid_x.to_numpy()
    assert windows.distance.iloc[0] == pytest.approx(distance[:16].mean())
    assert windows.distance.iloc[1] == pytest.approx(distance[8:24].mean())
    assert result.status == 'descriptive' and result.effect < -0.5 and (result.p_value is None)
    assert len(output['window_observations']) == 13 * 16
    assert output['window_lag_profiles'].empty

def test_only_between_pair_distance_differences_never_create_within_pair_effects():
    rng = np.random.default_rng(333)
    for separation in [2.0, 200.0]:
        resolved, _, source, temporal = proximity_fixture(positions=np.full(120, separation), values=rng.normal(size=(2, 120)))
        result = proximity.analyse(resolved, source, temporal, 'science')['pair_effects'].iloc[0]
        assert result.status == 'unresolvable' and result.effect is None and (result.p_value is None)
        assert result.mean_pair_distance == separation and result.distance_range == 0.0

def test_missing_geometry_is_excluded_at_its_original_time_and_never_interpolated():
    resolved, tables, _, _ = proximity_fixture()
    frame = tables['cell_frame']
    frame.loc[frame.identity.eq(2) & frame.frame_index.eq(20), 'centroid_x'] = np.nan
    source = prepared(resolved, tables)
    temporal, _ = simultaneous.analyse(resolved, source, 'temporal')
    temporal['provenance'] = {'adjustment_interpretation': ADJUSTMENT_MEANING}
    output = proximity.analyse(resolved, source, temporal, 'science')
    assert len(output['excluded_observations']) == 1
    assert output['excluded_observations'].iloc[0].reference_hours == 60.0
    assert not output['window_observations'].reference_hours.eq(60.0).any()
    assert output['windows'].loc[output['windows'].window_index.isin([1, 2]), 'joint_observations'].eq(15).all()
    assert not output['windows'].loc[output['windows'].window_index.isin([1, 2]), 'complete_original_support'].any()

def test_window_series_native_test_retains_overlap_and_uses_declared_physical_radius():
    rng = np.random.default_rng(6)
    values = rng.normal(size=100)
    rows = [{'hours': 50.0 + 4 * i, 'distance': value + 10.0, 'coordination': -value, 'joint_observations': 16, 'complete_original_support': True} for i, value in enumerate(values)]
    q = question(True)
    result = proximity.window_evidence(rows, {**q, 'coordination_measure': 'signed_coefficient'})
    assert result['status'] == 'tested' and result['p_value'] <= 0.05
    assert result['effect'] == pytest.approx(-1.0) and result['tested_windows'] == 52
    assert result['tested_start_hours'] == 146.0 and result['tested_end_hours'] == 350.0
    assert result['native_details']['stationary_series'] == 'reference'
    rows[30]['complete_original_support'] = False
    failed = proximity.window_evidence(rows, q)
    assert failed['status'] == 'untestable' and failed['p_value'] is None
    assert failed['descriptive_effect'] == pytest.approx(-1.0)

def test_lagged_window_measure_retains_the_complete_grid_and_no_window_delay_claim():
    resolved, _, source, temporal = proximity_fixture(q=question(coordination='delay'))
    output = proximity.analyse(resolved, source, temporal, 'science')
    assert len(output['window_lag_profiles']) == len(output['windows']) * 5
    assert output['windows'].coordination_measure.eq('maximum_absolute_lag_coefficient').all()
    assert output['window_lag_profiles'].p_value.isna().all() and (not output['window_lag_profiles'].delay_supported.any())
    for window in output['windows'].to_dict('records'):
        values = output['window_lag_profiles'].loc[output['window_lag_profiles'].window_id.eq(window['window_id'])]
        assert window['coordination'] == pytest.approx(values.effect.abs().max())

def test_too_few_windows_and_empty_cells_retain_explicit_results():
    resolved, _, source, temporal = proximity_fixture(q=question(window_hours=45.0, step_hours=4.0))
    result = proximity.analyse(resolved, source, temporal, 'science')['pair_effects'].iloc[0]
    assert result.valid_windows == 4
    resolved, _, source, temporal = proximity_fixture(q=question(window_hours=55.0, step_hours=4.0))
    result = proximity.analyse(resolved, source, temporal, 'science')['pair_effects'].iloc[0]
    assert result.status == 'insufficient' and result.valid_windows == 2 and (result.p_value is None)
    temporal['pair_views'] = temporal['pair_views'].iloc[:0]
    output = proximity.analyse(resolved, source, temporal, 'science')
    assert output['pair_effects'].empty and {'window_id', 'distance', 'coordination'} <= set(output['windows'])

def test_simultaneous_proximity_does_not_open_a_delay_result_and_reopens_without_science(tmp_path):
    resolved, tables, _, _ = proximity_fixture()
    paths = {}
    for name, frame in tables.items():
        paths[name] = tmp_path / (name + '.csv')
        frame.to_csv(paths[name], index=False)
    resolved = resolve_request(resolved.request, source_run='source', tables=tables, input_hashes={name: file_hash(path) for name, path in paths.items()})
    with patch('pymicroglia.pipelines.coordination.delay.produce', side_effect=AssertionError('No delay result required')):
        first = run_request(resolved, paths, tmp_path / 'pipeline', only=['changing-proximity'])
    assert first.successful, {key: value.outcome.reason for key, value in first.results.items()}
    assert 'lagged-coordination' not in first.results
    assert len(read_table(first.results['changing-proximity'].artifact('windows'))) == 13
    with patch.object(proximity, 'analyse', side_effect=AssertionError('No repeated proximity analysis')), patch.object(simultaneous, 'analyse', side_effect=AssertionError('No repeated preparation')):
        second = run_request(resolved, paths, tmp_path / 'pipeline', only=['changing-proximity'])
    assert second.successful and all((value.outcome.status == 'reused' for value in second.results.values()))
