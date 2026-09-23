"""Independent observation clocks survive preparation and delayed pairing."""
from pymicroglia._results import read_document
import json
import numpy as np
import pandas as pd
import pytest
from pymicroglia.pipelines._contracts import content_id
from pymicroglia.pipelines.relationships.inputs import match_observations, paired_support
from pymicroglia.pipelines.relationships.options import resolve_request, run_request
from pymicroglia.pipelines._screening import file_hash, read_table
from tests.test_relationship_options import request, lag

def prepared(tmp_path, table, req=None, summary=None, output='pipeline'):
    tmp_path.mkdir(exist_ok=True)
    paths = {'cell_frame': tmp_path / 'cell_frame.csv'}
    table.to_csv(paths['cell_frame'], index=False)
    tables = {'cell_frame': table}
    if summary is not None:
        paths['cell_summary'] = tmp_path / 'cell_summary.csv'
        summary.to_csv(paths['cell_summary'], index=False)
        tables['cell_summary'] = summary
    resolved = resolve_request(req or request(), source_run='source-one', tables=tables, input_hashes={name: file_hash(path) for name, path in paths.items()})
    execution = run_request(resolved, paths, tmp_path / output, only=('paired-inputs',))
    assert execution.successful, {key: saved.outcome.reason for key, saved in execution.results.items()}
    saved = execution.results['paired-inputs']
    return (resolved, paths, saved, {name: read_table(saved.artifact(name)) for name in ('inventory', 'traces', 'trace_inventory', 'scalars', 'same_time_pairs', 'support', 'lag_support')})

def table(hours, left, right, movie='a', identity=1):
    return pd.DataFrame({'stem': movie, 'identity': identity, 'frame_index': range(len(hours)), 'hours': hours, 'corrected_mean': left, 'area_px': right})

def two_point_request(**changes):
    return request(support={'min_observations': 2, 'min_span_hours': 1, 'max_gap_hours': 3, 'matching': 'exact', 'matching_tolerance_hours': 0}, **changes)

def test_shift_precedes_complete_cases_and_original_gaps_survive(tmp_path):
    source = table([0.0, 1.0, 2.0, 3.0], [1.0, np.nan, 2.0, np.nan], [np.nan, 4.0, np.nan, 5.0])
    _, _, saved, data = prepared(tmp_path, source, two_point_request(lag=lag(range_hours=[-1, 1], resolution_hours=1)))
    assert data['same_time_pairs'].empty
    assert len(data['traces']) == 8 and data['traces'].raw_kind.eq('missing').sum() == 4
    row = data['lag_support'].loc[data['lag_support'].lag_hours.eq(-1)].iloc[0]
    assert row.paired_observations == 2 and row.overlap_span_hours == 2 and (row.status == 'eligible')
    traces = data['traces']
    left, right = [traces.loc[traces.measurement.eq(name)] for name in ('corrected_mean', 'area_px')]
    matches, _ = match_observations(left, right, -1, two_point_request().support)
    assert matches[['reference_hours', 'target_hours']].values.tolist() == [[0.0, 1.0], [2.0, 3.0]]
    assert matches.reference_observation.is_unique and matches.target_observation.is_unique
    assert read_document(saved.artifact('provenance'))['scientific_tests_performed'] is False

def test_distinct_recording_intervals_and_pair_reversal_keep_physical_delay(tmp_path):
    a = table(np.arange(0.0, 4.0, 1.0), [1.0, np.nan, 2.0, np.nan], [np.nan, 4.0, np.nan, 5.0])
    b = table(np.arange(0.0, 4.0, 0.5), [1.0, np.nan, np.nan, np.nan, 2.0, np.nan, np.nan, np.nan], [np.nan, np.nan, 4.0, np.nan, np.nan, np.nan, 5.0, np.nan], movie='b')
    req = two_point_request(lag=lag(range_hours=[-1, 1], resolution_hours=1))
    _, _, _, data = prepared(tmp_path, pd.concat([a, b], ignore_index=True), req)
    selected = data['lag_support'].loc[data['lag_support'].lag_hours.eq(-1)]
    assert selected.paired_observations.tolist() == [2, 2]
    assert set(selected.movie) == {'a', 'b'}
    traces = data['traces'].loc[data['traces'].movie.eq('a')]
    left, right = [traces.loc[traces.measurement.eq(name)] for name in ('corrected_mean', 'area_px')]
    forward, _ = match_observations(left, right, -1, req.support)
    reverse, _ = match_observations(right, left, 1, req.support)
    assert reverse.target_hours.tolist() == forward.reference_hours.tolist()
    assert reverse.reference_hours.tolist() == forward.target_hours.tolist()

def test_complete_inventory_retains_absent_constant_and_invalid_values(tmp_path):
    source = table([0.0, 1.0, 2.0], [2.0, 2.0, 2.0], [1.0, np.inf, np.nan])
    summary = pd.DataFrame({'stem': ['a', 'a'], 'identity': [1, 2], 'area_px_median': [1.0, 0.0]})
    _, _, _, data = prepared(tmp_path, source, summary=summary)
    assert len(data['inventory']) == 2 and len(data['trace_inventory']) == 4
    assert len(data['support']) == 6
    assert data['traces'].raw_kind.eq('positive_infinity').sum() == 1
    missing = data['trace_inventory'].loc[data['trace_inventory'].identity.eq(2)]
    assert missing.status.eq('missing').all() and missing.valid_processed.eq(0).all()
    constant = data['trace_inventory'].loc[data['trace_inventory'].identity.eq(1) & data['trace_inventory'].measurement.eq('corrected_mean')]
    assert constant.status.eq('prepared').all()

def test_declared_time_range_is_half_open_and_preserves_source_observations(tmp_path):
    source = table([50.0, 51.0, 52.0, 53.0], [1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0])
    _, _, _, data = prepared(tmp_path, source, two_point_request(time_range_hours=[51.0, 53.0]))
    assert sorted(data['traces'].hours.unique()) == [50.0, 51.0, 52.0, 53.0]
    assert sorted(data['traces'].loc[data['traces'].within_range, 'hours'].unique()) == [51.0, 52.0]
    assert data['same_time_pairs'].reference_hours.tolist() == [51.0, 52.0]

def test_matching_ties_and_collisions_never_duplicate_a_target():

    def points(hours):
        return pd.DataFrame({'hours': hours, 'observation_id': [content_id(h) for h in hours], 'processed_value': np.arange(len(hours)), 'within_range': True, 'processed_valid': True})
    support = {**two_point_request().support, 'matching': 'nearest_unique', 'matching_tolerance_hours': 0.2}
    matches, info = match_observations(points([0.0]), points([-0.1, 0.1]), 0, support)
    assert matches.empty and info['ambiguous_observations'] == 1
    matches, info = match_observations(points([-0.1, 0.1]), points([0.0]), 0, support)
    assert matches.empty and info['ambiguous_observations'] == 2
    matches, _ = match_observations(points([0.3]), points([0.2]), 0.1, two_point_request().support)
    assert len(matches) == 1
    matches, _ = match_observations(points([0.3]), points([0.200001]), 0.1, two_point_request().support)
    assert matches.empty

def test_duplicate_times_cannot_create_multiple_observations(tmp_path):
    source = table([0.0, 1.0, 1.0, 2.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
    _, _, _, data = prepared(tmp_path, source)
    assert data['trace_inventory'].status.eq('invalid').all()
    assert data['same_time_pairs'].empty and len(data['traces']) == 8

def test_native_detrending_preserves_late_clock_gaps_and_settings_without_fits(tmp_path, monkeypatch):
    from pymicroglia import workbench as circadian
    monkeypatch.setattr(circadian, 'estimate_one', lambda *a, **k: pytest.fail('A relationship preparation fitted a period'))
    hours = np.arange(50.0, 62.0)
    left = np.arange(12.0) + np.sin(np.arange(12.0))
    left[4] = np.nan
    source = table(hours, left, np.arange(12.0) * 3 + np.cos(np.arange(12.0)))
    req = request(representation='detrended', detrending={'detrend': 'linear', 'detrend_window_hours': 7.5})
    resolved, paths, saved, data = prepared(tmp_path, source, req)
    traces = data['traces'].loc[data['traces'].measurement.eq('corrected_mean')]
    assert traces.hours.tolist() == hours.tolist()
    assert np.isnan(traces.iloc[4].processed_value) and (not traces.iloc[4].processed_valid)
    assert data['trace_inventory'].status.eq('prepared').all()
    details = read_document(saved.artifact('processing'))
    assert details[0]['settings']['detrend_window_hours'] == 7.5
    result = details[0]['workbench_result']
    assert result['processed_trace']['hours'] == hours.tolist()
    assert result['workbench_processed_trace']['hours'][0] == 0
    monkeypatch.setattr(circadian, 'detrend_trace', lambda *a, **k: pytest.fail('Reopening repeated preprocessing'))
    reopened = run_request(resolved, paths, tmp_path / 'pipeline', only=('paired-inputs',), presentation={'columns': 1})
    assert reopened.successful and reopened.results['paired-inputs'].outcome.status == 'reused'

def test_empty_trace_tables_keep_schema_and_expected_cells(tmp_path):
    source = table([], [], [])
    summary = pd.DataFrame({'stem': ['a'], 'identity': [1], 'area_px_median': [1.0]})
    _, _, _, data = prepared(tmp_path, source, summary=summary)
    assert len(data['inventory']) == 1 and len(data['trace_inventory']) == 2
    assert data['traces'].empty and 'observation_id' in data['traces'].columns
    assert data['same_time_pairs'].empty and 'reference_hours' in data['same_time_pairs'].columns

def test_processing_cannot_supply_observations_on_a_replaced_clock(tmp_path, monkeypatch):
    from pymicroglia import workbench as circadian
    monkeypatch.setattr(circadian, 'detrend_trace', lambda *a, **k: {'processed_trace': {'hours': [0.0, 1.0, 2.0]}, 'values': [1.0, 2.0, 3.0]})
    source = table([50.0, 51.0, 52.0], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
    req = request(representation='detrended', detrending={'detrend': 'linear'})
    _, _, _, data = prepared(tmp_path, source, req)
    assert data['trace_inventory'].status.eq('processing-unavailable').all()
    assert data['same_time_pairs'].empty and (not data['traces'].processed_valid.any())
    support = data['support'].set_index('question')
    assert support.loc['within_cell', 'status'] == 'processing-unavailable'
    assert 'changed the original observation clock' in support.loc['within_cell', 'reason']

def test_between_cell_summary_support_does_not_require_simultaneous_values(tmp_path):
    source = table([0.0, 1.0, 2.0, 3.0], [1.0, np.nan, 2.0, np.nan], [np.nan, 4.0, np.nan, 5.0])
    req = two_point_request(measurements=[{'column': 'corrected_mean', 'summary': 'mean'}, {'column': 'area_px', 'summary': 'mean'}], between_cells={'enabled': True, 'statistic': 'pearson', 'evidence': {'method': 'none'}, 'experimental_unit': 'cell'})
    _, _, _, data = prepared(tmp_path, source, req)
    statuses = data['support'].set_index('question').status.to_dict()
    assert statuses['between_cells'] == 'eligible' and statuses['within_cell'] == 'insufficient'
