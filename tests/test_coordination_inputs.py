"""Hand-checkable cross-cell membership, physical clocks and measured geometry."""
import json
import numpy as np
import pandas as pd
import pytest
import pymicroglia.pipelines.coordination.inputs as inputs
from pymicroglia.pipelines.coordination.options import resolve_request, run_request
from pymicroglia.pipelines._screening import file_hash, read_table
from tests.test_coordination_options import declaration, request, resolve, tables

def same_request(**changes):
    return request(target_measurements=['custom_signal'], **changes)

def test_pairs_never_cross_recordings_and_cross_measurement_roles_remain_distinct(tables):
    resolved = resolve(tables)
    outputs, definitions = inputs.prepare(resolved, tables)
    pairs = outputs['inventory']
    assert len(pairs) == 6 and pairs.pair_id.is_unique
    assert all((row['key']['reference']['cell']['movie'] == row['key']['target']['cell']['movie'] for row in definitions))
    for movie, frame in pairs.groupby('movie'):
        assert len(frame.loc[frame.reference.eq(frame.target)]) == 1
        cross = frame.loc[frame.reference.ne(frame.target)]
        assert set(zip(cross.reference_identity, cross.target_identity)) == {(7, 8), (8, 7)}
    assert len(outputs['support']) == 6 * 6
    assert len(outputs['traces']) == 4 * 4 * 8
    assert outputs['traces'].observation_id.is_unique
    assert all((row['key']['reference']['measurement']['table'] == 'cell_frame' for row in definitions))

def test_static_median_layout_is_distinct_from_contemporaneous_distance(tables):
    frame = tables['cell_frame']
    frame['centroid_x'] = 0.0
    frame['centroid_y'] = 0.0
    for movie in ['one', 'two']:
        index = frame.index[frame.stem.eq(movie) & frame.identity.eq(8)]
        frame.loc[index, 'centroid_x'] = [-3, -3, -3, 0, 0, 3, 3, 3]
        frame.loc[index, 'centroid_y'] = 4.0
    geometry = {**declaration()['geometry'], 'unit': 'um', 'scale': [0.5, 0.5]}
    outputs, _ = inputs.prepare(resolve(tables, same_request(geometry=geometry)), tables)
    np.testing.assert_allclose(outputs['inventory'].static_distance, 2.0)
    assert set(outputs['pair_positions'].distance) == {2.0, 2.5}
    assert set(outputs['geometry'].unit) == {'um'}
    assert outputs['geometry'].inside_field.isna().all()
    assert outputs['layouts'].position_meaning.str.contains('not its position at one shared time').all()
    np.testing.assert_allclose(outputs['inventory'].jointly_observed_hours, 3.5)

def test_missing_signal_and_position_have_different_eligibility(tables):
    tables['cell_frame'].loc[tables['cell_frame'].identity.eq(8), 'centroid_x'] = np.nan
    outputs, _ = inputs.prepare(resolve(tables, same_request()), tables)
    assert len(outputs['matches']) == 16
    assert outputs['pair_positions'].empty
    assert outputs['inventory'].static_distance.isna().all()
    support = outputs['support'].query("question == 'simultaneous'")
    assert set(support.status) == {'eligible'} and set(support.geometry_status) == {'insufficient'}
    assert outputs['geometry'].loc[outputs['geometry'].identity.eq(8), 'reason'].str.contains('Missing').all()

def test_removed_frame_and_missing_value_cannot_acquire_interval_exposure(tables):
    frame = tables['cell_frame']
    frame = frame.loc[~(frame.stem.eq('one') & frame.identity.eq(8) & frame.frame_index.eq(3))].copy()
    frame.loc[frame.stem.eq('two') & frame.identity.eq(8) & frame.frame_index.eq(3), 'custom_signal'] = np.nan
    tables['cell_frame'] = frame
    outputs, _ = inputs.prepare(resolve(tables, same_request()), tables)
    np.testing.assert_allclose(outputs['inventory'].jointly_observed_hours, 2.5)
    assert outputs['intervals'].loc[~outputs['intervals'].valid, 'jointly_observed_hours'].eq(0).all()
    assert outputs['matches'].groupby('movie').segment.max().eq(1).all()
    assert outputs['geometry_intervals'].query("movie == 'two'").valid.all()
    assert not outputs['geometry_intervals'].query("movie == 'one'").valid.all()
    assert len(outputs['traces']) == 31 * 3

def test_hour_only_observations_do_not_bridge_an_invalid_middle_value(tables):
    frame = tables.pop('cell_frame').drop(columns='frame_index')
    frame.loc[frame.identity.eq(8) & frame.hours.eq(51.5), 'custom_signal'] = np.nan
    tables['clocked'] = frame
    req = same_request(table_grains={'clocked': ['identity', 'hours']})
    outputs, _ = inputs.prepare(resolve(tables, req), tables)
    np.testing.assert_allclose(outputs['inventory'].jointly_observed_hours, 2.5)
    assert outputs['intervals'].loc[~outputs['intervals'].valid, 'reason'].str.contains('original observation').all()

def test_reset_clocks_nonoverlap_and_summary_only_cells_remain_explicit(tables):
    frame = tables['cell_frame']
    frame.loc[frame.stem.eq('one') & frame.identity.eq(8) & frame.frame_index.eq(3), 'hours'] = 49.0
    frame.loc[frame.stem.eq('two') & frame.identity.eq(8), 'hours'] += 20
    tables['cell_summary'] = pd.concat([tables['cell_summary'], pd.DataFrame([{'stem': 'one', 'identity': 9, 'trait': 9.0}])], ignore_index=True)
    outputs, _ = inputs.prepare(resolve(tables, same_request()), tables)
    assert len(outputs['cells']) == 5 and len(outputs['inventory']) == 4
    assert outputs['matches'].empty
    assert set(outputs['support'].query("question == 'simultaneous'").status) == {'invalid_clock', 'insufficient'}
    assert len(outputs['traces']) == 32 * 3
    assert outputs['endpoints'].loc[outputs['endpoints'].identity.eq(9), 'status'].eq('missing').all()

def test_neighbourhood_uses_geometry_and_preserves_distant_comparisons(tables):
    third = tables['cell_frame'].loc[tables['cell_frame'].identity.eq(8)].copy()
    third['identity'] = 9
    third['centroid_x'] += 100
    tables['cell_frame'] = pd.concat([tables['cell_frame'], third], ignore_index=True)
    geometry = {**declaration()['geometry'], 'neighbourhood': {'method': 'radius', 'distance': 20, 'comparison': 'retain_distant'}}
    req = same_request(geometry=geometry)
    resolved = resolve(tables, req)
    first, _ = inputs.prepare(resolved, tables)
    assert first['inventory'].near.sum() == 2 and first['inventory'].geometry_population_eligible.all()
    tables['cell_frame']['custom_signal'] *= -100
    second, _ = inputs.prepare(resolved, tables)
    assert first['inventory'].equals(second['inventory'])
    geometry['neighbourhood']['comparison'] = 'restrict'
    restricted, _ = inputs.prepare(resolve(tables, same_request(geometry=geometry)), tables)
    assert len(restricted['inventory']) == 6 and restricted['inventory'].geometry_population_eligible.sum() == 2
    assert (restricted['support'].query("question == 'simultaneous'").status == 'outside_geometry_population').sum() == 4

def test_field_edges_and_per_recording_calibration_are_saved(tables):
    geometry = {**declaration()['geometry'], 'unit': 'um', 'scale': [0.5, 0.5], 'recording_scales': {'two': [1.0, 1.0]}, 'field': {'bounds': [0, 45, 0, 45], 'edge_margin': 4}}
    outputs, _ = inputs.prepare(resolve(tables, same_request(geometry=geometry)), tables)
    assert outputs['geometry'].query("movie == 'two'").valid.eq(False).all()
    assert outputs['geometry'].query("movie == 'two'").inside_field.eq(False).all()
    assert outputs['geometry'].query("movie == 'one'").x_scale.eq(0.5).all()
    assert outputs['geometry'].query("movie == 'one'").near_field_edge.any()

def test_original_scalars_are_not_repeated_into_traces(tables):
    req = request(reference_measurements=['trait'], target_measurements=['trait'], questions={'characteristics': {'enabled': True, 'summary': 'median', 'statistic': 'absolute_difference', 'evidence': {'method': 'none'}}})
    outputs, _ = inputs.prepare(resolve(tables, req), tables)
    assert len(outputs['scalars']) == 4
    assert not outputs['traces'].measurement.eq('trait').any()
    assert outputs['matches'].empty
    assert outputs['support'].query("question == 'characteristics'").status.eq('eligible').all()

def test_input_identity_is_independent_of_later_evidence_method(tables):
    first = resolve(tables, same_request())
    second = resolve(tables, same_request(questions={'simultaneous': {'enabled': True, 'statistic': 'spearman', 'evidence': {'method': 'none'}}}))
    assert inputs.input_settings(first) == inputs.input_settings(second)
    assert first.scientific_id != second.scientific_id

def test_staggered_clocks_keep_true_joint_exposure_separate_from_point_matching(tables):
    tables['cell_frame'].loc[tables['cell_frame'].identity.eq(8), 'hours'] += 0.08
    support = {**declaration()['support'], 'matching': 'nearest_unique', 'matching_tolerance_hours': 0.1}
    outputs, _ = inputs.prepare(resolve(tables, same_request(support=support)), tables)
    assert len(outputs['matches']) == 16
    np.testing.assert_allclose(outputs['inventory'].jointly_observed_hours, 3.42)
    np.testing.assert_allclose(outputs['matching_intervals'].groupby('movie').matched_interval_hours.sum(), 2.94)
    exact, _ = inputs.prepare(resolve(tables, same_request()), tables)
    assert exact['matches'].empty
    np.testing.assert_allclose(exact['inventory'].jointly_observed_hours, 3.42)

def test_moving_neighbours_use_actual_positions_and_visibility(tables):
    frame = tables['cell_frame']
    frame['centroid_x'] = 0.0
    frame['centroid_y'] = 0.0
    frame.loc[frame.identity.eq(8), 'centroid_x'] = np.tile([0, 10, 20, 30, 40, 50, 60, 70], 2)
    geometry = {**declaration()['geometry'], 'neighbourhood': {'method': 'radius', 'distance': 25, 'comparison': 'retain_distant'}}
    outputs, _ = inputs.prepare(resolve(tables, same_request(geometry=geometry)), tables)
    assert outputs['inventory'].near.eq(False).all()
    assert outputs['moving_neighbours'].near.sum() == 6
    assert outputs['moving_neighbours'].reference_visible_cells.eq(1).all()
    assert outputs['pair_positions'].near.sum() == 6

def test_cross_measurement_direction_does_not_change_geometry_matching(tables):
    frame = tables['cell_frame'].query("stem == 'one' and frame_index < 3").copy()
    frame = frame.loc[~(frame.identity.eq(7) & frame.frame_index.eq(1))]
    frame.loc[frame.identity.eq(7), 'hours'] = [50.0, 50.5]
    frame.loc[frame.identity.eq(8), 'hours'] = [50.1, 50.2, 50.5]
    tables['cell_frame'] = frame
    tables['cell_summary'] = frame[['stem', 'identity']].drop_duplicates()
    support = {**declaration()['support'], 'matching': 'nearest_unique', 'matching_tolerance_hours': 0.15}
    outputs, _ = inputs.prepare(resolve(tables, request(support=support)), tables)
    cross = outputs['pair_positions'].query('reference != target')
    forward = cross.loc[cross.reference_identity.eq(7)]
    reverse = cross.loc[cross.reference_identity.eq(8)]
    assert set(zip(forward.reference_observation, forward.target_observation)) == set(zip(reverse.target_observation, reverse.reference_observation))
    assert len(forward) == 2

def test_empty_selected_population_keeps_complete_machine_readable_tables(tables):
    outputs, definitions = inputs.prepare(resolve(tables, same_request(cells=[])), tables)
    assert not definitions and outputs['inventory'].empty
    assert {'static_distance', 'geometry_population_eligible', 'jointly_observed_hours'} <= set(outputs['inventory'])
    assert {'reference_value', 'target_value', 'segment'} <= set(outputs['matches'])

def test_real_producer_reuses_inputs_without_opening_optional_sources(tables, tmp_path, monkeypatch):
    from pymicroglia import workbench as circadian

    def forbidden(*a, **k):
        raise AssertionError('Input preparation attempted scientific calculations')
    monkeypatch.setattr(circadian, 'estimate_one', forbidden)
    monkeypatch.setattr(circadian, 'detrend_trace', forbidden)
    paths = {}
    for name, frame in tables.items():
        paths[name] = tmp_path / (name + '.csv')
        frame.to_csv(paths[name], index=False)
    resolved = resolve_request(same_request(), source_run='source', tables=tables, input_hashes={k: file_hash(v) for k, v in paths.items()})
    result = run_request(resolved, paths, tmp_path / 'pipeline', only=('pair-inputs',))
    assert result.successful
    saved = result.results['pair-inputs']
    exported = inputs.read_inputs(saved)
    assert len(exported['inventory']) == 2 and (not exported['provenance']['scientific_tests_performed'])
    before = {ref.name: file_hash(saved.artifact(ref.name)) for ref in saved.outcome.artifacts}
    monkeypatch.setattr(inputs, 'prepare', forbidden)
    reopened = run_request(resolved, paths, tmp_path / 'pipeline', only=('pair-inputs',), presentation={'maps': {'edge_limit': 1}})
    assert reopened.results['pair-inputs'].outcome.status == 'reused'
    changed = resolve_request(same_request(questions={'simultaneous': {'enabled': True, 'statistic': 'spearman', 'evidence': {'method': 'none'}}}), source_run='source', tables=tables, input_hashes={k: file_hash(v) for k, v in paths.items()})
    other = run_request(changed, paths, tmp_path / 'pipeline', only=('pair-inputs',))
    assert other.results['coordination-design'].outcome.scientific_id != result.results['coordination-design'].outcome.scientific_id
    assert other.results['pair-inputs'].outcome.status == 'reused'
    assert all((file_hash(saved.artifact(name)) == fingerprint for name, fingerprint in before.items()))

def test_reader_rejects_an_observation_from_another_cell_even_with_a_fresh_file_hash(tables, tmp_path):
    from dataclasses import replace
    from pymicroglia.pipelines._runner import SavedResult
    from pymicroglia.pipelines._screening import write_table
    paths = {}
    for name, frame in tables.items():
        paths[name] = tmp_path / (name + '.csv')
        frame.to_csv(paths[name], index=False)
    resolved = resolve_request(same_request(), source_run='source', tables=tables, input_hashes={k: file_hash(v) for k, v in paths.items()})
    saved = run_request(resolved, paths, tmp_path / 'pipeline', only=('pair-inputs',)).results['pair-inputs']
    matched = read_table(saved.artifact('matches'))
    matched.loc[0, 'reference_observation'] = matched.loc[0, 'target_observation']
    path = saved.root / 'matches.json'
    path = write_table(path, matched)
    refs = tuple((replace(ref, sha256=file_hash(path)) if ref.name == 'matches' else ref for ref in saved.outcome.artifacts))
    altered = SavedResult(saved.root, replace(saved.outcome, artifacts=refs))
    with pytest.raises(ValueError, match='recorded endpoint'):
        inputs.read_inputs(altered)
