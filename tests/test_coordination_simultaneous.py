"""Cross-cell temporal evidence retains original clocks and reference contributors."""
import copy
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
import pymicroglia.measure.relationship_statistics as native
import pymicroglia.pipelines.coordination.inputs as inputs, pymicroglia.pipelines.coordination.simultaneous as simultaneous
from pymicroglia.pipelines.coordination.temporal_inputs import prepare_series
from pymicroglia.pipelines.coordination.options import resolve_request, run_request
from pymicroglia.pipelines._screening import file_hash
from tests.test_coordination_options import request, resolve

def question(formal=False):
    return {'enabled': True, 'statistic': 'pearson', 'evidence': {'method': 'truncated_time_shift', 'radius_hours': 12.0, 'stationary_series': 'target', 'stationarity_justification': 'Controlled stationary software-test sequences'} if formal else {'method': 'none'}}

def fixture(values=None, formal=False, **changes):
    rng = np.random.default_rng(921)
    base = rng.normal(size=180)
    values = np.asarray([base, base, -base]) if values is None else np.asarray(values, float)
    frame = pd.DataFrame([{'stem': 'movie', 'identity': cell + 1, 'frame_index': i, 'hours': 50.0 + 0.5 * i, 'custom_signal': value, 'centroid_x': float(5 * cell), 'centroid_y': 0.0} for cell, trace in enumerate(values) for i, value in enumerate(trace)])
    tables = {'cell_frame': frame, 'cell_summary': frame[['stem', 'identity']].drop_duplicates()}
    req = request(target_measurements=['custom_signal'], questions={'simultaneous': question(formal)}, **{'inference': {'alpha': 0.05, 'multiple_testing': 'bonferroni', 'correction_scope': 'question'}} if formal else {}, **changes)
    resolved = resolve(tables, req)
    return (resolved, tables, prepared(resolved, tables))

def prepared(resolved, tables):
    result, definitions = inputs.prepare(resolved, tables)
    result.update(pair_definitions=definitions, provenance={'scientific_id': 'controlled-input'})
    return result

def test_signed_pairs_and_native_central_evidence_keep_full_endpoint_keys():
    resolved, _, source = fixture(formal=True)
    output, _ = simultaneous.analyse(resolved, source, 'science')
    effects = output['pair_effects']
    assert len(effects) == 3 and len(output['pair_views']) == 3
    assert effects.reference.eq('custom_signal').all() and effects.target.eq('custom_signal').all()
    assert effects.evidence_level.eq('pair').all() and effects.status.eq('tested').all()
    assert effects.p_value.le(0.05).all()
    assert effects.effect.tolist() == pytest.approx([1.0, -1.0, -1.0])
    assert effects.tested_start_hours.eq(62.0).all() and effects.tested_end_hours.eq(127.5).all()
    assert effects.tested_observations.eq(132).all() and effects.paired_observations.eq(180).all()
    assert len(output['shift_reference']) == 3 * 49
    assert effects.confidence_low.isna().all()

def test_raw_support_survives_an_unavailable_leave_pair_out_reference():
    resolved, _, source = fixture(values=np.random.default_rng(3).normal(size=(2, 30)), shared_reference={'method': 'leave_pair_out_mean', 'min_cells': 1, 'adjustment': 'subtract'})
    output, _ = simultaneous.analyse(resolved, source, 'science')
    raw, adjusted = [output['pair_effects'].loc[output['pair_effects'].adjustment.eq(key)].iloc[0] for key in ['none', 'shared_reference']]
    assert raw.status == 'descriptive' and adjusted.status == 'insufficient'
    assert np.isnan(adjusted.effect) and adjusted.paired_observations == 0
    assert output['reference_membership'].empty
    assert output['reference_summary'].members.eq(0).all()

def test_leave_pair_out_membership_excludes_both_targets_and_tracks_disappearance():
    values = np.random.default_rng(77).normal(size=(4, 50))
    values[3, 10] = np.nan
    resolved, _, source = fixture(values=values, shared_reference={'method': 'leave_pair_out_mean', 'min_cells': 2, 'adjustment': 'subtract'})
    output, _ = simultaneous.analyse(resolved, source, 'science')
    pair = output['pair_views'].query('reference_identity == 1 and target_identity == 2').iloc[0]
    members = output['reference_membership'].loc[output['reference_membership'].pair_id.eq(pair.pair_id)]
    assert set(members.donor_identity) == {3, 4}
    refs = output['reference_summary'].loc[output['reference_summary'].pair_id.eq(pair.pair_id)]
    assert refs.loc[refs.hours.eq(55.0), 'members'].eq(1).all()
    assert refs.loc[refs.hours.eq(55.0), 'reference_value'].isna().all()
    assert refs.loc[~refs.hours.eq(55.0), 'members'].eq(2).all()
    assert len(output['pair_effects']) == 12
    assert output['pair_effects'].view_id.nunique() == 12

def test_subtracting_a_shared_reference_can_induce_association():
    rng = np.random.default_rng(777)
    independent = rng.normal(size=(2, 300))
    values = np.vstack([independent, rng.normal(scale=8.0, size=300)])
    resolved, _, source = fixture(values=values, shared_reference={'method': 'leave_pair_out_mean', 'min_cells': 1, 'adjustment': 'subtract'})
    output, _ = simultaneous.analyse(resolved, source, 'science')
    pair = output['pair_effects'].query('reference_identity == 1 and target_identity == 2')
    assert abs(pair.loc[pair.adjustment.eq('none'), 'effect'].iloc[0]) < 0.15
    assert pair.loc[pair.adjustment.eq('shared_reference'), 'effect'].iloc[0] > 0.95
    assert 'induce association' in pair.loc[pair.adjustment.eq('shared_reference'), 'shared_reference_interpretation'].iloc[0]

@pytest.mark.parametrize('problem', ['missing_value', 'frame_gap', 'unequal_clock', 'invalid_clock'])
def test_inference_refuses_original_gaps_or_incompatible_clocks(problem):
    resolved, tables, _ = fixture(formal=True)
    frame = tables['cell_frame']
    selected = frame.identity.eq(2)
    if problem == 'missing_value':
        frame.loc[selected & frame.frame_index.eq(30), 'custom_signal'] = np.nan
    elif problem == 'frame_gap':
        frame.loc[selected, 'frame_index'] *= 2
    elif problem == 'unequal_clock':
        frame.loc[selected, 'hours'] += 0.1
    else:
        frame.loc[selected & frame.frame_index.eq(30), 'hours'] = 50.0
    output, _ = simultaneous.analyse(resolved, prepared(resolved, tables), 'science')
    affected = output['pair_effects'].query('reference_identity == 2 or target_identity == 2')
    assert affected.p_value.isna().all()
    assert not affected.status.eq('tested').any()
    assert len(affected) == 2

def test_measured_increments_preserve_original_intervals_and_missing_endpoints():
    values = np.array([[0, 1, 3, np.nan, 7, 8, 9, 10], [0, 2, 6, 8, 14, 16, 18, 20]])
    resolved, _, source = fixture(values=values, representation='changes')
    output, _ = simultaneous.analyse(resolved, source, 'science')
    trace = output['series'].query('identity == 1 and representation == "changes"').sort_values('hours')
    assert trace.processed_value.tolist()[:3] == pytest.approx([np.nan, 1, 2], nan_ok=True)
    assert trace.loc[trace.frame_index.isin([3, 4]), 'processed_value'].isna().all()
    assert trace.loc[trace.frame_index.gt(0), 'increment_hours'].eq(0.5).all()
    assert not trace.iloc[0].within_range and trace.iloc[0].original_within_range
    assert output['pair_effects'].representation.tolist() == ['raw', 'changes']
    assert output['pair_effects'].is_hypothesis.tolist() == [False, True]

def test_native_detrending_is_recorded_and_restores_original_missing_mask():
    resolved, tables, _ = fixture(values=np.random.default_rng(9).normal(size=(2, 30)), representation='detrended', detrending={'detrend': 'linear'})
    frame = tables['cell_frame']
    frame.loc[frame.frame_index.eq(9) & frame.identity.eq(1), 'custom_signal'] = np.nan
    output, processing = simultaneous.analyse(resolved, prepared(resolved, tables), 'science')
    detrended = output['series'].query('identity == 1 and representation == "detrended"')
    assert detrended.loc[detrended.frame_index.eq(9), 'processed_value'].isna().all()
    records = [row for row in processing if row['representation'] == 'detrended']
    assert all((row['workbench_result'] is not None for row in records))
    assert len(output['pair_effects']) == 2

def test_disabled_simultaneous_prepares_delay_without_testing():
    delay = {'enabled': True, 'statistic': 'pearson', 'evidence': {'method': 'none'}, 'range_hours': [-2.0, 2.0], 'resolution_hours': 0.5, 'peak_resolution': {}}
    resolved, tables, _ = fixture()
    req = request(target_measurements=['custom_signal'], questions={'delay': delay})
    resolved = resolve(tables, req)
    with patch.object(native, 'coefficient', side_effect=AssertionError('No simultaneous calculation')):
        output, _ = simultaneous.analyse(resolved, prepared(resolved, tables), 'science')
    assert output['pair_effects'].empty and len(output['pair_views']) == 3
    assert not output['series'].empty

def test_empty_and_single_cell_populations_preserve_required_temporal_schema():
    resolved, tables, _ = fixture(values=np.zeros((1, 10)))
    for empty in [False, True]:
        if empty:
            tables = {name: frame.iloc[:0] for name, frame in tables.items()}
            resolved = resolve(tables, resolved.request)
        output, _ = simultaneous.analyse(resolved, prepared(resolved, tables), 'science')
        assert output['pair_effects'].empty and output['pair_views'].empty
        assert {'series_id', 'endpoint_id', 'measurement_id', 'hours'} <= set(output['series'])

def test_completed_temporal_artifacts_reopen_without_science(tmp_path):
    resolved, tables, _ = fixture(values=np.random.default_rng(44).normal(size=(3, 30)))
    paths = {}
    for name, frame in tables.items():
        path = tmp_path / (name + '.csv')
        frame.to_csv(path, index=False)
        paths[name] = path
    resolved = resolve_request(resolved.request, source_run='source', tables=tables, input_hashes={name: file_hash(path) for name, path in paths.items()})
    first = run_request(resolved, paths, tmp_path / 'pipeline', only=['simultaneous-coordination'])
    assert first.successful
    with patch.object(simultaneous, 'analyse', side_effect=AssertionError('No refitting')), patch.object(inputs, 'prepare', side_effect=AssertionError('No preparation')):
        second = run_request(resolved, paths, tmp_path / 'pipeline', only=['simultaneous-coordination'])
    assert second.successful and all((row.outcome.status == 'reused' for row in second.results.values()))
    reopened = simultaneous.read_temporal(second.results['simultaneous-coordination'])
    assert len(reopened['pair_effects']) == 3

def external_fixture():
    rng = np.random.default_rng(779)
    common = rng.normal(scale=10.0, size=200)
    resolved, tables, _ = fixture(values=common + rng.normal(size=(2, 200)))
    tables['recording_reference'] = pd.DataFrame({'stem': 'movie', 'frame_index': np.arange(200), 'hours': 50.0 + 0.5 * np.arange(200), 'measured_reference': common})
    req = request(target_measurements=['custom_signal'], table_grains={'recording_reference': ['frame_index']}, shared_reference={'method': 'external', 'measurements': {'custom_signal': 'measured_reference'}, 'adjustment': 'subtract', 'units': {'custom_signal': 'declared units', 'measured_reference': 'declared units'}})
    return (req, tables)

def test_original_recording_reference_removes_controlled_common_component_and_preserves_source_identity():
    req, tables = external_fixture()
    resolved = resolve(tables, req)
    assert len(resolved.inputs.cells) == 2 and len(resolved.reference_columns) == 1
    assert 'identity' not in resolved.reference_columns[0].grain
    output, _ = simultaneous.analyse(resolved, prepared(resolved, tables), 'science')
    raw, adjusted = [output['pair_effects'].loc[output['pair_effects'].adjustment.eq(key)].iloc[0] for key in ['none', 'shared_reference']]
    assert raw.effect > 0.98 and abs(adjusted.effect) < 0.15
    members = output['reference_membership']
    assert members.donor_source_scope.eq('recording_reference').all() and members.donor_identity.isna().all()
    assert members.donor_source_observation_id.nunique() == 200 and len(members) == 400
    assert members.donor_table.eq('recording_reference').all()
    refs = output['reference_summary']
    assert refs.members.eq(1).all() and refs.status.eq('available').all()
    tables['recording_reference'].loc[tables['recording_reference'].frame_index.eq(30), 'measured_reference'] = np.nan
    output, _ = simultaneous.analyse(resolved, prepared(resolved, tables), 'science')
    assert output['reference_summary'].loc[output['reference_summary'].hours.eq(65.0), 'reference_value'].isna().all()
    assert output['pair_effects'].loc[output['pair_effects'].adjustment.eq('none'), 'paired_observations'].iloc[0] == 200
    assert output['pair_effects'].loc[output['pair_effects'].adjustment.eq('shared_reference'), 'paired_observations'].iloc[0] == 199

def test_reference_resolution_rejects_unknown_units_wrong_grain_and_ambiguity():
    req, tables = external_fixture()
    declaration = req.declaration.as_dict()
    declaration['shared_reference'].pop('units')
    from pymicroglia.pipelines.coordination.options import CoordinationRequest
    with pytest.raises(ValueError, match='units'):
        resolve(tables, CoordinationRequest.from_dict(declaration, {}))
    declared = req.declaration.as_dict()
    declared['table_grains']['recording_reference'] = ['stem']
    with pytest.raises(ValueError, match='trace'):
        resolve(tables, CoordinationRequest.from_dict(declared, {}))
    tables['cell_frame']['measured_reference'] = 1.0
    with pytest.raises(ValueError, match='ambiguous'):
        resolve(tables, req)
    units = req.declaration.as_dict()
    units['shared_reference']['units']['custom_signal'] = 'different'
    tables['cell_frame'] = tables['cell_frame'].drop(columns=['measured_reference'])
    with pytest.raises(ValueError, match='units'):
        resolve(tables, CoordinationRequest.from_dict(units, {}))

def test_increment_rate_is_explicit_and_cannot_apply_to_levels():
    resolved, _, source = fixture(values=np.array([np.arange(10), 2 * np.arange(10)]), representation='changes', increment='rate')
    frames, _, _ = prepare_series(resolved, source)
    changed = [frame for (_, view), frame in frames.items() if view == 'changes'][0]
    assert changed.loc[changed.processed_valid, 'processed_value'].eq(2.0).all()
    with pytest.raises(ValueError, match='increment'):
        request(increment='rate')
