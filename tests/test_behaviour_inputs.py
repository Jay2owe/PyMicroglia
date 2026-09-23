"""State features preserve observations and protect complete validation groups."""
from pymicroglia._results import read_document
import json
import numpy as np
import pandas as pd
import pytest
from pymicroglia.pipelines.behaviour.options import BehaviourRequest, resolve_request, run_request
from pymicroglia.pipelines._screening import file_hash, read_table
from tests.test_behaviour_options import declaration

def fixture(tmp_path, *, change=None, request_changes=None):
    mapping = {'la': 'learn-a', 'lb': 'learn-a', 'lc': 'learn-b', 'd1': 'develop-a', 'd2': 'develop-b', 'c1': 'confirm-a', 'c2': 'confirm-b'}
    roles = {sample: 'learning' if sample.startswith('learn') else 'development' if sample.startswith('develop') else 'confirmation' for sample in mapping.values()}
    rows = []
    for movie in mapping:
        for identity in (7, 8):
            for frame in range(100 if movie == 'lc' else 60):
                rows.append({'stem': movie, 'identity': identity, 'frame_index': frame, 'hours': 50.0 + frame * (0.25 if movie == 'd1' else 0.5), 'custom_signal': float((frame * 7 + identity) % 19), 'custom_shape': frame * 0.1 + identity})
    frame = pd.DataFrame(rows)
    frame.loc[frame.stem.eq('la') & frame.identity.eq(7) & frame.frame_index.eq(5), 'custom_signal'] = np.nan
    frame = frame.loc[~(frame.stem.eq('lb') & frame.identity.eq(8) & frame.frame_index.eq(20))].reset_index(drop=True)
    tables = {'cell_frame': frame}
    block = declaration(biological_samples=mapping)
    block['learning']['max_observations_per_cell'] = 25
    block['validation'].update(split={'method': 'explicit', 'roles': roles}, min_groups={k: 2 for k in ('learning', 'development', 'confirmation')})
    if request_changes:
        block.update(request_changes)
    if change:
        change(tables, block)
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, table in tables.items():
        paths[name] = tmp_path / (name + '.csv')
        table.to_csv(paths[name], index=False)
    request = BehaviourRequest.from_dict(block, {})
    resolved = resolve_request(request, source_run='controlled-state-source', tables=tables, input_hashes={name: file_hash(path) for name, path in paths.items()})
    return (resolved, paths, tables)

def execute(tmp_path, **kwargs):
    resolved, paths, tables = fixture(tmp_path, **kwargs)
    execution = run_request(resolved, paths, tmp_path / 'pipeline', only=('feature-inputs',))
    assert execution.successful, {k: v.outcome.reason for k, v in execution.results.items()}
    saved = execution.results['feature-inputs']
    data = {name: read_table(saved.artifact(name)) for name in ('observations', 'raw_features', 'features', 'feature_masks', 'source_traces', 'feature_inventory', 'cell_inventory', 'partitions', 'learning_members')}
    return (resolved, paths, tables, saved, data)

def test_complete_observation_population_balanced_samples_and_no_model_fit(tmp_path, monkeypatch):
    from pymicroglia import workbench as circadian
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    def forbidden(*a, **k):
        raise AssertionError('Preparation attempted a learned transform or model fit')
    for obj, name in ((circadian, 'estimate_one'), (circadian, 'detrend_trace'), (GaussianMixture, 'fit'), (StandardScaler, 'fit')):
        monkeypatch.setattr(obj, name, forbidden)
    _, _, tables, saved, data = execute(tmp_path)
    obs, members, groups = (data['observations'], data['learning_members'], data['partitions'])
    assert len(obs) == len(tables['cell_frame']) == 919 and obs.observation_id.is_unique
    assert len(groups) == 14 and groups.groupby('sample').role.nunique().eq(1).all()
    assert groups.loc[groups.movie.isin(['la', 'lb']), 'group_id'].nunique() == 1
    assert len(members) == 100 and members.observation_id.is_unique
    assert members.groupby('balance_group_id').size().eq(50).all()
    assert members.groupby(['movie', 'identity']).size().max() <= 25
    assert not obs.loc[obs.role.ne('learning'), 'learning_selected'].any()
    assert obs.loc[obs.movie.eq('la') & obs.identity.eq(7) & obs.frame_index.eq(5), 'status'].iloc[0] == 'missing_features'
    gap = obs.loc[obs.movie.eq('lb') & obs.identity.eq(8) & obs.frame_index.eq(21)].iloc[0]
    assert not gap.adjacent and gap.adjacency_reason == 'Missing frame index'
    assert obs.loc[obs.movie.eq('d1') & obs.frame_index.eq(1), 'elapsed_hours'].eq(0.25).all()
    assert data['features'].isna().sum().sum() == data['raw_features'].isna().sum().sum() == 1
    design = read_document(saved.artifact('partition_design'))
    assert design['status'] == 'ready' and design['group_counts'] == {'learning': 2, 'development': 2, 'confirmation': 2}
    provenance = read_document(saved.artifact('provenance'))
    assert not provenance['model_fitted'] and (not provenance['learned_transform_fitted']) and (not provenance['imputation_performed'])

def test_confirmation_values_and_source_row_order_cannot_choose_learning_rows(tmp_path):
    _, _, _, _, original = execute(tmp_path / 'original')

    def changed(tables, block):
        frame = tables['cell_frame']
        chosen = frame.stem.isin(['d1', 'd2', 'c1', 'c2'])
        frame.loc[chosen, ['custom_signal', 'custom_shape']] = 1000000000.0
        tables['cell_frame'] = frame.sample(frac=1, random_state=51).reset_index(drop=True)
    _, _, _, _, altered = execute(tmp_path / 'altered', change=changed)
    pd.testing.assert_frame_equal(original['partitions'], altered['partitions'])
    pd.testing.assert_frame_equal(original['learning_members'], altered['learning_members'])
    selected = original['learning_members'].observation_id
    pd.testing.assert_frame_equal(original['features'].set_index('observation_id').loc[selected], altered['features'].set_index('observation_id').loc[selected])

def test_half_open_range_and_actual_clock_gaps_preserve_all_source_rows(tmp_path):
    _, _, _, _, data = execute(tmp_path, request_changes={'time_range_hours': [52.0, 65.0]})
    obs = data['observations']
    assert len(obs) == 919
    assert obs.loc[obs.hours.eq(52), 'within_range'].all()
    assert not obs.loc[obs.hours.eq(65), 'within_range'].any()
    assert obs.loc[~obs.within_range, 'status'].eq('outside_range').all()
    for frame in data['features'].loc[~obs.within_range, ['feature:custom_signal', 'feature:custom_shape']].to_numpy():
        assert np.isnan(frame).all()
    assert data['source_traces'].hours.min() == 50

def test_native_detrending_retains_clocks_masks_and_protected_roles(tmp_path, monkeypatch):
    from pymicroglia import workbench as circadian
    monkeypatch.setattr(circadian, 'estimate_one', lambda *a, **k: pytest.fail('Unexpected rhythm fit'))
    _, _, _, saved, data = execute(tmp_path, request_changes={'representation': 'detrended', 'detrending': {'detrend': 'linear'}})
    obs, traces = (data['observations'], data['source_traces'])
    assert obs.hours.min() == 50 and traces.hours.min() == 50
    assert traces.loc[traces.raw_kind.ne('finite'), 'processed_value'].isna().all()
    assert data['partitions'].groupby('sample').role.nunique().eq(1).all()
    processing = read_document(saved.artifact('processing'))
    assert len(processing) == 28 and all((row['action'] == 'detrended' for row in processing))
    assert all((row.get('original_missing_mask_reapplied') for row in processing))
    assert not np.allclose(data['features']['feature:custom_shape'], data['raw_features']['feature:custom_shape'])

def test_conflicting_feature_clocks_are_retained_as_unassignable_vectors(tmp_path):

    def changed(tables, block):
        extra = tables['cell_frame'][['stem', 'identity', 'frame_index', 'hours', 'custom_shape']].copy()
        extra.loc[extra.stem.eq('la') & extra.identity.eq(7) & extra.frame_index.eq(10), 'hours'] += 0.1
        tables['cell_frame'] = tables['cell_frame'].drop(columns='custom_shape')
        tables['extra'] = extra
        block['features'] = ['custom_signal', {'column': 'custom_shape', 'table': 'extra'}]
        block['table_grains'] = {'extra': ['identity', 'frame_index']}
    _, _, _, _, data = execute(tmp_path, change=changed)
    bad = data['observations'].loc[data['observations'].time_status.eq('conflicting_feature_clocks')]
    assert len(bad) == 1 and bad.hours.isna().all() and (not bad.assignment_feature_eligible.any())
    assert len(data['source_traces']) == 1838
    assert set(data['source_traces'].loc[data['source_traces'].movie.eq('la') & data['source_traces'].identity.eq(7) & data['source_traces'].frame_index.eq(10), 'hours']) == {55.0, 55.1}

def test_unconfirmed_samples_and_insufficient_groups_are_explicit(tmp_path):
    block = declaration()['validation']
    _, _, _, saved, data = execute(tmp_path, request_changes={'biological_samples': {}, 'validation': block})
    assert data['partitions'].role.eq('assignment_only').all()
    assert not data['partitions'].sample_confirmed.any() and data['learning_members'].empty
    assert len(data['observations']) == 919
    design = read_document(saved.artifact('partition_design'))
    assert design['status'] == 'inconclusive' and 'Fewer than three' in design['reason']

def test_group_shuffle_is_seeded_disjoint_and_value_independent(tmp_path):
    validation = declaration()['validation']
    _, _, _, _, first = execute(tmp_path / 'first', request_changes={'validation': validation})
    _, _, _, _, second = execute(tmp_path / 'second', request_changes={'validation': validation})
    pd.testing.assert_frame_equal(first['partitions'], second['partitions'])
    partition = first['partitions']
    assert partition.groupby('sample').role.nunique().eq(1).all()
    assert set(partition.role) == {'learning', 'development', 'confirmation'}

def test_summary_only_cells_and_empty_observation_schema_survive(tmp_path):

    def changed(tables, block):
        tables['cell_frame'] = tables['cell_frame'].iloc[:0]
        tables['cell_summary'] = pd.DataFrame({'stem': ['la'], 'identity': [99], 'arbitrary_scalar': [3.0]})
        block['biological_samples'] = {'la': 'one'}
        block['validation']['split'] = {'method': 'explicit', 'roles': {'one': 'learning'}}
    _, _, _, saved, data = execute(tmp_path, change=changed)
    assert data['observations'].empty and 'observation_id' in data['observations']
    assert data['learning_members'].empty and 'learning_weight' in data['learning_members']
    assert data['cell_inventory'].status.tolist() == ['missing_observations']
    assert data['feature_inventory'].status.eq('missing').all()
    assert read_document(saved.artifact('partition_design'))['status'] == 'inconclusive'

def test_reopening_inputs_uses_original_saved_membership(tmp_path, monkeypatch):
    import pymicroglia.pipelines.behaviour.inputs as behaviour_inputs
    resolved, paths, _, saved, data = execute(tmp_path)
    monkeypatch.setattr(behaviour_inputs, '_prepare_trace', lambda *a, **k: pytest.fail('Saved preparation was repeated'))
    reopened = run_request(resolved, paths, tmp_path / 'pipeline', only=('feature-inputs',), presentation={'anything': 'changed'})
    assert reopened.successful and reopened.results['feature-inputs'].outcome.status == 'reused'
    assert reopened.results['feature-inputs'].root == saved.root

def test_empty_population_keeps_all_saved_schemas(tmp_path):

    def changed(tables, block):
        tables['cell_frame'] = tables['cell_frame'].iloc[:0]
        block['biological_samples'] = {}
        block['validation'] = declaration()['validation']
    _, _, _, saved, data = execute(tmp_path, change=changed)
    assert all((frame.empty for frame in data.values()))
    assert all((len(frame.columns) > 0 for frame in data.values()))
    assert read_document(saved.artifact('partition_design'))['learning_observations'] == 0

@pytest.mark.parametrize('balance', ['sample_cell', 'cell'])
def test_recording_validation_cannot_invent_sample_balancing(tmp_path, balance):

    def changed(tables, block):
        block['biological_samples'] = {}
        block['learning']['balance'] = balance
        block['validation']['unit'] = 'recording'
        block['validation']['split'] = {'method': 'explicit', 'roles': {movie: 'learning' if movie.startswith('l') else 'development' if movie.startswith('d') else 'confirmation' for movie in tables['cell_frame'].stem.unique()}}
    _, _, _, _, data = execute(tmp_path, change=changed)
    assert not data['partitions'].sample_confirmed.any()
    if balance == 'sample_cell':
        assert data['learning_members'].empty
        assert data['cell_inventory'].loc[data['cell_inventory'].role.eq('learning'), 'learning_reason'].str.contains('Confirmed biological sample required').all()
    else:
        assert len(data['learning_members']) == 150
        assert data['learning_members'].groupby(['movie', 'identity']).size().eq(25).all()
