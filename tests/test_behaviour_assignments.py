"""Accepted definitions assign original rows and never manufacture profile members."""
import json
import numpy as np
import pandas as pd
import pytest
import pymicroglia.states.behaviour_models as backend
import pymicroglia.states.behaviour_support as behaviour_support
import pymicroglia.pipelines.behaviour.assignments as assignment, pymicroglia.pipelines._screening as screening
from pymicroglia.pipelines.behaviour.options import run_request
from pymicroglia.pipelines._screening import file_hash, read_table
from tests.test_behaviour_models import inputs
from tests.test_behaviour_validation import fixture

@pytest.fixture(autouse=True)
def isolated_source_fingerprint(monkeypatch):
    monkeypatch.setattr(screening, 'producer_identity', lambda: {'implementation': 'controlled-assignment-test'})

def native_fixture(*, missing=False, outlier=0.01):
    frame, definitions, learning, recipe, settings, members = inputs(missing=missing)
    settings['outlier_quantile'] = outlier
    model = backend.fit_candidate(frame, definitions, learning, recipe, settings, members, 'raw')['model']
    decision = {'status': 'accepted', 'accepted_model_id': model['model_id'], 'candidate_id': 'controlled-candidate', 'decision_id': 'controlled-acceptance'}
    values = pd.DataFrame([[-3.0, -1.0], [3.0, 2.0], [-3.1, -1.1], [1000000.0, 1000000.0], [-3.0, np.nan], [-3.0, -1.0], [3.0, 2.0]], columns=frame.columns, index=['observation-' + str(i) for i in range(7)])
    rows = []
    for index, key in enumerate(values.index):
        valid_time = index != 5
        inside = index != 6
        rows.append({'source_run': 'controlled-source', 'movie': 'a' if index < 3 else 'b', 'identity': 7, 'observation_id': key, 'frame_index': index, 'hours': 50.0 + 0.5 * index if valid_time else None, 'role': ['learning', 'development', 'confirmation', 'assignment_only'][index % 4], 'learning_selected': index == 0, 'time_status': 'recorded' if valid_time else 'invalid_time', 'within_range': inside, 'status': 'eligible' if valid_time and inside else 'invalid_time' if not valid_time else 'outside_range', 'reason': 'Controlled original observation', 'sample': 'one' if index < 3 else 'two', 'sample_confirmed': True, 'group_id': 'one' if index < 3 else 'two', 'vector_id': key + '-features'})
    observations = pd.DataFrame(rows).set_index('observation_id', drop=False)
    values.index.name = 'observation_id'
    return (model, decision, observations, values, definitions)

def test_all_original_observations_unknowns_and_shared_definition_survive(monkeypatch):
    model, decision, observations, values, definitions = native_fixture()
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    def forbidden(*a, **k):
        raise AssertionError('Assignment attempted model learning')
    monkeypatch.setattr(GaussianMixture, 'fit', forbidden)
    monkeypatch.setattr(StandardScaler, 'fit', forbidden)
    rows = assignment.apply_assignments(model, decision, observations, values, definitions)
    assert len(rows) == 7 and rows.observation_id.is_unique
    assert rows.status.tolist() == ['assigned', 'assigned', 'assigned', 'outside_training_distribution', 'missing_features', 'invalid_time', 'outside_range']
    assert rows.loc[rows.status.ne('assigned'), 'state_id'].isna().all()
    assert rows.loc[rows.status.ne('assigned'), 'component'].isna().all()
    assert rows.score_calibration.eq('uncalibrated_model_conditional').all()
    assert set(rows.movie) == {'a', 'b'} and rows.identity.eq(7).all()
    single = [assignment.apply_assignments(model, decision, observations.loc[[key]], values.loc[[key]], definitions) for key in reversed(values.index)]
    pd.testing.assert_frame_equal(rows.sort_values('observation_id').reset_index(drop=True), pd.concat(single).sort_values('observation_id').reset_index(drop=True))
    for rejected in ['no_supported_states', 'inconclusive']:
        with pytest.raises(ValueError, match='accepted model'):
            assignment.apply_assignments(model, {**decision, 'status': rejected}, observations, values, definitions)

def test_profiles_use_actual_observed_values_and_real_density_selected_members():
    model, decision, observations, values, definitions = native_fixture()
    rows = assignment.apply_assignments(model, decision, observations, values, definitions)
    states, profiles, examples = assignment.describe_states(model, rows, {'raw': values}, definitions)
    assert len(states) == 2 and len(profiles) == 4 and (len(examples) == 2)
    assert not states.centre_is_observed_cell.any() and (not profiles.independent_validation.any())
    for profile in profiles.to_dict('records'):
        actual = values.loc[profile['observed_members'], profile['measurement']]
        assert len(actual) == profile['observed_values']
        assert profile['mean'] == pytest.approx(actual.mean()) and profile['median'] == pytest.approx(actual.median())
        assert rows.loc[rows.observation_id.isin(profile['observed_members']), 'state_id'].eq(profile['state_id']).all()
    indexed = rows.set_index('observation_id')
    for example in examples.to_dict('records'):
        actual = indexed.loc[example['observation_id']]
        assert actual['status'] == 'assigned' and actual['state_id'] == example['state_id']
        assert actual['hours'] == example['hours'] and actual['movie'] == example['movie']
        assert actual['log_density'] == rows.loc[rows.state_id.eq(example['state_id']), 'log_density'].max()

def test_imputed_assignment_does_not_insert_imputed_values_into_raw_profiles():
    model, decision, observations, values, definitions = native_fixture(missing=True, outlier=0)
    keys = ['observation-0', 'observation-2', 'observation-4']
    rows = assignment.apply_assignments(model, decision, observations.loc[keys], values.loc[keys], definitions)
    assert rows.status.eq('assigned').all()
    states, profiles, examples = assignment.describe_states(model, rows, {'raw': values}, definitions)
    used = profiles.loc[profiles.assigned_observations.gt(0)]
    shape = used.loc[used.measurement.eq('shape')].iloc[0]
    assert shape['assigned_observations'] == 3 and shape['observed_values'] == 2 and (shape['missing_values'] == 1)
    assert 'observation-4' not in shape['observed_members']
    assert shape['mean'] == pytest.approx((-1.0 - 1.1) / 2)
    assert examples.loc[examples.status.eq('no_assigned_observation'), 'observation_id'].isna().all()

def test_ambiguous_native_membership_remains_unassigned():
    from scipy.optimize import brentq
    model, decision, observations, values, definitions = native_fixture(outlier=0)

    def membership(fraction):
        point = np.array([-3.0, -1.0]) + fraction * np.array([6.0, 3.0])
        probability = backend.predict_candidate(model, pd.DataFrame([point], columns=values.columns), definitions)['probabilities'][0, 0]
        return probability - 0.5
    boundary = brentq(membership, 0.0, 1.0)
    values = values.iloc[:1].copy()
    values.iloc[0] = np.array([-3.0, -1.0]) + boundary * np.array([6.0, 3.0])
    rows = assignment.apply_assignments(model, decision, observations.iloc[:1], values, definitions)
    assert rows.status.tolist() == ['ambiguous'] and rows.state_id.isna().all()
    assert rows.max_probability.iloc[0] == pytest.approx(0.5)

def test_empty_observations_keep_state_definitions_without_inventing_examples():
    model, decision, observations, values, definitions = native_fixture()
    rows = assignment.apply_assignments(model, decision, observations.iloc[:0], values.iloc[:0], definitions)
    states, profiles, examples = assignment.describe_states(model, rows, {'raw': values.iloc[:0]}, definitions)
    assert rows.empty and 'state_id' in rows and (len(states) == 2)
    assert states.assigned_observations.eq(0).all() and profiles.observed_values.eq(0).all() and profiles['mean'].isna().all()
    assert examples.status.eq('no_assigned_observation').all()

def test_real_accepted_pipeline_assigns_every_original_row_and_reopens_without_science(tmp_path, monkeypatch):
    resolved, paths, source = fixture(tmp_path)
    first = run_request(resolved, paths, tmp_path / 'pipeline', only=('state-assignments',))
    assert first.successful, {k: v.outcome.reason for k, v in first.results.items()}
    saved = first.results['state-assignments']
    rows, provenance = assignment.read_assignments(saved)
    assert len(rows) == len(source) == 960 and rows.observation_id.is_unique
    assert set(rows.role) == {'learning', 'development', 'confirmation'}
    assert not provenance['model_fitted'] and (not provenance['transform_fitted']) and (not provenance['temporal_smoothing'])
    members = read_table(saved.artifact('state_members'))
    assert set(members.observation_id) == set(rows.loc[rows.status.eq('assigned'), 'observation_id'])
    cells = read_table(saved.artifact('cell_inventory'))
    assert len(cells) == 124 and cells.observations.sum() == 960
    assert cells[[status + '_observations' for status in assignment.REASONS]].sum().sum() == 960
    before = {ref.name: file_hash(saved.artifact(ref.name)) for ref in saved.outcome.artifacts}

    def forbidden(*a, **k):
        raise AssertionError('Reopening performed fresh state science')
    monkeypatch.setattr(backend, 'fit_candidate', forbidden)
    monkeypatch.setattr(backend, 'predict_candidate', forbidden)
    monkeypatch.setattr(behaviour_support, 'measure', forbidden)
    second = run_request(resolved, paths, tmp_path / 'pipeline', only=('state-assignments',), presentation={'state_names': ['First', 'Second']})
    assert second.successful and all((result.outcome.status == 'reused' for result in second.results.values()))
    assert all((file_hash(saved.artifact(name)) == fingerprint for name, fingerprint in before.items()))
    with pytest.raises(ValueError, match='different model'):
        assignment.read_assignments(saved, expected_model='different-model')

def test_rejected_pipeline_never_produces_assignments(tmp_path):
    resolved, paths, _ = fixture(tmp_path, kind='uniform')
    result = run_request(resolved, paths, tmp_path / 'pipeline', only=('state-assignments',))
    assert result.successful and result.results['state-assignments'].outcome.status == 'skipped-empty'
    assert not result.results['state-assignments'].outcome.artifacts
