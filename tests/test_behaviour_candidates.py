"""Candidate integration keeps reserved confirmation out of all fitting and scores."""
from pymicroglia._results import read_document
import json
import numpy as np
import pandas as pd
import pytest
import pymicroglia.states.behaviour_models as backend
from pymicroglia.pipelines.behaviour.options import run_request
from pymicroglia.pipelines.behaviour.candidates import saved_inputs
from pymicroglia.pipelines._screening import read_table
from tests.test_behaviour_inputs import fixture

def execute(tmp_path, **kwargs):
    resolved, paths, _ = fixture(tmp_path, **kwargs)
    execution = run_request(resolved, paths, tmp_path / 'pipeline', only=('candidate-models',))
    return (resolved, paths, execution)

def artifacts(execution):
    assert execution.successful, {k: v.outcome.reason for k, v in execution.results.items()}
    saved = execution.results['candidate-models']
    return (saved, {name: read_table(saved.artifact(name)) for name in ('candidates', 'diagnostic_predictions', 'group_diagnostics', 'candidate_states')}, read_document(saved.artifact('models')))

def test_learning_models_and_development_diagnostics_exclude_confirmation(tmp_path):
    _, _, execution = execute(tmp_path)
    saved, tables, models = artifacts(execution)
    candidates, predictions = (tables['candidates'], tables['diagnostic_predictions'])
    assert candidates.status.eq('fitted').all() and len(models) == 2
    assert candidates.learning_observations.eq(100).all() and candidates.development_groups.eq(2).all()
    assert set(predictions.role) == {'learning', 'development'} and (not predictions.movie.str.startswith('c').any())
    assert len(predictions) == (919 - 240) * 2 and predictions.diagnostic_only.all()
    assert predictions.log_density.notna().sum() == (919 - 240 - 1) * 2
    assert predictions.loc[predictions.movie.eq('la') & predictions.identity.eq(7) & predictions.frame_index.eq(5), 'status'].eq('missing_features').all()
    assert len(tables['candidate_states']) == 3 and (not tables['candidate_states'].accepted.any())
    for model in models.values():
        assert len(model['learning_membership']) == 100
        assert all((row['movie'].startswith('l') for row in model['learning_membership']))
        backend.verify_model(model)
    provenance = read_document(saved.artifact('provenance'))
    assert not provenance['confirmation_evaluated'] and (not provenance['state_acceptance_performed'])

def test_heldout_values_and_source_order_change_diagnostics_but_not_fitted_definition(tmp_path):
    _, _, original = execute(tmp_path / 'original')

    def change(tables, block):
        frame = tables['cell_frame']
        protected = ~frame.stem.str.startswith('l')
        frame.loc[protected, ['custom_signal', 'custom_shape']] += 1000000.0
        tables['cell_frame'] = frame.sample(frac=1, random_state=73).reset_index(drop=True)
    _, _, altered = execute(tmp_path / 'altered', change=change)
    first, a, ma = artifacts(original)
    second, b, mb = artifacts(altered)
    assert first.outcome.scientific_id != second.outcome.scientific_id
    assert ma == mb
    pd.testing.assert_frame_equal(a['diagnostic_predictions'].loc[lambda d: d.role.eq('learning')].reset_index(drop=True), b['diagnostic_predictions'].loc[lambda d: d.role.eq('learning')].reset_index(drop=True))
    assert b['diagnostic_predictions'].loc[lambda d: d.role.eq('development'), 'status'].eq('outside_training_distribution').all()

def test_confirmation_never_enters_public_prediction(tmp_path, monkeypatch):
    original = backend.predict_candidate

    def guarded(model, frame, definitions):
        assert not (frame.to_numpy() == 918273.0).any(), 'Reserved confirmation reached native prediction'
        return original(model, frame, definitions)
    monkeypatch.setattr(backend, 'predict_candidate', guarded)

    def change(tables, block):
        frame = tables['cell_frame']
        frame.loc[frame.stem.str.startswith('c'), ['custom_signal', 'custom_shape']] = 918273.0
    _, _, execution = execute(tmp_path, change=change)
    artifacts(execution)

def test_frozen_candidate_reopening_requires_no_fit(tmp_path, monkeypatch):
    resolved, paths, first = execute(tmp_path)
    saved, _, models = artifacts(first)
    monkeypatch.setattr(backend, 'fit_candidate', lambda *a, **k: pytest.fail('Candidate refitted on reopen'))
    second = run_request(resolved, paths, tmp_path / 'pipeline', only=('candidate-models',), presentation={'palette': 'changed'})
    assert second.successful and second.results['candidate-models'].outcome.status == 'reused'
    assert second.results['candidate-models'].root == saved.root

def test_unknown_native_assignment_capability_is_unavailable(tmp_path):
    _, _, execution = execute(tmp_path, request_changes={'candidates': [{'method': 'no_new_observation_api'}]})
    assert not execution.successful
    assert execution.results['candidate-models'].outcome.status == 'unavailable'

def test_constant_and_empty_learning_keep_every_requested_candidate(tmp_path):

    def change(tables, block):
        tables['cell_frame'].loc[:, ['custom_signal', 'custom_shape']] = 1.0
    _, _, execution = execute(tmp_path / 'constant', change=change)
    _, tables, models = artifacts(execution)
    assert tables['candidates'].status.eq('ineligible').all() and len(tables['candidates']) == 2
    assert not models and tables['diagnostic_predictions'].empty and tables['group_diagnostics'].empty

    def empty(tables, block):
        tables['cell_frame'] = tables['cell_frame'].iloc[:0]

    def no_observations(tables, block):
        tables['cell_summary'] = tables['cell_frame'][['stem', 'identity']].drop_duplicates()
        empty(tables, block)
    _, _, execution = execute(tmp_path / 'empty', change=no_observations)
    _, tables, models = artifacts(execution)
    assert tables['candidates'].status.eq('ineligible').all() and (not models)

def test_input_feature_mapping_checks_and_half_open_diagnostics(tmp_path):
    resolved, _, execution = execute(tmp_path, request_changes={'time_range_hours': [52.0, 65.0]})
    _, tables, _ = artifacts(execution)
    predicted = tables['diagnostic_predictions']
    assert predicted.loc[predicted.hours.ge(65) | predicted.hours.lt(52), 'log_density'].isna().all()
    with pytest.raises(ValueError, match='feature definition'):
        saved_inputs(execution.results['feature-inputs'], [{**f.as_dict(), 'column': 'wrong'} for f in resolved.features])
