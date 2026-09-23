"""Whole-group support, immutable confirmation and legitimate no-state branches."""
from pymicroglia._results import read_document
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import pymicroglia.states.behaviour_models as backend
import pymicroglia.states.behaviour_support as support
import pymicroglia.pipelines.behaviour.validation as behaviour_validation, pymicroglia.pipelines._screening as screening
from pymicroglia.pipelines.behaviour.options import BehaviourRequest, resolve_request, run_request
from pymicroglia.pipelines._screening import file_hash, read_table
from tests.test_behaviour_options import declaration
from tests.test_behaviour_support import draw

@pytest.fixture(autouse=True)
def isolated_source_fingerprint(monkeypatch):
    monkeypatch.setattr(screening, 'producer_identity', lambda: {'implementation': 'controlled-state-support-test'})

def fixture(tmp_path, *, kind='two', confirmation_kind=None, groups=60, confirmation_groups=None, changes=None):
    rng = np.random.default_rng(77013)
    rows, samples, roles = ([], {}, {})
    for role, count in [('learning', 4), ('development', groups), ('confirmation', groups if confirmation_groups is None else confirmation_groups)]:
        for group in range(count):
            movie = role + '-' + str(group)
            samples[movie] = movie
            roles[movie] = role
            number = 120 if role == 'learning' else 4
            values = draw(confirmation_kind if role == 'confirmation' and confirmation_kind else kind, number, rng)
            for index, value in enumerate(values):
                rows.append({'stem': movie, 'identity': 7, 'frame_index': index, 'hours': 50.0 + index * 0.5, 'custom_signal': float(value[0]), 'custom_shape': float(value[1])})
    frame = pd.DataFrame(rows)
    block = declaration(biological_samples=samples)
    block['learning']['max_observations_per_cell'] = 60
    block['validation']['split'] = {'method': 'explicit', 'roles': roles}
    block['support'] = {'method': 'heldout_multimodality', 'sampling_population': 'Independent exchangeable controlled samples with the same measurement-generating distribution'}
    if changes:
        changes(frame, block)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / 'cell_frame.csv'
    frame.to_csv(path, index=False)
    request = BehaviourRequest.from_dict(block, {})
    resolved = resolve_request(request, source_run='controlled-state-validation', tables={'cell_frame': frame}, input_hashes={'cell_frame': file_hash(path)})
    return (resolved, {'cell_frame': path}, frame)

def execute(tmp_path, **kwargs):
    resolved, paths, frame = fixture(tmp_path, **kwargs)
    execution = run_request(resolved, paths, tmp_path / 'pipeline', only=('state-support',))
    assert execution.successful, {k: v.outcome.reason for k, v in execution.results.items()}
    saved = execution.results['state-support']
    decision = read_document(saved.artifact('support_decision'))
    return (resolved, paths, execution, saved, decision)

def test_separated_model_is_frozen_before_reserved_prediction(tmp_path, monkeypatch):
    original = support.measure
    observed_confirmations = []
    original_claim = behaviour_validation.claim_confirmation

    def guarded_claim(context, frozen, observations):
        on_disk = read_document(context.output / 'frozen_choice.json')
        model = read_document(context.output / 'frozen_model.json')
        assert on_disk == frozen and model['model_id'] == frozen['model_id']
        observed_confirmations.append(frozen['selection_id'])
        return original_claim(context, frozen, observations)
    monkeypatch.setattr(behaviour_validation, 'claim_confirmation', guarded_claim)
    _, _, execution, saved, decision = execute(tmp_path)
    assert decision['status'] == 'accepted' and decision['confirmation_evaluated']
    assert observed_confirmations == [decision['selection_id']]
    assert len(saved.outcome.selections[0].members) == 1
    reps = read_table(saved.artifact('validation_representatives'))
    assert reps.group_id.is_unique and len(reps) == 120 and reps.status.eq('selected').all()
    refits = read_document(saved.artifact('grouped_refits'))
    assert len(refits['checks']) == len(refits['models']) == 4
    assert all((row['correspondence']['adjusted_rand'] > 0.9 for row in refits['checks']))
    assert all((not row['confirmation_evaluated'] for row in refits['checks']))
    model = read_document(saved.artifact('frozen_model'))
    assert len(model['learning_membership']) == 240
    assert decision['accepted_model_id'] == model['model_id']
    checked, replay_model = behaviour_validation.read_support(saved, require_accepted=True)
    assert checked == decision and replay_model == model
    from dataclasses import replace
    from pymicroglia.pipelines._runner import SavedResult
    wrong_selection = replace(saved.outcome.selections[0], members=())
    with pytest.raises(ValueError, match='membership disagrees'):
        behaviour_validation.read_support(SavedResult(saved.root, replace(saved.outcome, selections=(wrong_selection,))))

@pytest.mark.parametrize('kind', ['cloud', 'uniform', 'ring'])
def test_continuous_controls_never_open_confirmation_or_accept_states(tmp_path, kind, monkeypatch):
    monkeypatch.setattr(behaviour_validation, 'claim_confirmation', lambda *a, **k: pytest.fail('Rejected candidate opened reserved samples'))
    _, _, _, saved, decision = execute(tmp_path, kind=kind)
    assert decision['status'] in {'no_supported_states', 'inconclusive'}
    assert not decision['confirmation_evaluated'] and (not saved.outcome.selections[0].members)
    assert read_document(saved.artifact('frozen_model')) is None
    assert behaviour_validation.read_support(saved)[0] == decision
    with pytest.raises(ValueError, match='accepted state model'):
        behaviour_validation.read_support(saved, require_accepted=True)

def test_changed_confirmation_cannot_select_a_different_model_or_threshold(tmp_path):
    _, _, _, first, passed = execute(tmp_path / 'matching')
    _, _, _, second, failed = execute(tmp_path / 'changed', confirmation_kind='cloud')
    assert passed['status'] == 'accepted' and failed['status'] != 'accepted'
    assert failed['confirmation_evaluated'] and (not second.outcome.selections[0].members)
    assert read_document(first.artifact('frozen_model')) == read_document(second.artifact('frozen_model'))
    assert read_document(first.artifact('frozen_choice')) == read_document(second.artifact('frozen_choice'))

def test_missing_population_or_too_few_groups_is_inconclusive_without_confirmation(tmp_path, monkeypatch):
    monkeypatch.setattr(behaviour_validation, 'claim_confirmation', lambda *a, **k: pytest.fail('Insufficient input opened confirmation'))
    _, _, _, saved, decision = execute(tmp_path / 'few', groups=3)
    assert decision['status'] == 'inconclusive' and (not saved.outcome.selections[0].members)

    def missing(frame, block):
        block['support'].pop('sampling_population')
    _, _, _, saved, decision = execute(tmp_path / 'unspecified', changes=missing)
    assert decision['status'] == 'inconclusive' and 'sampling_population' in decision['reason']

def test_insufficient_confirmation_does_not_consume_reserved_values(tmp_path, monkeypatch):
    monkeypatch.setattr(behaviour_validation, 'claim_confirmation', lambda *a, **k: pytest.fail('Insufficient confirmation was opened'))
    _, _, _, saved, decision = execute(tmp_path, confirmation_groups=3)
    assert decision['status'] == 'inconclusive' and decision['provisional_model_id'] is not None
    assert not decision['confirmation_evaluated'] and (not decision['confirmation_consumption'])
    assert read_document(saved.artifact('confirmation_evidence'))['independent_groups'] == 3
    assert behaviour_validation.read_support(saved)[0] == decision

def test_unidentifiable_requested_state_count_is_inconclusive(tmp_path):

    def insufficient_vectors(frame, block):
        sign = np.where(frame.custom_signal > 0, 1.0, -1.0)
        frame['custom_signal'] = 4.0 * sign
        frame['custom_shape'] = 2.0 * sign
        block['candidates'][1]['components'] = 3
    _, _, _, saved, decision = execute(tmp_path, changes=insufficient_vectors)
    assert decision['status'] == 'inconclusive' and (not decision['confirmation_evaluated'])
    rows = read_table(saved.artifact('candidate_support'))
    assert rows.loc[rows.components.eq(3), 'status'].tolist() == ['ineligible']
    assert not saved.outcome.selections[0].members

def test_exact_reopen_and_retuned_reservation_across_output_folders(tmp_path, monkeypatch):
    resolved, paths, first, saved, decision = execute(tmp_path)
    original_hashes = {ref.name: file_hash(saved.artifact(ref.name)) for ref in saved.outcome.artifacts}
    monkeypatch.setattr(backend, 'fit_candidate', lambda *a, **k: pytest.fail('Saved model was refitted'))
    monkeypatch.setattr(support, 'measure', lambda *a, **k: pytest.fail('Saved support was recomputed'))
    reopened = run_request(resolved, paths, tmp_path / 'pipeline', only=('state-support',), presentation={'palette': 'changed'})
    assert reopened.successful and reopened.results['state-support'].outcome.status == 'reused'
    assert all((file_hash(saved.artifact(key)) == value for key, value in original_hashes.items()))
    frozen = read_document(saved.artifact('frozen_choice'))
    observations = read_table(first.results['feature-inputs'].artifact('observations'))
    context = SimpleNamespace(table_paths=paths, request=resolved)
    assert behaviour_validation.claim_confirmation(context, frozen, observations) == decision['confirmation_consumption']
    with pytest.raises(ValueError, match='already consumed'):
        behaviour_validation.claim_confirmation(context, {**frozen, 'selection_id': 'retuned-rule'}, observations)

def test_no_support_skips_nested_production_branches_but_preserves_diagnostics(tmp_path):
    resolved, paths, _, _, decision = execute(tmp_path, kind='uniform')
    execution = run_request(resolved, paths, tmp_path / 'pipeline', only=('state-assignments', 'durations-and-switches', 'state-sample-comparisons'))
    assert execution.successful
    assert all((execution.results[name].outcome.status == 'skipped-empty' for name in ['state-assignments', 'durations-and-switches', 'state-sample-comparisons']))
    from pymicroglia.pipelines.behaviour.options import RECIPE
    diagnostic = next((step for step in RECIPE.steps if step.name == 'state-support-figures'))
    assert diagnostic.selection is None and 'state-assignments' not in diagnostic.prerequisites
