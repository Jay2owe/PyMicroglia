"""Development choices need evidence and cannot be manufactured from missing answers."""
from dataclasses import replace
from types import SimpleNamespace
import pandas as pd
import pytest
from pymicroglia import workbench as circadian
from pymicroglia.pipelines.audit.selection import assess_requirements, decide_scope, family_compatibility, paired_recovery, read_selection, select_candidates
from pymicroglia.pipelines._contracts import Settings
from pymicroglia.pipelines.audit.workflow import run_request
from tests.test_audit_options import declaration
from tests.test_audit_benchmarks import fixture
POLICY = declaration()['score_policy']

def score(candidate, recovery=1.0, false_alarm=0.0, n=1000, coverage=1.0):
    row = {'candidate_id': candidate, 'scope': 'measurement', 'measurement': 'signal', 'facet': 'overall', 'value': 'all'}
    for name, value in (('recovery', recovery), ('false_alarm', false_alarm)):
        units = [{'id': f'{name}-{i}', 'dependency_id': f'{name}-{i}', 'value': value, 'weight': 1.0} for i in range(n)]
        interval = circadian.benchmark_score_interval(units, confidence=POLICY['confidence'], bounds=[0.0, 1.0])
        row.update({name + '_denominator': n, name + '_independent_units': n, name + '_rate': interval['mean'], name + '_lower': interval['lower'], name + '_upper': interval['upper'], name + '_weighted_valid_fraction': coverage, name + '_units': units})
    return row

def decide(rows):
    return decide_scope([SimpleNamespace(candidate_id=c) for c in ('a', 'b')], pd.DataFrame(rows), POLICY, scope='measurement', measurement='signal')

@pytest.mark.parametrize('kind,state,chosen', [('clear', 'provisional_choice', ['a']), ('equal', 'tie', ['a', 'b']), ('unacceptable', 'no_acceptable_candidate', []), ('too-small', 'insufficient_evidence', [])])
def test_four_decision_states_have_explicit_gates(kind, state, chosen):
    if kind == 'clear':
        rows = [score('a'), score('b', recovery=0.0)]
    elif kind == 'equal':
        rows = [score('a', recovery=0.5), score('b', recovery=0.5)]
    elif kind == 'unacceptable':
        rows = [score('a', false_alarm=0.5), score('b', false_alarm=0.5)]
    else:
        rows = [score('a', n=3), score('b', n=3)]
    decision, assessments, comparisons = decide(rows)
    assert decision['state'] == state and decision['candidate_ids'] == chosen
    assert all((len(a['gates']) == 5 and a['score_id'] for a in assessments))
    assert all((g['reason'] for a in assessments for g in a['gates']))
    reordered = decide(list(reversed(rows)) + [rows[0]])[0]
    assert decision == reordered
    if kind == 'clear':
        assert comparisons[0]['uncertainty']['lower'] > POLICY['recovery_margin']

def test_zero_answers_and_high_recovery_with_false_alarms_cannot_win():
    decision, assessments, _ = decide([score('a', coverage=0.0), score('b', false_alarm=0.7)])
    assert decision['state'] == 'no_acceptable_candidate'
    assert assessments[0]['gates'][1]['state'] == 'fail'
    small = assess_requirements(score('a', n=30), POLICY)
    assert small['state'] == 'insufficient'

def test_small_paired_difference_remains_tied_and_unpaired_sources_are_rejected():
    a, b = (score('a', recovery=0.51), score('b', recovery=0.5))
    assert decide([a, b])[0]['state'] == 'tie'
    b['recovery_units'][0]['dependency_id'] = 'different-noise'
    with pytest.raises(ValueError, match='source dependencies'):
        paired_recovery(a, b, POLICY)

def test_conflicting_duplicate_science_is_not_a_presentation_copy():
    a = score('a')
    with pytest.raises(ValueError, match='Conflicting'):
        decide([a, {**a, 'recovery_rate': 0.1}, score('b')])

def test_measurement_and_dataset_scopes_preserve_family_limits(fixture):
    resolved, _ = fixture
    candidate = resolved.candidates[0]
    rows = pd.DataFrame([score(candidate.candidate_id)])
    stability = pd.DataFrame([{'scope': 'measurement', 'measurement': 'signal', 'candidate_id': candidate.candidate_id, 'status': 'not_requested', 'pairs': 0}])
    decisions, _, _ = select_candidates(resolved, rows, stability)
    assert len(decisions) == 1 and decisions[0]['measurement'] == 'signal'
    dataset = replace(resolved, request=replace(resolved.request, recommendation_scope='dataset'))
    rows['scope'], rows['measurement'] = ('dataset', None)
    assert select_candidates(dataset, rows, stability)[0][0]['scope'] == 'dataset'
    altered = replace(candidate, candidate_id='different')
    mixed = replace(resolved, candidates=(candidate, altered))
    compatibility = family_compatibility(mixed, [{'state': 'provisional_choice', 'candidate_ids': [c.candidate_id]} for c in mixed.candidates])
    assert compatibility['state'] == 'mixed_joint_profile_not_evaluated' and (not compatibility['equivalent_calibration'])

def test_production_selection_freezes_before_confirmation_access(fixture, tmp_path, monkeypatch):
    resolved, paths = fixture
    generator = circadian.generate_benchmark_cases

    def development(*args, **kwargs):
        assert kwargs['partition'] == 'development'
        return generator(*args, **kwargs)
    monkeypatch.setattr(circadian, 'generate_benchmark_cases', development)

    def estimate(hours, values, params, method, **kwargs):
        return {'method': method, 'status': 'ok', 'period_hours': 12.0, 'p_value': 0.5, 'components': [], 'diagnostics': {}}
    monkeypatch.setattr(circadian, 'estimate_one', estimate)
    execution = run_request(resolved, paths, tmp_path / 'selection', only=('candidate-shortlist',))
    saved = execution.results['candidate-shortlist']
    assert execution.successful, saved.outcome.reason
    record = read_selection(saved, expected_audit=resolved.scientific_id)
    assert record['confirmation_opened'] is False and (not record['candidate_ids_to_confirm'])
    assert record['decisions'][0]['state'] == 'insufficient_evidence'
    assert record['reservation_id'] and record['case_manifest_id'] and record['input_results']
    monkeypatch.setattr(circadian, 'generate_benchmark_cases', lambda *a, **k: pytest.fail('selection read generated cases'))
    assert read_selection(saved)['selection_id'] == record['selection_id']
