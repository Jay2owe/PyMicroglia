"""Reserved cases are consumed by one frozen choice, with no runner-up promotion."""
from pymicroglia._results import read_document
from dataclasses import replace
import json
from types import SimpleNamespace
import pandas as pd
import pytest
from pymicroglia import workbench as circadian
from pymicroglia.pipelines import parse
from pymicroglia.pipelines.audit.benchmarks import generate_development
from pymicroglia.pipelines.audit.confirmation import assess_confirmation, claim_reservation, consumption_root, generate_reserved, guard_selection, validate_separation, read_confirmation, CONFIRMATION_POPULATION
from pymicroglia.pipelines.audit.options import resolve_request
from pymicroglia.pipelines.audit.selection import confirmation_checks, decide_scope
from pymicroglia.pipelines._contracts import Settings, content_id
from pymicroglia.pipelines.audit.workflow import run_request
from pymicroglia.pipelines._screening import read_table
from tests.test_audit_benchmarks import fixture
from tests.test_audit_selection import score

def frozen(resolved, development, *, tie=False):
    candidates = list(resolved.candidates)
    rows = [score(c.candidate_id) for c in candidates]
    decision, assessments, _ = decide_scope(candidates, pd.DataFrame(rows), resolved.request.score_policy, scope='measurement', measurement='signal')
    selection = {'schema_version': 1, 'audit_id': resolved.scientific_id, 'case_manifest_id': development.manifest['manifest_id'], 'reservation_id': development.reservation['reservation_id'], 'candidates': [c.as_dict() for c in candidates], 'decisions': [decision], 'candidate_ids_to_confirm': decision['candidate_ids'], 'policy': resolved.request.score_policy.as_dict(), 'stability_policy': resolved.request.stability.as_dict(), 'confirmation_opened': False, 'confirmation_population': CONFIRMATION_POPULATION, 'confirmation_checks': confirmation_checks([decision], assessments, resolved.request.score_policy)}
    selection['selection_id'] = content_id(selection)
    return (selection, assessments)

def test_exact_freeze_is_required_before_reserved_generation(fixture, monkeypatch):
    resolved, _ = fixture
    development = generate_development(resolved)
    selection, assessments = frozen(resolved, development)
    monkeypatch.setattr(circadian, 'generate_benchmark_cases', lambda *a, **k: pytest.fail('guard opened reserved data'))
    assert guard_selection(resolved, selection, development, assessments) == selection['candidate_ids_to_confirm']
    with pytest.raises(ValueError, match='settings or policy'):
        guard_selection(resolved, {**selection, 'audit_id': 'changed'}, development, assessments)
    damaged = {**selection, 'confirmation_checks': [{**selection['confirmation_checks'][0], 'minimum_recovery': 0.0}]}
    with pytest.raises(ValueError, match='thresholds'):
        guard_selection(resolved, damaged, development, assessments)
    changed = replace(resolved, candidates=(replace(resolved.candidates[0], filtering=Settings({'method': 'changed'})),))
    with pytest.raises(ValueError, match='frozen candidate|settings or policy'):
        guard_selection(changed, selection, development, assessments)

def test_real_reserved_streams_are_fresh_and_deterministic_cases_are_declared(fixture):
    resolved, _ = fixture
    development = generate_development(resolved)
    reserved = generate_reserved(resolved, development)
    assert reserved.cases.fresh_confirmation_eligible.sum() == 6
    assert len(reserved.cases) == 12
    assert not set(reserved.cases.case_id) & set(development.cases.case_id)
    assert reserved.cases.loc[reserved.cases.scenario.eq('multiple'), 'fresh_confirmation_eligible'].eq(False).all()
    assert reserved.traces.iloc[0].hours == development.traces.iloc[0].hours

@pytest.mark.parametrize('field', ['case_id', 'observation_sha256', 'seed_entropy'])
def test_reusing_development_identity_content_or_stream_is_refused(fixture, field):
    resolved, _ = fixture
    development = generate_development(resolved)
    reserved = generate_reserved(resolved, development)
    index = reserved.cases.index[reserved.cases.fresh_confirmation_eligible][0]
    reserved.cases.at[index, field] = development.cases.iloc[0][field]
    with pytest.raises(ValueError, match='reuses|repeats'):
        validate_separation(reserved, development)

def test_consumption_claim_survives_interruption_and_cannot_be_reassigned(fixture, tmp_path):
    resolved, paths = fixture
    development = generate_development(resolved)
    selection, _ = frozen(resolved, development)
    root = consumption_root(SimpleNamespace(table_paths=paths, output=tmp_path / 'first'))
    assert root == consumption_root(SimpleNamespace(table_paths=paths, output=tmp_path / 'second'))
    first = claim_reservation(root, selection, development.reservation)
    assert first == claim_reservation(root, selection, development.reservation)
    assert first['opened_utc']
    with pytest.raises(ValueError, match='already opened'):
        claim_reservation(root, {**selection, 'selection_id': 'runner-up'}, development.reservation)

@pytest.mark.parametrize('kind,expected', [('passes', 'supported'), ('fails', 'failed'), ('small', 'insufficient')])
def test_confirmation_checks_frozen_recovery_without_changing_choice(fixture, kind, expected):
    resolved, _ = fixture
    selection, _ = frozen(resolved, generate_development(resolved))
    candidate = resolved.candidates[0].candidate_id
    evidence = score(candidate, recovery=0.0 if kind == 'fails' else 1.0, n=3 if kind == 'small' else 1000)
    checks, decisions = assess_confirmation(selection, pd.DataFrame([evidence]))
    assert checks[0]['status'] == decisions[0]['confirmation_status'] == expected
    assert decisions[0]['candidate_ids'] == selection['decisions'][0]['candidate_ids']
    assert decisions[0]['no_replacement']
    if kind == 'fails':
        assert checks[0]['gates'][-1]['state'] == 'fail' and (not decisions[0]['promotion_allowed'])

def test_tied_checks_cannot_select_the_best_confirmation_result(fixture):
    resolved, _ = fixture
    second = replace(resolved.candidates[0], candidate_id='other')
    resolved = replace(resolved, candidates=(*resolved.candidates, second))
    selection, _ = frozen(resolved, generate_development(resolved))
    assert selection['decisions'][0]['state'] == 'tie'
    evidence = pd.DataFrame([score(resolved.candidates[0].candidate_id), score('other', recovery=0.0)])
    _, decisions = assess_confirmation(selection, evidence)
    assert decisions[0]['final_state'] == 'tie' and len(decisions[0]['candidate_ids']) == 2
    assert not decisions[0]['promotion_allowed']

def test_complete_confirmation_recipe_reads_saved_evidence_on_resume(fixture, tmp_path, monkeypatch):
    resolved, paths = fixture
    declaration = resolved.request.declaration.as_dict()
    declaration['score_policy'].update(false_alarm_limit=1.0, confidence=0.1, min_positive=1, min_negative=1)
    resolved = resolve_request(parse([declaration])[0], source_run='run', tables={'cell_frame': pd.read_csv(paths['cell_frame'])}, input_hashes=resolved.inputs.table_hashes)
    calls = []

    def estimate(hours, values, params, method, **kwargs):
        calls.append(method)
        return {'method': method, 'status': 'ok', 'period_hours': 12.0, 'p_value': 0.001, 'components': [], 'diagnostics': {}}
    monkeypatch.setattr(circadian, 'estimate_one', estimate)
    first = run_request(resolved, paths, tmp_path / 'confirmed', only=('independent-confirmation',))
    saved = first.results['independent-confirmation']
    assert first.successful, saved.outcome.reason
    record = read_document(saved.artifact('confirmation_record'))
    assert read_confirmation(saved, expected_selection=record['selection_id']) == record
    assert record['confirmation_opened'] and record['fresh_cases'] == record['nonfresh_cases'] == 6
    assert len(record['candidate_ids']) == 1 and record['consumption']['selection_id'] == record['selection_id']
    results = read_table(saved.artifact('results'))
    scores = read_table(saved.artifact('case_scores'))
    assert len(results) == len(scores) == 12 and scores.confirmation_eligible.sum() == 6
    assert read_table(saved.artifact('families')).requested.sum() == 12
    count = len(calls)
    monkeypatch.setattr(circadian, 'generate_benchmark_cases', lambda *a, **k: pytest.fail('resume regenerated cases'))
    second = run_request(resolved, paths, tmp_path / 'confirmed', only=('independent-confirmation',), presentation={'columns': 2})
    assert second.results['independent-confirmation'].outcome.status == 'reused' and len(calls) == count
    assert read_document(second.results['independent-confirmation'].artifact('confirmation_record')) == record

def test_no_provisional_choice_skips_without_consuming_cases(fixture, tmp_path, monkeypatch):
    resolved, paths = fixture

    def estimate(hours, values, params, method, **kwargs):
        return {'method': method, 'status': 'ok', 'period_hours': 12.0, 'p_value': 0.5, 'components': [], 'diagnostics': {}}
    monkeypatch.setattr(circadian, 'estimate_one', estimate)
    original = circadian.generate_benchmark_cases

    def development(*args, **kwargs):
        assert kwargs['partition'] == 'development'
        return original(*args, **kwargs)
    monkeypatch.setattr(circadian, 'generate_benchmark_cases', development)
    execution = run_request(resolved, paths, tmp_path / 'skip', only=('independent-confirmation',))
    saved = execution.results['independent-confirmation']
    assert execution.successful, saved.outcome.reason
    record = read_document(saved.artifact('confirmation_record'))
    assert not record['confirmation_opened'] and record['consumption'] is None
    assert record['final_decisions'][0]['confirmation_status'] == 'skipped'
    assert not (tmp_path / '.pipeline-confirmation').exists()
