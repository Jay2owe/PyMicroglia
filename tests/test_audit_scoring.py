"""Known-outcome scores retain failed answers and independent simulation support."""
from dataclasses import replace
import pandas as pd
import pytest
from pymicroglia import workbench as circadian
from pymicroglia.pipelines.audit.benchmarks import generate_development
from pymicroglia.pipelines.audit.evaluator import Evaluation
from pymicroglia.pipelines.audit.scoring import match_components, rate_units, score_cases, summarize
from pymicroglia.pipelines.audit.workflow import run_request
from pymicroglia.pipelines._screening import read_table
from tests.test_audit_benchmarks import fixture

def responses(cases, **overrides):
    rows = []
    for case in cases.to_dict('records'):
        rows.append({'case_id': case['case_id'], 'period_hours': 12.0, 'period_available': True, 'period_underdetermined': False, 'test_status': 'ok', 'estimate_status': 'ok', 'p_value': 0.001, 'q_value': 0.002, 'components': [], **overrides})
    return Evaluation(pd.DataFrame(rows), pd.DataFrame(), pd.DataFrame())

def test_hand_checked_recovery_false_alarms_and_insufficiency(fixture):
    resolved, _ = fixture
    cases = generate_development(resolved)
    scores = score_cases(resolved.candidates[0], responses(cases.cases), cases, resolved.request.benchmark_design['truth_policy'])
    assert scores.eligible_positive.sum() == 6
    assert scores.eligible_negative.sum() == 4
    assert scores.recovered.sum() == 2
    assert scores.false_alarm.sum() == 4
    assert scores.insufficient_response.eq('unsupported_detection').sum() == 2
    assert scores.loc[scores.scenario.eq('multiple'), 'component_matching'].map(lambda x: len(x['missed'])).eq(2).all()

def test_missing_failed_and_unsupported_answers_stay_in_denominators(fixture):
    resolved, _ = fixture
    cases = generate_development(resolved)
    failed = responses(cases.cases, test_status='failed', p_value=None, q_value=None, period_available=False, period_hours=None)
    failed = replace(failed, results=failed.results.iloc[2:])
    scores = score_cases(resolved.candidates[0], failed, cases, resolved.request.benchmark_design['truth_policy'])
    assert len(scores) == 12 and scores.eligible_positive.sum() == 6
    assert not scores.valid_test.any() and (not scores.recovered.any())
    assert scores.test_status.eq('missing_evaluation').sum() == 2
    unsupported = score_cases(resolved.candidates[0], responses(cases.cases, period_underdetermined=True), cases, resolved.request.benchmark_design['truth_policy'])
    assert not unsupported.recovered.any() and unsupported.false_alarm.sum() == 4
    summary = summarize(scores, confidence=0.95)
    overall = summary[summary.scope.eq('dataset')].iloc[0]
    assert overall.recovery_denominator == 6 and overall.recovery_rate == 0
    assert overall.false_alarm_denominator == 4 and overall.false_alarm_valid_fraction == 0

@pytest.mark.parametrize('estimated,label', [(6.0, 'half-period'), (24.0, 'double-period')])
def test_period_aliases_remain_visible(fixture, estimated, label):
    resolved, _ = fixture
    cases = generate_development(resolved)
    scores = score_cases(resolved.candidates[0], responses(cases.cases, period_hours=estimated), cases, resolved.request.benchmark_design['truth_policy'])
    assert scores.loc[scores.scenario.eq('positive'), 'period_alias'].eq(label).all()

def test_component_assignment_is_one_to_one_and_respects_extra_policy(fixture):
    resolved, _ = fixture
    policy = resolved.request.benchmark_design['truth_policy'].copy()
    matched = match_components([{'period_hours': 10.0}, {'period_hours': 11.0}], [10.5], policy)
    assert len(matched['matches']) == 1 and len(matched['missed']) == 1
    cases = generate_development(resolved)
    evaluation = responses(cases.cases, period_hours=6.0, components=[{'period_hours': 6.0}, {'period_hours': 30.0}, {'period_hours': 42.0}])
    penalized = score_cases(resolved.candidates[0], evaluation, cases, policy)
    assert not penalized.loc[penalized.scenario.eq('multiple'), 'recovered'].any()
    ignored = score_cases(resolved.candidates[0], evaluation, cases, {**policy, 'extra_components': 'ignore'})
    assert ignored.loc[ignored.scenario.eq('multiple'), 'recovered'].all()

def test_duplicate_realizations_do_not_add_independent_support_and_weights_survive(fixture):
    resolved, _ = fixture
    cases = generate_development(resolved)
    scored = score_cases(resolved.candidates[0], responses(cases.cases), cases, resolved.request.benchmark_design['truth_policy'])
    mixed = scored[scored.scenario.eq('multiple')]
    units = rate_units(mixed, 'eligible_positive', 'recovered')
    assert len(units) == 2 and len({u['dependency_id'] for u in units}) == 1
    interval = circadian.benchmark_score_interval(units, confidence=0.95, bounds=[0.0, 1.0])
    assert interval['independent_units'] == 1 and interval['upper'] == 1
    scored.loc[scored.scenario.eq('positive'), 'weight'] = 4.0
    summary = summarize(scored, confidence=0.95)
    overall = summary[summary.scope.eq('dataset')].iloc[0]
    assert overall.recovery_rate == pytest.approx(8 / 12)
    assert overall.recovery_weighted_numerator == 8 and overall.recovery_weighted_denominator == 12
    assert {'periods', 'noise', 'waveforms', 'span_hours', 'source_profile_id'} <= set(summary.facet)

def test_real_scoring_recipe_saves_complete_evidence_without_confirmation(fixture, tmp_path, monkeypatch):
    resolved, paths = fixture
    called = []
    generate = circadian.generate_benchmark_cases

    def development(*args, **kwargs):
        called.append(kwargs['partition'])
        assert kwargs['partition'] == 'development'
        return generate(*args, **kwargs)
    monkeypatch.setattr(circadian, 'generate_benchmark_cases', development)
    first = run_request(resolved, paths, tmp_path / 'scoring', only=('development-scores',))
    assert first.successful, first.results['development-scores'].outcome.reason
    result = first.results['development-scores']
    scores = read_table(result.artifact('case_scores'))
    assert len(scores) == 12 and scores.partition.eq('development').all()
    assert scores.case_id.is_unique and scores.valid_test.any()
    before = scores.copy(deep=True)
    again = run_request(resolved, paths, tmp_path / 'scoring', only=('development-scores',), presentation={'columns': 3})
    assert again.results['development-scores'].outcome.status == 'reused'
    pd.testing.assert_frame_equal(before, read_table(again.results['development-scores'].artifact('case_scores')))
    assert called == ['development']
