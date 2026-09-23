"""Complete families, distinct evidence levels and saved report decisions."""
import json
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
import pytest
from pymicroglia import workbench as circadian
import pymicroglia.pipelines.coordination.evidence as evidence, pymicroglia.pipelines.coordination.inputs as inputs
from pymicroglia.pipelines._contracts import Settings, StepResult
from pymicroglia.pipelines.coordination.options import BRANCHES, resolve_request, run_request
from pymicroglia.pipelines._runner import SavedResult, Unavailable
from pymicroglia.pipelines._screening import file_hash
from tests.test_coordination_options import request, tables
INFERENCE = Settings({'alpha': 0.05, 'multiple_testing': 'bonferroni', 'correction_scope': 'all'})

def row(name, question='simultaneous', level='pair', p=0.01, **changes):
    return {'effect_id': name, 'source_run': 'source', 'movie': 'one', 'source_result_id': name, 'pair_id': 'pair', 'question': question, 'evidence_level': level, 'p_value': p, 'effect': 0.8, 'is_hypothesis': True, 'formal_hypothesis': True, 'status': 'tested' if p is not None else 'untestable', 'reason': 'Controlled saved result', 'method': 'native', 'representation': 'raw', 'adjustment': 'none', **changes}

def inventory():
    return pd.DataFrame([{**{key: 'declared-' + key for key in inputs.PAIR_KEYS}, 'source_run': 'source', 'movie': 'one', 'pair_id': 'pair', 'reference_identity': 7, 'target_identity': 8, 'sample_confirmed': True, 'oriented': False}])

def select(rows):
    members = [{'effect_id': item['effect_id'], 'pair_id': 'pair'} for item in rows if item['evidence_level'] == 'pair']
    return evidence.selections(inventory(), rows, members, 'evidence', INFERENCE)

def test_recording_support_never_becomes_pair_support():
    rows, families = evidence.resolve_evidence([row('field', 'characteristics', 'recording', 0.001), row('edge', 'characteristics', p=None, formal_hypothesis=False, status='descriptive')], INFERENCE)
    pairs, recordings, selected = select(rows)
    choices = {item.name: item for item in selected}
    assert recordings[0]['supported'] and len(choices['supported-recordings'].members) == 1
    assert not choices['supported-pairs'].members and (not pairs[0]['supported_effect_ids'])
    assert rows[1]['q_value'] is None and rows[1]['decision'] == 'descriptive'
    assert families[0]['evidence_level'] == 'recording' and families[0]['requested'] == 1

def test_untestable_slots_and_adjusted_views_stay_in_the_complete_family():
    original = [row('raw', p=0.02), row('adjusted', p=0.03, adjustment='leave_pair_out'), row('missing', p=None), row('baseline', p=None, is_hypothesis=False, formal_hypothesis=False, status='descriptive')]
    rows, families = evidence.resolve_evidence(original, INFERENCE)
    assert original[0].get('q_value') is None
    assert len(families) == 1 and families[0]['requested'] == 3 and (families[0]['tested'] == 2)
    assert rows[0]['q_value'] == pytest.approx(0.06) and rows[1]['q_value'] == pytest.approx(0.09)
    assert rows[2]['p_value'] is None and rows[2]['q_value'] is None and (rows[2]['family_requested'] == 3)
    assert rows[3]['family_id'] is None and (not any((item['supported'] for item in rows)))

def test_level_and_declared_question_scopes_are_distinct():
    original = [row('sim'), row('lag', 'delay'), row('spatial', 'characteristics', 'recording')]
    all_rows, families = evidence.resolve_evidence(original, INFERENCE)
    assert sorted((item['requested'] for item in families)) == [1, 2]
    assert all_rows[2]['q_value'] == 0.01 and all_rows[0]['q_value'] == 0.02
    _, families = evidence.resolve_evidence(original, {**INFERENCE.as_dict(), 'correction_scope': 'question'})
    assert len(families) == 3 and all((item['requested'] == 1 for item in families))

def test_full_search_support_does_not_certify_an_unresolved_delay():
    original = [row('resolved', 'delay', p=0.001, resolution_status='resolved', estimated_delay_hours=-1.0, delay_interval_hours=[-1.25, -0.75], candidate_lags_hours=[-1.25, -1.0, -0.75], lag_convention='Negative means reference leads'), row('boundary', 'delay', p=0.001, resolution_status='search-boundary', estimated_delay_hours=None), row('weak', 'delay', p=0.9, resolution_status='resolved', estimated_delay_hours=-1.0)]
    rows, _ = evidence.resolve_evidence(original, INFERENCE)
    assert rows[0]['delay_supported'] and rows[0]['delay_hours'] == -1.0
    assert rows[0]['delay_direction_supported'] and rows[0]['delay_direction'] == 'reference_leads_target'
    assert rows[1]['supported'] and (not rows[1]['delay_supported']) and (rows[1]['delay_hours'] is None)
    assert not rows[2]['supported'] and rows[2]['estimated_delay_hours'] == -1.0 and (rows[2]['delay_hours'] is None)
    choices = {item.name: item for item in select(rows)[2]}
    assert len(choices['resolved-delays'].members) == 1 and choices['resolved-delays'].members[0]['effect_id'] == 'resolved'

def test_a_resolved_region_containing_zero_has_no_supported_direction():
    rows, _ = evidence.resolve_evidence([row('near-zero', 'delay', p=0.001, resolution_status='resolved', estimated_delay_hours=0.25, candidate_lags_hours=[0.0, 0.25, 0.5]), row('positive', 'delay', p=0.001, resolution_status='resolved', estimated_delay_hours=1.0, candidate_lags_hours=[0.75, 1.0, 1.25])], INFERENCE)
    assert rows[0]['delay_supported'] and (not rows[0]['delay_direction_supported']) and (rows[0]['delay_direction'] == 'unresolved')
    assert rows[1]['delay_direction_supported'] and rows[1]['delay_direction'] == 'target_leads_reference'

def test_native_timing_comparability_is_separate_from_pair_significance():
    rows, families = evidence.resolve_evidence([row('timing', 'rhythm', p=None, formal_hypothesis=False, status='eligible', trace_q_value=0.001)], {})
    choices = {item.name: item for item in select(rows)[2]}
    assert not families and rows[0]['decision'] == 'comparable_timing' and (rows[0]['q_value'] is None)
    assert len(choices['timing-comparable-pairs'].members) == 1 and (not choices['supported-pairs'].members)

def test_state_measurement_memberships_do_not_multiply_hypotheses():
    rows, families = evidence.resolve_evidence([row('state', 'states', p=0.04, state_pair_id='whole-cell-state-pair')], INFERENCE)
    base = inventory()
    second = base.copy()
    second['pair_id'] = 'other-measurement'
    population = pd.concat([base, second], ignore_index=True)
    members = [{'effect_id': 'state', 'pair_id': pair} for pair in ['pair', 'other-measurement']]
    pairs, _, selected = evidence.selections(population, rows, members, 'evidence', INFERENCE)
    assert families[0]['requested'] == 1 and rows[0]['q_value'] == 0.04 and (len(pairs) == 2)
    assert all((item['supported_effect_ids'] == ['state'] for item in pairs))
    assert len(next((item for item in selected if item.name == 'supported-effects')).members) == 1

def test_disabled_branches_need_no_artifacts_but_enabled_missing_outputs_fail(tmp_path):
    saved = {step: SavedResult(tmp_path, StepResult(step, step, 'skipped-empty', 'Explicitly disabled')) for step in BRANCHES}
    context = SimpleNamespace(request=SimpleNamespace(request=SimpleNamespace(questions={q: {'enabled': False} for q in BRANCHES.values()})), dependencies=saved, saved=lambda step: saved[step])
    result = evidence.collect(context, {'inventory': inventory()})
    assert not result[0] and len(result[2]) == 6 and all((item['status'] == 'disabled' for item in result[2]))
    context.request.request.questions['simultaneous'] = {'enabled': True}
    with pytest.raises(Unavailable, match='did not complete'):
        evidence.collect(context, {'inventory': inventory()})
    saved['simultaneous-coordination'] = SavedResult(tmp_path, StepResult('simultaneous-coordination', 'sim', 'completed', 'Success without required artifact'))
    with pytest.raises(Unavailable, match='no unique saved artifact'):
        evidence.collect(context, {'inventory': inventory()})

def test_reordering_has_no_effect_on_families_or_scientific_selection():
    original = [row('b', p=0.001), row('a', p=0.9), row('c', p=None)]
    first, families = evidence.resolve_evidence(original, INFERENCE)
    second, again = evidence.resolve_evidence(list(reversed(original)), INFERENCE)
    assert families == again
    assert {item['effect_id']: item for item in first} == {item['effect_id']: item for item in second}
    assert select(first)[0] == select(second)[0]

def test_actual_evidence_producer_reopens_with_estimation_and_correction_blocked(tables, tmp_path):
    paths = {name: tmp_path / (name + '.csv') for name in tables}
    for name, table in tables.items():
        table.to_csv(paths[name], index=False)
    resolved = resolve_request(request(), source_run='source', tables=tables, input_hashes={name: file_hash(path) for name, path in paths.items()})
    first = run_request(resolved, paths, tmp_path / 'pipeline', only=['coordination-evidence'])
    assert first.successful, {key: value.outcome.reason for key, value in first.results.items()}
    output = evidence.read_evidence(first.results['coordination-evidence'], first.results['pair-inputs'].outcome.scientific_id)
    assert len(output['effects']) == 6 and len(output['pair_decisions']) == 6 and output['families'].empty
    assert output['effects'].decision.eq('descriptive').all()
    assert not next((item for item in first.results['coordination-evidence'].outcome.selections if item.name == 'supported-pairs')).members
    import pymicroglia.pipelines.coordination.simultaneous as coordination_simultaneous
    with patch.object(circadian, 'adjust_pvalues', side_effect=AssertionError('No new correction')), patch.object(evidence, 'collect', side_effect=AssertionError('No selection recomputation')), patch.object(coordination_simultaneous, 'analyse', side_effect=AssertionError('No repeated estimates')):
        second = run_request(resolved, paths, tmp_path / 'pipeline', only=['coordination-evidence'], presentation={'maps': {'edge_limit': 1}, 'cards': {'page_limit': 1}})
    assert second.successful and second.results['coordination-evidence'].outcome.status == 'reused'
    assert evidence.read_evidence(second.results['coordination-evidence'])['provenance'] == output['provenance']
