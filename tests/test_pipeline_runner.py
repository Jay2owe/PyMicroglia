"""Dependent execution, immutable reuse and declared figure-source integration."""
from __future__ import annotations
import json
from dataclasses import replace
from pathlib import Path
import pandas as pd
import pytest
import pymicroglia.pipelines._screening as screening
from pymicroglia.pipelines._contracts import ArtifactRef, CellKey, CellMeasurementKey, MeasurementPair, PipelineRecipe, SelectionRecord, Settings, StepResult, StepSpec, result_from_dict, result_to_dict
from pymicroglia.pipelines.rhythm.discovery import producers, run_request
from pymicroglia.pipelines._runner import Producer, Unavailable, dependency_order, figure_binding, register_figure_plan, run_pipeline
from tests.test_pipeline_screening import fake_methods, inputs

@pytest.fixture(autouse=True)
def fast_producer_identity(monkeypatch):
    monkeypatch.setattr(screening, 'producer_identity', lambda: {'implementation': 'screen-test-v1'})

def export_members(context):
    saved = screening.read_screen(context.saved('rhythm-screen').root)
    if context.selection is None:
        members = saved.results[['source_run', 'movie', 'identity', 'measurement']]
    else:
        members = pd.DataFrame([member.as_dict() for member in context.selection.members])
    context.output.mkdir(parents=True)
    path = context.output / 'members.json'
    path = screening.write_table(path, members)
    ref = ArtifactRef('members', path.name, screening.file_hash(path), context.scientific_id, columns=tuple(members.columns))
    return StepResult(context.step.name, context.scientific_id, 'completed', 'Saved selected members', (ref,), provenance=Settings({'presentation': context.presentation}))

def fail_producer(context):
    raise RuntimeError('deliberate producer failure')

def unavailable_producer(context):
    raise Unavailable('selected method exports no required timing capability')

def recipe():
    return PipelineRecipe('miniature', 1, (StepSpec('rhythm-screen', 'rhythm-screen', inputs=('measured-tables',)), StepSpec('overview', 'members', ('rhythm-screen',), inputs=('rhythm-screen:rhythm_results',), kind='render'), StepSpec('cards', 'members', ('rhythm-screen',), selection='rhythm-screen:any-significant', requires_selected_rows=True, kind='render')))

def registry():
    return {**producers(), 'members': Producer(export_members), 'failure': Producer(fail_producer), 'unavailable': Producer(unavailable_producer)}

@pytest.mark.parametrize('failed_prerequisite', [False, True])
def test_empty_scientific_selection_propagates_through_dependent_skips(tmp_path, failed_prerequisite):

    def support(context):
        return StepResult(context.step.name, context.scientific_id, 'completed', 'No supported states', selections=(SelectionRecord('accepted-model', context.scientific_id, Settings({'reason': 'Complete validation did not support discrete states'}), ()),))

    def should_not_run(context):
        raise AssertionError('An empty accepted-model selection attempted dependent work')
    steps = [StepSpec('support', 'support'), StepSpec('assignments', 'assignments', ('support',), selection='support:accepted-model', requires_selected_rows=True), StepSpec('durations', 'durations', ('support', 'assignments'), inputs=('assignments:assigned-observations',), selection='support:accepted-model', requires_selected_rows=True), StepSpec('figures', 'figures', ('support', 'durations'), inputs=('durations:duration-table',), selection='support:accepted-model', requires_selected_rows=True, kind='render')]
    producers = {name: Producer(should_not_run) for name in ('assignments', 'durations', 'figures')}
    producers['support'] = Producer(support)
    if failed_prerequisite:
        steps.insert(1, StepSpec('failure', 'failure'))
        steps[2] = replace(steps[2], prerequisites=('support', 'failure'))
        producers['failure'] = Producer(fail_producer)
    result = run_pipeline(PipelineRecipe('conditional-chain', 1, tuple(steps)), producers, request=Settings(), scientific_settings={}, output=tmp_path / 'pipeline')
    assert result.successful is not failed_prerequisite
    expected = 'unavailable' if failed_prerequisite else 'skipped-empty'
    assert all((result.results[name].outcome.status == expected for name in ('assignments', 'durations', 'figures')))

def execute(tmp_path, resolved, paths, **kwargs):
    return run_request(resolved, paths, tmp_path / 'pipeline', recipe=recipe(), registry=registry(), **kwargs)

def test_dependent_members_and_rerender_do_not_refit(tmp_path, fake_methods):
    resolved, paths = inputs(tmp_path)
    first = execute(tmp_path, resolved, paths, presentation={'columns': 3})
    assert first.successful
    assert [saved.outcome.status for saved in first.results.values()] == ['completed'] * 3
    cards = screening.read_table(first.results['cards'].artifact('members'))
    assert set(zip(cards.movie, cards.identity)) == {('a', 7), ('b', 7)}
    fitted = len(fake_methods)
    repeated = execute(tmp_path, resolved, paths, presentation={'columns': 3})
    assert [saved.outcome.status for saved in repeated.results.values()] == ['reused'] * 3
    assert len(fake_methods) == fitted
    changed = execute(tmp_path, resolved, paths, presentation={'columns': 5})
    assert changed.results['rhythm-screen'].outcome.status == 'reused'
    assert changed.results['cards'].outcome.status == 'completed'
    assert len(fake_methods) == fitted
    assert changed.results['cards'].outcome.scientific_id == first.results['cards'].outcome.scientific_id
    assert changed.results['cards'].root != first.results['cards'].root
    assert first.record_path.is_file() and repeated.record_path.is_file()

@pytest.mark.parametrize('request_options', [{'empty': True}, {'options': {'rhythmic_alpha': 1e-08}}, {'options': {'min_observations': 24}}])
def test_empty_selection_runs_overview_and_skips_cards(tmp_path, fake_methods, request_options):
    resolved, paths = inputs(tmp_path, **request_options)
    execution = execute(tmp_path, resolved, paths)
    assert execution.successful
    assert execution.results['rhythm-screen'].outcome.status == 'completed'
    assert execution.results['overview'].outcome.status == 'completed'
    assert execution.results['cards'].outcome.status == 'skipped-empty'
    record = execution.record()
    assert record['steps'][-1]['selection']
    assert record['steps'][-1]['selection_source'] == 'rhythm-screen:any-significant'

@pytest.mark.parametrize('steps, message', [((StepSpec('a', 'members', ('absent',)),), 'missing prerequisite'), ((StepSpec('a', 'members', ('b',)), StepSpec('b', 'members', ('a',))), 'cycle'), ((StepSpec('a', 'unknown'),), 'unregistered'), ((StepSpec('a', 'members', kind='render'), StepSpec('b', 'members', ('a',))), 'cannot depend')])
def test_invalid_recipe_fails_before_side_effects(tmp_path, steps, message):
    with pytest.raises(ValueError, match=message):
        run_pipeline(PipelineRecipe('bad', 1, steps), registry(), request=None, scientific_settings={}, output=tmp_path / 'untouched')
    assert not (tmp_path / 'untouched').exists()

def test_failure_and_unavailable_are_distinct_and_block_only_dependants(tmp_path):
    plan = PipelineRecipe('outcomes', 1, (StepSpec('failed', 'failure'), StepSpec('missing', 'unavailable'), StepSpec('child', 'failure', ('failed',))))
    execution = run_pipeline(plan, registry(), request=None, scientific_settings={}, output=tmp_path)
    assert not execution.successful
    assert execution.results['failed'].outcome.status == 'failed'
    assert execution.results['missing'].outcome.status == 'unavailable'
    assert execution.results['child'].outcome.status == 'unavailable'
    assert 'deliberate producer failure' in execution.results['child'].outcome.reason
    assert 'timing capability' in execution.results['missing'].outcome.reason

def test_changed_science_and_changed_source_cannot_reuse(tmp_path, fake_methods):
    resolved, paths = inputs(tmp_path)
    first = execute(tmp_path, resolved, paths)
    other, paths = inputs(tmp_path, scope='measurement')
    changed = execute(tmp_path, other, paths)
    assert changed.results['rhythm-screen'].outcome.status == 'completed'
    assert changed.results['rhythm-screen'].outcome.scientific_id != first.results['rhythm-screen'].outcome.scientific_id
    fitted = len(fake_methods)
    with paths['cell_frame'].open('a', encoding='utf-8') as stream:
        stream.write('\n')
    failed = execute(tmp_path, other, paths)
    assert failed.results['rhythm-screen'].outcome.status == 'failed'
    assert 'fingerprint' in failed.results['rhythm-screen'].outcome.reason
    assert len(fake_methods) == fitted

def test_partial_or_corrupt_results_recompute_without_overwriting(tmp_path, fake_methods):
    resolved, paths = inputs(tmp_path)
    first = execute(tmp_path, resolved, paths)
    original = first.results['rhythm-screen']
    original.artifact('trace_inputs').write_text('corrupt', encoding='utf-8')
    fitted = len(fake_methods)
    next_run = execute(tmp_path, resolved, paths)
    assert next_run.results['rhythm-screen'].outcome.status == 'completed'
    assert len(fake_methods) > fitted
    assert next_run.results['rhythm-screen'].root != original.root
    assert (original.root / 'trace_inputs.csv').read_text(encoding='utf-8') == 'corrupt'

def test_selective_regeneration_leaves_other_outputs_and_records(tmp_path, fake_methods):
    resolved, paths = inputs(tmp_path)
    first = execute(tmp_path, resolved, paths)
    record = first.record()
    overview = first.results['overview'].artifact('members')
    original = overview.read_bytes()
    subset = execute(tmp_path, resolved, paths, presentation={'columns': 9}, only=('cards',))
    assert set(subset.results) == {'rhythm-screen', 'cards'}
    assert first.record() == record
    assert overview.read_bytes() == original

def test_typed_selection_keys_round_trip():
    cell = CellKey('run', 'movie', 1)
    members = (cell, CellMeasurementKey(cell, 'speed'), MeasurementPair('a', 'b'), Settings({'state': 2, 'movie': 'movie'}))
    result = StepResult('step', 'science', 'completed', 'saved', selections=(SelectionRecord('members', 'science', Settings({'rule': 'test'}), members),))
    assert result_from_dict(json.loads(json.dumps(result_to_dict(result)))) == result

def test_figure_reads_track_science_settings_selections_and_leave_static_plan(tmp_path, fake_methods, monkeypatch):
    from pymicroglia.figure_tables.pipeline_inputs import SavedInputs
    from pymicroglia.visualisation.figures._declare import Figure,View,Table
    from pymicroglia._results import read_document
    resolved, paths = inputs(tmp_path)
    execution = execute(tmp_path, resolved, paths)
    static = tmp_path / 'figures.json'
    static.write_text(json.dumps({'plots': [{'name': 'ordinary', 'figure': 'demo', 'options': {}}]}), encoding='utf-8')
    original = static.read_bytes()
    binding = figure_binding({'screen': execution.results['rhythm-screen']}, inputs={'rhythm_results.json': ('screen', 'rhythm_results')}, selections={'cells': ('screen', 'any-significant')})
    item = {'name': 'saved-cells', 'figure': 'demo', 'options': {'bins': 9}, 'pipeline': binding}
    plan = register_figure_plan(tmp_path, [item])
    assert read_document(plan)['figure_plans']['saved-cells']==item
    assert static.read_bytes() == original
    spec = Figure('demo','Saved cells','review',views=(View('matrix',lambda *a:None),),reads=(Table('rhythm_results.json',scope='pipeline'),),prepare='unused')
    context = SavedInputs(run=tmp_path,spec=spec,item='saved-cells',options=item['options'],binding=binding)
    monkeypatch.setattr(screening, 'screen', lambda *a, **k: pytest.fail('display must not fit'))
    frame = context.table('rhythm_results.json')
    assert len(frame) == 6 and context.has('rhythm_results.json:period_hours')
    assert len(context.pipeline_selection('cells').members) == 2
    assert context.pipeline_metadata('screen')['resolved_request']['analysis_options']['fit_method'] == 'mesa'
    assert set(context.sources)=={'screen_provenance','rhythm_results.json'}
    with pytest.raises(KeyError, match='undeclared'):
        context.table('undeclared.json')
    with pytest.raises(ValueError, match='different plan'):
        register_figure_plan(tmp_path, [{**item, 'options': {'bins': 4}}])

def test_figure_plan_can_retry_with_reused_scientific_inputs(tmp_path, fake_methods):
    resolved, paths = inputs(tmp_path)
    first = execute(tmp_path, resolved, paths)
    original = figure_binding({'screen': first.results['rhythm-screen']}, inputs={'rhythm_results.json': ('screen', 'rhythm_results')})
    item = {'name': 'retryable-figure', 'figure': 'demo', 'options': {}, 'pipeline': original}
    plan = register_figure_plan(tmp_path, [item])
    calls = len(fake_methods)
    repeated = execute(tmp_path, resolved, paths)
    rebound = figure_binding({'screen': repeated.results['rhythm-screen']}, inputs={'rhythm_results.json': ('screen', 'rhythm_results')})
    assert repeated.results['rhythm-screen'].outcome.status == 'reused' and len(fake_methods) == calls
    assert rebound == original and register_figure_plan(tmp_path, [{**item, 'pipeline': rebound}]) == plan

def test_cli_runs_request_from_saved_pooled_tables(tmp_path, fake_methods):
    from pymicroglia.cli import main
    resolved, paths = inputs(tmp_path)
    run = tmp_path / 'run'
    tables = run / 'pooled' / 'tables'
    tables.mkdir(parents=True)
    for path in paths.values():
        (tables / path.name).write_bytes(path.read_bytes())
    (run / 'manifest.json').write_text(json.dumps({'movies': [{'stem': 'a', 'modules': []}, {'stem': 'b', 'modules': []}]}), encoding='utf-8')
    declaration = tmp_path / 'request.json'
    declaration.write_text(json.dumps(resolved.request.declaration.as_dict()), encoding='utf-8')
    assert main(['run','rhythm_discovery',f'run={run}',f'pipeline_request={declaration}',"only=['rhythm-screen']",'if_exists=skip','--claim','Synthetic pipeline command replay check']) == 0
    fitted = len(fake_methods)
    assert main(['run','rhythm_discovery',f'run={run}',f'pipeline_request={declaration}',"only=['rhythm-screen']",'if_exists=skip','--claim','Synthetic pipeline command replay check']) == 0
    assert len(fake_methods) == fitted
    from pymicroglia.pipelines import _records
    assert len(_records.read(run/'pipelines'/'rhythm-discovery')['invocations'])==2

def test_outcome_aware_render_chain_reuses_first_complete_run(tmp_path):
    calls = []

    def emit(context):
        calls.append(context.step.name)
        context.output.mkdir(parents=True)
        path = context.output / 'original.txt'
        path.write_text(context.step.name, encoding='utf-8')
        return StepResult(context.step.name, context.scientific_id, 'completed', 'Original completed outcome', (ArtifactRef('original', path.name, screening.file_hash(path), context.scientific_id),))
    recipe = PipelineRecipe('outcome-aware', 1, (StepSpec('science', 'science'), StepSpec('figure', 'figure', ('science',), kind='render'), StepSpec('index', 'index', ('science', 'figure'), kind='render')))
    registry = {'science': Producer(emit), **{name: Producer(emit, accepts_unavailable_dependencies=True) for name in ['figure', 'index']}}
    arguments = dict(request=Settings(), scientific_settings={}, output=tmp_path / 'pipeline')
    first = run_pipeline(recipe, registry, **arguments)
    second = run_pipeline(recipe, registry, **arguments)
    assert first.successful and second.successful and (calls == ['science', 'figure', 'index'])
    assert all((value.outcome.status == 'reused' for value in second.results.values()))
    assert all((second.results[name].root == value.root and second.results[name].outcome.scientific_id == value.outcome.scientific_id for name, value in first.results.items()))
    assert all((value.outcome.status == 'completed' and value.outcome.reason == 'Original completed outcome' for value in first.results.values()))

def test_changed_unavailable_reason_still_invalidates_outcome_aware_report(tmp_path):
    state = {'reason': 'No observations'}
    calls = []

    def source(context):
        return StepResult(context.step.name, context.scientific_id, 'unavailable', state['reason'])

    def report(context):
        calls.append(context.saved('science').outcome.reason)
        context.output.mkdir(parents=True)
        path = context.output / 'reason.txt'
        path.write_text(calls[-1], encoding='utf-8')
        return StepResult(context.step.name, context.scientific_id, 'completed', 'Saved original failure reason', (ArtifactRef('reason', path.name, screening.file_hash(path), context.scientific_id),))
    recipe = PipelineRecipe('unavailable-report', 1, (StepSpec('science', 'science'), StepSpec('report', 'report', ('science',), kind='render')))
    registry = {'science': Producer(source), 'report': Producer(report, accepts_unavailable_dependencies=True)}
    args = dict(request=Settings(), scientific_settings={}, output=tmp_path / 'pipeline')
    first = run_pipeline(recipe, registry, **args)
    again = run_pipeline(recipe, registry, **args)
    assert again.results['report'].outcome.status == 'reused' and (not again.successful)
    state['reason'] = 'Required method unavailable'
    changed = run_pipeline(recipe, registry, **args)
    assert not first.successful and (not changed.successful) and (calls == ['No observations', 'Required method unavailable'])
    assert changed.results['report'].outcome.scientific_id != first.results['report'].outcome.scientific_id
