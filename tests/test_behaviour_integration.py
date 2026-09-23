"""Complete public state controls, protected membership and scientific reuse."""
from pymicroglia._results import read_document
import copy
import json
from pathlib import Path
import pandas as pd
import pytest
import pymicroglia.states.behaviour_models as behaviour_models
import pymicroglia.states.behaviour_support as behaviour_support
from pymicroglia.pipelines import parse
import tests.pipelines.behaviour.demo as demo
import pymicroglia.pipelines.behaviour.options as options
import pymicroglia.pipelines.behaviour.durations as behaviour_durations
import pymicroglia.pipelines._screening as screening
from pymicroglia.pipelines._screening import read_table

@pytest.fixture(autouse=True)
def bounded_source_identity(monkeypatch):
    monkeypatch.setattr(screening, 'producer_identity', lambda: {'implementation': 'controlled-state-integration-test'})

def execute(output, only=('state-sample-comparisons',), request=None):
    resolved, paths = demo.resolve(output)
    if request is not None:
        resolved = options.resolve_request(parse([request])[0], source_run=resolved.inputs.source_run, tables={name: pd.read_csv(path) for name, path in paths.items()}, input_hashes=resolved.inputs.table_hashes.as_dict())
    run = options.run_request(resolved, paths, output / 'science-check', only=only)
    assert run.successful, {name: item.outcome.reason for name, item in run.results.items()}
    return (run, resolved, paths)

def test_public_accepted_control_retains_complete_native_model_time_and_sample_contract(tmp_path):
    output = demo.prepare(tmp_path / 'accepted')
    run, resolved, paths = execute(output)
    decision = read_document(run.results['state-support'].artifact('support_decision'))
    assert decision['status'] == 'accepted' and decision['confirmation_evaluated']
    assignments = read_table(run.results['state-assignments'].artifact('assignments'))
    original = pd.read_csv(paths['cell_frame'])
    assert len(assignments) == len(original) and assignments.observation_id.is_unique
    learning = read_table(run.results['feature-inputs'].artifact('learning_members'))
    assert learning.observation_id.is_unique and len(learning) == 240
    assert set(assignments.loc[assignments.observation_id.isin(learning.observation_id), 'role']) == {'learning'}
    assert {'missing_features', 'outside_training_distribution'} <= set(assignments.status)
    times = read_table(run.results['durations-and-switches'].artifact('cell_statistics'))
    assert len(times) == len(resolved.inputs.cells)
    invalid = times.loc[times.movie.eq('timeline-clock-reset')].iloc[0]
    assert invalid.status == 'invalid_clock_order' and invalid.observed_hours == 0 and pd.isna(invalid.switches_per_valid_transition_hour)
    exposures = read_table(run.results['durations-and-switches'].artifact('exposures'))
    assert not exposures.movie.eq('timeline-clock-reset').any()
    point = times.loc[times.movie.eq('timeline-point') & times.identity.eq(7)].iloc[0]
    assert point.observed_hours == 0 and point.observations == 1
    summary = times.loc[times.movie.eq('timeline-point') & times.identity.eq(8)].iloc[0]
    assert summary.observations == 0
    bouts = read_table(run.results['durations-and-switches'].artifact('bouts'))
    assert bouts.bout_id.is_unique and bouts.duration_status.eq('incomplete_observed_bout').any()
    units = read_table(run.results['state-sample-comparisons'].artifact('unit_inventory'))
    paired = units.loc[units['sample'].eq('control-independent-0')].iloc[0]
    assert paired.cells == paired.recordings == 2 and paired.independent_of_state_choice
    contrasts = read_table(run.results['state-sample-comparisons'].artifact('comparisons'))
    assert len(contrasts) == 4 and set(contrasts.family_requested) == {4}
    assert (contrasts.reference_samples == 6).all() and (contrasts.target_samples == 6).all()
    assert contrasts.loc[contrasts.metric.eq('occupancy_observed'), 'significant'].all()

@pytest.mark.parametrize('scenario,expected', [('continuous', 'no_supported_states'), ('cloud', 'no_supported_states'), ('insufficient', 'inconclusive')])
def test_public_controls_do_not_force_a_state_vocabulary(tmp_path, scenario, expected):
    output = demo.prepare(tmp_path / scenario, scenario)
    run, _, _ = execute(output, only=('state-assignments',))
    decision = read_document(run.results['state-support'].artifact('support_decision'))
    assert decision['status'] == expected and (not decision['confirmation_evaluated'])
    assert not run.results['state-support'].outcome.selections[0].members
    assert run.results['state-assignments'].outcome.status == 'skipped-empty'

def test_changed_time_and_comparison_questions_reuse_unaffected_frozen_science(tmp_path, monkeypatch):
    output = demo.prepare(tmp_path / 'accepted')
    original, resolved, paths = execute(output)
    request = read_document(output / 'behaviour.json')
    request['statistics']['transition_interval_hours'] = [0.2, 0.3]

    def forbidden(*a, **k):
        raise AssertionError('Changed time or sample question refitted or reassigned states')
    for owner, name in [(behaviour_models, 'fit_candidate'), (behaviour_models, 'predict_candidate'), (behaviour_support, 'measure')]:
        monkeypatch.setattr(owner, name, forbidden)
    changed, _, _ = execute(output, request=request)
    for name in ['feature-inputs', 'candidate-models', 'state-support', 'state-assignments']:
        assert changed.results[name].outcome.status == 'reused'
        assert changed.results[name].outcome.scientific_id == original.results[name].outcome.scientific_id
    assert changed.results['durations-and-switches'].outcome.status == 'completed'
    assert changed.results['durations-and-switches'].outcome.scientific_id != original.results['durations-and-switches'].outcome.scientific_id
    monkeypatch.setattr(behaviour_durations, 'statistics', forbidden)
    sample_request = copy.deepcopy(request)
    sample_request['statistics']['comparison'] = {'method': 'none'}
    sample, _, _ = execute(output, request=sample_request)
    assert all((sample.results[name].outcome.status == 'reused' for name in ['feature-inputs', 'candidate-models', 'state-support', 'state-assignments', 'durations-and-switches']))
    assert sample.results['state-sample-comparisons'].outcome.status == 'completed'
    descriptive = read_table(sample.results['state-sample-comparisons'].artifact('comparisons'))
    assert descriptive.method.eq('none').all() and descriptive.p_value.isna().all() and descriptive.q_value.isna().all()
    assert descriptive.status.eq('descriptive').all()

def test_feature_or_learning_changes_invalidate_original_preparation(tmp_path):
    output = demo.prepare(tmp_path / 'input')
    original, resolved, paths = execute(output, only=('feature-inputs',))
    initial = original.results['feature-inputs'].outcome.scientific_id
    request = read_document(output / 'behaviour.json')
    for key, value in [('features', ['custom_signal']), ('learning', {**request['learning'], 'max_observations_per_cell': 30})]:
        changed = copy.deepcopy(request)
        changed[key] = value
        result, _, _ = execute(output, only=('feature-inputs',), request=changed)
        assert result.results['feature-inputs'].outcome.status == 'completed' and result.results['feature-inputs'].outcome.scientific_id != initial

def test_public_example_parses_and_every_recipe_producer_is_implemented():
    from importlib.resources import files
    request = parse([json.loads(files('pymicroglia').joinpath('data/behaviour-states.example.json').read_text(encoding='utf-8'))])[0]
    assert tuple((feature.column for feature in request.features)) == ('corrected_mean', 'area_px')
    registry = options.producers()
    assert len(options.RECIPE.steps) == 13
    assert all((registry[step.producer].run is not options._pending for step in options.RECIPE.steps))
    assert registry['linked-results-index'].accepts_unavailable_dependencies

def test_demo_refuses_overwrite_and_detects_changed_source_before_execution(tmp_path, monkeypatch):
    output = demo.prepare(tmp_path / 'example', 'insufficient')
    with pytest.raises(ValueError, match='new empty folder'):
        demo.prepare(output)
    path = output / 'run/pooled/tables/cell_frame.csv'
    path.write_text(path.read_text() + '\n')

    def forbidden(*a, **k):
        raise AssertionError('Changed fixture launched a subprocess')
    monkeypatch.setattr(demo, 'invoke', forbidden)
    with pytest.raises(ValueError, match='Prepared fixture changed'):
        demo.verify(output)
