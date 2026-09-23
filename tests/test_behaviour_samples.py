"""State comparisons retain experimental units, unknowns and observation spacing."""
import numpy as np
import pandas as pd
import pytest
import pymicroglia.states.behaviour_models as behaviour_models
import pymicroglia.states.behaviour_support as behaviour_support
import pymicroglia.pipelines.behaviour.durations as duration, pymicroglia.pipelines.behaviour.samples as samples, pymicroglia.pipelines._screening as screening
from pymicroglia.pipelines.behaviour.options import run_request
from tests.test_behaviour_durations import sequence, SETTINGS
from tests.test_behaviour_sample_statistics import OPTIONS
from tests.test_behaviour_validation import fixture

def concatenate(*inputs):
    return {name: pd.concat([item[name] for item in inputs], ignore_index=True) for name in inputs[0]}

def times(labels, hours, movie='a', sample='one', condition='control', role='assignment_only', confirmed=True):
    rows, states, cells = sequence(labels, hours, movie=movie)
    cells['sample'] = sample
    cells['condition'] = condition
    cells['role'] = role
    cells['sample_confirmed'] = confirmed
    return duration.statistics(rows, states, cells, {**SETTINGS, 'transition_interval_hours': [0.0, 2.0]})

def test_multiple_movies_remain_one_sample_and_explicit_weightings_differ():
    tables = concatenate(times([0, 0], [0.0, 0.5], 'a'), times([1, 1], [0.0, 2.0], 'b'))
    equal, pooled = (samples.aggregate(tables, 'mean'), samples.aggregate(tables, 'pooled_exposure'))
    unit = equal['unit_inventory'].iloc[0]
    assert unit.cells == 2 and unit.recordings == 2 and (len(equal['unit_inventory']) == 1)
    assert len(unit.members) == 2 and {row['movie'] for row in unit.members} == {'a', 'b'}
    assert equal['unit_metrics'].loc[equal['unit_metrics'].metric.eq('occupancy_observed'), 'value'].tolist() == [0.5, 0.5]
    actual = pooled['unit_metrics'].loc[pooled['unit_metrics'].metric.eq('occupancy_observed')]
    assert sorted(actual.value) == [0.2, 0.8] and actual.denominator_total.eq(2.5).all()
    assert set(pooled['cell_metrics'].identity) == {7} and set(pooled['cell_metrics'].movie) == {'a', 'b'}

def test_unknown_fraction_and_unique_censored_bouts_are_retained():
    tables = times([0, 0, None, 1, 1], [0.0, 0.5, 1.0, 1.5, 2.0])
    result = samples.aggregate(tables, 'mean')
    assert result['unit_metrics'].loc[result['unit_metrics'].metric.eq('unknown_fraction'), 'value'].iloc[0] == 0.25
    assert result['unit_inventory'].unknown_hours.iloc[0] == 0.5
    assert len(result['bout_membership']) == 2 and result['bout_membership'].bout_id.is_unique
    assert not result['bout_membership'].complete.any() and result['unit_inventory'].incomplete_bouts.iloc[0] == 2

def test_different_interval_questions_cannot_silently_pool_probabilities_or_rates():
    tables = concatenate(times([0, 1, 1], [0.0, 0.25, 0.5], 'a', 'one'), times([0, 1, 1], [0.0, 0.5, 1.0], 'b', 'two', 'treated'))
    result = samples.aggregate(tables, 'mean')
    metrics = result['unit_metrics']
    rates = metrics.loc[metrics.metric.eq('switch_rate')]
    assert len(rates) == 4 and rates.value.isna().sum() == 2
    assert sorted(rates.loc[rates.value.notna(), 'interval_hours']) == [0.25, 0.5]
    comparison, family, _, _ = samples.compare_samples(metrics, {**OPTIONS, 'metrics': ['switch_rate', 'transition_probability']})
    assert comparison.p_value.isna().all() and comparison.effect.isna().all()
    assert set(comparison.interval_hours) == {0.25, 0.5} and family.tested.iloc[0] == 0

def test_unconfirmed_and_model_choice_samples_stay_descriptive():
    tables = concatenate(times([0, 1], [0.0, 0.5], 'a', 'one', role='learning'), times([0, 1], [0.0, 0.5], 'b', 'two', 'treated', confirmed=False))
    result = samples.aggregate(tables, 'mean')
    assert set(result['unit_inventory'].level) == {'biological_sample', 'recording'}
    assert not result['unit_inventory'].independent_of_state_choice.any()
    comparison, family, details, _ = samples.compare_samples(result['unit_metrics'], OPTIONS)
    assert comparison.p_value.isna().all() and comparison.status.eq('untestable').all()
    assert comparison.reference_samples.eq(0).all() and comparison.target_samples.eq(0).all()
    assert result['unit_metrics'].value.notna().any()

def test_model_and_condition_conflicts_are_refused_before_comparison():
    tables = times([0, 1], [0.0, 0.5])
    tables['occupancy']['model_id'] = 'other'
    with pytest.raises(ValueError, match='same accepted model'):
        samples.aggregate(tables, 'mean')
    conflicting = concatenate(times([0, 1], [0.0, 0.5], 'a'), times([0, 1], [0.0, 0.5], 'b', condition='treated'))
    with pytest.raises(ValueError, match='inconsistent condition'):
        samples.aggregate(conflicting, 'mean')

def test_full_family_and_missing_probabilities_survive_display_independent_selection():
    tables = []
    for condition, labels in [('control', [0, 0, 0, 1]), ('treated', [0, 1, 1, 1])]:
        for index in range(6):
            movie = condition + str(index)
            tables.append(times(labels, [0.0, 0.5, 1.0, 1.5], movie, movie, condition))
    population = samples.aggregate(concatenate(*tables), 'mean')['unit_metrics']
    options = {**OPTIONS, 'metrics': ['occupancy_observed', 'unknown_fraction', 'switch_rate']}
    result, family, _, _ = samples.compare_samples(population, options)
    assert len(result) == family.requested.iloc[0] == 4 and family.tested.iloc[0] == 4
    assert result.loc[result.metric.eq('occupancy_observed'), 'significant'].all()
    missing = population.copy()
    missing.loc[missing.metric.eq('unknown_fraction'), 'value'] = None
    changed, altered, _, _ = samples.compare_samples(missing, options)
    assert altered.requested.iloc[0] == 4 and altered.tested.iloc[0] == 3
    np.testing.assert_allclose(result.loc[result.metric.eq('occupancy_observed'), 'q_value'], changed.loc[changed.metric.eq('occupancy_observed'), 'q_value'])
    reordered, _, _, _ = samples.compare_samples(population.iloc[::-1], options)
    pd.testing.assert_frame_equal(result.sort_values('comparison_id').reset_index(drop=True), reordered.sort_values('comparison_id').reset_index(drop=True))

def add_independent_samples(frame, block):
    rng = np.random.default_rng(9071)
    for condition, base in [('control', 0.1), ('treated', 0.7)]:
        for group in range(6):
            movie = condition + '-independent-' + str(group)
            block['biological_samples'][movie] = movie
            block.setdefault('conditions', {})[movie] = condition
            for index in range(40):
                state = int(index < int(40 * (base + 0.025 * group)))
                value = np.asarray([[-4.0, -2.0], [4.0, 2.0]])[state] + rng.normal(0, 0.04, 2)
                frame.loc[len(frame)] = [movie, 7, index, 50.0 + 0.5 * index, *value]
    block['statistics']['comparison'] = {**OPTIONS, 'metrics': ['occupancy_observed', 'unknown_fraction', 'switch_rate'], 'interval': {'method': 'independent_bootstrap', 'confidence': 0.95, 'resamples': 1000, 'seed': 31}}

def test_native_accepted_pipeline_compares_only_independent_unused_samples_and_reuses(tmp_path, monkeypatch):
    monkeypatch.setattr(screening, 'producer_identity', lambda: {'implementation': 'controlled-sample-test'})
    resolved, paths, source = fixture(tmp_path, changes=add_independent_samples)
    first = run_request(resolved, paths, tmp_path / 'pipeline', only=('state-sample-comparisons',))
    assert first.successful, {key: item.outcome.reason for key, item in first.results.items()}
    saved = first.results['state-sample-comparisons']
    tables, provenance = samples.read_samples(saved)
    assert len(tables['unit_inventory']) == 136 and tables['unit_inventory'].independent_of_state_choice.sum() == 12
    comparisons = tables['comparisons']
    assert comparisons.reference_samples.eq(6).all() and comparisons.target_samples.eq(6).all()
    assert comparisons.loc[comparisons.metric.eq('occupancy_observed'), 'significant'].all()
    assert len(saved.outcome.selections[0].members) == 2
    assert tables['families'].requested.iloc[0] == tables['families'].tested.iloc[0] == 4
    before = {ref.name: screening.file_hash(saved.artifact(ref.name)) for ref in saved.outcome.artifacts}

    def forbidden(*a, **k):
        raise AssertionError('Reopen performed fresh state or population science')
    monkeypatch.setattr(behaviour_models, 'fit_candidate', forbidden)
    monkeypatch.setattr(behaviour_models, 'predict_candidate', forbidden)
    monkeypatch.setattr(behaviour_support, 'measure', forbidden)
    monkeypatch.setattr(samples.backend, 'compare', forbidden)
    second = run_request(resolved, paths, tmp_path / 'pipeline', only=('state-sample-comparisons',), presentation={'state_names': ['First', 'Second']})
    assert second.successful and all((item.outcome.status == 'reused' for item in second.results.values()))
    assert all((screening.file_hash(saved.artifact(name)) == fingerprint for name, fingerprint in before.items()))
    assert not provenance['plot_selection_applied']
