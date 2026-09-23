"""Independent support cannot be supplied by an attractive forced partition."""
import numpy as np
import pandas as pd
import pytest
import pymicroglia.states.behaviour_models as backend
import pymicroglia.states.behaviour_support as support
from tests.test_behaviour_models import inputs

def draw(kind, count, rng):
    if kind == 'two':
        return rng.normal(0, 0.25, (count, 2)) + np.array([[-4.0, -2.0], [4.0, 2.0]])[rng.integers(0, 2, count)]
    if kind == 'three':
        return rng.normal(0, 0.25, (count, 2)) + np.array([[-4.0, -2.0], [0.0, 4.0], [4.0, -2.0]])[rng.integers(0, 3, count)]
    if kind == 'cloud':
        return rng.normal(0, 1, (count, 2))
    if kind == 'uniform':
        x = rng.uniform(-4, 4, count)
        return np.column_stack([x, x + rng.normal(0, 0.12, count)])
    if kind == 'ring':
        angle = rng.uniform(0, 2 * np.pi, count)
        return np.column_stack([np.cos(angle), np.sin(angle)]) + rng.normal(0, 0.01, (count, 2))
    raise ValueError(kind)

def fitted(kind='two', count=2, seed=6101):
    frame, definitions, learning, recipe, assignment, _ = inputs()
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(draw(kind, 600, rng), columns=frame.columns)
    members = [{'observation_id': 'learning-' + str(i), 'learning_weight': 1.0} for i in range(len(data))]
    recipe.update(components=count, n_init=3, seed=seed)
    result = backend.fit_candidate(data, definitions, learning, recipe, assignment, members, 'raw')
    assert result['status'] == 'fitted', result['reason']
    return (result['model'], definitions)

def test_native_dip_and_separation_match_public_values_and_full_family():
    from diptest import diptest
    from sklearn.metrics import silhouette_samples
    model, definitions = fitted()
    settings = support.policy({'method': 'heldout_multimodality', 'sampling_population': 'Independent controlled observations'})
    data = pd.DataFrame(draw('two', 80, np.random.default_rng(72)), columns=['signal', 'shape'])
    groups = ['group-' + str(i) for i in range(len(data))]
    result = support.measure(model, data, definitions, groups, settings)
    transformed = backend.predict_candidate(model, data, definitions)
    axis = result['axes'][0]
    expected = diptest(transformed['values'] @ axis['direction'], boot_pval=False)
    np.testing.assert_allclose([axis['statistic'], axis['p_value']], expected)
    labels = transformed['probabilities'].argmax(axis=1)
    silhouettes = silhouette_samples(transformed['values'], labels)
    for component in result['components']:
        assert component['mean_silhouette'] == pytest.approx(silhouettes[labels == component['component']].mean())
    support.correct_axes([result], 10, settings)
    assert axis['q_value'] == pytest.approx(min(1, expected[1] * 10)) and axis['family_requested'] == 10
    with pytest.raises(ValueError, match='one observation'):
        support.measure(model, data, definitions, ['same'] * len(data), settings)

@pytest.mark.parametrize('kind,expected', [('two', 2), ('three', 3), ('cloud', None), ('uniform', None), ('ring', None)])
def test_independent_controls_allow_correct_count_and_refuse_forced_continuous_splits(kind, expected):
    settings = support.policy({'method': 'heldout_multimodality', 'sampling_population': 'Independent controlled observations'})
    data = pd.DataFrame(draw(kind, 80, np.random.default_rng(103)), columns=['signal', 'shape'])
    groups = [str(i) for i in range(len(data))]
    evaluated = []
    for count in [1, 2, 3, 4]:
        model, definitions = fitted(kind, count)
        evaluated.append(support.measure(model, data, definitions, groups, settings))
    support.correct_axes(evaluated, 10, settings)
    accepted = [len(row['components']) for row in evaluated[1:] if support.assess(row, settings, baseline=evaluated[0])['status'] == 'supported']
    assert accepted == ([expected] if expected else [])

def test_equivalent_label_permutations_and_missing_components_are_distinct():
    original = np.tile([0, 1, 2], 20)
    permuted = np.array([2, 0, 1])[original]
    result = support.correspondence(original, permuted, 3)
    assert result['adjusted_rand'] == 1 and all((row['recall'] == 1 for row in result['mapping']))
    assert not result['missing_components']
    missing = support.correspondence(original, np.zeros(len(original), dtype=int), 3)
    assert missing['missing_components'] == [1, 2] and missing['adjusted_rand'] == 0
    with pytest.raises(ValueError, match='finite candidate labels'):
        support.correspondence(original, np.full(len(original), -1), 3)

def test_representatives_are_value_independent_unique_and_complete():
    rows = []
    for role in ['development', 'confirmation']:
        for group in ['sample-a', 'sample-b']:
            for movie in ['one', 'two']:
                for identity in [7, 8]:
                    for frame in range(4):
                        rows.append({'role': role, 'group_id': role + group, 'source_run': 'source', 'movie': role + group + movie, 'identity': identity, 'observation_id': f'{role}-{group}-{movie}-{identity}-{frame}', 'frame_index': frame, 'within_range': True, 'time_status': 'recorded', 'feature_fraction': 1.0, 'arbitrary_value': 1.0})
    frame = pd.DataFrame(rows)
    frame.loc[frame.group_id.eq('confirmationsample-b'), 'feature_fraction'] = 0.5
    settings = support.policy({'method': 'heldout_multimodality'})
    first = support.representatives(frame, settings)
    second = support.representatives(frame.assign(arbitrary_value=1000000000.0).sample(frac=1, random_state=4), settings)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 4 and first.group_id.is_unique and (first.status.eq('selected').sum() == 3)
    assert first.loc[first.status.eq('selected'), 'eligible_cells'].eq(4).all()
    assert not first.selection_uses_feature_values.any()

def test_insufficient_samples_never_gain_power_from_repeated_frames():
    model, definitions = fitted()
    settings = support.policy({'method': 'heldout_multimodality'})
    data = pd.DataFrame(draw('two', 3, np.random.default_rng(5)), columns=['signal', 'shape'])
    result = support.measure(model, data, definitions, ['a', 'b', 'c'], settings)
    support.correct_axes([result], 1, settings)
    assert result['axes'][0]['p_value'] is None and result['axes'][0]['status'] == 'insufficient_groups'
    assert support.assess(result, settings)['status'] == 'inconclusive'
    repeated = pd.concat([data] * 20, ignore_index=True)
    with pytest.raises(ValueError, match='one observation'):
        support.measure(model, repeated, definitions, ['a', 'b', 'c'] * 20, settings)

def test_predeclared_independent_null_and_power_calibration():
    from diptest import diptest
    from threadpoolctl import threadpool_limits
    model, definitions = fitted()
    settings = support.policy({'method': 'heldout_multimodality'})
    groups = [str(i) for i in range(80)]
    detected = {}
    means = backend.native_model(model).means_[model['component_order']]
    axis = means[1] - means[0]
    axis /= np.linalg.norm(axis)
    for kind, replicates in [('cloud', 4096), ('uniform', 4096), ('two', 32)]:
        count = 0
        raw = np.vstack([draw(kind, 80, np.random.default_rng(20260910 + seed)) for seed in range(replicates)])
        with threadpool_limits(limits=1):
            values = backend.transform(model, pd.DataFrame(raw, columns=['signal', 'shape']), definitions)[0]
            projections = (values @ axis).reshape(replicates, 80)
            probabilities = [float(diptest(row, boot_pval=False)[1]) for row in projections]
        count = sum((value <= settings['alpha'] for value in probabilities))
        detected[kind] = count
    assert detected['cloud'] <= 0.065 * 4096 and detected['uniform'] <= 0.065 * 4096, detected
    assert detected['two'] >= 29, detected

def test_unexplained_population_and_unknown_policy_remain_explicit():
    settings = support.policy({'method': 'heldout_multimodality'})
    assert settings['sampling_population'] is None
    with pytest.raises(backend.UnsupportedModel):
        support.policy({'method': 'silhouette_only'})
    with pytest.raises(ValueError, match='family-wise'):
        support.policy({'method': 'heldout_multimodality', 'multiple_testing': 'none'})
    with pytest.raises(ValueError, match='Unknown'):
        support.policy({'method': 'heldout_multimodality', 'arbitrary_threshold': 3})
