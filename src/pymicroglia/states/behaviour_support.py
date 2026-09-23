"""Independent projected multimodality and explicit measurement-partition checks.

Acceptance concerns a separated, reproducible measurement vocabulary. It does
not identify latent biological categories or prove a particular latent state count.
"""
from __future__ import annotations

from importlib.metadata import version
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import pymicroglia.states.behaviour_models as backend
from pymicroglia.pipelines._contracts import content_id
from pymicroglia.pipelines.relationships.inputs import KEYS
from pymicroglia.pipelines._screening import _json_value, file_hash


DEFAULTS = {'method': 'heldout_multimodality', 'sampling_seed': 17, 'sampling_population': None,
    'min_validation_groups': 40, 'min_learning_groups': 3, 'min_component_groups': 6,
    'min_component_silhouette': .6, 'min_assignment_coverage': .8,
    'min_adjusted_rand': .9, 'min_component_recall': .8, 'min_log_density_gain': 0.,
    'alpha': .05, 'multiple_testing': 'bonferroni'}
LIMITATION = ('Supported separated measurement modes under the declared sampling population and representation; '
    'not a test of a particular number of latent biological states. A continuous latent process can also '
    'spend time in separated measurement regions. Membership probabilities are model-conditional, not biological confidence.')


def policy(settings):
    from pymicroglia.pipelines.behaviour.options import _number, _seed, _fraction
    if settings.get('method') != DEFAULTS['method']: raise backend.UnsupportedModel('Unsupported declared state-support method')
    if set(settings) - DEFAULTS.keys(): raise ValueError('Unknown state-support settings: ' + ', '.join(sorted(set(settings) - DEFAULTS.keys())))
    result = {**DEFAULTS, **dict(settings)}
    _seed(result['sampling_seed'], 'support.sampling_seed')
    for key, minimum in [('min_validation_groups', 4), ('min_learning_groups', 2), ('min_component_groups', 2)]:
        _number(result[key], 'support.' + key, minimum=minimum, integer=True)
    for key in ['min_component_silhouette', 'min_assignment_coverage', 'min_adjusted_rand', 'min_component_recall', 'alpha']:
        _fraction(result[key], 'support.' + key)
    if not 0 < result['alpha'] < 1: raise ValueError('support.alpha must be between zero and one')
    _number(result['min_log_density_gain'], 'support.min_log_density_gain', minimum=0)
    if result['multiple_testing'] != 'bonferroni': raise ValueError('State support requires the supported Bonferroni family-wise correction')
    population = result['sampling_population']
    if population is not None and (not isinstance(population, str) or not population.strip()):
        raise ValueError('support.sampling_population must describe the common validation sampling population')
    return result


def implementation_version():
    return {'code': file_hash(Path(__file__)), 'models': backend.implementation_version(),
        'diptest': version('diptest'), 'scikit_learn': version('scikit-learn'), 'scipy': version('scipy')}


def representatives(observations, settings):
    """One complete observation per protected group; values never choose the row."""
    rows = []
    for (role, group), frame in observations.loc[observations.role.isin(['development', 'confirmation'])].groupby(['role', 'group_id'], sort=True):
        eligible = frame.loc[frame.within_range & frame.time_status.eq('recorded') & frame.feature_fraction.eq(1)]
        choice = None
        if len(eligible):
            rng = np.random.default_rng(int(content_id({'seed': settings['sampling_seed'], 'group': group, 'role': role})[:16], 16))
            cells = list(eligible.groupby(KEYS, sort=True))
            cell = cells[int(rng.integers(len(cells)))][1].sort_values(['frame_index', 'observation_id'])
            choice = cell.iloc[int(rng.integers(len(cell)))].to_dict()
        rows.append({'role': role, 'group_id': group, 'observation_id': choice['observation_id'] if choice else None,
            **{key: choice[key] if choice else None for key in KEYS}, 'eligible_cells': len(eligible[KEYS].drop_duplicates()),
            'eligible_observations': len(eligible), 'status': 'selected' if choice else 'no_complete_observation',
            'selection': 'Uniform cell, then uniform complete original observation within that declared group',
            'selection_uses_feature_values': False})
    return pd.DataFrame(rows, columns=['role', 'group_id', 'observation_id', *KEYS, 'eligible_cells',
        'eligible_observations', 'status', 'selection', 'selection_uses_feature_values'])


@threadpool_limits.wrap(limits=1)
def measure(model, frame, definitions, groups, settings):
    """Describe/test independent representatives using the frozen candidate only."""
    from diptest import diptest
    from sklearn.metrics import silhouette_samples
    if len(groups) != len(frame) or len(set(groups)) != len(groups):
        raise ValueError('Formal state support requires exactly one observation per independent group')
    result = backend.predict_candidate(model, frame, definitions)
    if not result['eligible'].all() or not np.isfinite(result['probabilities']).all():
        raise ValueError('Support representatives must be complete finite transformed observations')
    count = len(model['component_order'])
    labels = result['probabilities'].argmax(axis=1) if len(frame) else np.array([], dtype=int)
    sizes = np.bincount(labels, minlength=count)
    enough = len(frame) >= settings['min_validation_groups']
    silhouette = silhouette_samples(result['values'], labels, metric='euclidean') if 1 < len(set(labels)) < len(labels) else np.full(len(frame), np.nan)
    components = [{'component': component, 'independent_groups': int(sizes[component]),
        'mean_silhouette': _json_value(np.mean(silhouette[labels == component])) if sizes[component] else None}
        for component in range(count)]
    means = backend.native_model(model).means_[model['component_order']]
    axes = []
    for a, b in combinations(range(count), 2):
        direction = means[b] - means[a]; length = float(np.linalg.norm(direction))
        valid = np.isfinite(length) and length > 0
        axis = direction / length if valid else np.full_like(direction, np.nan)
        projection = result['values'] @ axis if valid else np.full(len(frame), np.nan)
        statistic, probability = diptest(projection, boot_pval=False, sort_x=True, allow_zero=True) if valid and enough else (None, None)
        axes.append({'components': [a, b], 'direction': _json_value(axis.tolist()),
            'projection': _json_value(projection.tolist()), 'statistic': _json_value(statistic), 'p_value': _json_value(probability),
            'q_value': None, 'status': 'tested' if probability is not None else 'insufficient_groups' if not enough else 'coincident_centres'})
    return {'model_id': model['model_id'], 'independent_groups': len(frame), 'group_ids': list(groups),
        'observation_ids': list(frame.index), 'components': components, 'axes': axes,
        'assignment_coverage': float(np.mean(result['status'] == 'assigned')) if len(frame) else None,
        'mean_log_density': _json_value(np.mean(result['log_density'])) if len(frame) else None,
        'log_density': _json_value(result['log_density'].tolist()), 'labels': labels.tolist(),
        'prediction_status': result['status'].tolist(), 'geometry_population': 'All selected complete representatives, including ambiguous and out-of-domain predictions',
        'probability_null': 'Independent identically distributed observations from one declared sampling population; each tested fixed projection is unimodal',
        'scope': LIMITATION}


def correct_axes(evaluations, requested_axis_count, settings):
    """Keep failed requested candidates/axes in the family before any selection."""
    import pymicroglia.workbench as circadian
    axes = [axis for evaluation in evaluations for axis in evaluation['axes']]
    if requested_axis_count < len(axes): raise ValueError('Requested state-support family lost an axis')
    probabilities = [axis['p_value'] if axis['p_value'] is not None else 1. for axis in axes]
    probabilities += [1.] * (requested_axis_count - len(axes))
    adjusted = circadian.adjust_pvalues(probabilities, settings['multiple_testing']) if probabilities else []
    for axis, value in zip(axes, adjusted):
        axis['q_value'] = float(value) if axis['p_value'] is not None else None
        axis['family_requested'] = requested_axis_count
        axis['multiple_testing'] = settings['multiple_testing']


@threadpool_limits.wrap(limits=1)
def correspondence(reference, refitted, components):
    """Compare arbitrary labels through a native optimal correspondence and ARI."""
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_rand_score, confusion_matrix
    a, b = np.asarray(reference, dtype=int), np.asarray(refitted, dtype=int)
    if a.shape != b.shape or a.ndim != 1 or np.any((a < 0) | (a >= components) | (b < 0) | (b >= components)):
        raise ValueError('Correspondence requires aligned finite candidate labels')
    counts = confusion_matrix(a, b, labels=np.arange(components))
    left, right = linear_sum_assignment(counts, maximize=True)
    mapping = []
    for x, y in zip(left, right):
        denominator = int(counts[x].sum())
        mapping.append({'reference_component': int(x), 'refit_component': int(y), 'overlap': int(counts[x, y]),
            'reference_count': denominator, 'refit_count': int(counts[:, y].sum()),
            'recall': float(counts[x, y] / denominator) if denominator else None})
    return {'adjusted_rand': float(adjusted_rand_score(a, b)) if len(a) else None,
        'mapping': mapping, 'contingency': counts.tolist(),
        'missing_components': [i for i in range(components) if not counts[i].sum() or not counts[:, i].sum()]}


def assess(evaluation, settings, *, baseline=None, stability=None):
    """Keep insufficiency, failed operational criteria and acceptance distinct."""
    gates = []
    def add(name, status, observed, required): gates.append({'name': name, 'status': status, 'observed': observed, 'required': required})
    count = len(evaluation['components']); enough = evaluation['independent_groups'] >= settings['min_validation_groups']
    add('independent_groups', 'pass' if enough else 'insufficient', evaluation['independent_groups'], settings['min_validation_groups'])
    add('multiple_components', 'pass' if count > 1 else 'fail', count, 'At least two candidate components')
    for component in evaluation['components']:
        n = component['independent_groups']; value = component['mean_silhouette']
        add('component_group_count:' + str(component['component']), 'pass' if n >= settings['min_component_groups'] else 'insufficient', n, settings['min_component_groups'])
        add('component_separation:' + str(component['component']), 'insufficient' if value is None else 'pass' if value >= settings['min_component_silhouette'] else 'fail', value, settings['min_component_silhouette'])
    q = [axis['q_value'] for axis in evaluation['axes']]
    add('independent_multimodality', 'insufficient' if not q or any(value is None for value in q) else 'pass' if all(value <= settings['alpha'] for value in q) else 'fail', q, settings['alpha'])
    value = evaluation['assignment_coverage']
    add('assignment_coverage', 'insufficient' if value is None else 'pass' if value >= settings['min_assignment_coverage'] else 'fail', value, settings['min_assignment_coverage'])
    gain = evaluation['mean_log_density'] - baseline['mean_log_density'] if baseline and baseline['mean_log_density'] is not None and evaluation['mean_log_density'] is not None else None
    add('density_gain_over_single_component', 'insufficient' if gain is None else 'pass' if gain > settings['min_log_density_gain'] else 'fail', gain, settings['min_log_density_gain'])
    if stability is not None:
        for refit in stability:
            matched = refit.get('correspondence')
            valid = matched is not None and not matched['missing_components'] and matched['adjusted_rand'] is not None
            recall = [m['recall'] for m in matched['mapping']] if matched else []
            passes = valid and matched['adjusted_rand'] >= settings['min_adjusted_rand'] and all(v is not None and v >= settings['min_component_recall'] for v in recall) \
                and refit.get('refit_assignment_coverage', 0.) >= settings['min_assignment_coverage']
            add('grouped_refit:' + refit['excluded_group_id'], 'pass' if passes else 'fail', matched,
                {'minimum_adjusted_rand': settings['min_adjusted_rand'], 'minimum_component_recall': settings['min_component_recall'],
                 'minimum_assignment_coverage': settings['min_assignment_coverage'], 'all_components_required': True})
    # Too little independent support cannot become evidence of continuity.
    status = 'inconclusive' if not enough or any(g['status'] == 'insufficient' for g in gates) else \
        'no_supported_states' if any(g['status'] == 'fail' for g in gates) else 'supported'
    return {'status': status, 'gates': gates, 'scope': LIMITATION}
