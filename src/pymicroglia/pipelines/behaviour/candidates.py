"""Fit learning-only candidates; expose development diagnostics, never confirmation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import pymicroglia.states.behaviour_models as backend
from pymicroglia.pipelines.behaviour.options import StateKey
from pymicroglia.pipelines._contracts import ArtifactRef, StepResult, content_id
from pymicroglia.pipelines.relationships.inputs import KEYS
from pymicroglia.pipelines._screening import _json_value, _write_json, file_hash, read_table, write_table


CANDIDATE_COLUMNS = ['candidate_id', 'model_id', 'method', 'components', 'status', 'reason',
    'learning_observations', 'development_observations', 'development_groups', 'development_mean_log_density']
PREDICTION_COLUMNS = ['candidate_id', 'model_id', 'observation_id'] + KEYS + [
    'frame_index', 'hours', 'group_id', 'role', 'learning_selected', 'observed_fraction',
    'status', 'component', 'state_id', 'probabilities', 'log_density', 'diagnostic_only']
GROUP_COLUMNS = ['candidate_id', 'model_id', 'role', 'group_id', 'cells', 'observations',
    'predicted_observations', 'assigned_observations', 'mean_log_density', 'diagnostic_only']
STATE_COLUMNS = ['candidate_id', 'model_id', 'state_id', 'component', 'native_component', 'label', 'accepted']


def implementation_version():
    return content_id({'producer': file_hash(Path(__file__)), 'backend': backend.implementation_version()})


def saved_inputs(saved, definitions):
    """Verify exact observation alignment before accessing any model feature."""
    observations = read_table(saved.artifact('observations'))
    features = read_table(saved.artifact('features'))
    members = read_table(saved.artifact('learning_members')).sort_values('learning_order')
    for frame in (observations, features, members):
        if frame.observation_id.duplicated().any(): raise ValueError('Duplicate observation identity in saved state inputs')
    if set(observations.observation_id) != set(features.observation_id):
        raise ValueError('Saved feature observations do not match the full observation inventory')
    observations = observations.set_index('observation_id', drop=False)
    features = features.set_index('observation_id', drop=False).loc[observations.index]
    columns = ['feature:' + definition['column'] for definition in definitions]
    if set(features.columns) != {'observation_id', *columns}: raise ValueError('Saved feature definition changed')
    if not set(members.observation_id).issubset(observations.index): raise ValueError('Learning member is absent from observations')
    selected = observations.loc[members.observation_id]
    if not selected.role.eq('learning').all() or not selected.learning_selected.all() or not selected.learning_feature_eligible.all():
        raise ValueError('Learning membership contains an ineligible or protected observation')
    if set(observations.loc[observations.learning_selected, 'observation_id']) != set(members.observation_id):
        raise ValueError('Learning selection disagrees with exact saved membership')
    for key in KEYS + ['group_id']:
        if key in members and not selected[key].reset_index(drop=True).equals(members[key].reset_index(drop=True)):
            raise ValueError('Learning member ' + key + ' disagrees with original observation')
    matrix = features[columns].rename(columns=dict(zip(columns, [definition['column'] for definition in definitions])))
    return observations, matrix, members


def evaluate(model, observations, matrix, definitions, candidate_id, *, roles=('learning', 'development')):
    """Apply unchanged native parameters to explicitly allowed diagnostic roles."""
    chosen = observations.loc[observations.role.isin(roles)]
    prediction = backend.predict_candidate(model, matrix.loc[chosen.index], definitions)
    rows = []
    for index, record in enumerate(chosen.to_dict('records')):
        valid_time = record['time_status'] == 'recorded' and bool(record['within_range'])
        status = str(prediction['status'][index]) if valid_time else record['status']
        component = int(prediction['component'][index]) if valid_time else -1
        row = {key: record[key] for key in KEYS + ['observation_id', 'frame_index', 'hours', 'group_id', 'role', 'learning_selected']}
        rows.append({**row, 'candidate_id': candidate_id, 'model_id': model['model_id'],
            'observed_fraction': float(prediction['observed_fraction'][index]), 'status': status,
            'component': component if component >= 0 else None,
            'state_id': StateKey(model['model_id'], component).record_id if component >= 0 else None,
            'probabilities': _json_value(prediction['probabilities'][index].tolist()) if valid_time else None,
            'log_density': _json_value(prediction['log_density'][index]) if valid_time else None, 'diagnostic_only': True})
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def group_diagnostics(predictions):
    """Descriptive equal-cell means within each explicit independent group."""
    rows = []
    for (candidate, model, role, group), frame in predictions.groupby(['candidate_id', 'model_id', 'role', 'group_id'], dropna=False, sort=True):
        cells = frame.groupby(KEYS, dropna=False, sort=True).log_density.mean()
        rows.append({'candidate_id': candidate, 'model_id': model, 'role': role, 'group_id': group,
            'cells': len(cells), 'observations': len(frame), 'predicted_observations': int(frame.log_density.notna().sum()),
            'assigned_observations': int(frame.status.eq('assigned').sum()), 'mean_log_density': _json_value(cells.mean()), 'diagnostic_only': True})
    return pd.DataFrame(rows, columns=GROUP_COLUMNS)


def produce(context):
    from pymicroglia.pipelines._runner import Unavailable
    resolved = context.request; request = resolved.request
    definitions = [feature.as_dict() for feature in resolved.features]
    observations, matrix, members = saved_inputs(context.saved('feature-inputs'), definitions)
    learning = matrix.loc[members.observation_id]
    recipes = [candidate.as_dict() for candidate in request.candidates]
    # Check every requested capability before fitting; do not silently drop an unknown method.
    try:
        for recipe in recipes: backend.candidate_parameters(recipe)
    except backend.UnsupportedModel as error: raise Unavailable(str(error)) from error
    models, candidates, predictions, states = {}, [], [], []
    for recipe in recipes:
        candidate_id = content_id({'recipe': recipe, 'learning': request.learning.as_dict(),
            'features': backend.feature_signature(definitions), 'representation': request.representation,
            'detrending': resolved.detrending.as_dict(), 'assignment': request.assignment.as_dict()})
        result = backend.fit_candidate(learning, definitions, request.learning.as_dict(), recipe,
            request.assignment.as_dict(), members.to_dict('records'),
            {'kind': request.representation, 'detrending': resolved.detrending.as_dict()})
        model = result.pop('model')
        row = {'candidate_id': candidate_id, 'method': recipe['method'], 'components': recipe['components'],
            **result, 'model_id': model['model_id'] if model else None,
            'development_observations': 0, 'development_groups': 0, 'development_mean_log_density': None}
        if model:
            models[model['model_id']] = model
            predicted = evaluate(model, observations, matrix, definitions, candidate_id)
            predictions.append(predicted)
            development = predicted.loc[predicted.role.eq('development')]
            summary = group_diagnostics(development)
            row.update(development_observations=int(development.log_density.notna().sum()),
                development_groups=int(summary.mean_log_density.notna().sum()),
                development_mean_log_density=_json_value(summary.mean_log_density.mean()))
            for component, original in enumerate(model['component_order']):
                states.append({'candidate_id': candidate_id, 'model_id': model['model_id'],
                    'state_id': StateKey(model['model_id'], component).record_id, 'component': component,
                    'native_component': original, 'label': 'Candidate component ' + str(component + 1), 'accepted': False})
        candidates.append(row)
    predicted = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(columns=PREDICTION_COLUMNS)
    tables = {'candidates': pd.DataFrame(candidates, columns=CANDIDATE_COLUMNS), 'diagnostic_predictions': predicted,
        'group_diagnostics': group_diagnostics(predicted), 'candidate_states': pd.DataFrame(states, columns=STATE_COLUMNS)}
    documents = {'models': models, 'provenance': {'schema_version': 1, 'scientific_id': context.scientific_id,
        'input_id': context.saved('feature-inputs').outcome.scientific_id, 'recipes': recipes,
        'implementation': backend.implementation_version(), 'feature_definitions': definitions,
        'fitted_roles': ['learning'], 'diagnostic_roles': ['learning', 'development'],
        'confirmation_evaluated': False, 'state_acceptance_performed': False,
        'learning_membership_id': content_id(_json_value(members.to_dict('records'))),
        'density_scope': 'Transformed-feature native log density; compare only the same feature transformation and observation support',
        'development_aggregation': 'Mean of within-cell observation means, then equal means across declared validation groups; descriptive, no interval or test',
        'definition': 'One immutable learning-only transformation and native model per candidate; components are not accepted states'}}
    context.output.mkdir(parents=True)
    refs = []
    for name, frame in tables.items():
        path = context.output / (name + '.json'); path = write_table(path, frame)
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id, columns=tuple(frame.columns)))
    for name, document in documents.items():
        path = context.output / (name + '.json'); _write_json(path, document)
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id))
    return StepResult(context.step.name, context.scientific_id, 'completed',
        'Saved learning-only candidates and development diagnostics; reserved confirmation was not evaluated', tuple(refs))
