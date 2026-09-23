"""Apply one accepted state definition to original observations without learning."""
from __future__ import annotations
from pymicroglia._results import read_document
from pymicroglia._sources import source_file

from pathlib import Path
import json

import numpy as np
import pandas as pd

import pymicroglia.states.behaviour_models as backend
from pymicroglia.pipelines.behaviour.candidates import evaluate, saved_inputs
from pymicroglia.pipelines.behaviour.options import StateKey
from pymicroglia.pipelines.behaviour.validation import read_support
from pymicroglia.pipelines._contracts import ArtifactRef, StepResult, content_id
from pymicroglia.pipelines.relationships.inputs import KEYS
from pymicroglia.pipelines._screening import _json_value, _write_json, file_hash, read_table, write_table


PROFILE_COLUMNS = ['model_id', 'state_id', 'component', 'measurement', 'source_table', 'unit',
    'representation', 'assigned_observations', 'observed_values', 'missing_values', 'cells', 'movies',
    'confirmed_samples', 'mean', 'median', 'sd', 'q25', 'q75', 'minimum', 'maximum', 'observed_members', 'weighting', 'independent_validation']
STATE_COLUMNS = ['model_id', 'state_id', 'component', 'native_component', 'label', 'assigned_observations',
    'cells', 'movies', 'confirmed_samples', 'model_centre', 'centre_representation', 'centre_is_observed_cell']
REPRESENTATIVE_COLUMNS = ['model_id', 'state_id', 'component', 'status', 'observation_id', *KEYS,
    'frame_index', 'hours', 'max_probability', 'log_density', 'selection_rule']
REASONS = {'assigned': 'Applied the accepted frozen model and its assignment rules',
    'missing_features': 'The frozen observed-feature requirements are not met',
    'ambiguous': 'Largest native component membership is below the frozen threshold',
    'outside_training_distribution': 'Native fitted density is below the frozen training-density threshold',
    'invalid_prediction': 'The native model returned nonfinite prediction values',
    'invalid_time': 'Original observation time is invalid or conflicting',
    'outside_range': 'Original observation is outside the declared half-open analysis interval'}


def implementation_version():
    return content_id({'producer': file_hash(Path(__file__)), 'candidate_replay': file_hash(source_file('behaviour_candidates.py')),
        'models': backend.implementation_version()})


def apply_assignments(model, decision, observations, matrix, definitions):
    """Retain every original row; native membership never becomes a calibrated label."""
    if decision['status'] != 'accepted' or decision['accepted_model_id'] != model['model_id']:
        raise ValueError('Production assignments require the accepted model, never a rejected candidate')
    predicted = evaluate(model, observations, matrix, definitions, decision['candidate_id'],
        roles=('learning', 'development', 'confirmation', 'assignment_only'))
    if set(predicted.observation_id) != set(observations.observation_id):
        raise ValueError('Accepted-model replay lost an original observation or validation role')
    fields = ['observation_id', 'model_id', 'component', 'state_id', 'status', 'observed_fraction', 'probabilities', 'log_density']
    result = observations.rename(columns={'status': 'input_status', 'reason': 'input_reason'}).join(
        predicted.set_index('observation_id')[fields[1:]], how='left', validate='one_to_one')
    result['assignment_reason'] = result.status.map(REASONS)
    if result.assignment_reason.isna().any(): raise ValueError('Unknown native assignment status')
    result['max_probability'] = [max(value for value in values if value is not None) if values is not None and any(value is not None for value in values) else None
        for values in result.probabilities]
    # Nullable categorical indices and numerical scores keep the same schema
    # when replayed one observation at a time, including wholly unknown batches.
    result['component'] = result.component.astype('Int64')
    for name in ['log_density', 'max_probability', 'observed_fraction']:
        result[name] = result[name].astype(float)
    result['score_type'] = 'native_gaussian_mixture_component_membership'
    result['score_calibration'] = 'uncalibrated_model_conditional'
    result['accepted_decision_id'] = decision['decision_id']
    return result.reset_index(drop=True)


def describe_states(model, assignments, matrices, definitions):
    """Original-unit summaries and examples use observed members, never centres."""
    means = model['native_state']['means_']['array']
    states, profiles, examples = [], [], []
    rule = 'Highest native fitted density among assigned members, then membership probability and full observation identity; a real example, not a population representative'
    for component, native in enumerate(model['component_order']):
        state_id = StateKey(model['model_id'], component).record_id
        chosen = assignments.loc[assignments.status.eq('assigned') & assignments.component.eq(component)]
        key = {'model_id': model['model_id'], 'state_id': state_id, 'component': component}
        confirmed = chosen.loc[chosen.sample_confirmed, ['source_run', 'sample']].drop_duplicates()
        states.append({**key, 'native_component': native, 'label': 'State ' + str(component + 1),
            'assigned_observations': len(chosen), 'cells': len(chosen[KEYS].drop_duplicates()), 'movies': chosen.movie.nunique(),
            'confirmed_samples': len(confirmed), 'model_centre': means[native],
            'centre_representation': 'Learned model feature coordinates in the saved feature order; see frozen transformation', 'centre_is_observed_cell': False})
        for representation, matrix in matrices.items():
            values = matrix.loc[chosen.observation_id]
            for definition in definitions:
                series = values[definition['column']]
                finite = series.loc[np.isfinite(series.to_numpy(dtype=float))]
                members = chosen.loc[chosen.observation_id.isin(finite.index)]
                profiles.append({**key, 'measurement': definition['column'], 'source_table': definition['table'], 'unit': definition['unit'],
                    'representation': representation, 'assigned_observations': len(chosen), 'observed_values': len(finite),
                    'missing_values': len(chosen) - len(finite), 'cells': len(members[KEYS].drop_duplicates()), 'movies': members.movie.nunique(),
                    'confirmed_samples': len(members.loc[members.sample_confirmed, ['source_run', 'sample']].drop_duplicates()),
                    'mean': _json_value(finite.mean()) if len(finite) else None, 'median': _json_value(finite.median()) if len(finite) else None,
                    'sd': _json_value(finite.std(ddof=1)) if len(finite) > 1 else None,
                    'q25': _json_value(finite.quantile(.25)) if len(finite) else None, 'q75': _json_value(finite.quantile(.75)) if len(finite) else None,
                    'minimum': _json_value(finite.min()) if len(finite) else None, 'maximum': _json_value(finite.max()) if len(finite) else None,
                    'observed_members': finite.index.tolist(), 'weighting': 'Equal weight per observed assigned frame; not elapsed-time or biological-sample weighting',
                    'independent_validation': False})
        example = {**key, 'status': 'no_assigned_observation', 'selection_rule': rule}
        if len(chosen):
            member = chosen.sort_values(['log_density', 'max_probability', 'observation_id'], ascending=[False, False, True]).iloc[0]
            example.update(status='selected', **{name: _json_value(member[name]) for name in ['observation_id', *KEYS,
                'frame_index', 'hours', 'max_probability', 'log_density']})
        examples.append(example)
    return pd.DataFrame(states, columns=STATE_COLUMNS), pd.DataFrame(profiles, columns=PROFILE_COLUMNS), pd.DataFrame(examples, columns=REPRESENTATIVE_COLUMNS)


def read_assignments(saved, *, expected_model=None):
    provenance = read_document(saved.artifact('provenance'))
    if provenance.get('schema_version') != 1 or provenance.get('scientific_id') != saved.outcome.scientific_id:
        raise ValueError('Saved assignment provenance does not match its scientific identity')
    if expected_model is not None and provenance['model_id'] != expected_model: raise ValueError('Saved assignments use a different model definition')
    assignments = read_table(saved.artifact('assignments'))
    if assignments.observation_id.duplicated().any() or not assignments.model_id.eq(provenance['model_id']).all():
        raise ValueError('Saved assignments repeat observations or mix model definitions')
    assigned = assignments.status.eq('assigned')
    if not assignments.loc[~assigned, 'state_id'].isna().all() or not assignments.loc[~assigned, 'component'].isna().all():
        raise ValueError('An unassigned observation contains a production state label')
    for row in assignments.loc[assigned].to_dict('records'):
        if row['state_id'] != StateKey(row['model_id'], row['component']).record_id:
            raise ValueError('Saved categorical state does not match its model/component identity')
    return assignments, provenance


def produce(context):
    from pymicroglia.pipelines._runner import Unavailable
    decision, model = read_support(context.saved('state-support'), require_accepted=True)
    if context.selection is None or [member.as_dict() for member in context.selection.members] != [{'model_id': model['model_id']}]:
        raise ValueError('Requested assignment selection differs from the accepted model')
    request = context.request.request
    if request.assignment.as_dict() != model['assignment'] or request.learning.as_dict() != model['transformation']['learning']:
        raise ValueError('Assignment or learned transformation settings changed after state acceptance')
    if model['transformation']['representation'] != {'kind': request.representation, 'detrending': context.request.detrending.as_dict()}:
        raise ValueError('Requested representation differs from the frozen accepted transformation')
    candidates = read_document(context.saved('candidate-models').artifact('models'))
    if candidates.get(model['model_id']) != model: raise ValueError('Accepted model differs from its saved fitted candidate')
    definitions = [feature.as_dict() for feature in context.request.features]
    prepared = context.saved('feature-inputs')
    observations, matrix, _ = saved_inputs(prepared, definitions)
    raw = read_table(prepared.artifact('raw_features')).set_index('observation_id')
    expected_columns = {'feature:' + definition['column'] for definition in definitions}
    if not raw.index.is_unique or set(raw.index) != set(observations.index) or set(raw.columns) != expected_columns:
        raise ValueError('Original-unit feature values do not match the observation inventory')
    raw = raw.rename(columns={'feature:' + definition['column']: definition['column'] for definition in definitions}).loc[observations.index]
    try: assignments = apply_assignments(model, decision, observations, matrix, definitions)
    except backend.UnsupportedModel as error: raise Unavailable(str(error)) from error
    matrices = {'raw': raw}
    if request.representation != 'raw': matrices[request.representation] = matrix
    states, profiles, examples = describe_states(model, assignments, matrices, definitions)
    members = assignments.loc[assignments.status.eq('assigned'), ['model_id', 'state_id', 'component', 'observation_id', *KEYS,
        'frame_index', 'hours', 'vector_id', 'sample', 'sample_confirmed', 'role', 'max_probability', 'log_density']]
    cells = read_table(prepared.artifact('cell_inventory')).copy()
    counts = assignments.groupby(KEYS, sort=True).status.value_counts().to_dict()
    for status in REASONS:
        cells[status + '_observations'] = [counts.get((*[row[key] for key in KEYS], status), 0) for row in cells.to_dict('records')]
    cells['model_id'] = model['model_id']
    tables = {'assignments': assignments, 'state_definitions': states, 'state_profiles': profiles,
        'state_members': members, 'representatives': examples, 'cell_inventory': cells}
    provenance = {'schema_version': 1, 'scientific_id': context.scientific_id, 'model_id': model['model_id'],
        'decision_id': decision['decision_id'], 'support_id': context.saved('state-support').outcome.scientific_id,
        'feature_input_id': prepared.outcome.scientific_id, 'feature_definitions': definitions,
        'model_implementation': model['implementation'], 'assignment_implementation': backend.implementation_version(),
        'assignment': model['assignment'], 'transformation': model['transformation'],
        'model_fitted': False, 'transform_fitted': False, 'temporal_smoothing': False,
        'score_type': 'Native Gaussian mixture component membership conditional on this saved model',
        'score_calibration': 'No calibration to biological confidence', 'state_scope': decision['scope'],
        'profile_scope': 'Actual observed values in original feature units; features defined these states and do not independently validate them',
        'representative_scope': 'Actual observed assigned members; model centres are never substituted for real cells',
        'input_observations': len(observations), 'assignment_observations': len(assignments),
        'status_counts': assignments.status.value_counts().to_dict(), 'requested_cells': len(cells)}
    context.output.mkdir(parents=True)
    refs = []
    for name, frame in tables.items():
        path = context.output/(name+'.json'); path = write_table(path, frame)
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id, columns=tuple(frame.columns)))
    path = context.output/'provenance.json'; _write_json(path, provenance)
    refs.append(ArtifactRef('provenance', path.name, file_hash(path), context.scientific_id))
    return StepResult(context.step.name, context.scientific_id, 'completed',
        'Applied one accepted frozen model to all original observations; saved unknowns, original-unit profiles and real-member examples', tuple(refs))
