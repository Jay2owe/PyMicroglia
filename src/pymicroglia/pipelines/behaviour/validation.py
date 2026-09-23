"""Freeze a measurement-state choice before one protected confirmation check."""
from __future__ import annotations
from pymicroglia._results import read_document

from datetime import datetime, timezone
from pathlib import Path
import json
import os

import numpy as np
import pandas as pd

import pymicroglia.states.behaviour_models as backend
import pymicroglia.states.behaviour_support as support
from pymicroglia.pipelines.behaviour.candidates import saved_inputs
from pymicroglia.pipelines._contracts import ArtifactRef, SelectionRecord, Settings, StepResult, content_id
from pymicroglia.pipelines._screening import _json_value, _write_json, file_hash, read_table, write_table


SUMMARY_COLUMNS = ['candidate_id', 'model_id', 'components', 'status', 'reason', 'gates']


def implementation_version():
    from pymicroglia.pipelines._screening import producer_identity
    return content_id({'producer': file_hash(Path(__file__)), 'support': support.implementation_version(),
        'native_correction': producer_identity()})


def refit_checks(model, observations, matrix, members, definitions, representatives, settings):
    """Leave whole learning groups out; compare on the same development rows."""
    development = representatives.loc[representatives.role.eq('development') & representatives.status.eq('selected')]
    validation = matrix.loc[development.observation_id]
    reference = backend.predict_candidate(model, validation, definitions)['probabilities'].argmax(axis=1)
    rows, models = [], {}
    for group in sorted(members.group_id.unique()):
        kept = members.loc[members.group_id.ne(group)]
        result = backend.fit_candidate(matrix.loc[kept.observation_id], definitions, model['transformation']['learning'],
            model['candidate'], model['assignment'], kept.to_dict('records'), model['transformation']['representation'])
        fitted = result.pop('model')
        row = {'excluded_group_id': group, 'original_model_id': model['model_id'],
            'refit_model_id': fitted['model_id'] if fitted else None, **result, 'correspondence': None,
            'development_observation_ids': development.observation_id.tolist(), 'confirmation_evaluated': False}
        if fitted:
            models[fitted['model_id']] = fitted
            prediction = backend.predict_candidate(fitted, validation, definitions)
            if np.isfinite(prediction['probabilities']).all():
                row['correspondence'] = support.correspondence(reference, prediction['probabilities'].argmax(axis=1), len(model['component_order']))
                row['refit_assignment_coverage'] = float(np.mean(prediction['status'] == 'assigned'))
        rows.append(row)
    return rows, models


def claim_confirmation(context, frozen, observations):
    """Persist consumption beside original tables; changing output cannot reset it."""
    from pymicroglia.pipelines.audit.confirmation import consumption_root
    root = consumption_root(context) / 'behaviour-states'
    root.mkdir(parents=True, exist_ok=True)
    confirm = observations.loc[observations.role.eq('confirmation')]
    keys = [{'kind': 'movie', 'identity': movie} for movie in sorted(confirm.movie.unique())]
    keys += [{'kind': 'biological_sample', 'identity': sample} for sample in
        sorted(confirm.loc[confirm.sample_confirmed, 'sample'].dropna().unique())]
    immutable = {'schema_version': 1, 'selection_id': frozen['selection_id'],
        'model_id': frozen['model_id'], 'source_run': context.request.inputs.source_run,
        'verified_source_tables_id': content_id(context.request.inputs.table_hashes),
        'rule': 'These original samples/recordings are reserved to this exact frozen choice; no retuning can reuse them as fresh confirmation'}
    entries = []
    # Check known conflicts before claiming previously unopened groups.
    for key in keys:
        path = root / (content_id(key) + '.json')
        if path.exists():
            record = read_document(path)
            if record.get('group') != key or any(record.get(name) != value for name, value in immutable.items()):
                raise ValueError('A reserved behaviour-validation sample was already consumed by another frozen choice; new independent confirmation samples are required')
    for key in keys:
        path = root / (content_id(key) + '.json')
        record = {**immutable, 'group': key, 'opened_utc': datetime.now(timezone.utc).isoformat()}
        try:
            with path.open('x', encoding='utf-8') as handle:
                json.dump(record, handle, allow_nan=False, sort_keys=True); handle.flush(); os.fsync(handle.fileno())
        except FileExistsError:
            record = read_document(path)
            if record.get('group') != key or any(record.get(name) != value for name, value in immutable.items()):
                raise ValueError('Concurrent confirmation attempted a different frozen state choice')
        entries.append(record)
    return entries


def _subset(matrix, reps, role):
    selected = reps.loc[reps.role.eq(role) & reps.status.eq('selected')]
    return matrix.loc[selected.observation_id], selected.group_id.tolist()


def read_support(saved, *, require_accepted=False):
    """Validate saved decisions and selections without fitting or new inference."""
    decision = read_document(saved.artifact('support_decision'))
    expected = content_id({key: value for key, value in decision.items() if key != 'decision_id'})
    if decision.get('schema_version') != 1 or decision.get('decision_id') != expected or decision.get('scientific_id') != saved.outcome.scientific_id:
        raise ValueError('Saved state-support decision identity changed')
    selections = [selection for selection in saved.outcome.selections if selection.name == 'accepted-model']
    if len(selections) != 1: raise ValueError('Saved support needs one authoritative accepted-model selection')
    selection = selections[0]
    if selection.rule.get('decision_id') != expected or selection.rule.get('status') != decision['status']:
        raise ValueError('Saved acceptance selection refers to another scientific decision')
    accepted = decision['status'] == 'accepted'
    members = [member.as_dict() for member in selection.members]
    if members != ([{'model_id': decision['accepted_model_id']}] if accepted else []):
        raise ValueError('Saved accepted-model membership disagrees with the support decision')
    model = read_document(saved.artifact('frozen_model'))
    frozen = read_document(saved.artifact('frozen_choice'))
    confirmation = read_document(saved.artifact('confirmation_evidence'))
    if bool(model) != bool(frozen) or bool(model) != bool(decision['provisional_model_id']):
        raise ValueError('Saved provisional state definition is incomplete')
    if model:
        if model.get('model_id') != content_id({key: value for key, value in model.items() if key != 'model_id'}) or model['model_id'] != decision['provisional_model_id']:
            raise ValueError('Saved frozen state model identity changed')
        if frozen.get('selection_id') != content_id({key: value for key, value in frozen.items() if key != 'selection_id'}) or frozen['selection_id'] != decision['selection_id'] or frozen['model_id'] != model['model_id']:
            raise ValueError('Saved frozen choice disagrees with its model or decision')
    if confirmation['evaluated'] != decision['confirmation_evaluated']:
        raise ValueError('Saved confirmation consumption disagrees with the decision')
    if accepted and (not model or decision['accepted_model_id'] != model['model_id'] or not confirmation['evaluated']
            or confirmation.get('selection_id') != frozen['selection_id'] or confirmation['assessment']['status'] != 'supported'):
        raise ValueError('Accepted states lack matching supported reserved confirmation')
    if require_accepted and not accepted: raise ValueError('A completed accepted state model is required; ' + decision['reason'])
    return decision, model


def produce(context):
    from pymicroglia.pipelines._runner import Unavailable
    request = context.request.request
    try: settings = support.policy(request.support.as_dict())
    except backend.UnsupportedModel as error: raise Unavailable(str(error)) from error
    definitions = [feature.as_dict() for feature in context.request.features]
    observations, matrix, members = saved_inputs(context.saved('feature-inputs'), definitions)
    candidates_saved = context.saved('candidate-models')
    candidates = read_table(candidates_saved.artifact('candidates'))
    models = read_document(candidates_saved.artifact('models'))
    for model in models.values(): backend.verify_model(model)
    reps = support.representatives(observations.reset_index(drop=True), settings)
    design = read_document(context.saved('feature-inputs').artifact('partition_design'))
    needed_learning = max(settings['min_learning_groups'], request.validation['min_groups']['learning'])
    ready = design['status'] == 'ready' and members.group_id.nunique() >= needed_learning and bool(settings['sampling_population'])
    reason = 'Declared validation partitions lack sufficient learning groups or independent group counts' if not ready else ''
    if not settings['sampling_population']: reason = 'The common independent validation sampling population has not been declared in support.sampling_population'
    baseline_rows = candidates.loc[candidates.components.eq(1) & candidates.status.eq('fitted')]
    if baseline_rows.empty: ready = False; reason = 'No fitted declared one-component reference is available'
    development, confirmation, summaries, checks, refitted_models = [], {'evaluated': False}, [], [], {}
    chosen = baseline = frozen = None
    consumed = []
    # Retain every requested candidate, including failed/native-ineligible models.
    for row in candidates.to_dict('records'):
        summaries.append({key: row[key] for key in ['candidate_id', 'model_id', 'components']} | {
            'status': 'reference' if row['components'] == 1 and row['status'] == 'fitted' else
                'inconclusive' if row['status'] == 'fitted' else row['status'],
            'reason': reason if row['status'] == 'fitted' else row['reason'], 'gates': []})
    if ready:
        frame, groups = _subset(matrix, reps, 'development')
        for row in candidates.loc[candidates.status.eq('fitted')].to_dict('records'):
            evaluation = support.measure(models[row['model_id']], frame, definitions, groups, settings)
            evaluation['candidate_id'] = row['candidate_id']; development.append(evaluation)
        family_count = sum(int(k) * (int(k) - 1) // 2 for k in candidates.components)
        support.correct_axes(development, family_count, settings)
        references = [value for value in development if len(value['components']) == 1 and value['mean_log_density'] is not None]
        baseline = max(references, key=lambda value: (value['mean_log_density'], value['candidate_id'])) if references else None
        passing = []
        for summary in summaries:
            if summary['components'] == 1 or summary['model_id'] not in models: continue
            model = models[summary['model_id']]
            evaluation = next(value for value in development if value['model_id'] == summary['model_id'])
            if baseline is not None and model['transformation'] != models[baseline['model_id']]['transformation']:
                raise ValueError('Candidate log densities cannot compare different learned feature transformations')
            refits = []
            if evaluation['independent_groups'] >= settings['min_validation_groups']:
                refits, fitted = refit_checks(model, observations, matrix, members, definitions, reps, settings)
                checks += refits; refitted_models.update(fitted)
            assessed = support.assess(evaluation, settings, baseline=baseline, stability=refits)
            summary.update(status=assessed['status'], gates=assessed['gates'],
                reason='Predeclared independent projection, separation, coverage, predictive and grouped-refit criteria')
            if assessed['status'] == 'supported': passing.append((summary, evaluation))
        if passing:
            selected, evidence = min(passing, key=lambda pair: (pair[0]['components'], -pair[1]['mean_log_density'], pair[0]['candidate_id']))
            chosen = models[selected['model_id']]
            body = {'schema_version': 1, 'model_id': chosen['model_id'], 'reference_model_id': baseline['model_id'],
                'candidate_id': selected['candidate_id'], 'settings': settings, 'implementation': support.implementation_version(),
                'development_evidence_id': content_id(_json_value(development)),
                'refit_evidence_id': content_id(_json_value(checks)), 'representatives_id': content_id(_json_value(reps.to_dict('records'))),
                'selection_rule': 'Fewest supported components, then greatest independent-development mean log density, then candidate identity',
                'scope': support.LIMITATION}
            frozen = {**body, 'selection_id': content_id(body)}
            context.output.mkdir(parents=True)
            _write_json(context.output/'frozen_choice.json', frozen)
            # The actual model, rules and representative membership now exist on
            # disk before any confirmation prediction or dip probability.
            _write_json(context.output/'frozen_model.json', chosen)
            frame, groups = _subset(matrix, reps, 'confirmation')
            if len(groups) < settings['min_validation_groups']:
                confirmation = {'evaluated': False, 'selection_id': frozen['selection_id'],
                    'assessment': {'status': 'inconclusive', 'reason': 'Too few complete independent confirmation groups; reserved values remain unevaluated'},
                    'independent_groups': len(groups), 'required_groups': settings['min_validation_groups'], 'retuned': False}
            else:
                consumed = claim_confirmation(context, frozen, observations)
                evidence = support.measure(chosen, frame, definitions, groups, settings)
                reference = support.measure(models[baseline['model_id']], frame, definitions, groups, settings)
                support.correct_axes([evidence], len(evidence['axes']), settings)
                assessed = support.assess(evidence, settings, baseline=reference)
                confirmation = {'evaluated': True, 'selection_id': frozen['selection_id'], 'evidence': evidence,
                    'reference': reference, 'assessment': assessed, 'retuned': False}
    if chosen is not None:
        status = 'accepted' if confirmation['assessment']['status'] == 'supported' else confirmation['assessment']['status']
        reason = 'Frozen candidate passed reserved confirmation under every declared support criterion' if status == 'accepted' else \
            'Frozen candidate did not pass reserved confirmation; no replacement candidate or retuning is permitted' if confirmation['evaluated'] else \
            'Frozen candidate requires more independent confirmation groups; reserved values remain unevaluated'
    elif ready:
        alternatives = [row for row in summaries if row['components'] > 1]
        status = 'inconclusive' if not alternatives or any(row['status'] in {'inconclusive', 'ineligible', 'fit_failed', 'not_converged'} for row in alternatives) else 'no_supported_states'
        reason = 'No candidate passed the complete predeclared development support criteria; this does not establish a biological continuum'
    else: status = 'inconclusive'
    decision = {'schema_version': 1, 'scientific_id': context.scientific_id, 'status': status, 'reason': reason,
        'accepted_model_id': chosen['model_id'] if status == 'accepted' else None,
        'provisional_model_id': chosen['model_id'] if chosen else None,
        'selection_id': frozen['selection_id'] if frozen else None, 'settings': settings,
        'implementation': support.implementation_version(), 'scope': support.LIMITATION,
        'confirmation_evaluated': confirmation['evaluated'], 'confirmation_consumption': consumed,
        'learning_groups': int(members.group_id.nunique()), 'minimum_learning_groups': needed_learning,
        'candidate_id': selected['candidate_id'] if chosen else None}
    decision['decision_id'] = content_id(_json_value(decision))
    documents = {'support_decision': decision, 'frozen_choice': frozen, 'frozen_model': chosen,
        'development_evidence': development, 'grouped_refits': {'checks': checks, 'models': refitted_models},
        'confirmation_evidence': confirmation}
    tables = {'candidate_support': pd.DataFrame(summaries, columns=SUMMARY_COLUMNS), 'validation_representatives': reps}
    context.output.mkdir(parents=True, exist_ok=True)
    refs = []
    for name, document in documents.items():
        path = context.output/(name+'.json'); _write_json(path, document)
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id))
    for name, frame in tables.items():
        path = context.output/(name+'.json'); path = write_table(path, frame)
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id, columns=tuple(frame.columns)))
    selection = SelectionRecord('accepted-model', context.scientific_id,
        Settings({'decision_id': decision['decision_id'], 'status': status, 'reason': reason}),
        (Settings({'model_id': chosen['model_id']}),) if status == 'accepted' else ())
    return StepResult(context.step.name, context.scientific_id, 'completed', reason, tuple(refs), (selection,))
