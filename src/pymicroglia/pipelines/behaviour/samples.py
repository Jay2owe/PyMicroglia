"""Complete cell and sample populations for one accepted state vocabulary."""
from __future__ import annotations
from pymicroglia._results import read_document
from pymicroglia._sources import source_file

from itertools import combinations
from pathlib import Path
from importlib.metadata import version
import json

import numpy as np
import pandas as pd

import pymicroglia.measure.behaviour_sample_statistics as backend
from pymicroglia.pipelines.behaviour.durations import read_statistics, META
from pymicroglia.pipelines.behaviour.validation import read_support
from pymicroglia.pipelines._contracts import ArtifactRef, SelectionRecord, Settings, StepResult, content_id
from pymicroglia.pipelines.relationships.inputs import KEYS
from pymicroglia.pipelines._screening import _json_value, _write_json, file_hash, read_table, write_table


QUESTION = ['question_id', 'model_id', 'metric', 'state_id', 'target_state_id', 'interval_hours', 'denominator_unit']
CELL_COLUMNS = [*QUESTION, *KEYS, 'numerator', 'denominator', 'value', 'status']
UNIT_COLUMNS = ['unit_id', 'model_id', 'source_run', 'level', 'sample', 'sample_confirmed', 'condition', 'roles',
    'independent_of_state_choice', 'cells', 'recordings', 'members', 'movies', 'observed_hours', 'assigned_hours', 'unknown_hours',
    'unobserved_hours', 'bouts', 'complete_bouts', 'incomplete_bouts', 'inference_exclusion']
SUMMARY_COLUMNS = [*QUESTION, 'unit_id', 'source_run', 'level', 'sample', 'sample_confirmed', 'condition', 'roles',
    'independent_of_state_choice', 'aggregation', 'value', 'status', 'reason', 'cells_requested', 'cells_eligible',
    'recordings_requested', 'recordings_eligible', 'numerator_total', 'denominator_total', 'members', 'eligible_members', 'excluded_members']
RESULT_COLUMNS = [*QUESTION, 'comparison_id', 'reference_condition', 'target_condition', 'method', 'aggregation',
    'descriptive_effect', 'effect', 'effect_interval', 'interval_status', 'p_value', 'q_value', 'significant', 'status', 'reason',
    'reference_samples', 'target_samples', 'descriptive_reference_samples', 'descriptive_target_samples', 'eligible_units', 'excluded_units',
    'family_id', 'family_requested', 'family_tested', 'alpha', 'multiple_testing']
FAMILY_COLUMNS = ['family_id', 'requested', 'tested', 'unavailable', 'members', 'alpha', 'multiple_testing', 'missing_probability_policy']


def implementation_version():
    return content_id({'code': {path.name: file_hash(path) for path in [Path(__file__), Path(backend.__file__),
        source_file('behaviour_durations.py'), source_file('circadian.py')]},
        'libraries': {name: version(name) for name in ['numpy', 'pandas', 'scipy']}})


def _members(frame):
    return frame[KEYS].sort_values(KEYS).to_dict('records')


def cell_metrics(tables):
    """Densely retain every cell and state, including absent interval opportunities."""
    cells, occupancy, steps = [tables[name] for name in ['cell_statistics', 'occupancy', 'steps']]
    rows = []
    state_ids = sorted(occupancy.state_id.unique())
    strata = sorted(steps.loc[steps.valid_transition_opportunity, 'interval_stratum_hours'].unique()) if len(steps) else []
    strata = strata or [None]
    def add(cell, metric, numerator, denominator, state=None, target=None, spacing=None, unit='hours'):
        question = dict(model_id=cell['model_id'], metric=metric, state_id=state, target_state_id=target, interval_hours=spacing, denominator_unit=unit)
        rows.append({**question, 'question_id': content_id(question), **{key: cell[key] for key in KEYS},
            'numerator': numerator, 'denominator': denominator, 'value': numerator/denominator if denominator > 0 else None,
            'status': 'observed' if denominator > 0 else 'no_supported_denominator'})
    occupancy_groups = {key: group for key, group in occupancy.groupby(KEYS, sort=True)}
    step_groups = {key: group for key, group in steps.groupby(KEYS, sort=True)}
    for cell in cells.to_dict('records'):
        key = tuple(cell[name] for name in KEYS)
        for state in occupancy_groups.get(key, occupancy.iloc[:0]).to_dict('records'):
            add(cell, 'occupancy_observed', state['state_hours'], state['observed_hours'], state['state_id'])
            add(cell, 'occupancy_assigned', state['state_hours'], state['assigned_hours'], state['state_id'])
        add(cell, 'unknown_fraction', cell['unknown_hours'], cell['observed_hours'])
        local = step_groups.get(key, steps.iloc[:0])
        for spacing in strata:
            valid = local.loc[local.valid_transition_opportunity & local.interval_stratum_hours.eq(spacing)] if spacing is not None else local.iloc[:0]
            add(cell, 'switch_rate', int(valid.counted_switch.sum()), float(valid.elapsed_hours.sum()), spacing=spacing)
            for source in state_ids:
                source_steps = valid.loc[valid.source_state_id.eq(source)]
                for target in state_ids:
                    add(cell, 'transition_probability', int(source_steps.target_state_id.eq(target).sum()), len(source_steps),
                        source, target, spacing, 'observed_next_state_opportunities')
    return pd.DataFrame(rows, columns=CELL_COLUMNS)


def aggregate(tables, aggregation):
    """Each confirmed biological sample is one unit; unconfirmed movies stay labelled."""
    if aggregation not in {'mean', 'pooled_exposure'}: raise ValueError('Unknown biological-sample aggregation')
    cells = tables['cell_statistics'].copy()
    models = set(cells.model_id)
    if len(models) != 1 or any(not frame.model_id.isin(models).all() for frame in tables.values()):
        raise ValueError('Sample comparisons require the same accepted model throughout')
    if cells.duplicated(KEYS).any(): raise ValueError('Repeated full cell identities in sample population')
    cells['unit_id'] = [content_id({'source_run': row['source_run'], 'level': 'biological_sample' if row['sample_confirmed'] else 'recording',
        'group': row['sample'] if row['sample_confirmed'] else row['movie']}) for row in cells.to_dict('records')]
    units, summaries = [], []
    metrics = cell_metrics(tables)
    metrics = metrics.merge(cells[[*KEYS, 'unit_id']], on=KEYS, how='left', validate='many_to_one')
    for unit_id, group in cells.groupby('unit_id', sort=True):
        first = group.iloc[0]
        conditions = set(value for value in group.condition if pd.notna(value))
        if len(conditions) > 1 or len(conditions) == 1 and group.condition.isna().any():
            raise ValueError('One experimental unit has inconsistent condition assignments')
        roles = sorted(group.role.unique())
        confirmed = bool(first.sample_confirmed)
        independent = confirmed and roles == ['assignment_only']
        unit = {'unit_id': unit_id, 'model_id': first.model_id, 'source_run': first.source_run,
            'level': 'biological_sample' if confirmed else 'recording', 'sample': first['sample'], 'sample_confirmed': confirmed,
            'condition': next(iter(conditions)) if conditions else None, 'roles': roles, 'independent_of_state_choice': independent,
            'cells': len(group), 'recordings': group.movie.nunique(), 'members': _members(group), 'movies': sorted(group.movie.unique()),
            **{name: _json_value(group[name].sum(min_count=1)) for name in ['observed_hours', 'assigned_hours', 'unknown_hours', 'unobserved_hours', 'bouts', 'complete_bouts', 'incomplete_bouts']},
            'inference_exclusion': None if independent else 'Unconfirmed biological sample' if not confirmed else 'Sample contributed to model learning, development or confirmation'}
        units.append(unit)
        for question_id, question in metrics.loc[metrics.unit_id.eq(unit_id)].groupby('question_id', sort=True):
            eligible = question.loc[question.value.notna() & question.denominator.gt(0)]
            numerator, denominator = eligible.numerator.sum(), eligible.denominator.sum()
            value = float(eligible.value.mean()) if aggregation == 'mean' and len(eligible) else float(numerator/denominator) if denominator > 0 else None
            summaries.append({**{name: _json_value(question.iloc[0][name]) for name in QUESTION},
                **{name: unit[name] for name in ['unit_id', 'source_run', 'level', 'sample', 'sample_confirmed', 'condition', 'roles', 'independent_of_state_choice']},
                'aggregation': aggregation, 'value': value, 'status': 'descriptive' if value is not None else 'unavailable',
                'reason': 'Equal weight per eligible cell fraction/rate' if value is not None and aggregation == 'mean' else
                    'Ratio of summed eligible numerators to summed eligible denominators' if value is not None else 'No cell has a supported denominator for this question and observation spacing',
                'cells_requested': len(question), 'cells_eligible': len(eligible), 'recordings_requested': question.movie.nunique(), 'recordings_eligible': eligible.movie.nunique(),
                'numerator_total': float(numerator), 'denominator_total': float(denominator), 'members': _members(question),
                'eligible_members': _members(eligible), 'excluded_members': _members(question.loc[~question.index.isin(eligible.index)])})
    bouts = tables['bouts'].merge(cells[[*KEYS, 'unit_id']], on=KEYS, how='left', validate='many_to_one')
    bouts['summary_scope'] = 'Observed duration allocation with original censoring; no pooled complete-dwell estimate or biological-sample independence for bouts'
    return {'unit_inventory': pd.DataFrame(units, columns=UNIT_COLUMNS), 'unit_metrics': pd.DataFrame(summaries, columns=SUMMARY_COLUMNS),
        'cell_metrics': metrics, 'bout_membership': bouts}


def compare_samples(units, options):
    """Correct the entire requested family before choosing any report rows."""
    settings = backend.policy(options)
    contrasts = settings['contrasts']
    if not contrasts and settings['method'] == 'none':
        contrasts = [{'reference': a, 'target': b} for a, b in combinations(sorted(units.condition.dropna().unique()), 2)]
    rows, details = [], []
    for pair in contrasts:
        for question_id, population in units.loc[units.metric.isin(settings['metrics'])].groupby('question_id', sort=True):
            eligible = population.loc[population.independent_of_state_choice & population.value.notna()]
            left = eligible.loc[eligible.condition.eq(pair['reference'])].sort_values('unit_id')
            right = eligible.loc[eligible.condition.eq(pair['target'])].sort_values('unit_id')
            descriptive = population.loc[population.sample_confirmed & population.value.notna()]
            all_left = descriptive.loc[descriptive.condition.eq(pair['reference'])]
            all_right = descriptive.loc[descriptive.condition.eq(pair['target'])]
            evidence = backend.compare(left.value.to_numpy(float), right.value.to_numpy(float), settings)
            relevant = population.loc[population.condition.isin(pair.values())]
            used = set(left.unit_id) | set(right.unit_id)
            key = {name: _json_value(population.iloc[0][name]) for name in QUESTION}
            cid = content_id({'question': question_id, 'contrast': pair})
            rows.append({**key, 'comparison_id': cid, 'reference_condition': pair['reference'], 'target_condition': pair['target'],
                'method': settings['method'], 'aggregation': population.aggregation.iloc[0],
                'descriptive_effect': float(all_right.value.mean()-all_left.value.mean()) if len(all_left) and len(all_right) else None,
                **{name: evidence[name] for name in ['effect', 'effect_interval', 'interval_status', 'p_value', 'status', 'reason', 'reference_samples', 'target_samples']},
                'descriptive_reference_samples': len(all_left), 'descriptive_target_samples': len(all_right),
                'eligible_units': sorted(used), 'excluded_units': relevant.loc[~relevant.unit_id.isin(used), ['unit_id', 'roles', 'sample_confirmed', 'status', 'reason']].to_dict('records')})
            details.append({'comparison_id': cid, 'question_id': question_id, 'evidence': evidence, 'reference_unit_ids': left.unit_id.tolist(), 'target_unit_ids': right.unit_id.tolist()})
    family = content_id({'model_ids': sorted(units.model_id.unique()), 'members': [row['comparison_id'] for row in rows],
        'multiple_testing': settings['multiple_testing'], 'alpha': settings['alpha']})
    usable = [row['p_value'] is not None for row in rows]
    if any(usable):
        import pymicroglia.workbench as circadian
        adjusted = circadian.adjust_pvalues([row['p_value'] if valid else 1. for row, valid in zip(rows, usable)], settings['multiple_testing'])
    else: adjusted = [None]*len(rows)
    for row, valid, q in zip(rows, usable, adjusted):
        row.update(family_id=family, family_requested=len(rows), family_tested=sum(usable), alpha=settings['alpha'], multiple_testing=settings['multiple_testing'],
            q_value=float(q) if valid else None, significant=bool(q <= settings['alpha']) if valid else None)
        if valid:
            row['status'] = 'detected_difference' if row['significant'] else 'no_detected_difference'
            row['reason'] = 'Corrected independent-sample evidence; interpret the saved effect and conditional scope' if row['significant'] else 'The full-family corrected test did not detect a difference'
    families = [{'family_id': family, 'requested': len(rows), 'tested': sum(usable), 'unavailable': len(rows)-sum(usable),
        'members': [row['comparison_id'] for row in rows], 'alpha': settings['alpha'], 'multiple_testing': settings['multiple_testing'],
        'missing_probability_policy': 'Unavailable requested hypotheses occupy probability-one slots during correction; their reported probabilities remain null'}]
    return pd.DataFrame(rows, columns=RESULT_COLUMNS), pd.DataFrame(families, columns=FAMILY_COLUMNS), details, settings


def read_samples(saved, *, expected_model=None):
    provenance = read_document(saved.artifact('provenance'))
    if provenance.get('schema_version') != 1 or provenance.get('scientific_id') != saved.outcome.scientific_id:
        raise ValueError('Sample provenance does not match its scientific identity')
    if expected_model is not None and provenance['model_id'] != expected_model: raise ValueError('Sample results use a different state definition')
    names = ['unit_inventory', 'unit_metrics', 'cell_metrics', 'bout_membership', 'comparisons', 'families']
    tables = {name: read_table(saved.artifact(name)) for name in names}
    if any('model_id' in frame and not frame.model_id.eq(provenance['model_id']).all() for frame in tables.values()):
        raise ValueError('Sample tables mix state model identities')
    return tables, provenance


def produce(context):
    from pymicroglia.pipelines._runner import Unavailable
    decision, model = read_support(context.saved('state-support'), require_accepted=True)
    source = context.saved('durations-and-switches')
    tables, time_provenance = read_statistics(source, expected_model=model['model_id'])
    request = context.request.request
    from pymicroglia.pipelines.behaviour.options import time_settings
    if time_settings(time_provenance['settings']) != time_settings(request.statistics): raise ValueError('Sample statistics differ from saved time settings')
    try: settings = backend.policy(request.statistics['comparison'])
    except backend.UnsupportedComparison as error: raise Unavailable(str(error)) from error
    output = aggregate(tables, request.statistics['sample_aggregation'])
    comparisons, families, details, settings = compare_samples(output['unit_metrics'], settings)
    output.update(comparisons=comparisons, families=families)
    provenance = {'schema_version': 1, 'scientific_id': context.scientific_id, 'model_id': model['model_id'], 'decision_id': decision['decision_id'],
        'duration_id': source.outcome.scientific_id, 'settings': settings, 'aggregation': request.statistics['sample_aggregation'],
        'population_scope': 'Complete requested cell population; confirmed biological samples aggregate all their recordings; unconfirmed recordings remain descriptive',
        'formal_scope': 'Only independent assignment-only biological samples can enter formal contrasts; all state-choice sample roles remain descriptive',
        'interval_scope': 'Each switch-rate and next-state question uses exactly one saved observation-spacing stratum; missing common spacing retains unavailable comparisons',
        'duration_scope': 'Unique observed bout allocations and censoring remain visible; complete latent dwell-time inference is not implemented',
        'uncertainty_scope': 'Optional native approximate independent-sample percentile intervals are unadjusted; p-values are corrected over the entire requested family',
        'model_fitted': False, 'assignments_changed': False, 'plot_selection_applied': False, 'references': backend.REFERENCES,
        'libraries': {name: version(name) for name in ['numpy', 'pandas', 'scipy']}, 'counts': {name: len(frame) for name, frame in output.items()}}
    context.output.mkdir(parents=True)
    refs = []
    for name, frame in output.items():
        path = context.output/(name+'.json'); path = write_table(path, frame)
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id, columns=tuple(frame.columns)))
    for name, value in [('provenance', provenance), ('comparison_details', details)]:
        path = context.output/(name+'.json'); _write_json(path, value)
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id))
    selected = tuple(Settings({'comparison_id': row['comparison_id']}) for row in comparisons.to_dict('records') if row['significant'] is True)
    selection = SelectionRecord('supported-comparisons', context.scientific_id, Settings({'families': families.family_id.tolist(), 'scope': provenance['formal_scope']}), selected)
    return StepResult(context.step.name, context.scientific_id, 'completed',
        'Saved complete sample populations, explicit weighting and spacing, censored observed bouts and declared independent-sample contrasts', tuple(refs), (selection,))
