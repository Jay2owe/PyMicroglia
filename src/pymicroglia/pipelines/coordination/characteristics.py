"""Original characteristics, descriptive pair differences and spatial evidence."""
from __future__ import annotations
from pymicroglia._sources import source_file

from importlib.metadata import version as library_version
from pathlib import Path
import numpy as np
import pandas as pd

import pymicroglia.measure.coordination_statistics as native
from pymicroglia.pipelines._contracts import ArtifactRef, StepResult, content_id
from pymicroglia.pipelines.coordination.inputs import PAIR_KEYS, _adjacent, _table, read_inputs
from pymicroglia.pipelines._screening import _write_json, file_hash, write_table


def implementation_version():
    return content_id({'code': {p.name: file_hash(p) for p in [Path(__file__), Path(native.__file__),
        source_file('coordination_inputs.py')]},
        'libraries': {name: library_version(name) for name in ['numpy', 'pandas', 'scipy']}})


def identity(context):
    return content_id({'inputs': context.saved('pair-inputs').outcome.scientific_id,
        'question': context.request.request.questions['characteristics'], 'implementation': implementation_version()})


def characteristics(resolved, prepared):
    request = resolved.request; question = request.questions['characteristics']
    selected = {m.record_id: m for m in (*resolved.reference_measurements, *resolved.target_measurements)}
    traces = {endpoint: frame for endpoint, frame in prepared['traces'].groupby('endpoint_id', sort=False)}
    scalars = {row['endpoint_id']: row for row in prepared['scalars'].to_dict('records')}
    rows = []
    for endpoint in prepared['endpoints'].to_dict('records'):
        if endpoint['measurement_id'] not in selected: continue
        base = {**endpoint, 'summary': 'saved_scalar' if endpoint['kind'] == 'scalar' else question['summary'],
            'representation': 'original_characteristic', 'value': None, 'observed_interval_hours': None,
            'characteristic_id': content_id({'endpoint': endpoint['endpoint_id'], 'summary': question['summary'], 'inputs': prepared['provenance']['scientific_id']}),
            'status': endpoint['status'], 'reason': endpoint['reason']}
        if endpoint['kind'] == 'scalar':
            scalar = scalars[endpoint['endpoint_id']]
            base.update(value=scalar['value'] if scalar['value_kind'] == 'finite' else None,
                observations=endpoint['valid_observations'], excluded_observations=endpoint['source_observations']-endpoint['valid_observations'],
                start_hours=None, end_hours=None, eligible=scalar['value_kind'] == 'finite',
                window_meaning='Original saved whole-recording scalar; source table retains its provenance')
        else:
            frame = traces.get(endpoint['endpoint_id'], prepared['traces'].iloc[:0])
            observed = frame.loc[frame.within_range & frame.processed_valid]
            n = len(observed); span = float(observed.hours.max()-observed.hours.min()) if n else 0.
            records = frame.to_dict('records'); exposure = 0.
            for a, b in zip(records, records[1:]):
                if a['processed_valid'] and b['processed_valid'] and _adjacent(a, b, request.support['max_gap_hours'])[0]: exposure += b['hours']-a['hours']
            enough = n >= request.support['min_observations'] and span >= request.support['min_span_hours']
            # The original finite summary is retained even if support is too
            # short for the declared spatial comparison.
            base.update(value=float(getattr(observed.raw_value, question['summary'])()) if n else None,
                observations=n, excluded_observations=len(frame)-n, observed_interval_hours=exposure,
                start_hours=float(observed.hours.min()) if n else None, end_hours=float(observed.hours.max()) if n else None,
                span_hours=span, eligible=enough, status='eligible' if enough else endpoint['status'] if endpoint['status'] != 'available' else 'insufficient',
                reason='Original measured characteristic over declared supported observations' if enough else 'Insufficient original observations, invalid clock or time span',
                window_meaning='Declared half-open analysis window; actual observed start/end and gaps retained')
        rows.append(base)
    return _table(rows, ['source_run', 'movie', 'identity', 'measurement', 'measurement_id', 'endpoint_id', 'characteristic_id',
        'summary', 'representation', 'value', 'unit', 'eligible', 'status', 'reason', 'observations', 'observed_interval_hours'])


def analyse(resolved, prepared, scientific_id):
    question = resolved.request.questions['characteristics']; settings = native.validate_spatial(question)
    cells = characteristics(resolved, prepared); by_endpoint = {row['endpoint_id']: row for row in cells.to_dict('records')}
    by_cell = {(row['movie'], row['identity'], row['measurement_id']): row for row in cells.to_dict('records')}
    support = {row['pair_id']: row for row in prepared['support'].query("question == 'characteristics'").to_dict('records')}
    pair_rows = []
    for pair in prepared['inventory'].to_dict('records'):
        a, b = by_endpoint[pair['reference_endpoint_id']], by_endpoint[pair['target_endpoint_id']]
        value = float(native.pair_difference(a['value'], b['value'], question['statistic'])) if a['value'] is not None and b['value'] is not None else None
        if value is not None and not np.isfinite(value): value = None
        eligible = support[pair['pair_id']]['status'] == 'eligible' and a['eligible'] and b['eligible'] and value is not None and np.isfinite(pair['static_distance'])
        unit = a['unit'] if question['statistic'] == 'absolute_difference' else (a['unit']+' squared' if a['unit'] else '')
        pair_rows.append({**pair, 'result_id': content_id({'pair': pair['pair_id'], 'science': scientific_id}), 'question': 'characteristics',
            'statistic': question['statistic'], 'reference_characteristic_id': a['characteristic_id'], 'target_characteristic_id': b['characteristic_id'],
            'reference_value': a['value'], 'target_value': b['value'], 'estimate': value, 'estimate_unit': unit,
            'evidence_level': 'pair', 'p_value': None, 'confidence_low': None, 'confidence_high': None,
            'status': 'descriptive' if eligible else support[pair['pair_id']]['status'] if support[pair['pair_id']]['status'] != 'eligible' else 'insufficient',
            'eligible': bool(eligible), 'reason': 'Original-unit descriptive pair difference; recording probability cannot certify this edge' if eligible else support[pair['pair_id']]['reason']})
    pairs = _table(pair_rows, [*prepared['inventory'].columns, 'result_id', 'question', 'statistic', 'estimate', 'estimate_unit',
        'evidence_level', 'p_value', 'eligible', 'status', 'reason'])
    references = {m.column: m for m in resolved.reference_measurements}; targets = {m.column: m for m in resolved.target_measurements}
    layouts = {(row['movie'], row['identity']): row for row in prepared['layouts'].to_dict('records')}
    recordings = []; nulls = []; profiles = []; memberships = []
    for sample in resolved.inputs.samples:
        for chosen in resolved.request.measurement_pairs:
            ma, mb = references[chosen['reference']], targets[chosen['target']]
            group = pairs.loc[pairs.movie.eq(sample.movie) & pairs.reference_measurement_id.eq(ma.record_id) & pairs.target_measurement_id.eq(mb.record_id)]
            eligible = group.loc[group.eligible]
            record_id = content_id({'source': resolved.inputs.source_run, 'movie': sample.movie, 'reference': ma,
                'target': mb, 'question': question, 'science': scientific_id})
            base = {'source_run': resolved.inputs.source_run, 'movie': sample.movie, 'recording_result_id': record_id,
                'question': 'characteristics', 'reference': ma.column, 'target': mb.column, 'reference_measurement_id': ma.record_id,
                'target_measurement_id': mb.record_id, 'sample': sample.sample, 'sample_confirmed': sample.confirmed,
                'condition': resolved.request.conditions.get(sample.movie), 'statistic': question['statistic'], 'distance_effect': settings['distance_effect'],
                'representation': 'original_characteristic', 'distance_unit': resolved.request.geometry['unit'],
                'effect_unit': 'rank correlation' if settings['distance_effect'] == 'spearman' else (ma.unit if question['statistic'] == 'absolute_difference' else ma.unit+' squared'),
                'requested_pairs': len(group), 'descriptive_pairs': len(eligible), 'summary': question['summary'],
                'descriptive_effect': native.distance_effect(eligible.estimate, eligible.static_distance, eligible.near, settings['distance_effect']),
                'minimum_distance': float(eligible.static_distance.min()) if len(eligible) else None,
                'maximum_distance': float(eligible.static_distance.max()) if len(eligible) else None,
                'meaning': 'Positive rank correlation means larger characteristic differences at larger distance' if settings['distance_effect'] == 'spearman' else
                    'Mean characteristic difference among static neighbours minus the mean among retained distant pairs'}
            complete = []; marks = []
            for cell in resolved.inputs.cells:
                if cell.movie != sample.movie: continue
                a, b = by_cell[(cell.movie, cell.identity, ma.record_id)], by_cell[(cell.movie, cell.identity, mb.record_id)]
                layout = layouts[(cell.movie, cell.identity)]
                accepted = bool(a['eligible'] and b['eligible'] and layout['valid_positions'] > 0)
                memberships.append({**base, **cell.as_dict(), 'reference_characteristic_id': a['characteristic_id'],
                    'target_characteristic_id': b['characteristic_id'], 'complete_for_spatial_reference': accepted,
                    'reason': 'Both original marks and a recorded static position available' if accepted else 'Missing or insufficient original mark/position support'})
                if accepted: complete.append(cell.identity); marks.append([a['value'], b['value']])
            lookup = {cell: index for index, cell in enumerate(complete)}
            tested_pairs = eligible.loc[eligible.reference_identity.isin(complete) & eligible.target_identity.isin(complete)]
            tested = native.spatial_test(np.asarray(marks, float).reshape((-1, 2)),
                [lookup[cell] for cell in tested_pairs.reference_identity], [lookup[cell] for cell in tested_pairs.target_identity],
                tested_pairs.static_distance, tested_pairs.near, question)
            for index, value in enumerate(tested.pop('null_values')):
                nulls.append({'recording_result_id': record_id, 'iteration': index, 'value': value})
            if settings['method'] == 'none':
                tested.update(effect=base['descriptive_effect'], status='descriptive' if base['descriptive_effect'] is not None else 'unresolvable',
                    reason='Complete eligible descriptive pair population; no spatial inference requested', p_value=None)
            recordings.append({**base, **tested, 'complete_cell_pairs': len(tested_pairs), 'complete_cell_identities': complete,
                'independent_biological_samples': 1 if sample.confirmed else 0})
            # Exact-distance summaries are unbinned observed profiles. They are
            # descriptive means with counts, not independent-pair confidence bands.
            for distance, same in eligible.groupby('static_distance', sort=True):
                profiles.append({'recording_result_id': record_id, 'distance': float(distance), 'mean_difference': float(same.estimate.mean()),
                    'median_difference': float(same.estimate.median()), 'pairs': len(same), 'near_pairs': int(same.near.sum()),
                    'cells': len(set(same.reference_identity)|set(same.target_identity)), 'pair_ids': same.pair_id.tolist(),
                    'distance_unit': resolved.request.geometry['unit'], 'difference_unit': same.estimate_unit.iloc[0],
                    'confidence_low': None, 'confidence_high': None, 'meaning': 'Unbinned original pair differences grouped at exactly the same distance; no fitted curve or independent-pair interval'})
    return {'cell_characteristics': cells, 'pair_characteristics': pairs,
        'recording_effects': _table(recordings, ['recording_result_id', 'source_run', 'movie', 'question', 'reference', 'target', 'effect', 'p_value', 'status', 'reason']),
        'spatial_membership': _table(memberships, ['recording_result_id', 'source_run', 'movie', 'identity', 'complete_for_spatial_reference']),
        'null_reference': _table(nulls, ['recording_result_id', 'iteration', 'value']),
        'distance_profiles': _table(profiles, ['recording_result_id', 'distance', 'mean_difference', 'median_difference', 'pairs', 'pair_ids'])}


def produce(context):
    question = context.request.request.questions['characteristics']
    if not question['enabled']:
        return StepResult(context.step.name, context.scientific_id, 'skipped-empty', 'Characteristic spatial comparison was explicitly disabled')
    native.validate_spatial(question)
    prepared = read_inputs(context.saved('pair-inputs')); outputs = analyse(context.request, prepared, context.scientific_id)
    context.output.mkdir(parents=True); refs = []
    for name, frame in outputs.items():
        path = context.output/(name+'.json'); path = write_table(path, frame)
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id, columns=tuple(frame.columns)))
    path = context.output/'provenance.json'
    _write_json(path, {'schema_version': 1, 'scientific_id': context.scientific_id,
        'pair_inputs_id': context.saved('pair-inputs').outcome.scientific_id, 'question': question,
        'native_settings': native.validate_spatial(question), 'implementation': implementation_version(), 'sources': native.SOURCES,
        'representation': 'Original raw characteristic summaries; the separate temporal representation does not redefine these traits',
        'families': 'All requested recording/measurement-pair hypotheses are retained, including unavailable estimates; later evidence stage applies correction',
        'pair_evidence': 'Descriptive only: no recording-level probability is copied to an edge',
        'spatial_null': 'Complete cell mark vectors exchangeable among fixed positions, conditional on the declared eligible population; no test of independent spatially autocorrelated fields',
        'uncertainty': 'Native permutation probabilities; no population spatial-effect confidence interval supplied by this random-label reference'})
    refs.append(ArtifactRef('provenance', path.name, file_hash(path), context.scientific_id))
    return StepResult(context.step.name, context.scientific_id, 'completed',
        'Saved original characteristics, complete descriptive pair differences and separate recording-level spatial evidence', tuple(refs))
