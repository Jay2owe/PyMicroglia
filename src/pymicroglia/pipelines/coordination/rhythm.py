"""Optional cross-cell timing from a pinned, independently corrected rhythm screen."""
from __future__ import annotations
from pymicroglia._results import read_document
from pymicroglia._sources import source_file

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from pymicroglia.pipelines._contracts import ArtifactRef, StepResult, content_id
from pymicroglia.pipelines.coordination.inputs import PAIR_KEYS, _table, read_inputs
from pymicroglia.pipelines.coordination.sources import load_source
from pymicroglia.pipelines.rhythm.timing import validate_options
from pymicroglia.pipelines._runner import Unavailable
from pymicroglia.pipelines._screening import _write_json, file_hash, read_screen, write_table


def implementation_version():
    import pymicroglia.workbench as circadian
    return {'code': {path.name: file_hash(path) for path in [Path(__file__), source_file('coordination_sources.py'),
        source_file('rhythm_timing.py'), source_file('screening.py'), Path(circadian.__file__)]},
        'workbench': circadian.rhythm_environment()}


def validate_question(question, measurements):
    if question['statistic'] != 'phase_offset_hours' or question['evidence']['method'] != 'none':
        raise Unavailable('Saved rhythm coordination supports native phase_offset_hours with conditional timing uncertainty; the Workbench timing operation supplies no pair significance probability')
    return validate_options(dict(question['settings']), measurements)


def identity(context):
    question = context.request.request.questions['rhythm']
    if not question['enabled']: return content_id({'enabled': False, 'code': file_hash(__file__)})
    validate_question(question, [m.column for m in (*context.request.reference_measurements, *context.request.target_measurements)])
    _, binding = load_source(question['source'], recipe='rhythm-discovery', step='rhythm-screen')
    return content_id({'pair_inputs': context.saved('pair-inputs').outcome.scientific_id,
        'source_id': binding['scientific_id'], 'source_artifacts': binding['artifacts'],
        'question': {key: value for key, value in question.items() if key != 'source'}, 'implementation': implementation_version()})


def source_data(resolved):
    question = resolved.request.questions['rhythm']
    saved, binding = load_source(question['source'], recipe='rhythm-discovery', step='rhythm-screen')
    screen = read_screen(saved.root, expected_id=saved.outcome.scientific_id)
    provenance = read_document(saved.artifact('provenance')); source = provenance['resolved_request']
    if source['inputs']['source_run'] != resolved.inputs.source_run:
        raise ValueError('Saved rhythm results belong to a different original source run')
    source_window = source['request'].get('time_range_hours')
    if (tuple(source_window) if source_window is not None else None) != resolved.request.time_range_hours:
        raise ValueError('Saved rhythm evidence has a different analysis window; explicitly produce matching screening evidence first')
    definitions = {m['column']: m for m in source['test_measurements']}
    for measurement in (*resolved.reference_measurements, *resolved.target_measurements):
        recorded = definitions.get(measurement.column)
        if recorded is None: continue  # A requested unscreened endpoint stays visible as such.
        if recorded['table'] != measurement.table or tuple(recorded['grain']) != measurement.grain:
            raise ValueError('Saved rhythm measurement resolves to a different source table or observation grain')
        if source['inputs']['table_hashes'].get(measurement.table) != resolved.inputs.table_hashes[measurement.table]:
            raise ValueError('Original measured table changed since the selected rhythm screen')
    return screen, provenance, binding


def analyse(resolved, prepared, screen, source_provenance, settings, scientific_id):
    import pymicroglia.workbench as circadian
    keys = ['source_run', 'movie', 'identity', 'measurement']
    evidence = {tuple(row[key] for key in keys): row for row in screen.results.to_dict('records')}
    traces = {key: frame for key, frame in screen.traces.groupby(keys, sort=False)}
    endpoint_status = prepared['endpoints'].set_index('endpoint_id').status.to_dict()
    native_settings = dict(settings['phase_options']); source = source_provenance['resolved_request']
    native_settings.setdefault('min_observations', max(24, source['analysis_options']['min_observations']))
    native_settings.setdefault('min_cycles', max(3, math.ceil(source['analysis_options']['min_cycles'])))
    rows = []; endpoints = []; segments = []; points = []; details = []
    for pair in prepared['inventory'].to_dict('records'):
        result_id = content_id({'scientific_id': scientific_id, 'pair': pair['pair_id'], 'question': 'rhythm'})
        key = {**pair, 'result_id': result_id}; payloads = []; records = []; refusals = []
        for role in ['reference', 'target']:
            measured = (pair['source_run'], pair['movie'], pair[role+'_identity'], pair[role])
            record = evidence.get(measured, {'status': 'not_screened', 'reason': 'This exact cell/measurement endpoint was not included in the saved rhythm screen'})
            records.append(record)
            trace = traces.get(measured, screen.traces.iloc[:0]).copy()
            trace = trace.loc[np.isfinite(pd.to_numeric(trace.hours, errors='coerce'))].sort_values('hours', kind='stable')
            values = 'filtered_value' if settings['trace_representation'] == 'filtered' else 'value'
            supported = (record.get('period_available') is not None and bool(record.get('period_available'))
                and not bool(record.get('period_underdetermined')) and not bool(record.get('period_at_search_edge')))
            payloads.append({'hours': trace.hours.tolist(), 'values': trace[values].tolist(),
                'detected': record['status'] == 'significant', 'period_supported': supported,
                'estimate': record.get('estimate_result') if isinstance(record.get('estimate_result'), dict) else {},
                'significance': record.get('significance_result') if isinstance(record.get('significance_result'), dict) else {},
                **({'component_period_band': settings['component_bands'][pair[role]]} if pair[role] in settings['component_bands'] else {})})
            if endpoint_status[pair[role+'_endpoint_id']] == 'invalid_clock': refusals.append(role+': original acquisition clock is invalid')
            endpoints.append({**key, 'endpoint_role': role, 'endpoint_id': pair[role+'_endpoint_id'],
                'identity': pair[role+'_identity'], 'measurement': pair[role], 'measurement_id': pair[role+'_measurement_id'],
                'screen_status': record['status'], 'screen_reason': record.get('reason'), 'period_hours': record.get('period_hours'),
                'period_supported': supported, 'estimator': record.get('method'), 'significance_method': record.get('significance_method'),
                'trace_p_value': record.get('p_value'), 'trace_q_value': record.get('q_value'), 'trace_family_id': record.get('family_id'),
                'source_screen_id': screen.outcome.scientific_id, 'saved_observations': len(trace), 'saved_representation': settings['trace_representation'],
                'source_evidence': record})
        if not pair['geometry_population_eligible']: refusals.append('Outside the predeclared geometric pair population')
        if refusals:
            result = {'status': 'ineligible', 'reason': '; '.join(refusals), 'segments': [], 'timecourse': []}
        else:
            try: result = circadian.rhythm_pair_timing(*payloads, settings=native_settings)
            except circadian.cw.WorkbenchInputError as error:
                if any(np.any(np.diff(np.asarray(payload['hours'])) <= 0) for payload in payloads):
                    result = {'status': 'ineligible', 'reason': str(error), 'segments': [], 'timecourse': []}
                else: raise
        row = {**key, 'question': 'rhythm', 'evidence_level': 'pair', 'representation': 'saved_'+settings['trace_representation'],
            'adjustment': 'none', 'method': 'workbench_rhythm_pair_timing', 'statistic': 'phase_offset_hours',
            'status': result['status'], 'reason': result['reason'], 'effect': result.get('offset_hours'), 'p_value': None,
            'both_significant': all(record['status'] == 'significant' for record in records),
            'reference_screen_status': records[0]['status'], 'target_screen_status': records[1]['status'],
            'source_screen_id': screen.outcome.scientific_id, 'workbench_version': circadian.WORKBENCH_VERSION,
            'uncertainty_status': 'conditional_native_timing' if result['status'] == 'eligible' else 'unavailable',
            'pair_probability_meaning': 'No pair significance probability supplied; endpoint rhythm probabilities are retained separately',
            **{name: result.get(name) for name in ['period_hours', 'offset_hours', 'offset_interval_hours', 'descriptive_relation',
                'temporal_status', 'summary_segment', 'phase_definition', 'sign_convention', 'common_observations', 'jointly_finite_observations']}}
        rows.append(row)
        segments.extend({**key, **{name: value for name, value in segment.items() if name != 'native'}} for segment in result.get('segments', []))
        points.extend({**key, **point} for point in result.get('timecourse', [])); details.append({**key, 'result': result})
    return {'pair_effects': _table(rows, PAIR_KEYS+['result_id', 'question', 'evidence_level', 'effect', 'p_value', 'status', 'reason']),
        'endpoint_evidence': _table(endpoints, PAIR_KEYS+['result_id', 'endpoint_id', 'endpoint_role', 'identity', 'measurement', 'screen_status']),
        'segments': _table(segments, PAIR_KEYS+['result_id', 'segment', 'offset_hours']),
        'timecourse': _table(points, PAIR_KEYS+['result_id', 'hours'])}, details, native_settings


def produce(context):
    question = context.request.request.questions['rhythm']
    if not question['enabled']: return StepResult(context.step.name, context.scientific_id, 'skipped-empty', 'Optional rhythm timing was disabled; no source or Workbench analysis was opened')
    settings = validate_question(question, [m.column for m in (*context.request.reference_measurements, *context.request.target_measurements)])
    prepared = read_inputs(context.saved('pair-inputs')); screen, provenance, binding = source_data(context.request)
    outputs, details, native_settings = analyse(context.request, prepared, screen, provenance, settings, context.scientific_id)
    context.output.mkdir(parents=True); refs = []
    for name, frame in outputs.items():
        path = context.output/(name+'.json'); path = write_table(path, frame)
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id, columns=tuple(frame.columns)))
    for name, document in {'engine_details': details, 'provenance': {'schema_version': 1, 'scientific_id': context.scientific_id,
        'pair_inputs_id': context.saved('pair-inputs').outcome.scientific_id, 'source_binding': binding,
        'source_screen_provenance': provenance, 'settings': settings, 'effective_phase_options': native_settings,
        'implementation': implementation_version(), 'source_screen_recomputed': False,
        'source_trace_policy': 'Original saved unadjusted screen traces and recorded filtering; the separate core temporal/shared-reference choices do not redefine rhythm evidence',
        'inference': 'Corrected trace rhythmicity precedes native period comparability and conditional observed timing; no trace probability supplies a pair probability',
        'clock': 'Actual saved recording origins and supported segments; no local period averaging, daily folding or conversion of incomparable periods'}}.items():
        path = context.output/(name+'.json'); _write_json(path, document)
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id))
    return StepResult(context.step.name, context.scientific_id, 'completed',
        'Saved every cross-cell rhythm comparability decision, original endpoint evidence and public Workbench timing/segment limits', tuple(refs))
