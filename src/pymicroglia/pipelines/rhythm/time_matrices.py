"""Explicit saved display scaling, followed by shared-row time matrix figures."""
from pymicroglia._results import output_files
from pymicroglia._results import read_document
import inspect
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pymicroglia.pipelines._contracts import ArtifactRef, Settings, StepResult, content_id
from pymicroglia.pipelines._screening import _write_json, file_hash, read_screen, read_table, write_table
DEFAULTS = {'matrix_population': 'any-significant', 'matrix_order': 'saved', 'matrix_reference': '', 'matrix_representation': 'raw', 'matrix_scale': 'none', 'matrix_normalization_config': {}, 'matrix_rows': 30, 'matrix_metrics': [], 'matrix_page': 1}

def options(presentation):
    declared = presentation.as_dict().get('time_matrices', {})
    if not isinstance(declared, dict) or set(declared) - (set(DEFAULTS) - {'matrix_page'} | {'text'}):
        raise ValueError('Time matrices accept population/order, saved representation, explicit scaling, page size and text')
    result = {**DEFAULTS, **{k: v for k, v in declared.items() if k != 'text'}}
    if result['matrix_representation'] not in {'raw', 'filtered', 'detrended'}:
        raise ValueError('matrix_representation must be raw, filtered or detrended')
    if result['matrix_order'] not in {'saved', 'identity', 'period'}:
        raise ValueError('matrix_order must be saved, identity or period with an explicit reference measurement')
    if result['matrix_order'] == 'period' and (not result['matrix_reference']):
        raise ValueError('Period ordering requires matrix_reference')
    if isinstance(result['matrix_rows'], bool) or not isinstance(result['matrix_rows'], int) or result['matrix_rows'] < 1:
        raise ValueError('matrix_rows must be a positive integer')
    if not isinstance(result['matrix_metrics'], list) or any((not isinstance(m, str) for m in result['matrix_metrics'])):
        raise ValueError('matrix_metrics must list saved measurement names')
    if len(set(result['matrix_metrics'])) != len(result['matrix_metrics']):
        raise ValueError('matrix_metrics contains duplicate names')
    return result

def normalization(settings):
    """Resolve the live Workbench catalogue without invoking a rhythm model."""
    import pymicroglia.workbench as circadian
    catalogue = circadian.available_normalization_methods()
    matches = [m for m in catalogue if settings['matrix_scale'] in {m['key'], *m.get('aliases', [])}]
    if len(matches) != 1:
        raise ValueError('Unknown Workbench matrix_scale: ' + str(settings['matrix_scale']))
    method = matches[0]
    if method['key'] in {'own_daily_total', 'envelope'}:
        raise ValueError('Time matrices use observed traces; daily totals and fitted envelopes require a separate justified analysis')
    if settings['matrix_representation'] == 'detrended' and (not method['supports_detrended']):
        raise ValueError('The selected normalization does not support detrended values')
    params = settings['matrix_normalization_config']
    allowed = set(inspect.signature(circadian.normalize_trace).parameters) - {'hours', 'values', 'method', 'detrended'}
    if not isinstance(params, dict) or set(params) - allowed:
        raise ValueError('matrix_normalization_config must contain public normalization parameters')
    if method['requires_reference'] and params.get('reference_value') is None and (not all((params.get(k) is not None for k in ('reference_start_hours', 'reference_end_hours')))):
        raise ValueError('The selected normalization requires an explicit reference value or recording-time window')
    return (method, params)

def implementation_version():
    import pymicroglia.workbench as circadian
    return content_id({'gateway': file_hash(circadian.__file__), 'workbench': circadian.rhythm_environment()})

def produce_values(context):
    """Freeze requested display values before any figure is drawn.

    All original screen pairs remain in this artifact. Population, ordering and
    pagination are selections made later by the pure renderer.
    """
    import pymicroglia.workbench as circadian
    from pymicroglia.figure_tables.rhythm_saved import overview_records
    settings = options(context.presentation)
    method, params = normalization(settings)
    saved = context.saved('rhythm-screen')
    screen = read_screen(saved.root)
    provenance = read_document(saved.artifact('provenance'))
    rows = overview_records(screen.results, provenance)
    keys = ['source_run', 'movie', 'identity', 'measurement']
    displays = {tuple((row[k] for k in keys)): row for row in screen.display_inputs.to_dict('records')}
    grouped = {key: group.sort_values(['hours', 'input_row'], kind='stable') for key, group in screen.traces.groupby(keys, sort=False)}
    points, metadata, details = ([], [], [])
    for row in rows.to_dict('records'):
        key = tuple((row[k] for k in keys))
        original = grouped.get(key, screen.traces.iloc[:0])
        hours, values, unit = (original.hours.to_numpy(float), original.value.to_numpy(float), row['unit'])
        source_reason = ''
        representation = settings['matrix_representation']
        if representation == 'filtered':
            if original.empty or original.filter_status.eq('not requested').all():
                values = np.full(len(hours), np.nan)
                source_reason = 'No filtering was requested in the saved screen'
            else:
                values = original.filtered_value.to_numpy(float)
        elif representation == 'detrended':
            diagnostic = displays[key].get('processed_trace')
            if diagnostic:
                hours, values = (np.asarray(diagnostic['hours'], float), np.asarray(diagnostic['values'], float))
                unit = diagnostic.get('value_unit') or unit
            else:
                values = np.full(len(hours), np.nan)
                source_reason = displays[key].get('processed_trace_reason') or 'No saved detrending diagnostic'
        if len(hours) != len(values):
            raise ValueError('Saved matrix representation has inconsistent lengths')
        scaled = values.copy()
        finite_time = np.isfinite(hours)
        status = 'available' if (finite_time & np.isfinite(values)).any() else 'unavailable'
        reason = source_reason or ('No finite saved observations' if status == 'unavailable' else '')
        label = unit if method['key'] == 'none' else method['label']
        display_unit = unit if method['key'] in {'none', 'mean_center', 'median_center', 'reference_delta'} else 'normalized units'
        detail = {'method': 'none', 'source': 'unchanged saved representation'}
        if method['key'] != 'none' and status == 'available':
            try:
                detail = circadian.normalize_trace(hours[finite_time], values[finite_time], method=method['key'], detrended=representation == 'detrended', **params)
                returned_hours = np.asarray(detail['hours'], float)
                if not np.array_equal(returned_hours, hours[finite_time], equal_nan=True):
                    raise ValueError('Workbench normalization changed original recording coordinates')
                returned_values = np.asarray(detail['values'], float)
                if returned_values.shape != values[finite_time].shape or np.any(np.isfinite(returned_values) & ~np.isfinite(values[finite_time])):
                    raise ValueError('Workbench normalization changed missing observations')
                scaled = np.full(len(hours), np.nan)
                scaled[finite_time] = returned_values
                label = detail['y_axis_label']
                display_unit = unit if method['key'] in {'mean_center', 'median_center', 'reference_delta'} else detail.get('processed_trace', {}).get('value_unit', 'normalized units')
            except Exception as error:
                scaled = np.full(len(hours), np.nan)
                status, reason = ('unavailable', f'{type(error).__name__}: {error}')
                detail = {'method': method['key'], 'status': status, 'reason': reason}
        if status == 'unavailable':
            scaled[:] = np.nan
        key_record = {k: row[k] for k in keys}
        for position, (hour, value, displayed) in enumerate(zip(hours, values, scaled)):
            points.append({**key_record, 'position': position, 'hours': hour, 'source_value': value, 'display_value': displayed, 'source_unit': unit, 'display_unit': display_unit, 'representation': representation, 'scale': method['key']})
        metadata.append({**row, 'representation': representation, 'scale': method['key'], 'scaling_status': status, 'scaling_reason': reason, 'scale_label': label or 'Saved measurement units', 'source_unit': unit, 'display_unit': display_unit})
        details.append({**key_record, 'result': detail})
    context.output.mkdir(parents=True, exist_ok=True)
    point_columns = [*keys, 'position', 'hours', 'source_value', 'display_value', 'source_unit', 'display_unit', 'representation', 'scale']
    write_table(context.output / 'time_values.csv', pd.DataFrame(points, columns=point_columns))
    write_table(context.output / 'time_status.csv', pd.DataFrame(metadata, columns=[*rows.columns, 'representation', 'scale', 'scaling_status', 'scaling_reason', 'scale_label', 'source_unit', 'display_unit']))
    _write_json(context.output / 'normalization_details.json', details)
    _write_json(context.output / 'provenance.json', {'schema_version': 1, 'screen_id': saved.outcome.scientific_id, 'purpose': 'Explicit display preparation; original rhythmicity evidence is unchanged', 'representation': settings['matrix_representation'], 'normalization_method': method, 'normalization_config': params, 'workbench_version': context.request.workbench_version, 'environment': circadian.rhythm_environment(), 'source_screen_provenance': provenance, 'pairs': len(rows), 'available_pairs': sum((r['scaling_status'] == 'available' for r in metadata))})
    refs = tuple((ArtifactRef(name, filename, file_hash(context.output / filename), context.scientific_id) for name, filename in (('values', 'time_values.csv'), ('status', 'time_status.csv'), ('normalization_details', 'normalization_details.json'), ('provenance', 'provenance.json'))))
    return StepResult(context.step.name, context.scientific_id, 'completed', 'Prepared explicit display values for every saved cell/measurement pair', refs, provenance=Settings({'screen_id': saved.outcome.scientific_id, 'rhythm_analysis_recomputed': False}))

def figure_version():
    from pymicroglia.pipelines._versions import rendering
    return rendering(__file__)

def produce_figures(context):
    from pymicroglia.pipelines._runner import figure_binding, register_figure_plan
    from pymicroglia.pipelines.rhythm.images import original_run
    import pymicroglia.figure_tables.time_matrix as display
    import pymicroglia.figure_tables.ordering as matrix_ordering
    from pymicroglia.visualisation.panels import rhythm_time_matrix
    settings = options(context.presentation)
    saved = context.saved('time-matrix-values')
    values, status = (read_table(saved.artifact('values')), read_table(saved.artifact('status')))
    provenance = read_document(saved.artifact('provenance'))
    selections = {s.name: s for s in context.saved('rhythm-screen').outcome.selections}
    members = display.row_map(status, selections, settings)
    data = {'values': values, 'status': status, 'members': members, 'settings': settings, 'pages': display.pages(values, status, members, settings), 'provenance': provenance}
    wording = context.presentation.as_dict().get('time_matrices', {}).get('text', {})
    if not isinstance(wording, dict) or set(wording) - {'title', 'subtitle', 'footnote', 'note', 'claim'}:
        raise ValueError('Time matrix text accepts title/subtitle/footnote/note/claim')
    run = original_run(context) or context.output.parents[3]
    binding = figure_binding(context.dependencies, inputs=display.ALIASES)
    items = [{'name': f'rhythm-time-matrix-{context.presentation_id[:12]}-p{index}', 'figure': 'rhythm-time-matrix', 'options': {**settings, 'matrix_page': index}, 'pipeline': binding, 'text': wording} for index in range(1, len(data['pages']) + 1)]
    context.output.mkdir(parents=True, exist_ok=True)
    completed = {'check_output': 'No selected cells; no time matrix pages requested', 'registration_output': ''}
    if items:
        plan = register_figure_plan(run, items)
        sources = {plan, Path(display.__file__), Path(matrix_ordering.__file__), Path(rhythm_time_matrix.__file__), *(saved.artifact(ref.name) for ref in saved.outcome.artifacts), *(context.saved('rhythm-screen').artifact(name) for name in ('rhythm_results', 'trace_inputs', 'display_inputs'))}
        completed = render_items(context, run, items, sources)
    manifest = {'schema_version': 1, 'scientific_id': context.scientific_id, 'screen_id': provenance['screen_id'], 'prepared_values_id': saved.outcome.scientific_id, 'analysis_recomputed': False, 'population': settings['matrix_population'], 'order': settings['matrix_order'], 'reference': settings['matrix_reference'], 'shared_rows': members.to_dict('records'), 'representation': provenance['representation'], 'normalization_method': provenance['normalization_method'], 'pages': [{'measurement': page['measurement'], 'movie': page['movie'], 'part': page['part'], 'parts': page['parts'], 'members': page['members'][[*display.KEYS, 'matrix_row', 'panel_row']].to_dict('records'), 'master': item['name'] + '.svg'} for item, page in zip(items, data['pages'])], 'empty_reason': 'No cells in the requested saved population' if members.empty else '', 'check_output': completed['check_output'], 'registration_output': completed['registration_output']}
    _write_json(context.output / 'time_matrix_manifest.json', manifest)
    refs = tuple((ArtifactRef(p.name, p.name, file_hash(p), context.scientific_id) for p in sorted(output_files(context.output)) if p.is_file() and (not p.name.startswith('.'))))
    return StepResult(context.step.name, context.scientific_id, 'completed', 'Saved time matrices with a shared cell row map', refs, provenance=Settings({'screen_id': provenance['screen_id'], 'analysis_recomputed': False}))

from pymicroglia.visualisation.figures import get_figure
from pymicroglia.pipelines._saved_figures import render_items
