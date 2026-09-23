"""Complete saved response selections and original-cell trace reports."""
from pymicroglia._sources import source_file
from pathlib import Path
import pandas as pd
from pymicroglia.pipelines._contracts import StepResult, content_id
from pymicroglia.pipelines._screening import _json_value, file_hash
from pymicroglia.pipelines.intervention.windows import read_windows
from pymicroglia.pipelines.intervention.evidence import read_evidence
from pymicroglia.pipelines.intervention.timing import read_timing
from pymicroglia.pipelines.intervention.rhythms import read_rhythms
import pymicroglia.pipelines.intervention.display as display
SLUGS = {'cell_report': 'intervention-cell-report', 'trace_grid': 'intervention-response-traces'}
KEYS = ['source_run', 'movie', 'identity']

def options(presentation):
    block = presentation.as_dict().get('reports', {})
    if not isinstance(block, dict) or set(block) - {'views', 'cells_per_page', 'measurements', 'grid_clock', 'shared_y', 'text'}:
        raise ValueError('Reports accept views, cells_per_page, measurements, grid_clock, shared_y and text')
    chosen = {'views': ['cell_report', 'trace_grid'], 'cells_per_page': 6, 'measurements': 'responding', 'grid_clock': 'relative_hours', 'shared_y': False, **{k: v for k, v in block.items() if k != 'text'}}
    if not isinstance(chosen['views'], list) or not chosen['views'] or len(set(chosen['views'])) != len(chosen['views']) or set(chosen['views']) - set(SLUGS):
        raise ValueError('Unknown or repeated report view')
    n = chosen['cells_per_page']
    if isinstance(n, bool) or not isinstance(n, int) or (not 1 <= n <= 12):
        raise ValueError('cells_per_page must be an integer from 1 to 12')
    if chosen['measurements'] not in {'responding', 'all_requested'}:
        raise ValueError('Report measurements must be responding or all_requested')
    if chosen['grid_clock'] not in {'relative_hours', 'hours'}:
        raise ValueError('Grid clock must be relative_hours or hours')
    if not isinstance(chosen['shared_y'], bool):
        raise ValueError('shared_y must be boolean')
    return (chosen, block.get('text', {}))

def collect(context):
    available = lambda name: name in context.dependencies and context.saved(name).outcome.status in {'completed', 'reused'}
    windows = context.saved('aligned-windows') if available('aligned-windows') else None
    wid = windows.outcome.scientific_id if windows else None
    evidence = context.saved('response-evidence') if available('response-evidence') else None
    data = {'windows': read_windows(windows) if windows else None, 'evidence': read_evidence(evidence, wid) if evidence else None, 'timing': read_timing(context.saved('response-timing'), evidence.outcome.scientific_id if evidence else None) if available('response-timing') else None, 'rhythms': read_rhythms(context.saved('rhythm-changes'), wid) if available('rhythm-changes') else None, 'resolved': context.request, 'selections': [], 'selected': {}, 'branches': {}}
    for step, name in [('response-evidence', 'responding-comparisons'), ('rhythm-changes', 'rhythm-change-comparisons')]:
        if not available(step):
            continue
        saved = context.saved(step)
        selection = next((item for item in saved.outcome.selections if item.name == name), None)
        if selection is None:
            raise ValueError('Saved response source is missing its immutable comparison selection')
        data['selections'].append(selection.as_dict())
        for member in selection.members:
            row = member.as_dict()
            key = content_id({k: row[k] for k in KEYS})
            selected = data['selected'].setdefault(key, {'cell': {k: row[k] for k in KEYS}, 'measurements': {}, 'sources': []})
            selected['measurements'].setdefault(row['measurement'], []).append({'step': step, 'selection': name, **row})
            if step not in selected['sources']:
                selected['sources'].append(step)
    for step, enabled in [('response-timing', context.request.request.timing['enabled']), ('rhythm-changes', context.request.request.rhythms['enabled'])]:
        source = context.dependencies.get(step)
        data['branches'][step] = {'status': 'disabled' if not enabled else 'available' if available(step) else source.outcome.status if source else 'unavailable', 'reason': 'Explicitly disabled' if not enabled else source.outcome.reason if source else 'No saved result'}
    return data

def rows_for(data, cell, metric):
    result = []
    cell_id = content_id(cell)
    for source, table, kind in [('windows', 'traces', 'trace'), ('windows', 'windows', 'window'), ('windows', 'comparisons', 'comparison'), ('evidence', 'effects', 'effect'), ('timing', 'timing', 'timing'), ('rhythms', 'window_results', 'rhythm_window'), ('rhythms', 'direct_comparisons', 'rhythm_change')]:
        if data[source] is None:
            continue
        frame = data[source][table]
        if frame.empty:
            continue
        match = frame.measurement.eq(metric)
        for key in KEYS:
            match &= frame[key].eq(cell[key])
        for row in _json_value(frame.loc[match].to_dict('records')):
            result.append(display.entry(**{**row, 'kind': kind, 'cell_id': cell_id, 'cell_label': cell['movie'] + ' / cell ' + str(cell['identity'])}))
    return result

def pages(data, settings):
    values = {}
    statistics = {}
    planned = []
    resolved = data['resolved']

    def add(view, rows, **extra):
        for row in rows:
            values[row['entry_id']] = row
            if row['kind'] != 'trace':
                statistics[row['entry_id']] = row
        planned.append({'view': view, 'entry_ids': [r['entry_id'] for r in rows], 'claim': 'Every selected original cell retains its measured trace and independent saved change evidence without new tests or inferred rhythms.', 'footnote': 'Original observations only; lines stop at missing values or acquisition gaps. Selection is the display union of saved measurement-change and direct rhythm-change evidence, not a new cell-level test. Control anchors are not treatment administrations.', 'max_gap_hours': resolved.request.support['max_gap_hours'], 'branches': data['branches'], **extra})
    groups = {}
    for cell_id, selected in sorted(data['selected'].items(), key=lambda item: (item[1]['cell']['movie'], item[1]['cell']['identity'])):
        cell = selected['cell']
        metrics = [m.column for m in resolved.measurements if settings['measurements'] == 'all_requested' or m.column in selected['measurements']]
        for metric in metrics:
            rows = rows_for(data, cell, metric)
            if not rows:
                raise ValueError('A selected cell/measurement has no saved original inventory')
            groups.setdefault(metric, []).append((cell_id, rows))
            if 'cell_report' not in settings['views']:
                continue
            comparisons = [r for r in rows if r['kind'] == 'comparison']
            for comparison in comparisons or [None]:
                ids = {comparison[k] for k in ['baseline_window_id', 'target_window_id']} if comparison else set()
                scoped = [r for r in rows if r['kind'] == 'trace' or comparison is None or (r['kind'] in {'window', 'rhythm_window'} and r['window_id'] in ids) or (r['kind'] not in {'window', 'rhythm_window', 'trace'} and r.get('comparison_id') == comparison['comparison_id'])]
                add('cell_report', scoped, title=cell['movie'] + ' / cell ' + str(cell['identity']) + ' | ' + metric, cell_ids=[cell_id], measurement=metric, unit=next((m.unit for m in resolved.measurements if m.column == metric)), selection_sources=selected['sources'], selection_members=selected['measurements'].get(metric, []), baseline=comparison['baseline'] if comparison else None, target_window=comparison['target_window'] if comparison else None, anchor=resolved.recordings[cell['movie']]['anchor'], shared_y=settings['shared_y'])
    if 'trace_grid' in settings['views']:
        for metric, members in groups.items():
            original = [row['raw_value'] for _, rows in members for row in rows if row['kind'] == 'trace' and row['raw_valid'] and row['clock_valid']]
            limits = None
            if settings['shared_y'] and original:
                low, high = (min(original), max(original))
                padding = 0.05 * (high - low or abs(high) or 1.0)
                limits = [low - padding, high + padding]
            for begin in range(0, len(members), settings['cells_per_page']):
                chunk = members[begin:begin + settings['cells_per_page']]
                add('trace_grid', [r for _, rows in chunk for r in rows], title='Selected-cell traces | ' + metric, measurement=metric, unit=next((m.unit for m in resolved.measurements if m.column == metric)), cell_ids=[key for key, _ in chunk], anchors={key: resolved.recordings[data['selected'][key]['cell']['movie']]['anchor'] for key, _ in chunk}, grid_clock=settings['grid_clock'], shared_y=settings['shared_y'], y_limits=limits)
    return (pd.DataFrame(values.values()), pd.DataFrame(statistics.values()), planned)

def version():
    from pymicroglia.pipelines._versions import rendering
    return rendering(__file__)

def build(ctx):
    from pymicroglia.visualisation.panels import intervention_reports
    return display.build(ctx, intervention_reports)

def produce(context):
    data = collect(context)
    settings, text = options(context.presentation)
    if not data['selected']:
        return StepResult(context.step.name, context.scientific_id, 'skipped-empty', 'No original cell was selected by saved measurement-change or direct rhythm-change evidence')
    if data['windows'] is None:
        raise ValueError('Selected-cell reports require their original saved windows')
    from pymicroglia.visualisation.panels import intervention_reports
    values, statistics, planned = pages(data, settings)
    return display.produce(context, values=values, statistics=statistics, metadata={'settings': settings, 'selections': data['selections'], 'selection_meaning': 'Display union only; original within-cell measurement and rhythm-parameter families remain separate', 'selected_cells': len(data['selected']), 'requested_cells': len(context.request.inputs.cells), 'branches': data['branches']}, pages=planned, slugs=[SLUGS[page['view']] for page in planned], panel=intervention_reports, sources=[__file__], text=text)
