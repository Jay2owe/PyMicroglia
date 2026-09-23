"""Complete original effects and control comparisons, displayed without refits."""
from pymicroglia._sources import source_file
from pathlib import Path
import pandas as pd
from pymicroglia.pipelines._contracts import content_id
from pymicroglia.pipelines._screening import _json_value, file_hash
from pymicroglia.pipelines.intervention.evidence import read_evidence
from pymicroglia.pipelines.intervention.controls import read_controls
import pymicroglia.pipelines.intervention.display as display
SLUG = 'intervention-response-overview'
VIEWS = ['coverage', 'status_matrix', 'before_after', 'original_changes', 'model_effects', 'controls']
CLAIMS = {'coverage': 'Saved intervention outcomes retain the complete requested cell population and distinguish disabled, unavailable and evaluated questions.', 'status_matrix': 'Every original cell and measurement retains its saved change-evidence category; inconclusive evidence is distinct from no detected change.', 'before_after': 'Original baseline and follow-up summaries retain their measurement units and missing observations without a new paired test.', 'original_changes': 'Observed follow-up minus baseline changes retain all original cells and their saved evidence categories; observed changes are separate from model contrasts.', 'model_effects': 'Saved within-cell model contrasts retain pointwise uncertainty, full-family evidence and untestable outcomes without refitting.', 'controls': 'Saved treatment/control comparisons retain their actual biological units, original matching, conditional effect and uncertainty.'}

def options(presentation):
    block = presentation.as_dict().get('overview', {})
    if not isinstance(block, dict) or set(block) - {'views', 'rows_per_page', 'metrics_per_page', 'text'}:
        raise ValueError('Intervention overview accepts views, rows_per_page, metrics_per_page and text')
    settings = {'views': VIEWS, 'rows_per_page': 16, 'metrics_per_page': 6, **{key: value for key, value in block.items() if key != 'text'}}
    if not isinstance(settings['views'], list) or not settings['views'] or len(set(settings['views'])) != len(settings['views']) or set(settings['views']) - set(VIEWS):
        raise ValueError('Unknown or repeated intervention overview view')
    for key, maximum in [('rows_per_page', 32), ('metrics_per_page', 8)]:
        if isinstance(settings[key], bool) or not isinstance(settings[key], int) or (not 1 <= settings[key] <= maximum):
            raise ValueError(key + ' must be a positive integer no larger than ' + str(maximum))
    return (settings, block.get('text', {}))

def collect(context):
    available = lambda name: name in context.dependencies and context.saved(name).outcome.status in {'completed', 'reused'}
    data = {'evidence': read_evidence(context.saved('response-evidence')) if available('response-evidence') else None, 'controls': read_controls(context.saved('control-comparisons')) if available('control-comparisons') else None, 'request': context.request, 'coverage': []}
    request = context.request.request
    for step, label, enabled in [('aligned-windows', 'Original windows', True), ('response-evidence', 'Within-cell changes', True), ('control-comparisons', 'Treatment/control samples', request.controls['enabled']), ('response-timing', 'Observed timing and recovery', request.timing['enabled']), ('rhythm-changes', 'Before/after rhythms', request.rhythms['enabled']), ('coordinated-responses', 'Paired measurement changes', request.coordinated['enabled'])]:
        saved = context.dependencies.get(step)
        data['coverage'].append(display.entry(kind='coverage', source_step=step, label=label, status='disabled' if not enabled else 'available' if available(step) else saved.outcome.status if saved else 'unavailable', reason='Question explicitly disabled' if not enabled else saved.outcome.reason if saved else 'No saved result', source_scientific_id=saved.outcome.scientific_id if saved else None, requested_cells=len(context.request.inputs.cells), requested_measurements=len(context.request.measurements), effect=None, p_value=None, q_value=None))
    return data

def effects(data):
    if data['evidence'] is not None:
        rows = _json_value(data['evidence']['effects'].to_dict('records'))
    else:
        rows = []
        resolved = data['request']
        for cell in resolved.inputs.cells:
            for measured in resolved.measurements:
                for window in resolved.recordings[cell.movie]['windows']:
                    if window['baseline'] is not None:
                        rows.append({**cell.as_dict(), 'measurement': measured.column, 'unit': measured.unit, 'baseline': window['baseline'], 'target_window': window['name'], 'effect_id': None, 'comparison_id': None, 'absolute_change': None, 'estimate': None, 'interval_low': None, 'interval_high': None, 'baseline_value': None, 'target_value': None, 'outcome': 'inconclusive', 'response_supported': False, 'p_value': None, 'q_value': None, 'reason': 'Scientific evidence unavailable; this row preserves the original requested comparison'})
    return [display.entry(**row, kind='cell_effect', cell_id=content_id({key: row[key] for key in ['source_run', 'movie', 'identity']}), cell_label=row['movie'] + ' / cell ' + str(row['identity'])) for row in rows]

def pages(data, settings):
    values = {}
    statistics = {}
    pages = []
    size = settings['rows_per_page']
    selected = settings['views']

    def add(view, title, rows, **extra):
        if not rows:
            return
        for row in rows:
            values[row['entry_id']] = row
            statistics[row['entry_id']] = row
        pages.append({'view': view, 'title': title, 'claim': CLAIMS[view], 'entry_ids': [row['entry_id'] for row in rows], 'footnote': extra.pop('footnote', 'Saved original results only. Inconclusive is not zero; non-significance does not prove absence or recovery.'), **extra})
    if 'coverage' in selected:
        add('coverage', 'Intervention questions and saved coverage', data['coverage'])
    original = effects(data)
    comparisons = {}
    for row in original:
        comparisons.setdefault((row['baseline'], row['target_window']), []).append(row)
    for (baseline, target), rows in sorted(comparisons.items()):
        cell_ids = sorted({row['cell_id'] for row in rows}, key=lambda key: next((row['cell_label'] for row in rows if row['cell_id'] == key)))
        metrics = list(dict.fromkeys((row['measurement'] for row in rows)))
        label = baseline + ' to ' + target
        if 'status_matrix' in selected:
            for start in range(0, len(cell_ids), size):
                ids = cell_ids[start:start + size]
                for begin in range(0, len(metrics), settings['metrics_per_page']):
                    columns = metrics[begin:begin + settings['metrics_per_page']]
                    members = [row for row in rows if row['cell_id'] in ids and row['measurement'] in columns]
                    add('status_matrix', 'Change evidence | ' + label, members, cell_ids=ids, measurements=columns, cell_labels=[next((row['cell_label'] for row in rows if row['cell_id'] == key)) for key in ids], footnote='Colour shows the saved corrected model-evidence category, not effect magnitude. Each original cell/measurement remains present; inconclusive includes unavailable tests.')
        for metric in metrics:
            members = [row for row in rows if row['measurement'] == metric]
            for start in range(0, len(members), size):
                chunk = members[start:start + size]
                for view in ['before_after', 'original_changes', 'model_effects']:
                    if view in selected:
                        add(view, metric + ' | ' + label, chunk, measurement=metric, unit=chunk[0]['unit'], baseline=baseline, target_window=target, footnote='Each row is one original cell. Values are saved original-window summaries; no new inference.' if view == 'before_after' else 'Observed change = chosen follow-up summary minus chosen baseline summary. Colour retains saved within-cell evidence; these values are not substituted for the fitted model contrast.' if view == 'original_changes' else 'Pointwise conditional-model intervals and corrected probabilities are saved results. Repeated temporal observations are not biological sample replicates; a missing model estimate remains unavailable.')
    if 'controls' in selected and data['controls'] is not None:
        rows = [display.entry(**row, kind='control_comparison') for row in _json_value(data['controls']['comparisons'].to_dict('records'))]
        groups = {}
        for row in rows:
            groups.setdefault((row['measurement'], row['quantity']), []).append(row)
        for (measurement, quantity), members in sorted(groups.items()):
            for start in range(0, len(members), size):
                add('controls', 'Treatment/control changes | ' + measurement, members[start:start + size], measurement=measurement, quantity=quantity, footnote='Effect = target condition minus reference condition at the original biological-sample level. Original matching and cell/sample counts are retained. Intervals are pointwise; no causal conclusion follows from this contrast alone.')
    if not pages:
        add('coverage', 'No requested display has saved numerical results', data['coverage'])
    return (pd.DataFrame(values.values()), pd.DataFrame(statistics.values()), pages)

def version():
    from pymicroglia.pipelines._versions import rendering
    return rendering(__file__)

def build(ctx):
    from pymicroglia.visualisation.panels import intervention_overview
    return display.build(ctx, intervention_overview)

def produce(context):
    from pymicroglia.visualisation.panels import intervention_overview
    settings, text = options(context.presentation)
    data = collect(context)
    values, statistics, planned = pages(data, settings)
    return display.produce(context, values=values, statistics=statistics, metadata={'settings': settings, 'scientific_population': 'Every original requested cell/measurement comparison before any response selection', 'windows_id': context.saved('aligned-windows').outcome.scientific_id if 'aligned-windows' in context.dependencies else None, 'evidence_id': context.saved('response-evidence').outcome.scientific_id if 'response-evidence' in context.dependencies else None}, pages=planned, slugs=[SLUG] * len(planned), panel=intervention_overview, sources=[__file__], text=text)
