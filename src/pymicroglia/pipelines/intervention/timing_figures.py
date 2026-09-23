"""Saved sampling bounds and independent window/component rhythm evidence."""
from pymicroglia._sources import source_file
from pathlib import Path
import pandas as pd
from pymicroglia.pipelines._contracts import content_id
from pymicroglia.pipelines._screening import _json_value, file_hash
from pymicroglia.pipelines.intervention.timing import read_timing
from pymicroglia.pipelines.intervention.rhythms import read_rhythms
import pymicroglia.pipelines.intervention.display as display
SLUGS = {'response_delay': 'intervention-response-timing', 'recovery': 'intervention-response-timing', 'window_period': 'intervention-rhythm-changes', 'window_amplitude': 'intervention-rhythm-changes', 'direct_change': 'intervention-rhythm-changes', 'coverage': 'intervention-response-timing'}

def options(presentation):
    block = presentation.as_dict().get('timing_figures', {})
    if not isinstance(block, dict) or set(block) - {'views', 'rows_per_page', 'text'}:
        raise ValueError('Timing figures accept views, rows_per_page and text')
    chosen = {'views': list(SLUGS), 'rows_per_page': 12, **{k: v for k, v in block.items() if k != 'text'}}
    if not isinstance(chosen['views'], list) or not chosen['views'] or len(set(chosen['views'])) != len(chosen['views']) or set(chosen['views']) - set(SLUGS):
        raise ValueError('Unknown or repeated timing/rhythm view')
    n = chosen['rows_per_page']
    if isinstance(n, bool) or not isinstance(n, int) or (not 1 <= n <= 24):
        raise ValueError('rows_per_page must be an integer from 1 to 24')
    return (chosen, block.get('text', {}))

def collect(context):
    available = lambda step: step in context.dependencies and context.saved(step).outcome.status in {'completed', 'reused'}
    wid = context.saved('aligned-windows').outcome.scientific_id if available('aligned-windows') else None
    data = {'timing': read_timing(context.saved('response-timing')) if available('response-timing') else None, 'rhythms': read_rhythms(context.saved('rhythm-changes'), wid) if available('rhythm-changes') else None, 'coverage': []}
    if data['timing'] and wid and (data['timing']['provenance']['windows_id'] != wid):
        raise ValueError('Timing figure uses a different original window source')
    for step, key, enabled in [('response-timing', 'timing', context.request.request.timing['enabled']), ('rhythm-changes', 'rhythms', context.request.request.rhythms['enabled'])]:
        source = context.dependencies.get(step)
        original = data[key]
        data['coverage'].append(display.entry(kind='coverage', source_step=step, label='Observed response and recovery' if key == 'timing' else 'Independent window rhythms', status='disabled' if not enabled else 'available' if available(step) else source.outcome.status if source else 'unavailable', reason='Explicitly disabled' if not enabled else source.outcome.reason if source else 'No saved result', original_cells=len(context.request.inputs.cells), window_rows=len(original['window_results']) if key == 'rhythms' and original else None, comparison_rows=len(original['comparisons']) if key == 'rhythms' and original else len(original['timing']) if original else None, direct_rows=len(original['direct_comparisons']) if key == 'rhythms' and original else None))
    return data

def pages(data, settings):
    values = {}
    statistics = {}
    planned = []
    n = settings['rows_per_page']
    views = settings['views']

    def add(view, rows, title, footnote, **extra):
        for row in rows:
            values[row['entry_id']] = row
            statistics[row['entry_id']] = row
        planned.append({'view': view, 'title': title, 'footnote': footnote, 'claim': footnote, 'entry_ids': [r['entry_id'] for r in rows], **extra})

    def original(frame, kind):
        return [display.entry(**{**row, 'kind': kind, 'cell_id': content_id({k: row[k] for k in ['source_run', 'movie', 'identity']}), 'cell_label': row['movie'] + ' / cell ' + str(row['identity'])}) for row in _json_value(frame.to_dict('records'))]
    if 'coverage' in views:
        add('coverage', data['coverage'], 'Timing and rhythm questions | saved coverage', 'Disabled, unavailable and evaluated questions remain separate. All original requested cells are retained; observed timing is distinct from statistical response evidence.')
    if data['timing'] is not None:
        groups = {}
        for row in original(data['timing']['timing'], 'timing'):
            groups.setdefault((row['measurement'], row['baseline'], row['target_window']), []).append(row)
        for (metric, baseline, target), rows in groups.items():
            for begin in range(0, len(rows), n):
                chunk = rows[begin:begin + n]
                for view in ['response_delay', 'recovery']:
                    if view in views:
                        add(view, chunk, ('First observed response' if view == 'response_delay' else 'Observed sustained return') + ' | ' + metric + ' | ' + baseline + ' to ' + target, 'Dots are saved qualifying observed events; bars are between-observation sampling brackets, not confidence intervals. Grey ticks show last observations when an event was not observed. An unobserved return is not placed at zero or at the endpoint. Control anchors are not treatment administrations.', measurement=metric, baseline=baseline, target_window=target, total_requested=len(rows), timing_settings=data['timing']['provenance']['applied_settings'])
    if data['rhythms'] is not None:
        native = data['rhythms']
        details = {r['window_id']: r for r in _json_value(native['window_details'].to_dict('records'))}
        groups = {}
        frame = native['window_results'].copy()
        frame['applied_recipe'] = [details[key]['applied_recipe'] for key in frame.window_id]
        for row in original(frame, 'rhythm_window'):
            groups.setdefault(row['measurement'], []).append(row)
        for metric, rows in groups.items():
            for begin in range(0, len(rows), n):
                for view in ['window_period', 'window_amplitude']:
                    if view in views:
                        add(view, rows[begin:begin + n], ('Window period estimates' if view == 'window_period' else 'Window fitted amplitudes') + ' | ' + metric, 'Each row is an independently analysed original window. Its reported estimator value, period sufficiency and corrected rhythm test remain separate. A fitted amplitude is not general rhythm strength. Before/after significance labels do not establish gain, loss or parameter change.', measurement=metric, total_requested=len(rows))
        groups = {}
        for row in original(native['direct_comparisons'], 'rhythm_change'):
            groups.setdefault((row['measurement'], row['property'], row['baseline'], row['target_window']), []).append(row)
        for (metric, prop, baseline, target), rows in groups.items():
            if 'direct_change' not in views:
                continue
            for begin in range(0, len(rows), n):
                add('direct_change', rows[begin:begin + n], 'Direct ' + prop + ' comparison | ' + metric + ' | ' + baseline + ' to ' + target, 'Saved native component contrasts compare follow-up minus baseline using their original correspondence and conditional-model uncertainty. Phase is reported only when the native period-compatibility and shared physical-reference checks allow it. All untestable comparisons remain in their declared family.', measurement=metric, property=prop, baseline=baseline, target_window=target, total_requested=len(rows), direct_settings=native['provenance']['applied_policy']['direct_change'])
    if not planned:
        add('coverage', data['coverage'], 'Requested timing/rhythm displays have no saved numerical results', 'Requested original questions remain visible with their disabled or unavailable reasons; no event or rhythm value is invented.')
    return (pd.DataFrame(values.values()), pd.DataFrame(statistics.values()), planned)

def version():
    from pymicroglia.pipelines._versions import rendering
    return rendering(__file__)

def build(ctx):
    from pymicroglia.visualisation.panels import intervention_timing
    return display.build(ctx, intervention_timing)

def produce(context):
    from pymicroglia.visualisation.panels import intervention_timing
    settings, text = options(context.presentation)
    data = collect(context)
    values, statistics, planned = pages(data, settings)
    return display.produce(context, values=values, statistics=statistics, metadata={'settings': settings, 'population': 'All original requested timing comparisons and rhythm windows; no responder filter', 'coverage': data['coverage']}, pages=planned, slugs=[SLUGS[page['view']] for page in planned], panel=intervention_timing, sources=[__file__], text=text)
