"""Saved biological units, within-cell pairs and sample-level recurrence."""
from pymicroglia._sources import source_file
from pathlib import Path
import pandas as pd
from pymicroglia.pipelines._contracts import content_id
from pymicroglia.pipelines._screening import _json_value, file_hash
from pymicroglia.pipelines.intervention.controls import read_controls
from pymicroglia.pipelines.intervention.patterns import read_patterns
import pymicroglia.pipelines.intervention.display as display
SLUGS = {'sample_changes': 'intervention-sample-effects', 'control_effects': 'intervention-sample-effects', 'cell_pairs': 'intervention-response-patterns', 'joint_outcomes': 'intervention-response-patterns', 'sample_pairs': 'intervention-response-patterns', 'recurrence': 'intervention-response-patterns', 'coverage': 'intervention-sample-effects'}

def options(presentation):
    block = presentation.as_dict().get('sample_figures', {})
    if not isinstance(block, dict) or set(block) - {'views', 'rows_per_page', 'text'}:
        raise ValueError('Sample figures accept views, rows_per_page and text')
    chosen = {'views': list(SLUGS), 'rows_per_page': 12, **{k: v for k, v in block.items() if k != 'text'}}
    if not isinstance(chosen['views'], list) or not chosen['views'] or len(set(chosen['views'])) != len(chosen['views']) or set(chosen['views']) - set(SLUGS):
        raise ValueError('Unknown or repeated sample/pattern view')
    n = chosen['rows_per_page']
    if isinstance(n, bool) or not isinstance(n, int) or (not 1 <= n <= 24):
        raise ValueError('rows_per_page must be an integer from 1 to 24')
    return (chosen, block.get('text', {}))

def collect(context):
    available = lambda step: step in context.dependencies and context.saved(step).outcome.status in {'completed', 'reused'}
    wid = context.saved('aligned-windows').outcome.scientific_id if available('aligned-windows') else None
    data = {'controls': read_controls(context.saved('control-comparisons')) if available('control-comparisons') else None, 'patterns': read_patterns(context.saved('coordinated-responses')) if available('coordinated-responses') else None, 'coverage': []}
    for step, key, enabled in [('control-comparisons', 'controls', context.request.request.controls['enabled']), ('coordinated-responses', 'patterns', context.request.request.coordinated['enabled'])]:
        source = context.dependencies.get(step)
        original = data[key]
        if original and wid and (original['provenance']['windows_id'] != wid):
            raise ValueError('Sample display uses another original window population')
        data['coverage'].append(display.entry(kind='coverage', source_step=step, label='Treatment/control sample changes' if key == 'controls' else 'Paired measurement changes', status='disabled' if not enabled else 'available' if available(step) else source.outcome.status if source else 'unavailable', reason='Explicitly disabled' if not enabled else source.outcome.reason if source else 'No saved result', requested_cells=len(context.request.inputs.cells), saved_unit_rows=len(original['unit_inventory']) if key == 'controls' and original else None, saved_cell_pairs=len(original['cell_pairs']) if key == 'patterns' and original else None, saved_sample_pairs=len(original['sample_pairs']) if key == 'patterns' and original else None))
    if data['controls'] and data['patterns'] and (data['controls']['provenance']['evidence_id'] != data['patterns']['provenance']['evidence_id']):
        raise ValueError('Sample and paired displays refer to different original change evidence')
    return data

def pages(data, settings):
    values = {}
    statistics = {}
    planned = []
    n = settings['rows_per_page']
    views = settings['views']

    def rows(frame, kind, definitions=None):
        result = []
        for row in _json_value(frame.to_dict('records')):
            row = {**(definitions or {}).get(row.get('definition_id'), {}), **row, 'kind': kind}
            if all((key in row for key in ['source_run', 'movie', 'identity'])):
                row.update(cell_id=content_id({k: row[k] for k in ['source_run', 'movie', 'identity']}), cell_label=row['movie'] + ' / cell ' + str(row['identity']))
            result.append(display.entry(**row))
        return result

    def add(view, members, title, footnote, *, related=(), **extra):
        for row in [*members, *related]:
            values[row['entry_id']] = row
            statistics[row['entry_id']] = row
        planned.append({'view': view, 'title': title, 'footnote': footnote, 'claim': footnote, 'entry_ids': [r['entry_id'] for r in [*members, *related]], 'plotted_entry_ids': [r['entry_id'] for r in members], **extra})
    if 'coverage' in views:
        add('coverage', data['coverage'], 'Biological samples and paired changes | coverage', 'Saved sample and measurement-pair analyses retain original populations and distinguish disabled, unavailable and evaluated questions.')
    if data['controls'] is not None:
        source = data['controls']
        definitions = {r['definition_id']: r for r in _json_value(source['definitions'].to_dict('records'))}
        grouped = {}
        for row in rows(source['unit_summaries'], 'sample_change', definitions):
            grouped.setdefault(row['definition_id'], []).append(row)
        if 'sample_changes' in views:
            for key, members in grouped.items():
                definition = definitions[key]
                for start in range(0, len(members), n):
                    add('sample_changes', members[start:start + n], 'Original sample changes | ' + definition['measurement'] + ' | ' + definition['baseline'] + ' to ' + definition['target_window'], 'Each row is an original confirmed biological sample or an explicitly unconfirmed recording. Saved changes use equal available recording summaries within samples. Cells are counted separately, no responder filter is applied, and unavailable values remain missing.', definition=definition, total_requested=len(members), unit=definition['unit'])
        if 'control_effects' in views:
            grouped = {}
            for row in rows(source['comparisons'], 'control_comparison'):
                grouped.setdefault((row['measurement'], row['quantity']), []).append(row)
            for (metric, quantity), members in grouped.items():
                for start in range(0, len(members), n):
                    add('control_effects', members[start:start + n], 'Treatment/control contrasts | ' + metric, 'Saved contrasts compare original biological samples using the declared independent or matched design. Effects, pointwise intervals and full-family evidence remain separate from cell counts; before/after association alone does not establish causality.', measurement=metric, quantity=quantity, total_requested=len(members), matching=_json_value(source['matches'].to_dict('records')))
    if data['patterns'] is not None:
        source = data['patterns']
        definitions = {r['definition_id']: r for r in _json_value(source['definitions'].to_dict('records'))}
        associations = {r['definition_id']: r for r in rows(source['associations'], 'association')}
        families = {r['family_id']: r for r in _json_value(source['families'].to_dict('records'))}
        for key, association in list(associations.items()):
            family = families.get(association.get('family_id'), {})
            associations[key] = display.entry(**{**{k: v for k, v in association.items() if k != 'entry_id'}, 'correction': family.get('correction'), 'alpha': family.get('alpha'), 'family_requested': family.get('requested'), 'family_tested': family.get('tested')})
        for table, kind, available_views in [('cell_pairs', 'cell_pair', ['cell_pairs', 'joint_outcomes']), ('sample_pairs', 'sample_pair', ['sample_pairs', 'recurrence'])]:
            grouped = {}
            for row in rows(source[table], kind, definitions):
                grouped.setdefault(row['definition_id'], []).append(row)
            for key, members in grouped.items():
                definition = definitions[key]
                label = definition['reference'] + ' / ' + definition['target'] + ' | ' + str(definition['condition'] or 'No declared condition')
                for view in available_views:
                    if view not in views:
                        continue
                    for start in range(0, len(members), n):
                        add(view, members[start:start + n], {'cell_pairs': 'Paired original-cell changes', 'joint_outcomes': 'Original joint response outcomes', 'sample_pairs': 'Paired biological-sample changes', 'recurrence': 'Saved joint-response recurrence'}[view] + ' | ' + label, 'Original paired values and joint evidence categories are saved results. Inconclusive outcomes remain distinct from two absent detections; neither establishes coordination. Cell-level displays are descriptive; any native association test uses the complete original eligible biological samples.' if view in {'cell_pairs', 'joint_outcomes'} else 'Each pair uses the same original cells and windows, then equal available recording summaries within each biological sample. The saved association uses all original eligible samples; cells and repeated recordings are not independent sample replicates.' if view == 'sample_pairs' else 'Recurrence is the saved number of cells with both supported changes divided by the explicitly jointly tested cells in each original sample. A missing denominator remains unavailable. These fractions are descriptive and do not constitute an additional coordination test.', related=[associations[key]] if view == 'sample_pairs' and key in associations else [], definition=definition, total_requested=len(members), page_start=start + 1)
    if not planned:
        add('coverage', data['coverage'], 'Requested sample/pattern displays have no saved numerical results', 'All requested questions retain their disabled or unavailable outcomes; no sample effect or paired value is invented.')
    return (pd.DataFrame(values.values()), pd.DataFrame(statistics.values()), planned)

def version():
    from pymicroglia.pipelines._versions import rendering
    return rendering(__file__)

def build(ctx):
    from pymicroglia.visualisation.panels import intervention_samples
    return display.build(ctx, intervention_samples)

def produce(context):
    from pymicroglia.visualisation.panels import intervention_samples
    settings, text = options(context.presentation)
    data = collect(context)
    values, statistics, planned = pages(data, settings)
    return display.produce(context, values=values, statistics=statistics, metadata={'settings': settings, 'population': 'Complete original biological units and paired cell changes, before any response display selection', 'coverage': data['coverage']}, pages=planned, slugs=[SLUGS[page['view']] for page in planned], panel=intervention_samples, sources=[__file__], text=text)
