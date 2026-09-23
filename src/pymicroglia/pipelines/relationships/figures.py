"""Registered overview figures displaying immutable relationship outcomes."""
from pymicroglia._results import output_files
from pymicroglia._results import read_document
from pymicroglia._sources import source_file
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pymicroglia.pipelines._contracts import ArtifactRef, Settings, StepResult, content_id
from pymicroglia.pipelines._screening import _json_value, _write_json, file_hash, read_table, write_table
ALIASES = {'rel_within.json': ('within-cell-association', 'results'), 'rel_lag.json': ('lag-association', 'results'), 'rel_between.json': ('between-cell-association', 'results'), 'rel_summaries.json': ('sample-consistency', 'summaries'), 'rel_members.json': ('sample-consistency', 'members'), 'rel_units.json': ('sample-consistency', 'units'), 'rel_inventory.json': ('paired-inputs', 'inventory'), 'rel_measurements.json': ('paired-inputs', 'trace_inventory')}
DEFAULTS = {'matrix_block_size': 4, 'summary_level': 'all_cells', 'evidence_page': 1}
SLUG = 'measurement-relationship-overview'
CLAIM = 'Saved within-cell effects, cell-summary associations and delays answer separate questions with explicit support.'
CLAIMS = {'within_cell': 'Saved summaries retain all eligible within-cell association effects and their support.', 'between_cells': 'Associations between measured cell summaries retain their declared experimental unit and evidence.', 'delay': 'Only compatible supported delays receive numeric summaries; unresolved or unavailable timing stays explicit.'}
GRAMMAR = 'matrix'

def options(presentation):
    declared = presentation.as_dict().get('measurement_relationships', {})
    if not isinstance(declared, dict) or set(declared) - {'matrix_block_size', 'summary_level', 'measurement_order', 'text'}:
        raise ValueError('Relationship overview presentation accepts matrix_block_size, summary_level, measurement_order and text')
    settings = {**DEFAULTS, **{key: value for key, value in declared.items() if key != 'text'}}
    size = settings['matrix_block_size']
    if isinstance(size, bool) or not isinstance(size, int) or (not 1 <= size <= 8):
        raise ValueError('matrix_block_size must be an integer from one to eight')
    if settings['summary_level'] not in {'all_cells', 'across_samples'}:
        raise ValueError('summary_level must be all_cells or across_samples')
    order = settings.get('measurement_order')
    if order is not None and (not isinstance(order, list) or any((not isinstance(item, str) for item in order)) or len(set(order)) != len(order)):
        raise ValueError('measurement_order must be a list of unique measured columns')
    return (settings, declared.get('text', {}))

def version():
    from pymicroglia.pipelines._versions import rendering
    return rendering(__file__)

def pages(data, settings):
    resolved = data['preparation']['resolved_request']
    columns = [measurement['column'] for measurement in resolved['measurements']]
    order = settings.get('measurement_order') or columns
    if set(order) != set(columns):
        raise ValueError('measurement_order must contain every requested measurement exactly once')
    pairs = resolved['request']['pairs']
    size = settings['matrix_block_size']
    blocks = [order[i:i + size] for i in range(0, len(order), size)]
    return [{'view': view, 'rows': rows, 'columns': cols, 'summary_level': settings['summary_level']} for view in ('within_cell', 'between_cells', 'delay') for rows in blocks for cols in blocks if any((pair['reference'] in rows and pair['target'] in cols for pair in pairs))]

def load(ctx):
    cached = ctx.cache('_measurement_relationship_sources')
    if cached is not None:
        return cached
    data = {name.removesuffix('.json'): ctx.table(name) for name in ALIASES}
    data['preparation'] = ctx.pipeline_metadata('paired-inputs')
    data['between_provenance'] = ctx.pipeline_metadata('between-cell-association')
    data['summary_provenance'] = ctx.pipeline_metadata('sample-consistency')
    choices = {key: ctx.option(key) for key in DEFAULTS if key != 'evidence_page'}
    choices['measurement_order'] = ctx.option('measurement_order')
    settings, _ = options(Settings({'measurement_relationships': choices}))
    data['pages'] = pages(data, settings)
    return data

def matrix_values(data, page):
    request = data['preparation']['resolved_request']['request']
    source_run = data['preparation']['resolved_request']['inputs']['source_run']
    inventory = data['rel_inventory']
    pairs = {(pair['reference'], pair['target']): content_id(pair) for pair in request['pairs']}
    view = page['view']
    question = 'lag' if view == 'delay' else view
    outcomes = data['rel_between'] if view == 'between_cells' else data['rel_summaries']
    if view != 'between_cells':
        outcomes = outcomes.loc[outcomes.question.eq(question) & outcomes.level.eq(page['summary_level'])]
    lookup = {row['pair_id']: row for row in outcomes.to_dict('records')}
    records = []
    for y, reference in enumerate(page['rows']):
        for x, target in enumerate(page['columns']):
            pair_id = pairs.get((reference, target))
            row = lookup.get(pair_id, {})
            cell_rows = inventory.loc[inventory.pair_id.eq(pair_id)]
            value = row.get('delay_summary_hours' if view == 'delay' else 'effect')
            status = row.get('delay_summary_status' if view == 'delay' else 'status', 'unavailable')
            reason = row.get('delay_summary_reason' if view == 'delay' else 'reason', 'Requested summary was not saved')
            if not request[question]['enabled']:
                value = None
                status = 'disabled'
                reason = 'This relationship question was not requested'
            if not pair_id:
                status = 'not-requested'
                reason = 'This ordered measurement pair was not requested'
                value = None
            samples = row.get('confirmed_samples', 0)
            sid = row.get('source_scientific_id')
            if view == 'between_cells':
                sid = data['between_provenance']['scientific_id']
            elif row:
                sid = data['summary_provenance']['scientific_id']
            interval = row.get('effect_interval') if view == 'between_cells' else None
            record = {'row_index': y, 'column_index': x, 'source_run': source_run, 'pair_id': pair_id, 'reference': reference, 'target': target, 'view': view, 'summary_level': page['summary_level'], 'requested': bool(pair_id), 'value': value, 'status': status, 'reason': reason, 'requested_cells': row.get('cells_requested' if view == 'between_cells' else 'requested_cells', len(cell_rows)), 'eligible_cells': row.get('cells_eligible' if view == 'between_cells' else 'eligible_cells', 0), 'samples': samples, 'tested_cells': row.get('tested_cells'), 'supported_cells': row.get('supported_cells'), 'delay_supported_cells': row.get('delay_supported_cells'), 'delay_unresolved_cells': row.get('delay_unresolved_cells'), 'not_detected_cells': row.get('not_detected_cells'), 'untestable_cells': row.get('untestable_cells'), 'disabled_cells': row.get('disabled_cells'), 'source_scientific_id': sid, 'effect_interval': interval, 'p_value': row.get('p_value'), 'q_value': row.get('q_value'), 'evidence_meaning': 'No population delay test; cell full-search evidence and delay resolution remain separate' if view == 'delay' else 'Association between declared independent units' if view == 'between_cells' else 'Observed sample sign frequency; not a test or interval of the coloured coefficient' if page['summary_level'] == 'across_samples' else 'Descriptive all-cell summary', 'positive_fraction': row.get('positive_fraction'), 'positive_fraction_interval': row.get('positive_fraction_interval'), 'effect_population': row.get('estimand' if view == 'between_cells' else 'effect_population'), 'members': cell_rows[['source_run', 'movie', 'identity']].to_dict('records')}
            if view == 'delay':
                record.update(p_value=None, q_value=None, positive_fraction=None, positive_fraction_interval=None)
            record['entry_id'] = content_id({key: record[key] for key in ('source_run', 'source_scientific_id', 'pair_id', 'view', 'summary_level')}) if pair_id else None
            records.append(record)
    values = pd.DataFrame(records)
    values['value'] = pd.to_numeric(values.value, errors='coerce')
    return (values, outcomes)
STANDALONE = ''

def build(ctx):
    from pymicroglia.figure_tables.prepared import PreparedPage as FigureResult, Drawing
    from pymicroglia.visualisation.text import figure_text
    from pymicroglia.visualisation.panels import measurement_relationships
    data = load(ctx)
    index = ctx.option('evidence_page')
    if isinstance(index, bool) or not isinstance(index, int) or (not 1 <= index <= len(data['pages'])):
        raise ValueError('evidence_page is outside the saved overview pages')
    page = data['pages'][index - 1]
    values, statistics = matrix_values(data, page)
    titles = {'within_cell': 'Relationships within individual cells', 'between_cells': 'Associations between cell summaries', 'delay': 'Compatible supported delays'}
    settings = dict(page)
    request = data['preparation']['resolved_request']['request']
    settings['labels'] = {m['column']: m['label'] for m in data['preparation']['resolved_request']['measurements']}
    settings['lag_colour_limit_hours'] = max((abs(v) for v in request['lag']['range_hours'])) if request['lag']['enabled'] else 1.0
    footnote = 'Colours show saved values. Unavailable is not zero. Only requested directions are shown. No fresh analysis, rhythm, phase or causal claim.'
    note = 'Cell counts show eligible / requested cells; samples are confirmed biological samples. '
    if page['view'] == 'delay':
        note += 'Negative lag means the reference leads the target. Timing counts refer to cells with supported lag associations. A compatible description is not a new delay confidence interval.'
    elif page['view'] == 'within_cell':
        note += 'Summary: ' + page['summary_level'].replace('_', ' ') + '. Sample evidence, when requested, tests sign frequency rather than the coloured mean or median.'
    else:
        note += 'Cell descriptions and inference across independently aggregated biological samples remain distinct in the saved results.'
    title = titles[page['view']]
    question = 'lag' if page['view'] == 'delay' else page['view']
    if page['view'] != 'delay' and request[question]['enabled']:
        title += ' | ' + request[question]['statistic'].capitalize()
    if page['view'] == 'between_cells' and request[question].get('experimental_unit') == 'biological_sample':
        title = 'Associations between biological-sample summaries | ' + request[question]['statistic'].capitalize()
    wording = figure_text(ctx.run, ctx.spec.slug, explicit=ctx.text, item=ctx.item, title=title, footnote=footnote, claim=CLAIMS[page['view']])
    settings.update(title=wording.title + ('\n' + wording.subtitle if wording.subtitle else ''), footnote=wording.footnote + ('\n' + wording.note if wording.note else ''), evidence_note=note, claim=wording.claim, grammar=GRAMMAR)
    from pymicroglia.figure_tables import relationship_matrix_display,page_views
    prepared=relationship_matrix_display.prepare(values,settings)
    drawing=Drawing(measurement_relationships.draw,(prepared,settings),{})
    views=page_views.views(ctx.spec.slug.replace('-','_'),page['view'],drawing,values)
    auxiliary = {'display.csv': pd.DataFrame([{'settings_json': json.dumps(_json_value(settings))}])}
    if not statistics.empty:
        auxiliary['statistics.csv'] = statistics
    return FigureResult(views=views,wording=wording, drawing=drawing, heading=wording.claim, figure_data=values, auxiliary=auxiliary, producer_sources={'relationship_matrix_display.py':Path(relationship_matrix_display.__file__),'page_views.py':Path(page_views.__file__),'measurement_relationships.py': Path(measurement_relationships.__file__), 'relationship_figures.py': Path(__file__), ctx.spec.source.name: ctx.spec.source}, readme=f'# {wording.title}\n\n{wording.claim}\n\n{note}\n\n{footnote}\n\nplot.py replays frozen values without Motion or Workbench.\n')

def produce_overview(context):
    from pymicroglia.visualisation.panels import measurement_relationships
    from pymicroglia.pipelines._saved_figures import draw_batch
    settings, text = options(context.presentation)
    data = {name.removesuffix('.json'): read_table(context.saved(step).artifact(artifact)) for name, (step, artifact) in ALIASES.items()}
    data['preparation'] = read_document(context.saved('paired-inputs').artifact('provenance'))
    data['between_provenance'] = read_document(context.saved('between-cell-association').artifact('provenance'))
    data['summary_provenance'] = read_document(context.saved('sample-consistency').artifact('provenance'))
    data['pages'] = pages(data, settings)
    completed = draw_batch(context, slug=SLUG, options=[{**settings, 'evidence_page': i} for i in range(1, len(data['pages']) + 1)], aliases=ALIASES, data=data, cache_name='_measurement_relationship_sources', sources=[__file__, measurement_relationships.__file__], text=text, claim=CLAIM, grammar=GRAMMAR, figure_claims=[CLAIMS[page['view']] for page in data['pages']])
    entries = []
    pages_out = []
    for page, master in zip(data['pages'], completed['masters']):
        values, _ = matrix_values(data, page)
        values = values.loc[values.requested]
        values['master'] = master
        entries.extend(values.to_dict('records'))
        pages_out.append({**page, 'master': master, 'entry_ids': values.entry_id.tolist()})
    write_table(context.output / 'entries.csv', pd.DataFrame(entries))
    _write_json(context.output / 'relationship_overview_manifest.json', {'schema_version': 1, 'pages': pages_out, 'settings': settings, 'source_results': {name: saved.outcome.scientific_id for name, saved in context.dependencies.items()}, 'analysis_recomputed': False, 'check_output': completed['check_output'], 'registration_output': completed['registration_output']})
    refs = tuple((ArtifactRef(path.name, path.name, file_hash(path), context.scientific_id) for path in sorted(output_files(context.output)) if path.is_file() and (not path.name.startswith('.'))))
    return StepResult(context.step.name, context.scientific_id, 'completed', 'Rendered separate saved relationship matrices with complete statuses and semantic pair entries', refs)
