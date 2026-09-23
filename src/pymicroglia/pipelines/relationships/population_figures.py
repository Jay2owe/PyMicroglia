"""Complete saved cell distributions and distinct sample/scalar descriptions."""
from pymicroglia._results import output_files
from pymicroglia._results import read_document
from pymicroglia._sources import source_file
import json
from pathlib import Path
import pandas as pd
from pymicroglia.pipelines._contracts import ArtifactRef, Settings, StepResult, content_id
from pymicroglia.pipelines.relationships.inputs import PAIR_KEYS
from pymicroglia.pipelines._screening import _json_value, _write_json, file_hash, read_table, write_table
ALIASES = {'population_members.json': ('sample-consistency', 'members'), 'population_summaries.json': ('sample-consistency', 'summaries'), 'population_units.json': ('sample-consistency', 'units'), 'scalar_pairs.json': ('between-cell-association', 'paired_scalars'), 'scalar_units.json': ('between-cell-association', 'units'), 'scalar_results.json': ('between-cell-association', 'results'), 'scalar_measurements.json': ('between-cell-association', 'cell_scalars')}
SLUG = 'measurement-relationship-populations'
GRAMMAR = 'small-multiples'
CLAIM = 'All eligible cell effects and distinct sample summaries remain visible alongside their actual contributors.'
CLAIMS = {'within_cell': 'Every eligible same-time cell effect is retained alongside the saved biological-sample summaries.', 'lag': 'Every eligible descriptive lag-search cell effect is retained alongside the saved biological-sample summaries.', 'between_cells': 'Cell-summary associations retain distinct cell descriptions and declared biological-sample units.', 'delay': 'Only supported resolved delays receive numeric marks; unresolved and incompatible outcomes remain explicit.'}
DEFAULTS = {'population_groups_per_page': 8, 'evidence_page': 1, 'population_group_order': None}

def options(presentation):
    declared = presentation.as_dict().get('relationship_populations', {})
    if not isinstance(declared, dict) or set(declared) - {'population_groups_per_page', 'population_group_order', 'text'}:
        raise ValueError('relationship_populations accepts population_groups_per_page, population_group_order and text')
    settings = {k: declared.get(k, v) for k, v in DEFAULTS.items() if k != 'evidence_page'}
    if type(settings['population_groups_per_page']) is not int or not 1 <= settings['population_groups_per_page'] <= 16:
        raise ValueError('population_groups_per_page must be an integer from one to sixteen')
    order = settings['population_group_order']
    if order is not None and (not isinstance(order, list) or any((not isinstance(x, str) for x in order)) or len(set(order)) != len(order)):
        raise ValueError('population_group_order must contain each displayed sample/recording group exactly once')
    return (settings, declared.get('text', {}))

def group_id(row):
    return 'sample:' + str(row['sample']) if row['sample_confirmed'] else 'recording:' + str(row['movie'])

def pages(data, settings):
    members = data['population_members']
    groups = sorted({group_id(row) for row in members.to_dict('records')})
    order = settings['population_group_order'] or groups
    if set(order) != set(groups):
        raise ValueError('population_group_order must contain every saved group exactly once')
    size = settings['population_groups_per_page']
    blocks = [order[i:i + size] for i in range(0, len(order), size)]
    request = data['preparation']['resolved_request']['request']
    return [{'view': view, 'pair': pair, 'pair_id': content_id(pair), 'groups': block, 'all_groups': order, 'page_number': index + 1} for pair in request['pairs'] for view in ('within_cell', 'lag', 'between_cells', 'delay') for index, block in enumerate(blocks)]

def _grouped(frame):
    result = frame.copy()
    result['display_group'] = [group_id(row) for row in result.to_dict('records')]
    return result

def values(data, page):
    pid = page['pair_id']
    question = 'lag' if page['view'] == 'delay' else page['view']
    if question == 'between_cells':
        cells = _grouped(data['scalar_pairs'].loc[lambda f: f.pair_id.eq(pid)])
        cells = cells.loc[cells.display_group.isin(page['groups'])].assign(kind='cell_scalar')
        units = data['scalar_units'].loc[lambda f: f.pair_id.eq(pid)].copy()
        units = units.loc[units.experimental_unit.eq('biological_sample')]
        units['display_group'] = 'sample:' + units['sample'].astype(str)
        units = units.loc[units.display_group.isin(page['groups'])].assign(kind='sample_scalar')
        statistics = data['scalar_results'].loc[lambda f: f.pair_id.eq(pid)].copy()
        return (pd.concat([cells, units], ignore_index=True), statistics)
    members = data['population_members'].loc[lambda f: f.pair_id.eq(pid) & f.question.eq(question)]
    members = _grouped(members)
    members = members.loc[members.display_group.isin(page['groups'])].assign(kind='cell_effect')
    summaries = data['population_summaries'].loc[lambda f: f.pair_id.eq(pid) & f.question.eq(question)].copy()
    units = summaries.loc[summaries.level.eq('biological_sample')].copy()
    units['display_group'] = 'sample:' + units.group_id.astype(str)
    units = units.loc[units.display_group.isin(page['groups'])].assign(kind='sample_effect')
    return (pd.concat([members, units], ignore_index=True), summaries)

def load(ctx):
    cached = ctx.cache('_relationship_population_sources')
    if cached is not None:
        return cached
    data = {name.removesuffix('.json'): ctx.table(name) for name in ALIASES}
    data['preparation'] = ctx.pipeline_metadata('paired-inputs')
    data['summary_provenance'] = ctx.pipeline_metadata('sample-consistency')
    data['between_provenance'] = ctx.pipeline_metadata('between-cell-association')
    settings, _ = options(Settings({'relationship_populations': {k: ctx.option(k) for k in DEFAULTS if k != 'evidence_page'}}))
    data['pages'] = pages(data, settings)
    return data

def version():
    from pymicroglia.pipelines._versions import rendering
    return rendering(__file__)
STANDALONE = ''

def build(ctx):
    from pymicroglia.figure_tables.prepared import PreparedPage as FigureResult, Drawing
    from pymicroglia.visualisation.text import figure_text
    from pymicroglia.visualisation.panels import relationship_populations
    data = load(ctx)
    index = ctx.option('evidence_page')
    if type(index) is not int or not 1 <= index <= len(data['pages']):
        raise ValueError('evidence_page is outside the complete population pages')
    page = data['pages'][index - 1]
    frame, statistics = values(data, page)
    resolved = data['preparation']['resolved_request']
    titles = {'within_cell': 'All eligible same-time cell effects', 'lag': 'All eligible descriptive lag-search cell effects', 'between_cells': 'Cell measurement summaries and independent sample units', 'delay': 'Supported delays with complete resolution accounting'}
    note = 'Every eligible cell is retained, including opposite and nonsignificant effects. Biological samples and recordings remain distinct. No new aggregation, interval or test.'
    wording = figure_text(ctx.run, ctx.spec.slug, explicit=ctx.text, item=ctx.item, title=titles[page['view']], footnote=note, claim=CLAIMS[page['view']])
    settings = {**page, 'request': resolved['request'], 'measurements': {m['column']: m for m in resolved['measurements']}, 'summary_provenance': data['summary_provenance'], 'between_provenance': data['between_provenance'], 'title': wording.title + ('\n' + wording.subtitle if wording.subtitle else ''), 'footnote': wording.footnote + ('\n' + wording.note if wording.note else ''), 'claim': wording.claim, 'grammar': GRAMMAR}
    from pymicroglia.figure_tables import population_display
    prepared=population_display.prepare(frame,statistics,settings)
    drawing=Drawing(relationship_populations.draw,(prepared,settings),{})
    views={name:(Drawing(relationship_populations.draw,(prepared,settings),{'selected_view':name}),frame[frame.kind.str.startswith('cell' if name=='cells' else 'sample')].copy()) for name in ('cells','samples')}
    return FigureResult(views=views,wording=wording, drawing=drawing, heading=wording.claim, figure_data=frame, auxiliary={'statistics.csv': statistics, 'display.csv': pd.DataFrame([{'settings_json': json.dumps(_json_value(settings))}])}, producer_sources={'population_display.py':Path(population_display.__file__),'relationship_populations.py': Path(relationship_populations.__file__), 'relationship_population_figures.py': Path(__file__), ctx.spec.source.name: ctx.spec.source}, readme=f'# {wording.title}\n\n{wording.claim}\n\n{note}\n\nSigned effects remain on linear axes. Sample sign evidence concerns the observed sample-sign frequency, not a confidence interval of the average coefficient. Compatible-delay regions are descriptions, not newly estimated confidence intervals.\n')

def produce(context):
    from pymicroglia.visualisation.panels import relationship_populations
    from pymicroglia.pipelines._saved_figures import draw_batch
    settings, text = options(context.presentation)
    data = {name.removesuffix('.json'): read_table(context.saved(step).artifact(artifact)) for name, (step, artifact) in ALIASES.items()}
    for name, step in (('preparation', 'paired-inputs'), ('summary_provenance', 'sample-consistency'), ('between_provenance', 'between-cell-association')):
        data[name] = read_document(context.saved(step).artifact('provenance'))
    data['pages'] = pages(data, settings)
    completed = draw_batch(context, slug=SLUG, options=[{**settings, 'evidence_page': i} for i in range(1, len(data['pages']) + 1)], aliases=ALIASES, data=data, cache_name='_relationship_population_sources', sources=[__file__, relationship_populations.__file__], text=text, claim=CLAIM, grammar=GRAMMAR, figure_claims=[CLAIMS[p['view']] for p in data['pages']])
    entries = []
    for page, master in zip(data['pages'], completed['masters']):
        frame, _ = values(data, page)
        for group in page['groups']:
            semantic = {'source_run': context.request.inputs.source_run, 'pair_id': page['pair_id'], **page['pair'], 'view': page['view'], 'group_id': group, 'source_scientific_id': context.saved('between-cell-association' if page['view'] == 'between_cells' else 'sample-consistency').outcome.scientific_id}
            members = frame.loc[frame.display_group.eq(group) & frame.kind.isin(['cell_effect', 'cell_scalar'])]
            entries.append({**semantic, 'entry_id': content_id(semantic), 'master': master, 'members': members[PAIR_KEYS].to_dict('records')})
    write_table(context.output / 'entries.csv', pd.DataFrame(entries))
    _write_json(context.output / 'relationship_populations_manifest.json', {'schema_version': 1, 'settings': settings, 'analysis_recomputed': False, 'pages': [{**p, 'master': m} for p, m in zip(data['pages'], completed['masters'])], **{k: completed[k] for k in ('check_output', 'registration_output')}})
    refs = tuple((ArtifactRef(p.name, p.name, file_hash(p), context.scientific_id) for p in sorted(output_files(context.output)) if p.is_file() and (not p.name.startswith('.'))))
    return StepResult(context.step.name, context.scientific_id, 'completed', 'Rendered complete cell populations and separately saved sample and scalar summaries', refs)
