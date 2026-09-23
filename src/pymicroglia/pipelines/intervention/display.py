"""Frozen, source-qualified display receipts shared by intervention figures."""
from pymicroglia._results import output_files
from dataclasses import replace
from pathlib import Path
import json
import pandas as pd
from pymicroglia.pipelines._contracts import ArtifactRef, StepResult, content_id, result_to_dict
from pymicroglia.pipelines._screening import _json_value, _write_json, file_hash, read_table, write_table
ALIASES = {'intervention_display_values.json': ('intervention-display', 'values'), 'intervention_display_statistics.json': ('intervention-display', 'statistics')}
DEFAULTS = {'evidence_page': 1}
GRAMMAR = 'small-multiples'

def entry(**record):
    record = _json_value(record)
    return {**record, 'entry_id': content_id(record)}

def snapshot(context, values, statistics, metadata):
    from pymicroglia.pipelines._runner import SavedResult
    context.output.mkdir(parents=True)
    metadata = {**metadata, 'schema_version': 1, 'scientific_id': context.scientific_id, 'snapshot_kind': 'saved_display_inputs', 'analysis_recomputed': False, 'source_outcomes': {name: result_to_dict(saved.outcome) for name, saved in context.dependencies.items()}, 'source_artifacts': [{'step': name, **ref.as_dict()} for name, saved in context.dependencies.items() for ref in saved.outcome.artifacts]}
    refs = []
    for name, frame in [('values', values), ('statistics', statistics)]:
        path = context.output / ('intervention_display_' + name + '.json')
        path = write_table(path, frame)
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id, columns=tuple(frame.columns)))
    path = context.output / 'intervention_display_provenance.json'
    _write_json(path, metadata)
    refs.append(ArtifactRef('provenance', path.name, file_hash(path), context.scientific_id))
    outcome = StepResult('intervention-display', context.scientific_id, 'completed', 'Frozen display inputs preserve the actual scientific outcomes, including unavailable analysis', tuple(refs))
    _write_json(context.output / 'result.json', result_to_dict(outcome))
    return (SavedResult(context.output, outcome), metadata)

def load(ctx):
    cached = ctx.cache('_intervention_display')
    if cached is not None:
        return cached
    return {'values': ctx.table('intervention_display_values.json'), 'statistics': ctx.table('intervention_display_statistics.json'), 'metadata': ctx.pipeline_metadata('intervention-display')}

def standalone(panel):
    return '"""Replay frozen intervention values without scientific analysis imports."""\nimport sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport argparse,json\nimport pandas as pd\nfrom src_' + panel + " import draw\nBUNDLE=Path(__file__).resolve().parent\nsys.path.insert(0,str(Path.home()/'.claude/skills/plot-that/scripts'))\nfrom plot_style import save\nparser=argparse.ArgumentParser();parser.add_argument('--slug');args=parser.parse_args()\nfor item in pd.read_csv(BUNDLE/'figures.csv').to_dict('records'):\n    slug=Path(item['figure']).stem\n    if args.slug and slug!=args.slug:continue\n    settings=json.loads(pd.read_csv(BUNDLE/('der_display_'+slug+'.csv')).iloc[0].settings_json)\n    values=pd.read_csv(BUNDLE/('figure_data_'+slug+'.csv'))\n    statistics=pd.read_csv(BUNDLE/('statistics_'+slug+'.csv'))\n    figure,axes=draw(values,statistics,settings)\n    save(figure,BUNDLE/(slug+'.svg'),claim=settings['claim'],grammar=settings['grammar'],producer='plot.py',statistics_status='complete')\n"

def build(ctx, panel):
    from pymicroglia.figure_tables.prepared import PreparedPage as FigureResult, Drawing
    from pymicroglia.visualisation.text import figure_text
    data = load(ctx)
    pages = data['metadata']['pages']
    index = ctx.option('evidence_page')
    if isinstance(index, bool) or not isinstance(index, int) or (not 1 <= index <= len(pages)):
        raise ValueError('evidence_page is outside the saved intervention display pages')
    page = pages[index - 1]
    values = data['values'].loc[data['values'].entry_id.isin(page['entry_ids'])].copy()
    statistics = data['statistics'].loc[data['statistics'].entry_id.isin(page['entry_ids'])].copy()
    if statistics.empty:
        statistics = pd.DataFrame([{'entry_id': page['entry_ids'][0], 'status': 'display_only', 'reason': 'Saved geometry and display coverage; no additional test', 'effect': None, 'p_value': None, 'q_value': None}])
    wording = figure_text(ctx.run, ctx.spec.slug, explicit=ctx.text, item=ctx.item, title=page['title'], footnote=page['footnote'], claim=page['claim'])
    settings = {**{key: value for key, value in page.items() if key not in {'entry_ids', 'effect_entries'}}, 'metadata': {key: value for key, value in data['metadata'].items() if key not in {'pages', 'source_artifacts', 'source_outcomes'}}, 'title': wording.title + ('\n' + wording.subtitle if wording.subtitle else ''), 'footnote': wording.footnote + ('\n' + wording.note if wording.note else ''), 'claim': wording.claim, 'grammar': GRAMMAR}
    from pymicroglia.figure_tables import page_views
    if panel.__name__.endswith(('.intervention_timing','.intervention_samples','.intervention_overview')):
        from pymicroglia.figure_tables import intervention_display
        prepared=intervention_display.prepare(values,settings,panel.__name__.rsplit('.',1)[-1])
        drawing=Drawing(panel.draw,(prepared,settings),{})
    elif panel.__name__.endswith('.intervention_reports'):
        from pymicroglia.figure_tables import intervention_report_display
        prepared=intervention_report_display.prepare(values,settings)
        drawing=Drawing(panel.draw,(prepared,settings),{})
    else:
        drawing = Drawing(panel.draw, (values, statistics, settings), {})
    views=page_views.views(ctx.spec.slug.replace('-','_'),page['view'],drawing,values)
    if panel.__name__.endswith('.intervention_reports'):
        names=('original','anchored','evidence') if settings['view']=='cell_report' else ('original','anchored')
        views={name:(Drawing(panel.draw,(prepared,settings),{'selected_view':name}),values[values.kind.ne('trace') if name=='evidence' else values.kind.isin(['trace','window'])].copy()) for name in names}
    sources = {Path(panel.__file__).name: Path(panel.__file__), 'intervention_display.py': Path(__file__), ctx.spec.source.name: ctx.spec.source}
    if 'intervention_report_display' in locals():sources['intervention_report_display.py']=Path(intervention_report_display.__file__)
    sources['page_views.py']=Path(page_views.__file__)
    if 'intervention_display' in locals():sources['intervention_display_geometry.py']=Path(intervention_display.__file__)
    for source in getattr(panel, 'SOURCES', []):
        sources[Path(source).name] = Path(source)
    return FigureResult(views=views,wording=wording, drawing=drawing, heading=wording.claim, figure_data=values, auxiliary={'statistics.csv': statistics, 'display.csv': pd.DataFrame([{'settings_json': json.dumps(_json_value(settings))}])}, producer_sources=sources, readme=f'# {wording.title}\n\n{wording.claim}\n\n{wording.footnote}\n\nFrozen original results; rendering does not estimate, test, correct, adjust, select scientific support or infer a shared rhythm.\n')

def produce(context, *, values, statistics, metadata, pages, slugs, panel, sources, text=None):
    from pymicroglia.pipelines._saved_figures import draw_batch
    if values.entry_id.duplicated().any():
        raise ValueError('Display rows repeat a semantic identity')
    original_sources = {Path(__file__), Path(panel.__file__), *map(Path, sources), *(saved.artifact(ref.name) for saved in context.dependencies.values() for ref in saved.outcome.artifacts)}
    original_sources.update(map(Path, getattr(panel, 'SOURCES', [])))
    saved, metadata = snapshot(context, values, statistics, {**metadata, 'pages': pages})
    completed = draw_batch(replace(context, dependencies={'intervention-display': saved}), slug=slugs[0], figure_slugs=slugs, options=[{'evidence_page': index} for index in range(1, len(pages) + 1)], aliases=ALIASES, data={'values': values, 'statistics': statistics, 'metadata': metadata}, cache_name='_intervention_display', sources=original_sources, text=text or {}, claim=pages[0]['claim'], grammar=GRAMMAR, figure_claims=[page['claim'] for page in pages])
    targets = []
    for page, master in zip(pages, completed['masters']):
        targets.extend(({**row, 'master': master, 'view': page['view']} for row in values.loc[values.entry_id.isin(page['entry_ids'])].to_dict('records')))
    write_table(context.output / 'entries.csv', pd.DataFrame(targets))
    _write_json(context.output / 'intervention_display_manifest.json', {'schema_version': 1, 'pages': [{**page, 'master': master} for page, master in zip(pages, completed['masters'])], 'settings': metadata.get('settings', {}), 'source_outcomes': metadata['source_outcomes'], 'analysis_recomputed': False, 'check_output': completed['check_output'], 'registration_output': completed['registration_output']})
    refs = tuple((ArtifactRef(path.name, path.name, file_hash(path), context.scientific_id) for path in sorted(output_files(context.output)) if path.is_file() and (not path.name.startswith('.'))))
    return StepResult(context.step.name, context.scientific_id, 'completed', 'Rendered frozen intervention evidence with exact sources and semantic targets', refs)
