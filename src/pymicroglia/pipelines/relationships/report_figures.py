"""Reports and trace grids from the frozen supported-relationship union."""
from pymicroglia._results import output_files
from pymicroglia._results import read_document
from pymicroglia._sources import source_file
import json
from pathlib import Path
import pandas as pd
from pymicroglia.pipelines._contracts import ArtifactRef, Settings, StepResult, content_id
from pymicroglia.pipelines.relationships.inputs import PAIR_KEYS, KEYS
from pymicroglia.pipelines._screening import _json_value, _write_json, file_hash, read_table, write_table
ALIASES = {'report_members.json': ('relationship-report-selection', 'report_members'), 'report_traces.json': ('paired-inputs', 'traces'), 'report_pairs.json': ('paired-inputs', 'same_time_pairs'), 'report_measurements.json': ('paired-inputs', 'trace_inventory'), 'report_within.json': ('within-cell-association', 'results'), 'report_lag.json': ('lag-association', 'results'), 'report_profiles.json': ('lag-association', 'profiles')}
SLUG = 'measurement-relationship-reports'
GRAMMAR = 'small-multiples'
CLAIM = 'Supported relationships retain their measured traces, matched observations and separate delay uncertainty.'
DEFAULTS = {'cells_per_page': 2, 'evidence_page': 1}

def options(presentation):
    declared = presentation.as_dict().get('relationship_reports', {})
    if not isinstance(declared, dict) or set(declared) - {'cells_per_page', 'reverse_cells', 'text'}:
        raise ValueError('relationship_reports accepts cells_per_page, reverse_cells and text')
    settings = {'cells_per_page': declared.get('cells_per_page', 2), 'reverse_cells': declared.get('reverse_cells', False)}
    if type(settings['cells_per_page']) is not int or not 1 <= settings['cells_per_page'] <= 4:
        raise ValueError('cells_per_page must be an integer from one to four')
    if type(settings['reverse_cells']) is not bool:
        raise ValueError('reverse_cells must be a boolean')
    return (settings, declared.get('text', {}))

def version():
    from pymicroglia.pipelines._versions import rendering
    return rendering(__file__)

def subset(frame, member, keys=PAIR_KEYS):
    for key in keys:
        frame = frame.loc[frame[key].eq(member[key])]
    return frame.copy()

def pages(data, settings):
    result = []
    members = data['report_members'].sort_values(PAIR_KEYS, ascending=not settings['reverse_cells'])
    for pair_id, group in members.groupby('pair_id', sort=False):
        records = group.to_dict('records')
        for start in range(0, len(records), settings['cells_per_page']):
            for view in ('pair', 'cells'):
                result.append({'view': view, 'pair_id': pair_id, 'members': records[start:start + settings['cells_per_page']], 'page_number': start // settings['cells_per_page'] + 1, 'selected_pair_cells': len(records)})
    return result

def load(ctx):
    cached = ctx.cache('_relationship_report_sources')
    if cached is not None:
        return cached
    data = {name.removesuffix('.json'): ctx.table(name) for name in ALIASES}
    data['preparation'] = ctx.pipeline_metadata('paired-inputs')
    settings, _ = options(Settings({'relationship_reports': {k: ctx.option(k) for k in ('cells_per_page', 'reverse_cells')}}))
    data['pages'] = pages(data, settings)
    return data

def values(data, page):
    frames, statistics = ([], [])
    for row_index, member in enumerate(page['members']):
        keys = {key: member[key] for key in PAIR_KEYS}
        within = subset(data['report_within'], member)
        lag = subset(data['report_lag'], member)
        if len(within) != 1 or len(lag) != 1:
            raise ValueError('Each selected full key must have exactly one result per question')
        for name, frame in (('within', within), ('lag', lag)):
            frame = frame.assign(kind=name, row_index=row_index)
            statistics.append(frame)
            frames.append(frame)
        if page['view'] == 'cells':
            traces = subset(data['report_traces'], member, KEYS)
            traces = traces.loc[traces.measurement.isin([member['reference'], member['target']])]
            for name, frame in (('trace', traces), ('scatter', subset(data['report_pairs'], member)), ('profile', subset(data['report_profiles'], member))):
                frames.append(frame.assign(**keys, kind=name, row_index=row_index))
    return (pd.concat(frames, ignore_index=True), pd.concat(statistics, ignore_index=True))
STANDALONE = ''

def build(ctx):
    from pymicroglia.figure_tables.prepared import PreparedPage as FigureResult, Drawing
    from pymicroglia.visualisation.text import figure_text
    from pymicroglia.visualisation.panels import relationship_reports
    data = load(ctx)
    index = ctx.option('evidence_page')
    if type(index) is not int or not 1 <= index <= len(data['pages']):
        raise ValueError('evidence_page is outside the selected report pages')
    page = data['pages'][index - 1]
    frame, statistics = values(data, page)
    resolved = data['preparation']['resolved_request']
    settings = {**page, 'measurements': {m['column']: m for m in resolved['measurements']}, 'max_gap_hours': resolved['request']['support']['max_gap_hours'], 'representation': resolved['request']['representation']}
    title = 'Selected relationship evidence' if page['view'] == 'pair' else 'Selected cell traces and paired observations'
    note = 'Selected by supported same-time OR full-search lag association. Negative lag means the reference leads. No rhythm or causal claim.'
    wording = figure_text(ctx.run, ctx.spec.slug, explicit=ctx.text, item=ctx.item, title=title, footnote=note, claim=CLAIM)
    settings.update(title=wording.title + ('\n' + wording.subtitle if wording.subtitle else ''), footnote=wording.footnote + ('\n' + wording.note if wording.note else ''), claim=wording.claim, grammar=GRAMMAR)
    from pymicroglia.figure_tables import relationship_report_display
    prepared=relationship_report_display.prepare(frame,statistics,settings)
    drawing=Drawing(relationship_reports.draw,(prepared,settings),{})
    names=('traces','scatter','lags') if settings['view']=='cells' else ('coefficients',)
    views={name:(Drawing(relationship_reports.draw,(prepared,settings),{'selected_view':name}),frame if name=='coefficients' else frame[frame.kind.eq({'traces':'trace','scatter':'scatter','lags':'profile'}[name])]) for name in names}
    return FigureResult(views=views,wording=wording, drawing=drawing, heading=wording.claim, figure_data=frame, auxiliary={'statistics.csv': statistics, 'display.csv': pd.DataFrame([{'settings_json': json.dumps(_json_value(settings))}])}, producer_sources={'relationship_report_display.py':Path(relationship_report_display.__file__),'relationship_reports.py': Path(relationship_reports.__file__), 'relationship_report_figures.py': Path(__file__), ctx.spec.source.name: ctx.spec.source}, readme=f'# {wording.title}\n\n{wording.claim}\n\n{note}\n\nOriginal clocks and matched observation identities are frozen. No new joins, fits, tests or delay searches.\n')

def produce(context):
    from pymicroglia.visualisation.panels import relationship_reports
    from pymicroglia.pipelines._saved_figures import draw_batch
    settings, text = options(context.presentation)
    data = {name.removesuffix('.json'): read_table(context.saved(step).artifact(artifact)) for name, (step, artifact) in ALIASES.items()}
    data['preparation'] = read_document(context.saved('paired-inputs').artifact('provenance'))
    data['pages'] = pages(data, settings)
    completed = draw_batch(context, slug=SLUG, options=[{**settings, 'evidence_page': i} for i in range(1, len(data['pages']) + 1)], aliases=ALIASES, data=data, cache_name='_relationship_report_sources', sources=[__file__, relationship_reports.__file__], text=text, claim=CLAIM, grammar=GRAMMAR)
    entries = []
    for page, master in zip(data['pages'], completed['masters']):
        for member in page['members']:
            key = {name: member[name] for name in PAIR_KEYS}
            semantic = {**key, 'view': page['view'], 'selection_id': context.saved('relationship-report-selection').outcome.scientific_id}
            entries.append({**semantic, 'entry_id': content_id(semantic), 'master': master, 'pair_target_id': content_id({'source_run': key['source_run'], 'pair_id': key['pair_id'], 'view': 'pair'}), 'supporting_questions': member['supporting_questions']})
    write_table(context.output / 'entries.csv', pd.DataFrame(entries))
    _write_json(context.output / 'relationship_reports_manifest.json', {'schema_version': 1, 'pages': [{**page, 'master': master} for page, master in zip(data['pages'], completed['masters'])], 'settings': settings, 'analysis_recomputed': False, **{key: completed[key] for key in ('check_output', 'registration_output')}})
    refs = tuple((ArtifactRef(p.name, p.name, file_hash(p), context.scientific_id) for p in sorted(output_files(context.output)) if p.is_file() and (not p.name.startswith('.'))))
    return StepResult(context.step.name, context.scientific_id, 'completed', 'Rendered every selected full cell/pair key with frozen observations and evidence', refs)
