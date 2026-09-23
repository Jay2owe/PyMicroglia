"""Display every saved lag hypothesis and its original physical search grid."""
from pymicroglia._results import output_files
from pymicroglia._results import read_document
from pymicroglia._sources import source_file
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pymicroglia.pipelines._contracts import ArtifactRef, Settings, StepResult, content_id
from pymicroglia.pipelines.relationships.inputs import PAIR_KEYS
from pymicroglia.pipelines._screening import _json_value, _write_json, file_hash, read_table, write_table
ALIASES = {'lag_results.json': ('lag-association', 'results'), 'lag_profiles.json': ('lag-association', 'profiles')}
SLUG = 'measurement-lag-profiles'
GRAMMAR = 'small-multiples'
CLAIM = 'Full saved lag profiles separate association evidence, compatible delay candidates and observation support.'
DEFAULTS = {'cells_per_page': 2, 'evidence_page': 1, 'display_lag_range_hours': None}

def options(presentation):
    declared = presentation.as_dict().get('relationship_lag_profiles', {})
    if not isinstance(declared, dict) or set(declared) - {'cells_per_page', 'display_lag_range_hours', 'text'}:
        raise ValueError('relationship_lag_profiles accepts cells_per_page, display_lag_range_hours and text')
    settings = {k: declared.get(k, v) for k, v in DEFAULTS.items() if k != 'evidence_page'}
    if type(settings['cells_per_page']) is not int or not 1 <= settings['cells_per_page'] <= 4:
        raise ValueError('cells_per_page must be an integer from one to four')
    bounds = settings['display_lag_range_hours']
    if bounds is not None:
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2 or any((type(x) not in (int, float) or not np.isfinite(x) for x in bounds)) or (bounds[0] >= bounds[1]):
            raise ValueError('display_lag_range_hours must contain two increasing finite hours')
        settings['display_lag_range_hours'] = list(map(float, bounds))
    return (settings, declared.get('text', {}))

def version():
    from pymicroglia.pipelines._versions import rendering
    return rendering(__file__)

def pages(data, settings):
    question = data['provenance']['question']
    bounds = settings['display_lag_range_hours']
    if bounds is not None and question['enabled'] and (bounds[0] < question['range_hours'][0] or bounds[1] > question['range_hours'][1]):
        raise ValueError('Display bounds must remain inside the saved declared lag search')
    records = data['lag_results'].sort_values(PAIR_KEYS).to_dict('records')
    size = settings['cells_per_page']
    return [{'members': records[i:i + size], 'page_number': i // size + 1, **settings} for i in range(0, len(records), size)]

def load(ctx):
    cached = ctx.cache('_relationship_lag_figure_sources')
    if cached is not None:
        return cached
    data = {name.removesuffix('.json'): ctx.table(name) for name in ALIASES}
    data['provenance'] = ctx.pipeline_metadata('lag-association')
    settings, _ = options(Settings({'relationship_lag_profiles': {k: ctx.option(k) for k in DEFAULTS if k != 'evidence_page'}}))
    data['pages'] = pages(data, settings)
    return data

def values(data, page):
    frames = []
    for index, member in enumerate(page['members']):
        profile = data['lag_profiles']
        for key in PAIR_KEYS:
            profile = profile.loc[profile[key].eq(member[key])]
        if profile.empty:
            frames.append(pd.DataFrame([{**{k: member[k] for k in PAIR_KEYS}, 'row_index': index, 'lag_hours': None, 'status': member['status'], 'reason': member['reason'], 'kind': 'unavailable'}]))
        else:
            frames.append(profile.assign(row_index=index, kind='profile'))
    return (pd.concat(frames, ignore_index=True), pd.DataFrame(page['members']).assign(row_index=range(len(page['members']))))
STANDALONE = ''

def build(ctx):
    from pymicroglia.figure_tables.prepared import PreparedPage as FigureResult, Drawing
    from pymicroglia.visualisation.text import figure_text
    from pymicroglia.visualisation.panels import relationship_lag_profiles
    data = load(ctx)
    index = ctx.option('evidence_page')
    if type(index) is not int or not 1 <= index <= len(data['pages']):
        raise ValueError('evidence_page is outside the saved lag pages')
    page = data['pages'][index - 1]
    frame, statistics = values(data, page)
    provenance = data['provenance']
    note = 'Negative lag: reference leads target. Inverse association is separate from lag direction. Full-search probabilities are unchanged by display cropping.'
    wording = figure_text(ctx.run, ctx.spec.slug, explicit=ctx.text, item=ctx.item, title='Full saved lag searches and observation support', footnote=note, claim=CLAIM)
    settings = {**page, 'question': provenance['question'], 'source_scientific_id': provenance['scientific_id'], 'title': wording.title + ('\n' + wording.subtitle if wording.subtitle else ''), 'footnote': wording.footnote + ('\n' + wording.note if wording.note else ''), 'claim': wording.claim, 'grammar': GRAMMAR}
    from pymicroglia.figure_tables import lag_display
    prepared=lag_display.prepare(frame,settings)
    drawing=Drawing(relationship_lag_profiles.draw,(prepared,settings),{})
    views={name:(Drawing(relationship_lag_profiles.draw,(prepared,settings),{'selected_view':name}),frame) for name in ('coefficients','support')}
    return FigureResult(views=views,wording=wording, drawing=drawing, heading=wording.claim, figure_data=frame, auxiliary={'statistics.csv': statistics, 'display.csv': pd.DataFrame([{'settings_json': json.dumps(_json_value(settings))}])}, producer_sources={'lag_display.py':Path(lag_display.__file__),'relationship_lag_profiles.py': Path(relationship_lag_profiles.__file__), 'relationship_lag_figures.py': Path(__file__), ctx.spec.source.name: ctx.spec.source}, readme=f'# {wording.title}\n\n{wording.claim}\n\n{note}\n\nBands are approximate simultaneous within-curve coefficient confidence bounds from the saved stationary bootstrap. They are not null bands. No pointwise null bands are provided by this backend. Delay candidates are discrete grid values; their hull is not a sub-grid interval. No cross-cell selection-adjusted coverage is claimed.\n')

def produce(context):
    from pymicroglia.visualisation.panels import relationship_lag_profiles
    from pymicroglia.pipelines._saved_figures import draw_batch
    settings, text = options(context.presentation)
    data = {name.removesuffix('.json'): read_table(context.saved(step).artifact(artifact)) for name, (step, artifact) in ALIASES.items()}
    data['provenance'] = read_document(context.saved('lag-association').artifact('provenance'))
    data['pages'] = pages(data, settings)
    completed = draw_batch(context, slug=SLUG, options=[{**settings, 'evidence_page': i} for i in range(1, len(data['pages']) + 1)], aliases=ALIASES, data=data, cache_name='_relationship_lag_figure_sources', sources=[__file__, relationship_lag_profiles.__file__], text=text, claim=CLAIM, grammar=GRAMMAR)
    entries = []
    for page, master in zip(data['pages'], completed['masters']):
        for member in page['members']:
            semantic = {**{k: member[k] for k in PAIR_KEYS}, 'source_scientific_id': context.saved('lag-association').outcome.scientific_id, 'view': 'lag-profile'}
            entries.append({**semantic, 'entry_id': content_id(semantic), 'master': master, 'status': member['status'], 'resolution_status': member['resolution_status'], 'declared_range_hours': data['provenance']['question'].get('range_hours'), 'display_range_hours': settings['display_lag_range_hours'], 'p_value': member['p_value'], 'q_value': member['q_value']})
    write_table(context.output / 'entries.csv', pd.DataFrame(entries))
    _write_json(context.output / 'relationship_lag_profiles_manifest.json', {'schema_version': 1, 'settings': settings, 'analysis_recomputed': False, 'pages': [{**p, 'master': m} for p, m in zip(data['pages'], completed['masters'])], **{k: completed[k] for k in ('check_output', 'registration_output')}})
    refs = tuple((ArtifactRef(p.name, p.name, file_hash(p), context.scientific_id) for p in sorted(output_files(context.output)) if p.is_file() and (not p.name.startswith('.'))))
    return StepResult(context.step.name, context.scientific_id, 'completed', 'Rendered every saved lag outcome with full search evidence and physical observation support', refs)
