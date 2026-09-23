"""Render saved state time statistics and independent-sample comparisons."""
from pymicroglia._results import output_files
from pymicroglia._sources import source_file
from pathlib import Path
import json
import pandas as pd
from pymicroglia.pipelines._contracts import ArtifactRef, StepResult, content_id
from pymicroglia.pipelines.relationships.inputs import KEYS
from pymicroglia.pipelines._screening import _json_value, _write_json, file_hash, read_table, write_table
SLUGS = {'cells': 'cell-state-time-summaries', 'samples': 'state-sample-comparisons'}
GRAMMAR = 'small-multiples'
CLAIMS = {'cells': 'Saved cell occupancy, interval-specific transitions and censored observed bouts retain their original values and denominators.', 'samples': 'Sample summaries and declared condition contrasts preserve experimental-unit membership, model-choice exclusions and saved uncertainty.'}
ALIASES = {**{'summary_' + name + '.json': ('durations-and-switches', name) for name in ['cell_statistics', 'occupancy', 'transitions', 'bouts']}, **{'summary_' + name + '.json': ('state-sample-comparisons', name) for name in ['unit_inventory', 'unit_metrics', 'cell_metrics', 'comparisons']}, 'summary_states.json': ('state-assignments', 'state_definitions')}
DEFAULTS = {'state_summary_cells_per_page': 20, 'state_summary_columns_per_page': 6, 'state_summary_views': ['occupancy', 'switch_rates', 'transitions', 'bouts', 'samples', 'contrasts'], 'state_sample_questions_per_page': 3, 'state_contrasts_per_page': 10, 'state_display_order': None, 'state_display_names': {}, 'evidence_page': 1}

def version():
    from pymicroglia.pipelines._versions import rendering
    return rendering(__file__)

def options(presentation):
    declared = presentation.as_dict().get('state_summaries', {})
    if not isinstance(declared, dict) or set(declared) - (set(DEFAULTS) - {'evidence_page'} | {'text'}):
        raise ValueError('State summary presentation accepts saved views, page sizes, state labels/order and text')
    settings = {**DEFAULTS, **{name: value for name, value in declared.items() if name != 'text'}}
    for name, maximum in [('state_summary_cells_per_page', 32), ('state_summary_columns_per_page', 8), ('state_sample_questions_per_page', 6), ('state_contrasts_per_page', 16)]:
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, int) or (not 1 <= value <= maximum):
            raise ValueError(name + ' must be an integer from one to ' + str(maximum))
    views = settings['state_summary_views']
    if not isinstance(views, list) or not views or any((view not in DEFAULTS['state_summary_views'] for view in views)) or (len(views) != len(set(views))):
        raise ValueError('state_summary_views must list unique saved occupancy, switch_rates, transitions, bouts, samples or contrasts views')
    order = settings['state_display_order']
    if order is not None and (not isinstance(order, list) or any((not isinstance(value, str) for value in order)) or len(set(order)) != len(order)):
        raise ValueError('state_display_order must list unique accepted state IDs')
    labels = settings['state_display_names']
    if not isinstance(labels, dict) or any((not isinstance(key, str) or not isinstance(value, str) or (not value.strip()) for key, value in labels.items())):
        raise ValueError('state_display_names must map accepted states to nonempty labels')
    return (settings, declared.get('text', {}))

def chunks(rows, size):
    return [rows[start:start + size] for start in range(0, len(rows), size)]

def pages(data, settings):
    cells, states = (data['cell_statistics'], data['states'])
    if cells.duplicated(KEYS).any():
        raise ValueError('State summaries repeat a full cell identity')
    model_id = data['time_provenance']['model_id']
    if data['sample_provenance']['model_id'] != model_id:
        raise ValueError('Cell and sample results use different models')
    for name in ['cell_statistics', 'occupancy', 'transitions', 'bouts', 'unit_inventory', 'unit_metrics', 'cell_metrics', 'comparisons', 'states']:
        if not data[name].model_id.eq(model_id).all():
            raise ValueError('State summary sources mix vocabularies')
    state_ids = settings['state_display_order'] or states.sort_values('component').state_id.tolist()
    if set(state_ids) != set(states.state_id):
        raise ValueError('state_display_order must include every accepted state')
    if set(settings['state_display_names']) - set(state_ids):
        raise ValueError('State names refer to unknown accepted states')
    cell_pages = chunks(cells.sort_values(KEYS)[KEYS].to_dict('records'), settings['state_summary_cells_per_page'])
    column_pages = chunks(state_ids, settings['state_summary_columns_per_page'])
    rates = data['cell_metrics'].loc[data['cell_metrics'].metric.eq('switch_rate')]
    spacings = [_json_value(value) for value in sorted(rates.interval_hours.dropna().unique())] or [None]
    result = []
    for view in settings['state_summary_views']:
        if view == 'occupancy':
            result.extend(({'kind': 'cells', 'view': view, 'cells': members, 'states': columns} for members in cell_pages for columns in column_pages))
        elif view == 'switch_rates':
            result.extend(({'kind': 'cells', 'view': view, 'cells': members, 'spacing': spacing} for members in cell_pages for spacing in spacings))
        elif view == 'transitions':
            pairs = [{'source': a, 'target': b} for a in state_ids for b in state_ids]
            result.extend(({'kind': 'cells', 'view': view, 'cells': members, 'spacing': spacing, 'pairs': columns} for members in cell_pages for spacing in spacings for columns in chunks(pairs, settings['state_summary_columns_per_page'])))
        elif view == 'bouts':
            result.extend(({'kind': 'cells', 'view': view, 'cells': members} for members in cell_pages))
        elif view == 'samples':
            questions = data['unit_metrics'].sort_values(['metric', 'question_id']).question_id.drop_duplicates().tolist()
            result.extend(({'kind': 'samples', 'view': view, 'questions': members} for members in chunks(questions, settings['state_sample_questions_per_page'])))
        else:
            members = data['comparisons'].sort_values(['reference_condition', 'target_condition', 'metric', 'comparison_id']).comparison_id.tolist()
            result.extend(({'kind': 'samples', 'view': view, 'comparisons': selected} for selected in chunks(members, settings['state_contrasts_per_page']) or [[]]))
    return [{**page, 'page': index + 1} for index, page in enumerate(result)]

def _cells(frame, keys):
    selected = {tuple((key[name] for name in KEYS)) for key in keys}
    return frame.loc[[key in selected for key in frame[KEYS].itertuples(index=False, name=None)]].copy()

def _spacing(frame, column, spacing):
    return frame.loc[frame[column].isna() if spacing is None else frame[column].eq(spacing)].copy()

def values(data, page):
    frames = []

    def add(name, frame):
        result = frame.copy()
        result['kind'] = name
        frames.append(result)
    view = page['view']
    if page['kind'] == 'cells':
        ledger = _cells(data['cell_statistics'], page['cells'])
        add('cell', ledger)
        if view == 'occupancy':
            source = _cells(data['occupancy'], page['cells'])
            add('occupancy', source.loc[source.state_id.isin(page['states'])])
        elif view == 'switch_rates':
            source = _cells(data['cell_metrics'], page['cells'])
            add('switch_rate', _spacing(source.loc[source.metric.eq('switch_rate')], 'interval_hours', page['spacing']))
        elif view == 'transitions':
            source = _spacing(_cells(data['transitions'], page['cells']), 'interval_stratum_hours', page['spacing'])
            pairs = {(pair['source'], pair['target']) for pair in page['pairs']}
            add('transition', source.loc[[pair in pairs for pair in source[['source_state_id', 'target_state_id']].itertuples(index=False, name=None)]])
        else:
            add('bout', _cells(data['bouts'], page['cells']))
    elif view == 'samples':
        selected = data['unit_metrics'].loc[data['unit_metrics'].question_id.isin(page['questions'])]
        add('sample', selected)
        add('unit', data['unit_inventory'].loc[data['unit_inventory'].unit_id.isin(selected.unit_id)])
        ledger = selected.copy()
    else:
        ledger = data['comparisons'].loc[data['comparisons'].comparison_id.isin(page['comparisons'])].copy()
        add('contrast', ledger)
        if ledger.empty:
            ledger = pd.DataFrame([{'status': 'not_requested', 'reason': 'No condition contrasts were declared or available', 'effect': None, 'p_value': None}])
            add('contrast_status', ledger)
    return (pd.concat(frames, ignore_index=True, sort=False), ledger)

def load(ctx):
    cached = ctx.cache('_behaviour_summary_sources')
    if cached is not None:
        return cached
    data = {alias.removeprefix('summary_').removesuffix('.json'): ctx.table(alias) for alias in ALIASES}
    data['time_provenance'] = ctx.pipeline_metadata('durations-and-switches')
    data['sample_provenance'] = ctx.pipeline_metadata('state-sample-comparisons')
    data['pages'] = pages(data, {name: ctx.option(name) for name in DEFAULTS})
    return data
STANDALONE = ''
NOTES = {'occupancy': 'Fractions use the saved all-observed or assigned-time denominator. Unknown support is shown separately. No renormalisation; row labels give observed / assigned hours.', 'switch_rates': 'Each point is the saved number of switches per eligible transition hour at this observation spacing. Self-steps remain in exposure. Unavailable exposure has no rate.', 'transitions': 'Counts and next-state probabilities are distinct saved quantities at one observation spacing. Labels inside the count matrix give count / source opportunities. Diagonal self-transitions remain included; unavailable probabilities remain undefined.', 'bouts': 'One mark per unique observed bout; hours are saved midpoint allocations. Filled circles: both boundaries observed; left/right triangles: onset/ending censored; crosses: both censored. These are observed durations, not inferred complete hidden dwell times. Point-only and invalid-clock bouts have no duration mark.', 'samples': 'Each point is one saved experimental-unit value using the declared cell/time weighting. Filled circles: independent assignment-only biological samples; open circles: samples used in model choice; crosses: unconfirmed recordings. Horizontal positions only separate units within each condition. Missing values remain in the inventory. No sample aggregation or inference is performed here.', 'contrasts': 'Formal effects are target minus reference among independent assignment-only samples. Saved bootstrap intervals are approximate and unadjusted; probabilities retain the declared whole-family correction. The all-sample descriptive effect is labelled separately and never receives a formal interval.'}

def build(ctx):
    from pymicroglia.figure_tables.prepared import PreparedPage as FigureResult, Drawing
    from pymicroglia.visualisation.text import figure_text
    from pymicroglia.visualisation.panels import behaviour_summaries, behaviour_profiles
    data = load(ctx)
    index = ctx.option('evidence_page')
    if isinstance(index, bool) or not isinstance(index, int) or (not 1 <= index <= len(data['pages'])):
        raise ValueError('evidence_page is outside saved summary pages')
    page = data['pages'][index - 1]
    if ctx.spec.slug != SLUGS[page['kind']]:
        raise ValueError('Saved summary page does not match the requested figure')
    frame, statistics = values(data, page)
    title = {'occupancy': 'Time spent in each accepted state', 'switch_rates': 'Observed state switching rates', 'transitions': 'Observed next-state counts and probabilities', 'bouts': 'Observed state bouts and censoring', 'samples': 'State statistics by experimental unit', 'contrasts': 'Saved independent-sample contrasts'}[page['view']]
    wording = figure_text(ctx.run, ctx.spec.slug, explicit=ctx.text, item=ctx.item, title=title, footnote=NOTES[page['view']], claim=CLAIMS[page['kind']])
    settings = {**page, 'state_definitions': data['states'].to_dict('records'), 'state_display_names': ctx.option('state_display_names'), 'model_id': data['time_provenance']['model_id'], 'sample_provenance': data['sample_provenance'], 'title': wording.title + ('\n' + wording.subtitle if wording.subtitle else ''), 'footnote': wording.footnote + ('\n' + wording.note if wording.note else ''), 'claim': wording.claim, 'grammar': GRAMMAR}
    from pymicroglia.figure_tables import state_summary_display
    prepared = state_summary_display.prepare(frame, settings)
    drawing = Drawing(behaviour_summaries.draw, (prepared, settings), {})
    from pymicroglia.figure_tables import page_views
    view_map=page_views.views(ctx.spec.slug.replace('-','_'),page['view'],drawing,frame)
    return FigureResult(views=view_map,wording=wording, drawing=drawing, heading=wording.claim, figure_data=frame, auxiliary={'statistics.csv': statistics, 'display.csv': pd.DataFrame([{'settings_json': json.dumps(_json_value(settings))}])}, producer_sources={'state_summary_display.py':Path(state_summary_display.__file__),'page_views.py':Path(page_views.__file__),'behaviour_summaries.py': Path(behaviour_summaries.__file__), 'behaviour_profiles.py': Path(behaviour_profiles.__file__), 'behaviour_summary_figures.py': Path(__file__), ctx.spec.source.name: ctx.spec.source}, readme=f"# {wording.title}\n\n{wording.claim}\n\n{NOTES[page['view']]}\n\nAll values, memberships and original censoring are frozen scientific inputs.\n")

def produce(context):
    from pymicroglia.pipelines.behaviour.validation import read_support
    from pymicroglia.pipelines.behaviour.durations import read_statistics
    from pymicroglia.pipelines.behaviour.samples import read_samples
    from pymicroglia.pipelines._saved_figures import draw_batch
    from pymicroglia.visualisation.panels import behaviour_summaries, behaviour_profiles
    decision, model = read_support(context.saved('state-support'), require_accepted=True)
    _, time_provenance = read_statistics(context.saved('durations-and-switches'), expected_model=model['model_id'])
    _, sample_provenance = read_samples(context.saved('state-sample-comparisons'), expected_model=model['model_id'])
    if sample_provenance['duration_id'] != context.saved('durations-and-switches').outcome.scientific_id:
        raise ValueError('Sample summaries use a different saved time analysis')
    data = {name.removeprefix('summary_').removesuffix('.json'): read_table(context.saved(step).artifact(artifact)) for name, (step, artifact) in ALIASES.items()}
    data.update(time_provenance=time_provenance, sample_provenance=sample_provenance)
    settings, text = options(context.presentation)
    data['pages'] = pages(data, settings)
    completed = draw_batch(context, slug='state-summaries', options=[{**settings, 'evidence_page': index} for index in range(1, len(data['pages']) + 1)], aliases=ALIASES, data=data, cache_name='_behaviour_summary_sources', sources=[__file__, behaviour_summaries.__file__, behaviour_profiles.__file__], text=text, claim=CLAIMS['cells'], grammar=GRAMMAR, figure_slugs=[SLUGS[page['kind']] for page in data['pages']], figure_claims=[CLAIMS[page['kind']] for page in data['pages']])
    entries = []
    for page, master in zip(data['pages'], completed['masters']):
        semantic = {**page, 'model_id': model['model_id'], 'time_id': context.saved('durations-and-switches').outcome.scientific_id, 'sample_id': context.saved('state-sample-comparisons').outcome.scientific_id}
        entries.append({**semantic, 'entry_id': content_id(semantic), 'master': master})
    write_table(context.output / 'entries.csv', pd.DataFrame(entries))
    _write_json(context.output / 'state_summaries_manifest.json', {'schema_version': 1, 'settings': settings, 'model_id': model['model_id'], 'decision_id': decision['decision_id'], 'analysis_recomputed': False, 'pages': [{**page, 'master': master} for page, master in zip(data['pages'], completed['masters'])], 'check_output': completed['check_output'], 'registration_output': completed['registration_output']})
    refs = tuple((ArtifactRef(path.name, path.name, file_hash(path), context.scientific_id) for path in sorted(output_files(context.output)) if path.is_file() and (not path.name.startswith('.'))))
    return StepResult(context.step.name, context.scientific_id, 'completed', 'Rendered saved cell time, censored bouts, interval-specific transitions and complete experimental-unit comparisons', refs)
