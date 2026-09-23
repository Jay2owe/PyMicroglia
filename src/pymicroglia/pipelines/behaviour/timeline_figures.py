"""Draw frozen state intervals on each cell's original recording clock."""
from __future__ import annotations
from pymicroglia._results import output_files
from pymicroglia._sources import source_file
from pathlib import Path
import json
import pandas as pd
from pymicroglia.pipelines._contracts import ArtifactRef, CellKey, Settings, StepResult, content_id
from pymicroglia.pipelines.relationships.inputs import KEYS
from pymicroglia.pipelines._screening import _json_value, _write_json, file_hash, read_table, write_table
SLUG = 'cell-state-timelines'
GRAMMAR = 'timeline'
CLAIM = 'Each cell timeline preserves its own recorded physical-time support, categorical assignments, typed unknown observations and unobserved gaps without smoothing or alignment.'
ALIASES = {'timeline_exposures.json': ('durations-and-switches', 'exposures'), 'timeline_cells.json': ('durations-and-switches', 'cell_statistics'), 'timeline_assignments.json': ('state-assignments', 'assignments'), 'timeline_states.json': ('state-assignments', 'state_definitions'), 'timeline_inventory.json': ('state-assignments', 'cell_inventory')}
DEFAULTS = {'state_cells_per_page': 6, 'state_cell_order': None, 'state_show_membership': False, 'state_display_order': None, 'state_display_names': {}, 'evidence_page': 1}

def version():
    from pymicroglia.pipelines._versions import rendering
    return rendering(__file__)

def options(presentation):
    declared = presentation.as_dict().get('state_timelines', {})
    if not isinstance(declared, dict) or set(declared) - (set(DEFAULTS) - {'evidence_page'} | {'text'}):
        raise ValueError('State timeline presentation accepts cell page size/order, state order/names, optional membership and text')
    result = {**DEFAULTS, **{key: value for key, value in declared.items() if key != 'text'}}
    size = result['state_cells_per_page']
    if isinstance(size, bool) or not isinstance(size, int) or (not 1 <= size <= 12):
        raise ValueError('state_cells_per_page must be an integer from one to twelve')
    if not isinstance(result['state_show_membership'], bool):
        raise ValueError('state_show_membership must be true or false')
    order = result['state_cell_order']
    if order is not None:
        if not isinstance(order, list):
            raise ValueError('state_cell_order must list complete source/movie/cell keys')
        canonical = []
        for key in order:
            if not isinstance(key, dict) or set(key) != set(KEYS):
                raise ValueError('state_cell_order requires full source_run, movie and identity keys')
            canonical.append(CellKey(**key).as_dict())
        if len({content_id(key) for key in canonical}) != len(canonical):
            raise ValueError('state_cell_order repeats a full cell identity')
        result['state_cell_order'] = canonical
    state_order = result['state_display_order']
    if state_order is not None and (not isinstance(state_order, list) or any((not isinstance(value, str) for value in state_order)) or len(set(state_order)) != len(state_order)):
        raise ValueError('state_display_order must contain unique accepted state IDs')
    names = result['state_display_names']
    if not isinstance(names, dict) or any((not isinstance(key, str) or not isinstance(value, str) or (not value.strip()) for key, value in names.items())):
        raise ValueError('state_display_names must map accepted state IDs to nonempty labels')
    return (result, declared.get('text', {}))

def pages(data, settings):
    cells, states = (data['timeline_cells'], data['timeline_states'])
    if cells.duplicated(KEYS).any():
        raise ValueError('Saved timeline inventory repeats a full cell identity')
    model = data['time_provenance']['model_id']
    for name in ALIASES:
        if not data[name.removesuffix('.json')].model_id.eq(model).all():
            raise ValueError('Timeline sources use different state vocabularies')
    original = cells.sort_values(KEYS)[KEYS].to_dict('records')
    order = settings['state_cell_order'] if settings['state_cell_order'] is not None else original
    if {content_id(key) for key in order} != {content_id(key) for key in original}:
        raise ValueError('state_cell_order must include every requested full cell identity exactly once')
    state_ids = states.sort_values('component').state_id.tolist()
    state_order = settings['state_display_order'] or state_ids
    if set(state_order) != set(state_ids):
        raise ValueError('state_display_order must include every accepted state')
    if set(settings['state_display_names']) - set(state_ids):
        raise ValueError('State labels refer to an unknown accepted state')
    inventory_keys = set(data['timeline_inventory'][KEYS].itertuples(index=False, name=None))
    if inventory_keys != set(cells[KEYS].itertuples(index=False, name=None)):
        raise ValueError('Assignment and time inventories contain different requested cells')
    if settings['state_show_membership']:
        assigned = data['timeline_assignments']
        if not assigned.score_type.eq('native_gaussian_mixture_component_membership').all() or not assigned.score_calibration.eq('uncalibrated_model_conditional').all():
            raise ValueError('The saved score type cannot be displayed as model-conditional component membership')
    size = settings['state_cells_per_page']
    return [{'cells': order[start:start + size], 'states': state_order, 'page': start // size + 1, 'show_membership': settings['state_show_membership']} for start in range(0, len(order), size)]

def values(data, page):
    members = {tuple((key[name] for name in KEYS)) for key in page['cells']}
    frames = []
    for kind, name in [('cell', 'timeline_cells'), ('interval', 'timeline_exposures'), ('observation', 'timeline_assignments')]:
        source = data[name]
        chosen = source.loc[[key in members for key in source[KEYS].itertuples(index=False, name=None)]].copy()
        if kind == 'cell':
            inventory = data['timeline_inventory']
            counts = [name for name in inventory if name.endswith('_observations') and name not in chosen]
            chosen = chosen.merge(inventory[[*KEYS, *counts]], on=KEYS, how='left', validate='one_to_one')
        chosen['kind'] = kind
        frames.append(chosen)
    return (pd.concat(frames, ignore_index=True, sort=False), frames[0].copy())

def load(ctx):
    cached = ctx.cache('_behaviour_timeline_sources')
    if cached is not None:
        return cached
    data = {name.removesuffix('.json'): ctx.table(name) for name in ALIASES}
    data['time_provenance'] = ctx.pipeline_metadata('durations-and-switches')
    data['assignment_provenance'] = ctx.pipeline_metadata('state-assignments')
    settings = {name: ctx.option(name) for name in DEFAULTS}
    data['pages'] = pages(data, settings)
    return data
STANDALONE = ''

def build(ctx):
    from pymicroglia.figure_tables.prepared import PreparedPage as FigureResult, Drawing
    from pymicroglia.visualisation.text import figure_text
    from pymicroglia.visualisation.panels import behaviour_timelines, behaviour_profiles
    data = load(ctx)
    index = ctx.option('evidence_page')
    if isinstance(index, bool) or not isinstance(index, int) or (not 1 <= index <= len(data['pages'])):
        raise ValueError('evidence_page is outside saved timeline pages')
    page = data['pages'][index - 1]
    frame, statistics = values(data, page)
    note = 'Each row uses its own recording clock. Coloured intervals are saved midpoint allocations, not exact hidden dwell times. Gaps stay unobserved; ticks mark original observations. No smoothing, phase alignment or new switch counting.'
    if page['show_membership']:
        note += ' Grey dots below each ribbon show native model-conditional component membership, not calibrated biological confidence.'
    wording = figure_text(ctx.run, ctx.spec.slug, explicit=ctx.text, item=ctx.item, title='Cell states in original recording time', footnote=note, claim=CLAIM)
    settings = {**page, 'state_definitions': data['timeline_states'].to_dict('records'), 'state_display_names': ctx.option('state_display_names'), 'model_id': data['time_provenance']['model_id'], 'title': wording.title + ('\n' + wording.subtitle if wording.subtitle else ''), 'footnote': wording.footnote + ('\n' + wording.note if wording.note else ''), 'claim': wording.claim, 'grammar': GRAMMAR}
    from pymicroglia.figure_tables import timeline_display
    prepared=timeline_display.prepare(frame,settings)
    drawing=Drawing(behaviour_timelines.draw,(prepared,settings),{})
    view_tables={'intervals':frame[frame.kind.isin(['cell','interval'])],'observations':frame[frame.kind.isin(['cell','observation'])]}
    if settings['show_membership']:view_tables['membership']=view_tables['observations']
    views={name:(Drawing(behaviour_timelines.draw,(prepared,settings),{'selected_view':name}),table) for name,table in view_tables.items()}
    return FigureResult(views=views,wording=wording, drawing=drawing, heading=wording.claim, figure_data=frame, auxiliary={'statistics.csv': statistics, 'display.csv': pd.DataFrame([{'settings_json': json.dumps(_json_value(settings))}])}, producer_sources={'timeline_display.py':Path(timeline_display.__file__),'behaviour_timelines.py': Path(behaviour_timelines.__file__), 'behaviour_profiles.py': Path(behaviour_profiles.__file__), 'behaviour_timeline_figures.py': Path(__file__), ctx.spec.source.name: ctx.spec.source}, readme=f'# {wording.title}\n\n{wording.claim}\n\n{note}\n\nEvery cell uses its original source/movie identity and its own physical-time axis. Missing or nonincreasing clocks do not create time exposure.\n')

def produce(context):
    from pymicroglia.pipelines.behaviour.validation import read_support
    from pymicroglia.pipelines.behaviour.assignments import read_assignments
    from pymicroglia.pipelines.behaviour.durations import read_statistics
    from pymicroglia.pipelines._saved_figures import draw_batch
    from pymicroglia.visualisation.panels import behaviour_timelines, behaviour_profiles
    decision, model = read_support(context.saved('state-support'), require_accepted=True)
    read_assignments(context.saved('state-assignments'), expected_model=model['model_id'])
    _, time_provenance = read_statistics(context.saved('durations-and-switches'), expected_model=model['model_id'])
    data = {name.removesuffix('.json'): read_table(context.saved(step).artifact(artifact)) for name, (step, artifact) in ALIASES.items()}
    data['time_provenance'] = time_provenance
    settings, text = options(context.presentation)
    data['pages'] = pages(data, settings)
    completed = draw_batch(context, slug=SLUG, options=[{**settings, 'evidence_page': index} for index in range(1, len(data['pages']) + 1)], aliases=ALIASES, data=data, cache_name='_behaviour_timeline_sources', sources=[__file__, behaviour_timelines.__file__, behaviour_profiles.__file__], text=text, claim=CLAIM, grammar=GRAMMAR)
    entries = []
    cells = data['timeline_cells'].set_index(KEYS)
    for page, master in zip(data['pages'], completed['masters']):
        for key in page['cells']:
            row = cells.loc[tuple((key[name] for name in KEYS))]
            semantic = {**key, 'model_id': model['model_id'], 'source_scientific_id': context.saved('durations-and-switches').outcome.scientific_id}
            entries.append({**semantic, 'entry_id': content_id(semantic), 'master': master, 'page': page['page'], 'status': row['status'], 'start_hours': _json_value(row['reference_start_hours']), 'end_hours': _json_value(row['reference_end_hours'])})
    write_table(context.output / 'entries.csv', pd.DataFrame(entries, columns=[*KEYS, 'model_id', 'source_scientific_id', 'entry_id', 'master', 'page', 'status', 'start_hours', 'end_hours']))
    _write_json(context.output / 'state_timelines_manifest.json', {'schema_version': 1, 'settings': settings, 'model_id': model['model_id'], 'decision_id': decision['decision_id'], 'analysis_recomputed': False, 'clock_scope': 'Separate original recording clock for every cell', 'pages': [{**page, 'master': master} for page, master in zip(data['pages'], completed['masters'])], 'check_output': completed['check_output'], 'registration_output': completed['registration_output']})
    refs = tuple((ArtifactRef(path.name, path.name, file_hash(path), context.scientific_id) for path in sorted(output_files(context.output)) if path.is_file() and (not path.name.startswith('.'))))
    return StepResult(context.step.name, context.scientific_id, 'completed', 'Rendered complete original-clock cell timelines with saved interval support and typed unknown assignments', refs)
