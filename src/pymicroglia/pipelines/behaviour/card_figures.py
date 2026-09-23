"""State cards connect frozen profiles to exact original observations."""
from pymicroglia._results import output_files
from pymicroglia._results import read_document
from pymicroglia._sources import source_file
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pandas as pd
from pymicroglia.pipelines._contracts import ArtifactRef, Settings, StepResult, content_id, result_to_dict
from pymicroglia.pipelines.relationships.inputs import KEYS
from pymicroglia.pipelines.rhythm.images import IMAGE_DEFAULTS
from pymicroglia.pipelines._screening import _json_value, _write_json, file_hash, read_table, write_table
SLUG = 'accepted-state-card'
GRAMMAR = 'small-multiples'
CLAIM = 'Each accepted-state card connects its complete saved observed profile to a recorded real member, original-time traces, censored bout context and available source imagery.'
ALIASES = {**{'card_' + name + '.json': ('state-assignments', name) for name in ['state_definitions', 'state_profiles', 'representatives', 'assignments']}, **{'card_' + name + '.json': ('durations-and-switches', name) for name in ['bouts', 'exposures', 'occupancy', 'cell_statistics', 'steps']}, 'card_source_traces.json': ('feature-inputs', 'source_traces')}
IMAGE_STEP = 'state-card-images'
DEFAULTS = {'state_card_features': None, 'state_card_features_per_page': 4, 'state_card_low_membership': False, 'state_card_images': True, 'state_display_order': None, 'state_display_names': {}, 'evidence_page': 1, **{key: value for key, value in IMAGE_DEFAULTS.items() if key not in {'images', 'image_hours'}}}

def version():
    from pymicroglia.pipelines._versions import rendering
    return rendering(__file__)

def options(presentation):
    declared = presentation.as_dict().get('state_cards', {})
    if not isinstance(declared, dict) or set(declared) - (set(DEFAULTS) - {'evidence_page'} | {'text'}):
        raise ValueError('State cards accept saved features, pagination, example/image display options, state labels/order and text')
    settings = {**DEFAULTS, **{key: value for key, value in declared.items() if key != 'text'}}
    count = settings['state_card_features_per_page']
    if isinstance(count, bool) or not isinstance(count, int) or (not 1 <= count <= 6):
        raise ValueError('state_card_features_per_page must be an integer from one to six')
    for key in ['state_card_images', 'state_card_low_membership']:
        if not isinstance(settings[key], bool):
            raise ValueError(key + ' must be true or false')
    for key in ['state_card_features', 'state_display_order']:
        value = settings[key]
        if value is not None and (not isinstance(value, list) or not value or any((not isinstance(item, str) or not item.strip() for item in value)) or (len(set(value)) != len(value))):
            raise ValueError(key + ' must be a nonempty list of unique saved names')
    labels = settings['state_display_names']
    if not isinstance(labels, dict) or any((not isinstance(key, str) or not isinstance(value, str) or (not value.strip()) for key, value in labels.items())):
        raise ValueError('state_display_names must map accepted states to nonempty labels')
    return (settings, declared.get('text', {}))

def select_examples(data, settings):
    states, assignments = (data['state_definitions'], data['assignments'])
    if assignments.observation_id.duplicated().any():
        raise ValueError('State cards repeat an original assignment observation')
    lookup = assignments.set_index('observation_id')
    examples = []
    inventory = []
    for state in states.sort_values('component').to_dict('records'):
        representatives = data['representatives'].loc[data['representatives'].state_id.eq(state['state_id'])]
        if len(representatives) != 1:
            raise ValueError('Each accepted state requires one saved representative outcome')
        primary = representatives.iloc[0].to_dict()
        choices = []
        if primary['status'] == 'selected':
            choices.append(('representative', primary, primary['selection_rule']))
        elif primary['status'] != 'no_assigned_observation':
            raise ValueError('Unknown saved state representative outcome')
        if settings['state_card_low_membership']:
            candidates = assignments.loc[assignments.state_id.eq(state['state_id']) & assignments.status.eq('assigned')]
            if choices:
                candidates = candidates.loc[~candidates.observation_id.eq(primary['observation_id'])]
            if len(candidates):
                member = candidates.sort_values(['max_probability', 'log_density', 'observation_id']).iloc[0].to_dict()
                choices.append(('low_membership_assigned', member, 'Lowest saved native maximum membership among other assigned observations; then log density and observation identity. This remains an assigned example, not biological confidence.'))
        selected = []
        for role, reference, rule in choices:
            if reference['observation_id'] not in lookup.index:
                raise ValueError('State representative is absent from saved assignments')
            actual = lookup.loc[reference['observation_id']]
            for name in ['model_id', 'state_id', *KEYS, 'frame_index', 'hours']:
                if actual[name] != reference[name]:
                    raise ValueError('State representative disagrees with its full saved observation identity')
            if actual.status != 'assigned' or actual.state_id != state['state_id'] or actual.model_id != state['model_id']:
                raise ValueError('State representative is not a confirmed saved member of this accepted model/state')
            key = {name: _json_value(reference[name]) for name in ['model_id', 'state_id', 'observation_id', *KEYS, 'frame_index', 'hours']}
            selected_bouts = data['bouts'].loc[data['bouts'].observation_ids.map(lambda members: reference['observation_id'] in members)]
            if len(selected_bouts) > 1:
                raise ValueError('One state representative belongs to multiple unique observed bouts')
            if len(selected_bouts) and (selected_bouts.state_id.iloc[0] != key['state_id'] or selected_bouts.model_id.iloc[0] != key['model_id']):
                raise ValueError('Representative bout refers to another state vocabulary')
            example = {**key, 'example_id': content_id({**key, 'role': role}), 'example_role': role, 'selection_rule': rule, 'max_probability': _json_value(actual.max_probability), 'log_density': _json_value(actual.log_density), 'bout_id': selected_bouts.bout_id.iloc[0] if len(selected_bouts) else None, 'bout_status': selected_bouts.duration_status.iloc[0] if len(selected_bouts) else 'no_saved_bout'}
            examples.append(example)
            selected.append(example['example_id'])
        inventory.append({'model_id': state['model_id'], 'state_id': state['state_id'], 'status': 'selected' if selected else 'no_assigned_observation', 'primary_rule': primary['selection_rule'], 'example_ids': selected, 'full_state_assigned_observations': state['assigned_observations']})
    return (examples, inventory)

def pages(data, settings):
    states = data['state_definitions']
    model = data['assignment_provenance']['model_id']
    for name in ['state_definitions', 'state_profiles', 'representatives', 'assignments', 'bouts', 'exposures', 'occupancy', 'cell_statistics', 'steps']:
        if not data[name].model_id.eq(model).all():
            raise ValueError('State card sources use different accepted models')
    order = settings['state_display_order'] or states.sort_values('component').state_id.tolist()
    if set(order) != set(states.state_id):
        raise ValueError('state_display_order must include every accepted state')
    if set(settings['state_display_names']) - set(order):
        raise ValueError('Card labels refer to an unknown accepted state')
    profiles = data['state_profiles']
    measurements = settings['state_card_features'] or profiles.measurement.drop_duplicates().tolist()
    if set(measurements) - set(profiles.measurement):
        raise ValueError('state_card_features names an unavailable saved feature')
    available = profiles[['measurement', 'source_table', 'representation', 'unit']].drop_duplicates()
    features = [row for measurement in measurements for row in available.loc[available.measurement.eq(measurement)].to_dict('records')]
    count = settings['state_card_features_per_page']
    result = []
    for state in order:
        examples = [row['example_id'] for row in data['examples'] if row['state_id'] == state]
        for start in range(0, len(features), count):
            result.append({'state_id': state, 'features': features[start:start + count], 'example_ids': examples, 'feature_part': start // count + 1})
    return [{**page, 'page': index + 1} for index, page in enumerate(result)]

def _cell(frame, key):
    mask = pd.Series(True, index=frame.index)
    for name in KEYS:
        mask &= frame[name].eq(key[name])
    return frame.loc[mask].copy()

def values(data, page):
    frames = []

    def add(kind, frame, example=None):
        result = frame.copy()
        result['kind'] = kind
        result['example_id'] = example
        frames.append(result)
    add('state', data['state_definitions'].loc[data['state_definitions'].state_id.eq(page['state_id'])])
    features = {(row['measurement'], row['source_table'], row['representation']) for row in page['features']}
    profiles = data['state_profiles']
    profiles = profiles.loc[profiles.state_id.eq(page['state_id']) & profiles[['measurement', 'source_table', 'representation']].apply(tuple, axis=1).isin(features)]
    add('profile', profiles)
    add('population_occupancy', data['occupancy'].loc[data['occupancy'].state_id.eq(page['state_id'])])
    add('population_cell', data['cell_statistics'])
    for example in [row for row in data['examples'] if row['example_id'] in page['example_ids']]:
        add('example', pd.DataFrame([example]), example['example_id'])
        for kind, name in [('observation', 'assignments'), ('interval', 'exposures'), ('cell', 'cell_statistics'), ('step', 'steps')]:
            add(kind, _cell(data[name], example), example['example_id'])
        traces = _cell(data['source_traces'], example)
        add('trace', traces.loc[traces.measurement.isin([row['measurement'] for row in page['features']])], example['example_id'])
        if example['bout_id']:
            add('example_bout', data['bouts'].loc[data['bouts'].bout_id.eq(example['bout_id'])], example['example_id'])
    return (pd.concat(frames, ignore_index=True, sort=False), profiles.copy())

def image_receipt(context, archive, metadata):
    from pymicroglia.pipelines._runner import SavedResult
    references = tuple((ArtifactRef(name, path.name, file_hash(path), context.scientific_id) for name, path in [('archive', archive), ('inventory', metadata)]))
    outcome = StepResult(IMAGE_STEP, context.scientific_id, 'completed', 'Frozen exact-observation image availability and display pixels; no scientific calculation', references)
    _write_json(context.output / 'result.json', result_to_dict(outcome))
    return SavedResult(context.output, outcome)

def load(ctx):
    cached = ctx.cache('_behaviour_card_sources')
    if cached is not None:
        return cached
    data = {alias.removeprefix('card_').removesuffix('.json'): ctx.table(alias) for alias in ALIASES}
    data['assignment_provenance'] = ctx.pipeline_metadata('state-assignments')
    data['time_provenance'] = ctx.pipeline_metadata('durations-and-switches')
    settings = {name: ctx.option(name) for name in DEFAULTS}
    data['examples'], data['selection_inventory'] = select_examples(data, settings)
    data['pages'] = pages(data, settings)
    _, saved = ctx._pipeline_sources()
    images = saved[IMAGE_STEP]
    data['image_archive'] = images.artifact('archive')
    data['image_inventory'] = images.artifact('inventory')
    ctx.record_source('state_example_images.json', data['image_inventory'])
    data['images'] = read_document(data['image_inventory'])
    return data
STANDALONE = ''

def build(ctx):
    from pymicroglia.figure_tables.prepared import PreparedPage as FigureResult, Drawing
    from pymicroglia.visualisation.text import figure_text
    from pymicroglia.visualisation.panels import behaviour_cards, behaviour_profiles, behaviour_timelines
    data = load(ctx)
    index = ctx.option('evidence_page')
    if isinstance(index, bool) or not isinstance(index, int) or (not 1 <= index <= len(data['pages'])):
        raise ValueError('evidence_page is outside saved state cards')
    page = data['pages'][index - 1]
    frame, ledger = values(data, page)
    state = data['state_definitions'].loc[data['state_definitions'].state_id.eq(page['state_id'])].iloc[0]
    title = ctx.option('state_display_names').get(state.state_id, state.label) + ' | observed state card'
    note = 'Profiles and occupancy describe the complete saved population. Examples are real assigned observations selected by the recorded rule; their subset never defines population statistics. These measurements defined the states and are descriptive. Traces retain original observations and saved interval support; no fitting, smoothing, gap filling or reassignment.'
    wording = figure_text(ctx.run, ctx.spec.slug, explicit=ctx.text, item=ctx.item, title=title, footnote=note, claim=CLAIM)
    settings = {**page, 'state_definitions': data['state_definitions'].to_dict('records'), 'state_display_names': ctx.option('state_display_names'), 'images': [row for row in data['images']['examples'] if row['example_id'] in page['example_ids']], 'model_id': state.model_id, 'assignment_provenance': data['assignment_provenance'], 'title': wording.title + ('\n' + wording.subtitle if wording.subtitle else ''), 'footnote': wording.footnote + ('\n' + wording.note if wording.note else ''), 'claim': wording.claim, 'grammar': GRAMMAR}
    ctx.record_source('state_example_tiles.npz', data['image_archive'])
    ctx.record_source('state_example_images.json', data['image_inventory'])
    with np.load(data['image_archive'], allow_pickle=False) as archive:
        from pymicroglia.figure_tables import state_card_display
        prepared=state_card_display.prepare(frame,settings,{key:archive[key] for key in archive.files})
        drawing=Drawing(behaviour_cards.draw,(prepared,settings),{})
    views={name:(Drawing(behaviour_cards.draw,(prepared,settings),{'selected_view':name}),frame[frame.kind.isin(kinds)].copy()) for name,kinds in {'profiles':['state','profile'],'population':['state','population_occupancy','population_cell'],'traces':['example','cell','interval','observation','trace','step'],'images':['example']}.items()}
    return FigureResult(views=views,wording=wording, drawing=drawing, heading=wording.claim, figure_data=frame, auxiliary={'statistics.csv': ledger, 'display.csv': pd.DataFrame([{'settings_json': json.dumps(_json_value(settings))}])}, producer_sources={'state_card_display.py':Path(state_card_display.__file__),'behaviour_cards.py': Path(behaviour_cards.__file__), 'behaviour_profiles.py': Path(behaviour_profiles.__file__), 'behaviour_timelines.py': Path(behaviour_timelines.__file__), 'behaviour_card_figures.py': Path(__file__), ctx.spec.source.name: ctx.spec.source}, readme=f'# {wording.title}\n\n{wording.claim}\n\n{note}\n\nImage crops retain exact observation/source-frame mappings; missing imagery remains explicitly unavailable.\n')

def produce(context):
    from pymicroglia.pipelines.behaviour.validation import read_support
    from pymicroglia.pipelines.behaviour.assignments import read_assignments
    from pymicroglia.pipelines.behaviour.durations import read_statistics
    from pymicroglia.pipelines.behaviour.card_images import prepare_examples
    from pymicroglia.pipelines._saved_figures import draw_batch
    from pymicroglia.visualisation.panels import behaviour_cards, behaviour_profiles, behaviour_timelines
    from pymicroglia.figure_tables import images as intensity
    decision, model = read_support(context.saved('state-support'), require_accepted=True)
    _, provenance = read_assignments(context.saved('state-assignments'), expected_model=model['model_id'])
    _, time_provenance = read_statistics(context.saved('durations-and-switches'), expected_model=model['model_id'])
    if provenance['feature_input_id'] != context.saved('feature-inputs').outcome.scientific_id:
        raise ValueError('State card traces belong to a different prepared input')
    data = {name.removeprefix('card_').removesuffix('.json'): read_table(context.saved(step).artifact(artifact)) for name, (step, artifact) in ALIASES.items()}
    data.update(assignment_provenance=provenance, time_provenance=time_provenance)
    settings, text = options(context.presentation)
    data['examples'], data['selection_inventory'] = select_examples(data, settings)
    data['pages'] = pages(data, settings)
    archive, metadata, sources = prepare_examples(context, data['examples'], settings, context.output)
    data['image_archive'] = archive
    data['image_inventory'] = metadata
    data['images'] = read_document(metadata)
    snapshot = image_receipt(context, archive, metadata)
    render_context = replace(context, dependencies={**context.dependencies, IMAGE_STEP: snapshot})
    completed = draw_batch(render_context, slug=SLUG, options=[{**settings, 'evidence_page': index} for index in range(1, len(data['pages']) + 1)], aliases=ALIASES, data=data, cache_name='_behaviour_card_sources', sources=[__file__, behaviour_cards.__file__, behaviour_profiles.__file__, behaviour_timelines.__file__, intensity.__file__, source_file('behaviour_card_images.py'), source_file('rhythm_images.py'), *sources.values()], text=text, claim=CLAIM, grammar=GRAMMAR)
    entries = []
    for page, master in zip(data['pages'], completed['masters']):
        semantic = {**page, 'model_id': model['model_id'], 'assignment_id': context.saved('state-assignments').outcome.scientific_id, 'examples': [row for row in data['examples'] if row['example_id'] in page['example_ids']]}
        entries.append({**semantic, 'entry_id': content_id(semantic), 'master': master})
    write_table(context.output / 'entries.csv', pd.DataFrame(entries))
    _write_json(context.output / 'state_card_selection.json', {'schema_version': 1, 'display_only': True, 'states': data['selection_inventory'], 'examples': data['examples']})
    _write_json(context.output / 'state_cards_manifest.json', {'schema_version': 1, 'settings': settings, 'model_id': model['model_id'], 'decision_id': decision['decision_id'], 'analysis_recomputed': False, 'pages': [{**page, 'master': master} for page, master in zip(data['pages'], completed['masters'])], 'images': data['images'], 'selection': data['selection_inventory'], 'check_output': completed['check_output'], 'registration_output': completed['registration_output']})
    refs = tuple((ArtifactRef(path.name, path.name, file_hash(path), context.scientific_id) for path in sorted(output_files(context.output)) if path.is_file() and (not path.name.startswith('.'))))
    return StepResult(context.step.name, context.scientific_id, 'completed', 'Rendered accepted-state cards with exact real members, full population profiles, original traces and recorded image availability', refs)
