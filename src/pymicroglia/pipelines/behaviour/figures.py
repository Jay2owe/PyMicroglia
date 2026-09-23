"""Saved support evidence and original-unit state profiles, including refusals."""
from __future__ import annotations
from pymicroglia._results import output_files
from pymicroglia._results import read_document
from pymicroglia._sources import source_file
from dataclasses import replace
from pathlib import Path
import json
import pandas as pd
from pymicroglia.pipelines._contracts import ArtifactRef, Settings, StepResult, content_id, result_to_dict
from pymicroglia.pipelines._screening import _json_value, _write_json, file_hash, read_table, write_table
GRAMMAR = 'small-multiples'
SLUGS = {'support': 'state-support-summary', 'profiles': 'state-profiles'}
CLAIMS = {'support': 'Saved separation, repeatability, coverage and independent confirmation checks determine whether the proposed state vocabulary is supported.', 'profiles': 'Accepted state profiles describe actual observed measurements in their original units, with missing values and member counts retained.'}
ALIASES = {'state_display_values.json': ('state-display', 'values'), 'state_display_statistics.json': ('state-display', 'statistics')}
DEFAULTS = {'state_page_size': 10, 'state_feature_page_size': 4, 'state_display_order': None, 'state_display_names': {}, 'evidence_page': 1}
VALUE_COLUMNS = ['entry_id', 'view', 'kind', 'group', 'candidate_id', 'model_id', 'state_id', 'component', 'criterion', 'observed', 'required', 'status', 'reason', 'role', 'source_scientific_id', 'measurement', 'source_table', 'unit', 'representation', 'mean', 'median', 'q25', 'q75', 'observed_values', 'missing_values', 'cells', 'movies', 'confirmed_samples', 'observed_members']

def options(presentation):
    declared = presentation.as_dict().get('state_profiles', {})
    if not isinstance(declared, dict) or set(declared) - (set(DEFAULTS) - {'evidence_page'} | {'text'}):
        raise ValueError('State presentation accepts page sizes, display order/names and text')
    result = {**DEFAULTS, **{key: value for key, value in declared.items() if key != 'text'}}
    for name, maximum in [('state_page_size', 16), ('state_feature_page_size', 6)]:
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, int) or (not 1 <= value <= maximum):
            raise ValueError(name + ' must be an integer from one to ' + str(maximum))
    order = result['state_display_order']
    if order is not None and (not isinstance(order, list) or any((not isinstance(value, str) for value in order)) or len(set(order)) != len(order)):
        raise ValueError('state_display_order must contain unique saved state IDs')
    names = result['state_display_names']
    if not isinstance(names, dict) or any((not isinstance(key, str) or not isinstance(value, str) or (not value.strip()) for key, value in names.items())):
        raise ValueError('state_display_names maps saved state IDs to nonempty display labels')
    return (result, declared.get('text', {}))

def version():
    from pymicroglia.pipelines._versions import rendering
    return rendering(__file__)

def _row(**values):
    values = _json_value(values)
    return {'entry_id': content_id(values), **values}

def support_data(context):
    """Read evidence as saved, without assuming that any candidate was accepted."""
    from pymicroglia.pipelines.behaviour.validation import read_support
    records, statistics = ([], [])
    support = context.saved('state-support')
    metadata = {'view': 'support', 'status': support.outcome.status, 'reason': support.outcome.reason, 'model_id': None, 'scope': 'Candidate groupings are not accepted biological states', 'states': [], 'source_outcomes': {name: result_to_dict(saved.outcome) for name, saved in context.dependencies.items()}}
    for name, saved in context.dependencies.items():
        records.append(_row(view='support', kind='execution', group='Analysis outcome', criterion=name.replace('-', ' '), status='completed' if saved.outcome.status == 'reused' else saved.outcome.status, reason='Saved result available' if saved.outcome.status in {'completed', 'reused'} else saved.outcome.reason, role='execution', source_scientific_id=saved.outcome.scientific_id))
    if support.outcome.status in {'completed', 'reused'}:
        decision, model = read_support(support)
        metadata.update(status=decision['status'], reason=decision['reason'], model_id=decision.get('accepted_model_id'), scope=decision['scope'], support_settings=decision['settings'])
        records.append(_row(view='support', kind='decision', group='Analysis outcome', criterion='State vocabulary decision', status=decision['status'], reason=decision['reason'], model_id=decision.get('accepted_model_id'), role='decision', source_scientific_id=support.outcome.scientific_id))
        candidates = read_table(support.artifact('candidate_support'))
        for number, candidate in enumerate(candidates.to_dict('records'), 1):
            group = 'Development: candidate ' + str(number) + ' (' + str(candidate['components']) + ' components)'
            base = dict(view='support', group=group, candidate_id=candidate['candidate_id'], model_id=candidate['model_id'], role='development', source_scientific_id=support.outcome.scientific_id)
            records.append(_row(**base, kind='candidate', criterion='Candidate support', status=candidate['status'], reason=candidate['reason']))
            for gate in candidate['gates']:
                records.append(_row(**base, kind='criterion', criterion=gate['name'], observed=gate['observed'], required=gate['required'], status=gate['status']))
            statistics.append({**base, **candidate, 'kind': 'candidate_support'})
        development = read_document(support.artifact('development_evidence'))
        confirmation = read_document(support.artifact('confirmation_evidence'))
        for evidence in development:
            for axis in evidence.get('axes', []):
                statistics.append({**evidence, **axis, 'role': 'development', 'kind': 'native_projection_test'})
        if confirmation.get('evaluated'):
            frozen = read_document(support.artifact('frozen_choice'))
            base = dict(view='support', group='Reserved confirmation of the frozen choice', candidate_id=frozen['candidate_id'], model_id=frozen['model_id'], role='confirmation', source_scientific_id=support.outcome.scientific_id)
            for gate in confirmation['assessment']['gates']:
                records.append(_row(**base, kind='criterion', criterion=gate['name'], observed=gate['observed'], required=gate['required'], status=gate['status']))
            statistics.append({**base, **confirmation['assessment'], 'kind': 'confirmation_support'})
            for axis in confirmation['evidence'].get('axes', []):
                statistics.append({**confirmation['evidence'], **axis, 'role': 'confirmation', 'kind': 'native_projection_test'})
        else:
            records.append(_row(view='support', kind='decision', group='Reserved confirmation', criterion='Confirmation evaluation', status='not_evaluated', reason=confirmation.get('reason', decision['reason']), role='confirmation', source_scientific_id=support.outcome.scientific_id))
    if not statistics:
        statistics = [{'kind': 'analysis_availability', 'question': name, 'status': saved.outcome.status, 'reason': saved.outcome.reason, 'estimate': None, 'p_value': None, 'q_value': None, 'evidence_method': 'not_evaluated', 'source_scientific_id': saved.outcome.scientific_id} for name, saved in context.dependencies.items()]
    return (pd.DataFrame(records, columns=VALUE_COLUMNS), pd.DataFrame(statistics), metadata)

def profile_data(context):
    from pymicroglia.pipelines.behaviour.assignments import read_assignments
    from pymicroglia.pipelines.behaviour.validation import read_support
    decision, model = read_support(context.saved('state-support'), require_accepted=True)
    saved = context.saved('state-assignments')
    _, provenance = read_assignments(saved, expected_model=model['model_id'])
    profiles = read_table(saved.artifact('state_profiles'))
    states = read_table(saved.artifact('state_definitions'))
    expected = [feature['column'] for feature in provenance['feature_definitions']]
    if set(profiles.measurement) != set(expected) or not profiles.model_id.eq(model['model_id']).all():
        raise ValueError('Saved profiles do not match the accepted feature/model definition')
    records = [_row(**row, view='profiles', kind='profile', group=row['measurement'], status='observed' if row['observed_values'] else 'no_observed_values', criterion=row['measurement'], role='descriptive_assignment', source_scientific_id=saved.outcome.scientific_id) for row in profiles.to_dict('records')]
    metadata = {'view': 'profiles', 'status': 'accepted', 'reason': decision['reason'], 'model_id': model['model_id'], 'scope': decision['scope'], 'states': states.to_dict('records'), 'features': provenance['feature_definitions'], 'assignment_status_counts': provenance['status_counts'], 'profile_scope': provenance['profile_scope'], 'score_scope': provenance['score_calibration'], 'source_outcomes': {name: result_to_dict(item.outcome) for name, item in context.dependencies.items()}}
    return (pd.DataFrame(records, columns=VALUE_COLUMNS), profiles, metadata)

def snapshot(context, values, statistics, metadata):
    """An immutable display-input receipt can describe unavailable scientific steps.

    It records the actual upstream outcomes, not a replacement scientific success.
    The ordinary strict figure binding remains unchanged. Everything stays flat;
    result.json belongs to this display snapshot, while the runner later writes
    the separate execution-result.json for the finished figure producer.
    """
    from pymicroglia.pipelines._runner import SavedResult
    context.output.mkdir(parents=True)
    metadata = {**metadata, 'schema_version': 1, 'scientific_id': context.scientific_id, 'snapshot_kind': 'saved_display_inputs', 'analysis_recomputed': False, 'source_artifacts': [{'step': name, 'artifact': ref.name, 'sha256': ref.sha256, 'scientific_id': ref.scientific_id} for name, saved in context.dependencies.items() for ref in saved.outcome.artifacts]}
    refs = []
    for name, table in [('values', values), ('statistics', statistics)]:
        path = context.output / ('state_display_' + name + '.json')
        path = write_table(path, table)
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id, columns=tuple(table.columns)))
    path = context.output / 'state_display_provenance.json'
    _write_json(path, metadata)
    refs.append(ArtifactRef('provenance', path.name, file_hash(path), context.scientific_id))
    result = StepResult('state-display', context.scientific_id, 'completed', 'Frozen display inputs preserve upstream scientific outcomes, including unavailable or failed analysis', tuple(refs))
    _write_json(context.output / 'result.json', result_to_dict(result))
    return (SavedResult(context.output, result), metadata)

def pages(values, metadata, settings):
    if metadata['view'] == 'support':
        output = []
        for group in values.group.drop_duplicates():
            entries = values.loc[values.group.eq(group), 'entry_id'].tolist()
            for start in range(0, len(entries), settings['state_page_size']):
                output.append({'view': 'support', 'group': group, 'entries': entries[start:start + settings['state_page_size']], 'part': start // settings['state_page_size'] + 1})
        return output
    state_ids = [state['state_id'] for state in metadata['states']]
    order = settings['state_display_order'] or state_ids
    if set(order) != set(state_ids):
        raise ValueError('state_display_order must include every accepted state exactly once')
    if set(settings['state_display_names']) - set(state_ids):
        raise ValueError('Display labels name unknown accepted states')
    features = values[['measurement', 'source_table', 'representation']].drop_duplicates().to_dict('records')
    size = settings['state_feature_page_size']
    return [{'view': 'profiles', 'features': features[start:start + size], 'states': order, 'part': start // size + 1} for start in range(0, len(features), size)]

def load(ctx):
    cached = ctx.cache('_behaviour_display_sources')
    if cached is not None:
        return cached
    metadata = ctx.pipeline_metadata('state-display')
    values = ctx.table('state_display_values.json')
    statistics = ctx.table('state_display_statistics.json')
    settings = {name: ctx.option(name) for name in DEFAULTS}
    return {'values': values, 'statistics': statistics, 'metadata': metadata, 'pages': pages(values, metadata, settings)}
STANDALONE = ''

def build(ctx):
    from pymicroglia.figure_tables.prepared import PreparedPage as FigureResult, Drawing
    from pymicroglia.visualisation.text import figure_text
    from pymicroglia.visualisation.panels import behaviour_profiles
    data = load(ctx)
    index = ctx.option('evidence_page')
    if isinstance(index, bool) or not isinstance(index, int) or (not 1 <= index <= len(data['pages'])):
        raise ValueError('evidence_page is outside the saved state display pages')
    page = data['pages'][index - 1]
    metadata = data['metadata']
    values = data['values']
    if page['view'] == 'support':
        values = values.loc[values.entry_id.isin(page['entries'])].copy()
    else:
        wanted = {(item['measurement'], item['source_table'], item['representation']) for item in page['features']}
        values = values.loc[[tuple(row) in wanted for row in values[['measurement', 'source_table', 'representation']].itertuples(index=False, name=None)]].copy()
    note = 'Candidate evidence and the final decision remain separate. A zero tabulated probability means the table endpoint. No biological state count or activation label is established.' if page['view'] == 'support' else 'Points: saved medians. Lines: observed interquartile ranges, not confidence intervals. Missing values are excluded from profiles and counted. Features defined these states; profiles are descriptive.'
    title = 'Evidence for a shared state vocabulary' if page['view'] == 'support' else 'Observed profiles of the accepted states'
    wording = figure_text(ctx.run, ctx.spec.slug, explicit=ctx.text, item=ctx.item, title=title, footnote=note, claim=CLAIMS[page['view']])
    settings = {**page, 'metadata': metadata, 'state_display_names': ctx.option('state_display_names'), 'title': wording.title + ('\n' + wording.subtitle if wording.subtitle else ''), 'footnote': wording.footnote + ('\n' + wording.note if wording.note else ''), 'claim': wording.claim, 'grammar': GRAMMAR}
    from pymicroglia.figure_tables import state_profile_display
    prepared=state_profile_display.prepare(values,settings)
    drawing=Drawing(behaviour_profiles.draw,(prepared,settings),{})
    names=('criteria','outcomes') if page['view']=='support' else ('values','membership')
    views={name:(Drawing(behaviour_profiles.draw,(prepared,settings),{'selected_view':name}),values) for name in names}
    return FigureResult(views=views,wording=wording, drawing=drawing, heading=wording.claim, figure_data=values, auxiliary={'statistics.csv': data['statistics'], 'display.csv': pd.DataFrame([{'settings_json': json.dumps(_json_value(settings))}])}, producer_sources={'state_profile_display.py':Path(state_profile_display.__file__),'behaviour_profiles.py': Path(behaviour_profiles.__file__), 'behaviour_figures.py': Path(__file__), ctx.spec.source.name: ctx.spec.source}, readme=f'# {wording.title}\n\n{wording.claim}\n\n{note}\n\nThis is a display-only snapshot of saved results, including their actual success, refusal or unavailable status. No model is fitted or accepted by the renderer.\n')

def _produce(context, view):
    from pymicroglia.visualisation.panels import behaviour_profiles
    from pymicroglia.pipelines._saved_figures import draw_batch
    settings, text = options(context.presentation)
    values, statistics, metadata = support_data(context) if view == 'support' else profile_data(context)
    display_pages = pages(values, metadata, settings)
    original_sources = {Path(__file__), Path(behaviour_profiles.__file__), *(saved.artifact(ref.name) for saved in context.dependencies.values() for ref in saved.outcome.artifacts)}
    saved_snapshot, metadata = snapshot(context, values, statistics, metadata)
    data = {'values': values, 'statistics': statistics, 'metadata': metadata, 'pages': display_pages}
    display_context = replace(context, dependencies={'state-display': saved_snapshot})
    completed = draw_batch(display_context, slug=SLUGS[view], options=[{**settings, 'evidence_page': index} for index in range(1, len(display_pages) + 1)], aliases=ALIASES, data=data, cache_name='_behaviour_display_sources', sources=original_sources, text=text, claim=CLAIMS[view], grammar=GRAMMAR)
    entries = []
    for page, master in zip(display_pages, completed['masters']):
        if view == 'support':
            selected = values.loc[values.entry_id.isin(page['entries'])]
        else:
            wanted = {(item['measurement'], item['source_table'], item['representation']) for item in page['features']}
            selected = values.loc[[tuple(row) in wanted for row in values[['measurement', 'source_table', 'representation']].itertuples(index=False, name=None)]]
        entries.extend(({**row, 'master': master} for row in selected.to_dict('records')))
    write_table(context.output / 'entries.csv', pd.DataFrame(entries, columns=[*VALUE_COLUMNS, 'master']))
    _write_json(context.output / 'state_display_manifest.json', {'schema_version': 1, 'view': view, 'settings': settings, 'pages': [{**page, 'master': master} for page, master in zip(display_pages, completed['masters'])], 'source_outcomes': metadata['source_outcomes'], 'analysis_recomputed': False, 'check_output': completed['check_output'], 'registration_output': completed['registration_output']})
    refs = tuple((ArtifactRef(path.name, path.name, file_hash(path), context.scientific_id) for path in sorted(output_files(context.output)) if path.is_file() and (not path.name.startswith('.'))))
    return StepResult(context.step.name, context.scientific_id, 'completed', 'Rendered saved support diagnostics with original outcome statuses' if view == 'support' else 'Rendered original-unit profiles of the accepted state vocabulary', refs)

def produce_support(context):
    return _produce(context, 'support')

def produce_profiles(context):
    return _produce(context, 'profiles')
