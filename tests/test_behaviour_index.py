"""Navigation follows exact saved vocabularies and members, including refusals."""
from pymicroglia._results import read_document
from dataclasses import replace
import copy
import json
import shutil
from types import SimpleNamespace
import pandas as pd
import pytest
import pymicroglia.pipelines.behaviour.index as index
from pymicroglia.pipelines._contracts import Settings, StepSpec
from pymicroglia.pipelines._runner import ExecutionContext
from pymicroglia.pipelines._screening import file_hash, write_table, _write_json
from tests.test_behaviour_cards import card_fixture, card_data
from tests.test_relationship_index import saved

def fixture(tmp_path):
    data = card_fixture('source <literal>')
    card_data(data, True)
    cells = data['cell_statistics'][['source_run', 'movie', 'identity']].to_dict('records')
    request = {'inputs': {'source_run': 'source <literal>', 'cells': cells}, 'request': {'features': ['custom_length', 'custom_texture']}}
    model = 'controlled-model'
    decision = {'status': 'accepted', 'accepted_model_id': model, 'decision_id': 'controlled-decision', 'reason': 'Controlled saved support'}
    deps = {'feature-inputs': saved(tmp_path, 'feature-inputs', {'provenance': {'resolved_request': request}}), 'state-support': saved(tmp_path, 'state-support', {'support_decision': decision}), 'state-assignments': saved(tmp_path, 'state-assignments', {**{name: data[name] for name in ['assignments', 'state_definitions', 'state_profiles']}, 'provenance': {'model_id': model}}), 'durations-and-switches': saved(tmp_path, 'durations-and-switches', {**{name: data[name] for name in ['cell_statistics', 'occupancy', 'bouts']}, 'provenance': {'model_id': model}})}
    units = [{'unit_id': 'unit ' + row['movie'], 'model_id': model, 'source_run': row['source_run'], 'level': 'biological_sample', 'sample': row['movie'], 'sample_confirmed': True, 'condition': 'control', 'independent_of_state_choice': True, 'cells': 1, 'recordings': 1, 'members': [row]} for row in cells]
    deps['state-sample-comparisons'] = saved(tmp_path, 'state-sample-comparisons', {'unit_inventory': pd.DataFrame(units), 'unit_metrics': pd.DataFrame([{'unit_id': row['unit_id'], 'question_id': 'occupancy', 'value': 0.5} for row in units]), 'comparisons': pd.DataFrame(columns=['comparison_id']), 'provenance': {'model_id': model, 'duration_id': deps['durations-and-switches'].outcome.scientific_id}})
    pages = []
    entries = []
    files = {}
    for i, cell in enumerate(cells):
        master = 'timeline ' + str(i) + '.txt'
        files[master] = 'Navigation test stand-in; no chart generated'
        pages.append({'page': i + 1, 'cells': [cell], 'master': master})
        entries.append({**cell, 'model_id': model, 'master': master, 'source_scientific_id': deps['durations-and-switches'].outcome.scientific_id})
    deps['state-timelines'] = saved(tmp_path, 'state-timelines', {**files, 'state_timelines_manifest.json': {'model_id': model, 'pages': pages}, 'entries.json': pd.DataFrame(entries)})
    pages = []
    entries = []
    files = {}
    for i, page in enumerate(data['pages']):
        master = 'card ' + str(i) + '.txt'
        files[master] = 'Navigation test stand-in; no chart generated'
        pages.append({**page, 'master': master})
        entries.append({**page, 'master': master, 'model_id': model, 'assignment_id': deps['state-assignments'].outcome.scientific_id, 'examples': [row for row in data['examples'] if row['example_id'] in page['example_ids']]})
    images = [{**row, 'image_status': 'unavailable', 'image_reason': 'Original image not included in controlled navigation fixture', 'tiles': []} for row in data['examples']]
    deps['state-report-cards'] = saved(tmp_path, 'state-report-cards', {**files, 'state_cards_manifest.json': {'model_id': model, 'decision_id': decision['decision_id'], 'pages': pages, 'images': {'examples': images}}, 'entries.json': pd.DataFrame(entries), 'state_example_tiles.npz': 'Navigation test stand-in; no pixels generated', 'state_example_images.json': {'examples': images}})
    return (deps, request)

def context(tmp_path, deps, request):
    return ExecutionContext(StepSpec('linked-results-index', 'index', tuple(deps), kind='render'), SimpleNamespace(as_dict=lambda: request), Settings(), {}, deps, None, tmp_path / 'report', 'index-science', Settings({'report': {'title': 'Saved <script>alert(1)</script> & states'}}), 'display')

def test_moved_report_retains_full_model_cell_bout_image_and_sample_targets(tmp_path, monkeypatch):
    import pymicroglia.states.behaviour_models as behaviour_models
    import pymicroglia.states.behaviour_support as behaviour_support
    import pymicroglia.pipelines.behaviour.durations as behaviour_durations, pymicroglia.pipelines.behaviour.samples as behaviour_samples, pymicroglia.pipelines.behaviour.card_images as behaviour_card_images
    deps, request = fixture(tmp_path)

    def forbidden(*a, **k):
        raise AssertionError('Navigation attempted scientific analysis or source pixels')
    for owner, name in [(behaviour_models, 'fit_candidate'), (behaviour_models, 'predict_candidate'), (behaviour_support, 'measure'), (behaviour_durations, 'statistics'), (behaviour_samples, 'aggregate'), (behaviour_card_images, 'prepare_examples')]:
        monkeypatch.setattr(owner, name, forbidden)
    before = {str(item.artifact(ref.name)): file_hash(item.artifact(ref.name)) for item in deps.values() for ref in item.outcome.artifacts}
    ctx = context(tmp_path, deps, request)
    outcome = index.produce(ctx)
    assert outcome.status == 'completed'
    data = read_document(ctx.output / 'navigation.json')
    assert (len(data['states']), len(data['cells']), len(data['bouts']), len(data['examples']), len(data['samples'])) == (2, 2, 6, 4, 2)
    assert len({row['id'] for row in data['cells']}) == 2
    assert all((row['pages'] and row['bout'] and row['cell'] for row in data['examples']))
    assert all((row['pages'] for row in data['bouts']))
    assert all((file_hash(path) == fingerprint for path, fingerprint in before.items()))
    text = (ctx.output / 'index.html').read_text()
    assert '<script>alert(1)</script>' not in text and '&lt;script&gt;' in text
    moved = tmp_path / 'moved report'
    shutil.copytree(ctx.output, moved)
    assert index.validate_links(moved)['links'] > 100
    assert index.cell_id(request['inputs']['cells'][0], 'another-model') != data['cells'][0]['id']

@pytest.mark.parametrize('status', ['no_supported_states', 'inconclusive', 'unavailable', 'failed'])
def test_no_model_retains_real_decision_or_failure_without_state_targets(tmp_path, status):
    cells = [{'source_run': 'source', 'movie': 'a', 'identity': 7}]
    request = {'inputs': {'source_run': 'source', 'cells': cells}, 'request': {}}
    files = {'support_decision': {'status': status, 'accepted_model_id': None, 'reason': 'Controlled absence of accepted states'}} if status not in {'unavailable', 'failed'} else {}
    deps = {'state-support': saved(tmp_path, 'state-support', files, status=status if status in {'unavailable', 'failed'} else 'completed', reason='Controlled recorded outcome'), 'state-report-cards': saved(tmp_path, 'state-report-cards', {}, status='unavailable' if status in {'unavailable', 'failed'} else 'skipped-empty', reason='No accepted model')}
    ctx = context(tmp_path, deps, request)
    index.produce(ctx)
    data = read_document(ctx.output / 'navigation.json')
    assert not data['states'] and (not data['examples']) and (not data['bouts']) and (not data['pages'])
    assert len(data['cells']) == 1 and index.validate_links(ctx.output)['links'] > 10
    if status in {'unavailable', 'failed'}:
        assert 'remain unavailable or failed' in (ctx.output / 'index.html').read_text()
    else:
        assert data['decision']['status'] == status

@pytest.mark.parametrize('corruption', ['cell', 'model', 'bout', 'frame', 'page'])
def test_real_but_wrong_member_or_page_is_refused(tmp_path, corruption):
    deps, request = fixture(tmp_path)
    ctx = context(tmp_path, deps, request)
    index.produce(ctx)
    data = read_document(ctx.output / 'navigation.json')
    inventory = copy.deepcopy(data['inputs'])
    step = 'state-report-cards'
    name = 'state_cards_manifest.json' if corruption == 'frame' else 'entries.json'
    path = ctx.output / inventory[step]['artifacts'][name]['path']
    if corruption == 'frame':
        manifest = read_document(path)
        image = manifest['images']['examples'][0]
        image['frame_index'] += 1
        _write_json(path, manifest)
    else:
        from pymicroglia.pipelines._screening import read_table
        rows = read_table(path)
        if corruption == 'model':
            rows.loc[0, 'model_id'] = 'another-model'
        elif corruption == 'page':
            rows.loc[0, 'master'] = rows.master.iloc[1]
        else:
            example = rows.examples.iloc[0][0]
            if corruption == 'cell':
                example['movie'] = 'b' if example['movie'] == 'a' else 'a'
            else:
                example['bout_id'] = next((row['bout_id'] for row in data['bouts'] if example['observation_id'] not in row['observation_ids']))
        path = write_table(path, rows)
    inventory[step]['artifacts'][name]['sha256'] = file_hash(path)
    with pytest.raises(ValueError):
        index.build(ctx.output, inventory, request)

def test_changed_copied_evidence_is_refused(tmp_path):
    deps, request = fixture(tmp_path)
    ctx = context(tmp_path, deps, request)
    index.produce(ctx)
    inventory = read_document(ctx.output / 'navigation.json')['inputs']
    path = ctx.output / inventory['state-assignments']['artifacts']['state_definitions']['path']
    path.write_text('{}')
    with pytest.raises(ValueError, match='Missing or changed'):
        index.build(ctx.output, inventory, request)
