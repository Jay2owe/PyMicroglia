from pymicroglia.figure_tables.state_card_display import prepare as prepare_state_card
"""Exact real-member provenance in state cards and optional source imagery."""
from tests.panel_helpers import panel_canvas
from pymicroglia._results import read_document
from pathlib import Path
from types import SimpleNamespace
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import tifffile
import pymicroglia.pipelines.behaviour.card_images as behaviour_card_images
from pymicroglia.pipelines._contracts import Settings, content_id
from pymicroglia.pipelines._screening import file_hash, _write_json

@pytest.fixture(autouse=True)
def unchanged_style():
    with plt.rc_context():
        yield
    plt.close('all')

def image_fixture(tmp_path):
    run = tmp_path / 'run'
    tables = run / 'pooled/tables'
    tables.mkdir(parents=True)
    rows, movies = ([], [])
    for movie, offset in [('a', 0), ('b', 10000)]:
        raw = np.arange(4 * 12 * 12, dtype=np.float32).reshape(4, 12, 12) + offset
        labels = np.zeros((3, 12, 12), dtype=np.uint16)
        labels[:, 3:6, 4:7] = 7
        declared = {}
        for kind, values in [('raw', raw), ('labels', labels)]:
            path = run / (movie + '-' + kind + '.tif')
            tifffile.imwrite(path, values, photometric='minisblack')
            declared[kind] = {'path': path.name, 'sha256': file_hash(path)}
        movies.append({'stem': movie, 'provenance': {'inputs': declared}})
        rows.extend(({'stem': movie, 'identity': 7, 'frame_index': index, 'hours': 50.0 + index * 0.5, 'imagej_frame': index + 1, 'source_imagej_frame': index + 2} for index in range(3)))
    path = tables / 'cell_frame.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    _write_json(run / 'manifest.json', {'synthetic': True, 'biological_result': False, 'movies': movies})
    source = file_hash(run / 'manifest.json')
    inputs = SimpleNamespace(source_run=source, table_hashes=Settings({'cell_frame': file_hash(path)}))
    context = SimpleNamespace(request=SimpleNamespace(inputs=inputs), table_paths={'cell_frame': path})
    examples = [{'source_run': source, 'movie': movie, 'identity': 7, 'frame_index': 1, 'hours': 50.5, 'observation_id': 'controlled-' + movie, 'state_id': 'controlled-state', 'model_id': 'controlled-model', 'example_id': content_id({'movie': movie})} for movie in ['a', 'b']]
    return (context, examples, run)

def test_exact_example_images_keep_movie_local_cell_and_source_frame(tmp_path):
    context, examples, _ = image_fixture(tmp_path)
    archive, metadata, sources = behaviour_card_images.prepare_examples(context, examples, {'state_card_images': True}, tmp_path / 'display')
    rows = read_document(metadata)['examples']
    assert len(rows) == 2 and all((row['image_status'] == 'available' for row in rows))
    assert sources
    with np.load(archive, allow_pickle=False) as saved:
        for row in rows:
            assert row['tiles'] == [{'hours': 50.5, 'frame_index': 1, 'source_frame_index': 2, 'cell_present': True}]
            top, _, left, _ = row['crop_box']
            expected = 2 * 144 + top * 12 + left + (10000 if row['movie'] == 'b' else 0)
            assert saved[row['archive_key'] + '_raw'][0, 0, 0] == expected
            assert saved[row['archive_key'] + '_mask'].sum() == 9

@pytest.mark.parametrize('change', [{'hours': 50.6}, {'frame_index': 2}])
def test_nearest_image_cannot_substitute_for_the_selected_observation(tmp_path, change):
    context, examples, _ = image_fixture(tmp_path)
    archive, metadata, _ = behaviour_card_images.prepare_examples(context, [{**examples[0], **change}], {'state_card_images': True}, tmp_path / 'display')
    row = read_document(metadata)['examples'][0]
    assert row['image_status'] == 'unavailable' and 'nearest-frame substitution refused' in row['image_reason']
    with np.load(archive, allow_pickle=False) as saved:
        assert saved.files == []

def test_unavailable_images_keep_representative_identity_and_other_movies(tmp_path):
    context, examples, run = image_fixture(tmp_path)
    with (run / 'a-raw.tif').open('ab') as stream:
        stream.write(b'changed')
    _, metadata, _ = behaviour_card_images.prepare_examples(context, examples, {'state_card_images': True}, tmp_path / 'display')
    rows = read_document(metadata)['examples']
    assert [row['image_status'] for row in rows] == ['unavailable', 'available']
    assert rows[0]['observation_id'] == examples[0]['observation_id'] and rows[0]['image_reason']

def test_disabled_images_read_no_pixels_and_wrong_source_identity_is_refused(tmp_path, monkeypatch):
    context, examples, _ = image_fixture(tmp_path)

    def forbidden(*a, **k):
        raise AssertionError('Disabled image branch read source pixels')
    monkeypatch.setattr(behaviour_card_images, 'prepare', forbidden)
    _, metadata, _ = behaviour_card_images.prepare_examples(context, examples, {'state_card_images': False}, tmp_path / 'display')
    assert all((row['image_status'] == 'not_requested' for row in read_document(metadata)['examples']))
    with pytest.raises(ValueError, match='different source run'):
        behaviour_card_images.prepare_examples(context, [{**examples[0], 'source_run': 'other'}], {'state_card_images': True}, tmp_path / 'other')

def card_fixture(source='controlled-source'):
    from tests.test_behaviour_durations import sequence, SETTINGS
    from pymicroglia.pipelines.behaviour.durations import statistics
    from pymicroglia.pipelines.behaviour.assignments import describe_states
    from pymicroglia.pipelines.behaviour.options import ObservationKey
    from pymicroglia.pipelines._contracts import CellKey
    observations = []
    inventories = []
    for movie in ['a', 'b']:
        rows, states, cells = sequence([0, 1, 0], [50.0, 50.5, 51.0], movie=movie)
        rows['source_run'] = source
        cells['source_run'] = source
        rows['observation_id'] = [ObservationKey(CellKey(source, movie, 7), row.frame_index, row.hours).record_id for row in rows.itertuples()]
        rows['log_density'] = [-1.0, -2.0, -3.0] if movie == 'a' else [-4.0, -5.0, -6.0]
        rows['max_probability'] = [0.99, 0.98, 0.97] if movie == 'a' else [0.94, 0.93, 0.92]
        observations.append(rows)
        inventories.append(cells)
    observations = pd.concat(observations, ignore_index=True)
    inventories = pd.concat(inventories, ignore_index=True)
    matrix = pd.DataFrame({'custom_length': [2.0, 8.0, 3.0, 2.2, 8.2, 3.2], 'custom_texture': [1.0, 4.0, 1.5, 1.2, np.nan, 1.7]}, index=observations.observation_id)
    definitions = [{'column': 'custom_length', 'table': 'cell_frame', 'unit': 'microns'}, {'column': 'custom_texture', 'table': 'cell_frame', 'unit': 'arb.'}]
    model = {'model_id': 'controlled-model', 'native_state': {'means_': {'array': [[9999.0, 9999.0], [8888.0, 8888.0]]}}, 'component_order': [0, 1]}
    states, profiles, representatives = describe_states(model, observations, {'raw': matrix}, definitions)
    time = statistics(observations, states, inventories, SETTINGS)
    traces = []
    for row in observations.to_dict('records'):
        for feature in definitions:
            value = matrix.loc[row['observation_id'], feature['column']]
            traces.append({**{key: row[key] for key in ['source_run', 'movie', 'identity', 'frame_index', 'hours', 'observation_id']}, 'measurement': feature['column'], 'trace_id': row['movie'] + '-' + feature['column'], 'source_position': row['frame_index'], 'raw_value': value, 'processed_value': value, 'raw_valid': np.isfinite(value), 'processed_valid': np.isfinite(value), 'within_range': True, 'time_kind': 'finite'})
    return {**time, 'assignments': observations, 'state_definitions': states, 'state_profiles': profiles, 'representatives': representatives, 'source_traces': pd.DataFrame(traces), 'assignment_provenance': {'model_id': 'controlled-model', 'status_counts': {'assigned': 6}}, 'time_provenance': {'model_id': 'controlled-model'}}

def card_data(data, low=False):
    import pymicroglia.pipelines.behaviour.card_figures as cards
    settings, _ = cards.options(Settings({'state_cards': {'state_card_low_membership': low}}))
    data['examples'], data['selection_inventory'] = cards.select_examples(data, settings)
    data['pages'] = cards.pages(data, settings)
    return settings

def drawing(data, page, images, archive):
    from pymicroglia.visualisation.figures import load as _figure_schema
    import pymicroglia.pipelines.behaviour.card_figures as cards
    _figure_schema()
    from pymicroglia.visualisation.panels import behaviour_cards
    frame, ledger = cards.values(data, page)
    settings = {**page, 'state_definitions': data['state_definitions'].to_dict('records'), 'state_display_names': {}, 'images': images, 'title': 'Controlled observed state card', 'footnote': 'Saved profiles and exact examples; no model centre is an observed cell.'}
    return (*behaviour_cards.draw(prepare_state_card(frame,settings,archive), settings, canvas=panel_canvas()), frame)

def test_examples_resolve_exact_saved_members_and_unique_bouts_without_model_centres():
    import pymicroglia.pipelines.behaviour.card_figures as cards
    data = card_fixture()
    settings = card_data(data, True)
    assert len(data['examples']) == 4
    members = data['assignments'].set_index('observation_id')
    for example in data['examples']:
        member = members.loc[example['observation_id']]
        assert member.state_id == example['state_id'] and member.movie == example['movie']
        assert member.frame_index == example['frame_index'] and member.hours == example['hours']
        assert example['bout_id'] in set(data['bouts'].bout_id)
        assert example['example_role'] in {'representative', 'low_membership_assigned'}
    repeated, _ = cards.select_examples(data, settings)
    assert repeated == data['examples']
    primary = [example for example in repeated if example['example_role'] == 'representative']
    assert all((example['movie'] == 'a' for example in primary))
    assert all((example['max_probability'] < 1 for example in repeated))

def test_card_profiles_and_population_occupancy_do_not_follow_example_limits():
    import pymicroglia.pipelines.behaviour.card_figures as cards
    data = card_fixture()
    settings = card_data(data)
    first = [cards.values(data, page)[0] for page in data['pages']]
    card_data(data, True)
    second = [cards.values(data, page)[0] for page in data['pages']]
    for a, b in zip(first, second):
        fields = ['model_id', 'state_id', 'source_run', 'movie', 'identity', 'fraction_observed', 'fraction_assigned', 'state_hours']
        pd.testing.assert_frame_equal(a.loc[a.kind.eq('population_occupancy'), fields].reset_index(drop=True), b.loc[b.kind.eq('population_occupancy'), fields].reset_index(drop=True))
    settings['state_card_features_per_page'] = 1
    assert len(cards.pages(data, settings)) == 4
    with pytest.raises(ValueError, match='unavailable saved feature'):
        cards.pages(data, {**settings, 'state_card_features': ['missing_feature']})

def test_numerical_card_retains_missing_images_and_original_feature_values():
    data = card_fixture()
    card_data(data)
    images = [{**row, 'image_status': 'unavailable', 'image_reason': 'Controlled missing original image'} for row in data['examples']]
    figure, axes, frame = drawing(data, data['pages'][0], images, {})
    primary = data['examples'][0]
    offsets = axes['trace_' + primary['example_id'] + '_0'].collections[0].get_offsets()
    np.testing.assert_allclose(offsets[:, 0], [50.0, 50.5, 51.0])
    np.testing.assert_allclose(offsets[:, 1], [2.0, 8.0, 3.0])
    assert any(('Controlled missing original image' in text.get_text() for text in axes['image_' + primary['example_id']].texts))
    assert len(frame.loc[frame.kind.eq('population_occupancy')]) == 2
    assert all((float(collection.get_offsets()[0, 0]) < 100 for name, axis in axes.items() if name.startswith('profile_') for collection in axis.collections))
    plt.close(figure)

def test_card_draws_verified_frozen_image_pixels_for_the_exact_real_example(tmp_path):
    context, _, _ = image_fixture(tmp_path)
    data = card_fixture(context.request.inputs.source_run)
    card_data(data)
    archive, metadata, _ = behaviour_card_images.prepare_examples(context, data['examples'], {'state_card_images': True}, tmp_path / 'display')
    images = read_document(metadata)['examples']
    assert all((row['image_status'] == 'available' for row in images))
    with np.load(archive, allow_pickle=False) as saved:
        figure, axes, _ = drawing(data, data['pages'][0], images, saved)
        example = data['examples'][0]
        record = next((row for row in images if row['example_id'] == example['example_id']))
        np.testing.assert_array_equal(axes['image_' + example['example_id']].images[0].get_array(), saved[record['archive_key'] + '_display'][0])
    plt.close(figure)

def test_mismatched_example_identity_or_state_is_refused():
    import pymicroglia.pipelines.behaviour.card_figures as cards
    data = card_fixture()
    settings = card_data(data)
    data['representatives'].loc[0, 'movie'] = 'b'
    with pytest.raises(ValueError, match='full saved observation identity'):
        cards.select_examples(data, settings)

def test_trace_does_not_connect_across_a_saved_unsupported_interval():
    data = card_fixture()
    card_data(data)
    example = data['examples'][0]
    data['steps'].loc[data['steps'].movie.eq(example['movie']), 'valid_observed_interval'] = False
    images = [{**row, 'image_status': 'unavailable', 'image_reason': 'Controlled missing image'} for row in data['examples']]
    figure, axes, _ = drawing(data, data['pages'][0], images, {})
    axis = axes['trace_' + example['example_id'] + '_0']
    assert len(axis.lines) == 1 and axis.lines[0].get_linestyle() == ':'
    assert len(axis.collections[0].get_offsets()) == 3
    plt.close(figure)
