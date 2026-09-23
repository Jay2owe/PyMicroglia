from pymicroglia.figure_tables.timeline_display import prepare as prepare_timeline
"""Original-time geometry and complete identities in saved categorical timelines."""
from tests.panel_helpers import panel_canvas
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np
import pandas as pd
import pytest
import pymicroglia.pipelines.behaviour.timeline_figures as timeline
from pymicroglia.pipelines.behaviour.options import RECIPE
from pymicroglia.pipelines._contracts import Settings, content_id
from tests.test_behaviour_durations import sequence, SETTINGS
from pymicroglia.pipelines.behaviour.durations import statistics

@pytest.fixture(autouse=True)
def plot_defaults():
    with plt.rc_context():
        yield
    plt.close('all')

def fixture():
    cases = [('record-a', [0, 0, 1, 1], [50.0, 50.5, 51.0, 52.0], None, 'controlled-source'), ('record-b', [0, None, None, None, 1], [80.0, 80.5, 81.0, 81.5, 82.0], None, 'controlled-source'), ('record-gap', [0, 0, 1, 1], [100.0, 100.5, 102.0, 102.5], [0, 1, 4, 5], 'controlled-source'), ('record-point', [0], [200.0], None, 'controlled-source'), ('record-empty', [], [], None, 'controlled-source'), ('record-clock', [0, 0, 1], [250.0, 251.0, 250.5], None, 'controlled-source'), ('record-a', [0, 1], [350.0, 351.0], None, 'other-source')]
    collected = []
    assignments, inventories = ([], [])
    for movie, labels, hours, frames, source in cases:
        rows, states, cells = sequence(labels, hours, frames, movie)
        rows['source_run'] = source
        cells['source_run'] = source
        rows['observation_id'] = [content_id({key: row[key] for key in ['source_run', 'movie', 'identity', 'frame_index', 'hours']}) for row in rows.to_dict('records')]
        if movie == 'record-b':
            rows.loc[rows.index[1:4], 'status'] = ['missing_features', 'ambiguous', 'outside_training_distribution']
        rows['score_type'] = 'native_gaussian_mixture_component_membership'
        rows['score_calibration'] = 'uncalibrated_model_conditional'
        rows['max_probability'] = [0.55 if status == 'ambiguous' else 0.95 for status in rows.status]
        tables = statistics(rows, states, cells, SETTINGS)
        inventory = tables['cell_statistics'].copy()
        for status in ['missing_features', 'ambiguous', 'outside_training_distribution', 'invalid_prediction']:
            inventory[status + '_observations'] = int(rows.status.eq(status).sum())
        collected.append(tables)
        assignments.append(rows)
        inventories.append(inventory)
    states['label'] = ['State 1', 'State 2']
    return {'timeline_exposures': pd.concat([item['exposures'] for item in collected], ignore_index=True), 'timeline_cells': pd.concat([item['cell_statistics'] for item in collected], ignore_index=True), 'timeline_inventory': pd.concat(inventories, ignore_index=True), 'timeline_assignments': pd.concat(assignments, ignore_index=True), 'timeline_states': states, 'time_provenance': {'model_id': 'controlled-model', 'interval_rule': 'adjacent_midpoint'}}

def drawing(data, page, settings):
    from pymicroglia.visualisation.figures import load as _figure_schema
    _figure_schema()
    from pymicroglia.visualisation.panels import behaviour_timelines
    values, stats = timeline.values(data, page)
    settings = {**page, 'state_definitions': data['timeline_states'].to_dict('records'), 'state_display_names': settings['state_display_names'], 'title': 'Controlled original recording clocks', 'footnote': 'Saved interval support, no inferred switches'}
    figure, axes = behaviour_timelines.draw(prepare_timeline(values,settings), settings, canvas=panel_canvas())
    return (figure, axes, values)

def test_every_full_cell_appears_once_with_no_display_selection_or_clock_alignment():
    data = fixture()
    settings, _ = timeline.options(Settings({'state_timelines': {'state_cells_per_page': 3}}))
    pages = timeline.pages(data, settings)
    assert [len(page['cells']) for page in pages] == [3, 3, 1]
    shown = [key for page in pages for key in page['cells']]
    assert len({content_id(key) for key in shown}) == 7 and sum((key['movie'] == 'record-a' for key in shown)) == 2
    assert all((key['identity'] == 7 for key in shown))
    reversed_pages = timeline.pages(data, {**settings, 'state_cell_order': shown[::-1]})
    assert [key for page in reversed_pages for key in page['cells']] == shown[::-1]
    with pytest.raises(ValueError, match='every requested'):
        timeline.pages(data, {**settings, 'state_cell_order': shown[:-1]})
    with pytest.raises(ValueError, match='full source_run'):
        timeline.options(Settings({'state_timelines': {'state_cell_order': [{'movie': 'record-a', 'identity': 7}]}}))

def test_irregular_intervals_and_gaps_use_saved_physical_rectangle_bounds():
    data = fixture()
    settings, _ = timeline.options(Settings({'state_timelines': {'state_cells_per_page': 12}}))
    page = timeline.pages(data, settings)[0]
    figure, axes, values = drawing(data, page, settings)
    for index, key in enumerate(page['cells']):
        expected = data['timeline_exposures']
        for name in timeline.KEYS:
            expected = expected.loc[expected[name].eq(key[name])]
        rectangles = axes['cell_' + str(index)].patches
        assert [(patch.get_x(), patch.get_width()) for patch in rectangles] == list(zip(expected.start_hours, expected.duration_hours))
        if key['movie'] == 'record-gap':
            assert any((patch.get_x() == 100.5 and patch.get_width() == 1.5 and (patch.get_linestyle() == ':') for patch in rectangles))
        if key['movie'] == 'record-a':
            assert axes['cell_' + str(index)].get_xlim()[0] > (349 if key['source_run'] == 'other-source' else 49)
    assert len(values.loc[values.kind.eq('cell')]) == 7
    plt.close(figure)

def test_typed_unknowns_and_point_observations_are_not_confidence_colours():
    data = fixture()
    settings, _ = timeline.options(Settings({'state_timelines': {'state_cells_per_page': 12}}))
    page = timeline.pages(data, settings)[0]
    figure, axes, _ = drawing(data, page, settings)
    index = next((index for index, key in enumerate(page['cells']) if key['movie'] == 'record-b'))
    axis = axes['cell_' + str(index)]
    assert {'..', '//', 'xx'} <= {patch.get_hatch() for patch in axis.patches}
    assert {'x', 'o', 's'} <= {line.get_marker() for line in axis.lines}
    assert not axis.collections
    point_index = next((index for index, key in enumerate(page['cells']) if key['movie'] == 'record-point'))
    point = axes['cell_' + str(point_index)]
    assert not point.patches and any((np.array_equal(line.get_xdata(), [200.0, 200.0]) for line in point.lines))
    assert any(('duration unavailable' in text.get_text() for text in point.texts))
    plt.close(figure)

def test_nonincreasing_clocks_and_empty_cells_keep_visible_unavailable_rows():
    data = fixture()
    settings, _ = timeline.options(Settings({'state_timelines': {'state_cells_per_page': 12}}))
    page = timeline.pages(data, settings)[0]
    figure, axes, _ = drawing(data, page, settings)
    for movie, message in [('record-clock', 'Nonincreasing recording clock'), ('record-empty', 'No usable observations')]:
        index = next((index for index, key in enumerate(page['cells']) if key['movie'] == movie))
        axis = axes['cell_' + str(index)]
        assert not axis.patches and (not axis.lines) and any((message in text.get_text() for text in axis.texts))
    plt.close(figure)

def test_optional_native_membership_preserves_saved_values_and_rejects_other_score_types():
    data = fixture()
    settings, _ = timeline.options(Settings({'state_timelines': {'state_show_membership': True}}))
    page = timeline.pages(data, settings)[0]
    figure, axes, _ = drawing(data, page, settings)
    index = next((index for index, key in enumerate(page['cells']) if key['movie'] == 'record-b'))
    offsets = axes['cell_' + str(index)].collections[0].get_offsets()
    np.testing.assert_allclose(offsets[:, 0], [80.0, 80.5, 81.0, 81.5, 82.0])
    np.testing.assert_allclose(offsets[:, 1], -0.35 + 0.5 * np.array([0.95, 0.95, 0.55, 0.95, 0.95]))
    plt.close(figure)
    data['timeline_assignments']['score_type'] = 'uncalibrated_distance'
    with pytest.raises(ValueError, match='score type'):
        timeline.pages(data, settings)

def test_extended_state_colours_are_unique_stable_and_shared_with_profiles():
    from pymicroglia.visualisation.figures import load as _figure_schema
    _figure_schema()
    from pymicroglia.visualisation.panels import behaviour_profiles, behaviour_timelines
    colours = [to_hex(behaviour_profiles.state_colour(index), keep_alpha=True) for index in range(40)]
    assert len(set(colours)) == 40 and colours[:6] == [to_hex(colour, keep_alpha=True) for colour in behaviour_profiles.COLOURS]
    assert behaviour_timelines.state_colour is behaviour_profiles.state_colour
    assert behaviour_timelines._hatch(6) != behaviour_timelines._hatch(0)

def test_timeline_recipe_requires_saved_time_support_and_the_accepted_model():
    step = next((step for step in RECIPE.steps if step.name == 'state-timelines'))
    assert 'durations-and-switches' in step.prerequisites and step.requires_selected_rows and (step.selection == 'state-support:accepted-model')

def test_no_supported_model_saves_timeline_skip_without_assignment_or_rendering(tmp_path):
    from pymicroglia.pipelines._contracts import StepResult, SelectionRecord
    from pymicroglia.pipelines._runner import Producer, run_pipeline

    def complete(context):
        selection = (SelectionRecord('accepted-model', context.scientific_id, Settings({'reason': 'Controlled complete refusal'}), ()),) if context.step.name == 'state-support' else ()
        return StepResult(context.step.name, context.scientific_id, 'completed', 'Controlled saved prerequisite', selections=selection)

    def forbidden(context):
        raise AssertionError('An unsupported model attempted assignment, timing or rendering')
    registry = {step.producer: Producer(forbidden) for step in RECIPE.steps}
    for name in ['behaviour-design', 'feature-inputs', 'candidate-models', 'state-support']:
        registry[name] = Producer(complete)
    result = run_pipeline(RECIPE, registry, request=Settings(), scientific_settings={}, output=tmp_path / 'pipeline', only=('state-timelines',))
    assert result.successful
    for name in ['state-assignments', 'durations-and-switches', 'state-timelines']:
        assert result.results[name].outcome.status == 'skipped-empty' and result.results[name].outcome.reason
