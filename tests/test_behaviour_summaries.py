"""Saved-value and denominator fidelity in behaviour time/sample figures."""
from tests.panel_helpers import panel_canvas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import pymicroglia.pipelines.behaviour.summary_figures as summary
import pymicroglia.pipelines.behaviour.samples as behaviour_samples, pymicroglia.pipelines.behaviour.durations as behaviour_durations
from pymicroglia.pipelines._contracts import Settings
from tests.test_behaviour_durations import sequence, SETTINGS
from tests.test_behaviour_samples import times, concatenate

@pytest.fixture(autouse=True)
def unchanged_style():
    with plt.rc_context():
        yield
    plt.close('all')

def fixture():
    observations, states, inventory = sequence(movie='a')
    first = behaviour_durations.statistics(observations, states, inventory, SETTINGS)
    tables = concatenate(first, times([0, 0, None, 1, 1], [0.0, 0.5, 1.0, 1.5, 2.0], 'b', 'b', 'treated', role='learning'), times([0], [50.0], 'point', 'point', 'treated', confirmed=False), times([0, 1, 1], [80.0, 0.25 + 80, 0.5 + 80], 'fast', 'fast', 'treated'))
    units = behaviour_samples.aggregate(tables, 'mean')
    comparisons, _, _, _ = behaviour_samples.compare_samples(units['unit_metrics'], {'method': 'none'})
    states['label'] = ['State 1', 'State 2']
    return {**tables, **units, 'states': states, 'comparisons': comparisons, 'time_provenance': {'model_id': 'controlled-model'}, 'sample_provenance': {'model_id': 'controlled-model', 'aggregation': 'mean', 'settings': {'interval': {'method': 'independent_bootstrap', 'confidence': 0.95}}}}

def drawing(data, page):
    from pymicroglia.visualisation.figures import load as _figure_schema
    _figure_schema()
    from pymicroglia.visualisation.panels import behaviour_summaries
    values, ledger = summary.values(data, page)
    settings = {**page, 'state_definitions': data['states'].to_dict('records'), 'state_display_names': {}, 'sample_provenance': data['sample_provenance'], 'title': 'Controlled saved summary', 'footnote': summary.NOTES[page['view']]}
    from pymicroglia.figure_tables.state_summary_display import prepare
    return (*behaviour_summaries.draw(prepare(values, settings), settings, canvas=panel_canvas()), values)

def test_pages_preserve_full_cells_unique_bouts_and_complete_sample_questions():
    data = fixture()
    settings, _ = summary.options(Settings({'state_summaries': {'state_summary_cells_per_page': 2, 'state_summary_columns_per_page': 1}}))
    pages = summary.pages(data, settings)
    bouts = pd.concat([summary.values(data, page)[0] for page in pages if page['view'] == 'bouts'])
    ids = bouts.loc[bouts.kind.eq('bout'), 'bout_id']
    assert ids.is_unique and set(ids) == set(data['bouts'].bout_id)
    plotted = pd.concat([summary.values(data, page)[0] for page in pages if page['view'] == 'samples'])
    source = data['unit_metrics']
    rows = plotted.loc[plotted.kind.eq('sample')]
    keys = ['unit_id', 'question_id']
    pd.testing.assert_frame_equal(source[[*keys, 'value', 'numerator_total', 'denominator_total', 'members']].sort_values(keys).reset_index(drop=True), rows[[*keys, 'value', 'numerator_total', 'denominator_total', 'members']].sort_values(keys).reset_index(drop=True), check_dtype=False)
    assert all((len(page['pairs']) == 1 for page in pages if page['view'] == 'transitions'))
    assert {page['spacing'] for page in pages if page['view'] == 'switch_rates'} == {0.25, 0.5}

def test_occupancy_uses_both_saved_denominators_and_keeps_unknown_fraction():
    data = fixture()
    settings, _ = summary.options(Settings())
    page = next((page for page in summary.pages(data, settings) if page['view'] == 'occupancy'))
    figure, axes, _ = drawing(data, page)
    actual = axes['observed'].images[0].get_array()
    np.testing.assert_allclose(actual[0], [0.4, 0.6])
    np.testing.assert_allclose(actual[1], [0.375, 0.375])
    np.testing.assert_allclose(axes['assigned'].images[0].get_array()[1], [0.5, 0.5])
    assert axes['unknown'].images[0].get_array()[1, 0] == 0.25
    assert np.ma.getmaskarray(actual)[3].all()
    plt.close(figure)

def test_transition_counts_keep_self_steps_and_unavailable_probability_rows():
    data = fixture()
    settings, _ = summary.options(Settings())
    page = next((page for page in summary.pages(data, settings) if page['view'] == 'transitions' and page['spacing'] == 0.5))
    figure, axes, _ = drawing(data, page)
    expected = data['transitions'].loc[data['transitions'].movie.eq('a') & data['transitions'].interval_stratum_hours.eq(0.5)]
    probabilities = axes['probabilities'].images[0].get_array()
    for index, pair in enumerate(page['pairs']):
        row = expected.loc[expected.source_state_id.eq(pair['source']) & expected.target_state_id.eq(pair['target'])].iloc[0]
        assert probabilities[0, index] == row.probability
    assert probabilities[0, 0] == 0.5 and probabilities[0, 3] == 0.0
    assert np.ma.getmaskarray(probabilities)[3].all()
    assert any((text.get_text() == '1 / 2' for text in axes['counts'].texts))
    plt.close(figure)

def test_rates_use_saved_spacing_specific_values_with_true_zeros_and_missing_rates():
    data = fixture()
    settings, _ = summary.options(Settings())
    page = next((page for page in summary.pages(data, settings) if page['view'] == 'switch_rates' and page['spacing'] == 0.5))
    figure, axes, _ = drawing(data, page)
    offsets = np.concatenate([item.get_offsets() for item in axes['switch_rate'].collections])
    np.testing.assert_allclose(offsets[:, 0], [4 / 3, 0.0])
    assert sum((text.get_text() == 'Unavailable' for text in axes['switch_rate'].texts)) == 2
    assert any(('2 switches / 1.5 h' == text.get_text() for text in axes['switch_rate'].texts))
    plt.close(figure)

def test_each_unique_bout_is_drawn_once_in_hours_with_both_censoring_boundaries():
    data = fixture()
    settings, _ = summary.options(Settings())
    page = next((page for page in summary.pages(data, settings) if page['view'] == 'bouts'))
    figure, axes, _ = drawing(data, page)
    actual = sorted((float(item.get_offsets()[0, 0]) for item in axes['bouts'].collections))
    np.testing.assert_allclose(actual, sorted(data['bouts'].observed_duration_hours.dropna()))
    assert any(('1 without duration' == text.get_text() for text in axes['bouts'].texts))
    from pymicroglia.visualisation.panels.behaviour_summaries import bout_marker
    assert [bout_marker({'onset_observed': a, 'ending_observed': b}) for a, b in [(True, True), (False, True), (True, False), (False, False)]] == ['o', '<', '>', 'x']
    plt.close(figure)

def test_every_experimental_unit_value_is_drawn_and_model_choice_role_is_explicit():
    data = fixture()
    settings, _ = summary.options(Settings())
    for page in [page for page in summary.pages(data, settings) if page['view'] == 'samples']:
        figure, axes, _ = drawing(data, page)
        for question in page['questions']:
            source = data['unit_metrics'].loc[data['unit_metrics'].question_id.eq(question)].sort_values(['condition', 'sample', 'unit_id'], na_position='last')
            actual = [float(item.get_offsets()[0, 1]) for item in axes[question].collections]
            np.testing.assert_allclose(actual, source.value.dropna())
            assert 'eligible cells/unit' in axes[question].get_title()
        plt.close(figure)

def test_contrast_effects_intervals_and_descriptive_values_remain_distinct():
    data = fixture()
    row = data['comparisons'].iloc[0].copy()
    row['effect'] = 0.4
    row['effect_interval'] = [0.15, 0.7]
    row['descriptive_effect'] = 0.2
    row['p_value'] = 0.01
    row['q_value'] = 0.04
    row['status'] = 'detected_difference'
    row['interval_status'] = 'approximate'
    data['comparisons'] = pd.DataFrame([row])
    settings, _ = summary.options(Settings())
    page = next((page for page in summary.pages(data, settings) if page['view'] == 'contrasts'))
    figure, axes, plotted = drawing(data, page)
    axis = axes[row.comparison_id]
    assert [float(item.get_offsets()[0, 0]) for item in axis.collections] == [0.4, 0.2]
    np.testing.assert_allclose(axis.lines[0].get_xdata(), [0.15, 0.7])
    assert any(('corrected p=0.04' in text.get_text() for text in axis.texts))
    assert any(('95% unadjusted' in text.get_text() for text in axis.texts))
    assert plotted.loc[plotted.kind.eq('contrast'), 'effect_interval'].iloc[0] == [0.15, 0.7]
    plt.close(figure)

def test_no_contrasts_remains_an_explicit_untested_page():
    data = fixture()
    data['comparisons'] = data['comparisons'].iloc[:0]
    settings, _ = summary.options(Settings())
    page = next((page for page in summary.pages(data, settings) if page['view'] == 'contrasts'))
    values, ledger = summary.values(data, page)
    assert ledger.status.tolist() == ['not_requested'] and ledger.p_value.isna().all()
    figure, axes, _ = drawing(data, page)
    assert set(axes) == {'contrast_status'}
    plt.close(figure)

def test_rendering_calls_no_time_counting_sample_aggregation_or_inference(monkeypatch):
    data = fixture()
    settings, _ = summary.options(Settings())

    def forbidden(*a, **k):
        raise AssertionError('Saved figure attempted scientific calculation')
    monkeypatch.setattr(behaviour_durations, 'statistics', forbidden)
    monkeypatch.setattr(behaviour_samples, 'aggregate', forbidden)
    monkeypatch.setattr(behaviour_samples, 'compare_samples', forbidden)
    for page in summary.pages(data, settings):
        figure, _, _ = drawing(data, page)
        plt.close(figure)

def test_options_and_models_cannot_change_scientific_membership():
    data = fixture()
    settings, _ = summary.options(Settings())
    before = data['occupancy'].copy(deep=True)
    changed = {**settings, 'state_display_order': data['states'].state_id.tolist()[::-1]}
    summary.pages(data, changed)
    pd.testing.assert_frame_equal(data['occupancy'], before)
    with pytest.raises(ValueError, match='every accepted'):
        summary.pages(data, {**changed, 'state_display_order': changed['state_display_order'][:1]})
    with pytest.raises(ValueError, match='unknown accepted'):
        summary.pages(data, {**changed, 'state_display_names': {'invented': 'State'}})
    with pytest.raises(ValueError):
        summary.options(Settings({'state_summaries': {'state_summary_views': ['new_inference']}}))
    data['sample_provenance']['model_id'] = 'other'
    with pytest.raises(ValueError, match='different models'):
        summary.pages(data, settings)

def test_transition_titles_and_unbalanced_condition_labels_do_not_overlap():
    data = fixture()
    settings, _ = summary.options(Settings())
    page = next((page for page in summary.pages(data, settings) if page['view'] == 'transitions'))
    figure, axes, _ = drawing(data, page)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    title = figure._suptitle.get_window_extent(renderer)
    assert all((not title.overlaps(axis.title.get_window_extent(renderer)) for axis in axes.values()))
    plt.close(figure)
    exemplar = data['unit_metrics'].loc[data['unit_metrics'].value.notna()].iloc[0].to_dict()
    rows = [{**exemplar, 'unit_id': condition + str(index), 'sample': condition + str(index), 'condition': condition} for condition, count in [('control', 6), ('treated', 6), ('unspecified', 129)] for index in range(count)]
    data['unit_metrics'] = pd.DataFrame(rows)
    page = {'kind': 'samples', 'view': 'samples', 'questions': [exemplar['question_id']]}
    figure, axes, _ = drawing(data, page)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axis = axes[exemplar['question_id']]
    boxes = [label.get_window_extent(renderer) for label in axis.get_xticklabels()]
    assert all((not first.overlaps(second) for index, first in enumerate(boxes) for second in boxes[index + 1:]))
    assert not axis.xaxis.label.get_window_extent(renderer).overlaps(figure.texts[-1].get_window_extent(renderer))
    assert len(axis.collections) == 141
    np.testing.assert_allclose([collection.get_offsets()[0, 1] for collection in axis.collections], [exemplar['value']] * 141)
    plt.close(figure)
