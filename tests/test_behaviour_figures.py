from pymicroglia.figure_tables.state_profile_display import prepare as prepare_state_display
"""Saved state diagnostics preserve scientific refusals and observed profiles."""
from tests.panel_helpers import panel_canvas
from pymicroglia._results import read_document
import json
from types import SimpleNamespace
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import pymicroglia.pipelines.behaviour.figures as display
from pymicroglia.pipelines._contracts import Settings, StepResult
from pymicroglia.pipelines._runner import SavedResult, figure_binding
from pymicroglia.pipelines._screening import read_table

@pytest.fixture(autouse=True)
def unchanged_plot_defaults():
    with plt.rc_context():
        yield
    plt.close('all')

def unavailable_context(tmp_path, status='unavailable'):
    dependencies = {name: SavedResult(tmp_path / name, StepResult(name, 'controlled-' + name, status, 'Controlled unavailable native capability')) for name in ['feature-inputs', 'candidate-models', 'state-support']}
    return SimpleNamespace(dependencies=dependencies, saved=dependencies.__getitem__, output=tmp_path / 'display', scientific_id='controlled-display')

@pytest.mark.parametrize('status', ['unavailable', 'failed'])
def test_diagnostic_snapshot_preserves_failure_without_promoting_scientific_sources(tmp_path, status):
    context = unavailable_context(tmp_path, status)
    values, statistics, metadata = display.support_data(context)
    assert len(values) == 3 and values.status.eq(status).all() and (metadata['status'] == status)
    assert statistics.status.eq(status).all() and statistics.estimate.isna().all() and statistics.p_value.isna().all()
    assert not values.kind.eq('profile').any() and metadata['model_id'] is None
    frozen, recorded = display.snapshot(context, values, statistics, metadata)
    assert all((item['status'] == status for item in recorded['source_outcomes'].values()))
    assert not recorded['analysis_recomputed'] and recorded['snapshot_kind'] == 'saved_display_inputs'
    binding = figure_binding({'state-display': frozen}, inputs=display.ALIASES)
    from pathlib import Path
    source = binding['results']['state-display']
    assert Path(source['ledger']).name == 'result.json'
    assert source['entry'] is None
    assert read_document(frozen.artifact('provenance'))['status'] == status
    pd.testing.assert_frame_equal(read_table(frozen.artifact('values')), values)
    assert all((item.outcome.status == status for item in context.dependencies.values()))
    with pytest.raises(Exception, match='not completed'):
        figure_binding(context.dependencies, inputs={})

def test_support_pages_retain_every_criterion_and_never_require_assignments(tmp_path):
    values, _, metadata = display.support_data(unavailable_context(tmp_path))
    settings, _ = display.options(Settings({'state_profiles': {'state_page_size': 2}}))
    pages = display.pages(values, metadata, settings)
    assert len(pages) == 2 and sum([page['entries'] for page in pages], []) == values.entry_id.tolist()
    assert all((page['view'] == 'support' for page in pages))

def profile_fixture():
    states = [{'state_id': 'state-a', 'component': 0, 'label': 'State 1'}, {'state_id': 'state-b', 'component': 1, 'label': 'State 2'}]
    values = pd.DataFrame([{'entry_id': state['state_id'] + '-' + measurement, 'view': 'profiles', 'kind': 'profile', 'state_id': state['state_id'], 'model_id': 'one-accepted-model', 'measurement': measurement, 'source_table': 'cell_frame', 'representation': 'raw', 'unit': 'microns' if measurement == 'custom_length' else '', 'component': state['component'], 'median': median, 'q25': median - 0.25, 'q75': median + 0.5, 'mean': median + 0.1, 'observed_values': 24, 'missing_values': 2, 'cells': 3, 'movies': 2, 'confirmed_samples': 1} for state, median in zip(states, [-2.0, 5.0]) for measurement in ['custom_length', 'custom_texture']])
    metadata = {'view': 'profiles', 'status': 'accepted', 'model_id': 'one-accepted-model', 'states': states, 'reason': 'Controlled frozen acceptance', 'scope': 'Observed measurement vocabulary only'}
    return (values, metadata)

def test_profile_display_order_changes_no_values_and_cannot_select_a_new_vocabulary():
    values, metadata = profile_fixture()
    before = values.copy(deep=True)
    settings, _ = display.options(Settings({'state_profiles': {'state_display_order': ['state-b', 'state-a'], 'state_feature_page_size': 1}}))
    pages = display.pages(values, metadata, settings)
    assert len(pages) == 2 and all((page['states'] == ['state-b', 'state-a'] for page in pages))
    assert [page['features'][0]['measurement'] for page in pages] == ['custom_length', 'custom_texture']
    pd.testing.assert_frame_equal(values, before)
    with pytest.raises(ValueError, match='every accepted state'):
        display.pages(values, metadata, {**settings, 'state_display_order': ['state-a']})
    with pytest.raises(ValueError, match='unknown accepted'):
        display.pages(values, metadata, {**settings, 'state_display_names': {'unknown': 'invented'}})

def test_unsaved_profile_marks_are_exact_saved_medians_and_observed_quartiles():
    from pymicroglia.visualisation.figures import load as _figure_schema
    _figure_schema()
    from pymicroglia.visualisation.panels import behaviour_profiles
    values, metadata = profile_fixture()
    options, _ = display.options(Settings())
    page = display.pages(values, metadata, options)[0]
    settings = {**page, 'metadata': metadata, 'state_display_names': {}, 'title': 'Controlled observed profiles', 'footnote': 'Observed interquartile ranges, not confidence intervals'}
    figure, axes = behaviour_profiles.draw(prepare_state_display(values,settings), settings, canvas=panel_canvas())
    assert len(axes) == 2
    for axis in axes.values():
        assert [float(collection.get_offsets()[0, 0]) for collection in axis.collections] == [-2.0, 5.0]
        np.testing.assert_allclose(axis.lines[0].get_xdata(), [-2.25, -1.5])
        np.testing.assert_allclose(axis.lines[1].get_xdata(), [4.75, 5.5])
        assert any(('24 values; 2 missing' in text.get_text() for text in axis.texts))
    plt.close(figure)

def test_unsaved_failed_diagnostic_keeps_status_and_has_no_candidate_profiles(tmp_path):
    from pymicroglia.visualisation.figures import load as _figure_schema
    _figure_schema()
    from pymicroglia.visualisation.panels import behaviour_profiles
    values, statistics, metadata = display.support_data(unavailable_context(tmp_path, 'failed'))
    options, _ = display.options(Settings())
    settings = {**display.pages(values, metadata, options)[0], 'metadata': metadata, 'state_display_names': {}, 'title': 'Controlled unavailable analysis', 'footnote': 'No accepted model'}
    figure, axes = behaviour_profiles.draw(prepare_state_display(values,settings), settings, canvas=panel_canvas())
    assert set(axes) == {'criteria'}
    assert any(('Saved decision: failed' == text.get_text() for text in figure.texts))
    assert sum((text.get_text() == 'failed' for text in axes['criteria'].texts)) == 3
    plt.close(figure)

def test_missing_profile_values_are_unavailable_marks_and_assignment_limits_remain_visible():
    from pymicroglia.visualisation.figures import load as _figure_schema
    _figure_schema()
    from pymicroglia.visualisation.panels import behaviour_profiles
    values, metadata = profile_fixture()
    values.loc[values.state_id.eq('state-a'), ['median', 'q25', 'q75']] = None
    values.loc[values.state_id.eq('state-a'), 'observed_values'] = 0
    metadata['assignment_status_counts'] = {'assigned': 48, 'ambiguous': 3}
    options, _ = display.options(Settings())
    settings = {**display.pages(values, metadata, options)[0], 'metadata': metadata, 'state_display_names': {}, 'title': 'Controlled missing profiles', 'footnote': 'Observed values only'}
    figure, axes = behaviour_profiles.draw(prepare_state_display(values,settings), settings, canvas=panel_canvas())
    assert all((len(axis.collections) == 1 and any((text.get_text() == 'No observed values' for text in axis.texts)) for axis in axes.values()))
    assert any(('3 without a state' in text.get_text() for text in figure.texts))
    plt.close(figure)
