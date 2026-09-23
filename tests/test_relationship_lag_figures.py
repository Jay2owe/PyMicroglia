from tests.panel_helpers import reopen_page
from pymicroglia.figure_tables.lag_display import prepare as prepare_display
"""Saved physical lag grids retain band meaning and full-search evidence."""
from tests.panel_helpers import panel_canvas
import copy
import numpy as np
import pandas as pd
import pytest
import pymicroglia.pipelines.relationships.lag_figures as figures
from pymicroglia.pipelines._contracts import content_id
from pymicroglia.pipelines.relationships.options import run_request
from pymicroglia.pipelines._screening import read_table
from tests.test_relationship_reports import inputs
from tests.test_rhythm_relationship_figures import capture_figures

@pytest.fixture(autouse=True)
def restore_style():
    import matplotlib as mpl
    with mpl.rc_context():
        yield

def saved_data(status='multiple-candidates', reverse=False):
    lags = np.arange(-2, 3, dtype=float)
    pair = {'reference': 'area' if reverse else 'signal', 'target': 'signal' if reverse else 'area'}
    key = {'source_run': 'source-one', 'movie': 'a', 'identity': 1, 'pair_id': content_id(pair), **pair}
    effect = [0.1, -0.8, 0.2, -0.8, 0.1]
    candidates = [-1.0, 1.0] if status == 'multiple-candidates' else [-2.0] if status == 'search-boundary' else [-1.0, 0.0, 1.0]
    result = {**key, 'status': 'negative-association', 'resolution_status': status, 'resolution_reason': 'Saved reason: ' + status, 'delay_hours': None, 'p_value': 0.02, 'q_value': 0.04, 'candidate_lags_hours': candidates, 'empirical_peak_lags_hours': [-1.0, 1.0]}
    profiles = pd.DataFrame([{**key, 'lag_hours': lag, 'effect': r, 'tested_effect': r, 'coefficient_lower': r - 0.05, 'coefficient_upper': r + 0.05, 'interval_kind': 'Saved approximate grid uncertainty', 'status': 'descriptive', 'reason': 'Saved overlap', 'paired_observations': 100 - abs(lag), 'overlap_span_hours': 99 - abs(lag), 'gap_count': 0} for lag, r in zip(lags, effect)])
    return {'lag_results': pd.DataFrame([result]), 'lag_profiles': profiles, 'provenance': {'scientific_id': 'saved-lag-science', 'question': {'enabled': True, 'range_hours': [-2, 2], 'peak_resolution': {'method': 'stationary_bootstrap'}}}}

@pytest.mark.parametrize('status', ['broad', 'multiple-candidates', 'search-boundary'])
def test_integer_hours_ambiguity_and_coefficient_bands_are_never_reinterpreted(status):
    from pymicroglia.visualisation.figures import load as _figure_schema
    _figure_schema()
    from pymicroglia.visualisation.panels.relationship_lag_profiles import draw
    import matplotlib.pyplot as plt
    data = saved_data(status)
    page = figures.pages(data, {'cells_per_page': 2, 'display_lag_range_hours': None})[0]
    values, stats = figures.values(data, page)
    settings = {**page, 'question': data['provenance']['question'], 'title': 'Saved test', 'footnote': 'Physical hours'}
    figure, axes = draw(prepare_display(values,settings), settings, canvas=panel_canvas())
    np.testing.assert_array_equal(axes['profile_0'].lines[1].get_xdata(), [-2, -1, 0, 1, 2])
    assert stats.resolution_status.iloc[0] == status and stats.delay_hours.isna().all()
    labels = axes['profile_0'].get_legend_handles_labels()[1]
    assert 'Within-curve coefficient confidence band' in labels and (not any(('null' in label.lower() for label in labels)))
    all_text = ' '.join((text.get_text() for axis in figure.axes for text in axis.texts))
    assert 'not a null band' in all_text and 'Saved reason: ' + status in all_text
    plt.close(figure)

def test_missing_uncertainty_and_reversed_pairs_keep_source_direction():
    from pymicroglia.visualisation.figures import load as _figure_schema
    _figure_schema()
    from pymicroglia.visualisation.panels.relationship_lag_profiles import draw
    import matplotlib.pyplot as plt
    data = saved_data(reverse=True)
    data['lag_profiles'][['coefficient_lower', 'coefficient_upper']] = np.nan
    data['lag_profiles'].loc[0, 'status'] = 'untestable'
    page = figures.pages(data, {'cells_per_page': 1, 'display_lag_range_hours': [-1, 1]})[0]
    values, stats = figures.values(data, page)
    figure, axes = draw(prepare_display(values,{**page, 'question': data['provenance']['question'], 'title': 'Saved test', 'footnote': 'Hours'}), {**page, 'question': data['provenance']['question'], 'title': 'Saved test', 'footnote': 'Hours'}, canvas=panel_canvas())
    assert 'area -> signal' in axes['profile_0'].get_title()
    assert len(values) == 5 and tuple(axes['profile_0'].get_xlim()) == (-1, 1)
    text = ' '.join((t.get_text() for a in figure.axes for t in a.texts))
    assert 'uncertainty unavailable' in text and '1/5' in text
    assert stats.p_value.iloc[0] == 0.02 and stats.q_value.iloc[0] == 0.04
    plt.close(figure)

def test_cropped_registered_profiles_use_all_saved_results_without_science(tmp_path, monkeypatch):
    from pymicroglia import workbench as circadian
    import pymicroglia.measure.relationship_statistics as relationship_statistics
    import pymicroglia.measure.relationship_lag_statistics as relationship_lag_statistics
    resolved, paths = inputs(tmp_path)
    science = run_request(resolved, paths, tmp_path / 'pipeline', only=('lag-association',))
    assert science.successful, {k: v.outcome.reason for k, v in science.results.items()}
    captured = capture_figures(monkeypatch)

    def forbidden(*a, **k):
        raise AssertionError('Lag display repeated scientific analysis')
    for obj, name in ((circadian, 'estimate_one'), (circadian, 'adjust_pvalues'), (circadian, 'detrend_trace'), (relationship_statistics, 'same_time_evidence'), (relationship_lag_statistics, 'evaluate')):
        monkeypatch.setattr(obj, name, forbidden)
    rendered = run_request(resolved, paths, tmp_path / 'pipeline', only=('relationship-lag-profiles',))
    assert rendered.successful, {k: v.outcome.reason for k, v in rendered.results.items()}
    assert 'within-cell-association' not in rendered.results
    saved = rendered.results['relationship-lag-profiles']
    entries = read_table(saved.artifact('entries.json'))
    assert len(entries) == 3 and len(captured) == 2
    for ctx, drawing in captured:
        ctx.cached.clear()
        cold = reopen_page(ctx)
        pd.testing.assert_frame_equal(cold.figure_data, drawing.figure_data)
        import matplotlib.pyplot as plt
    cropped = run_request(resolved, paths, tmp_path / 'pipeline', only=('relationship-lag-profiles',), presentation={'relationship_lag_profiles': {'cells_per_page': 1, 'display_lag_range_hours': [-0.25, 0.25]}})
    assert cropped.successful, {k: v.outcome.reason for k, v in cropped.results.items()}
    changed = read_table(cropped.results['relationship-lag-profiles'].artifact('entries.json'))
    assert set(changed.entry_id) == set(entries.entry_id)
    pd.testing.assert_frame_equal(changed.sort_values('entry_id')[['entry_id', 'p_value', 'q_value']].reset_index(drop=True), entries.sort_values('entry_id')[['entry_id', 'p_value', 'q_value']].reset_index(drop=True))
    for ctx, drawing in captured[-3:]:
        assert set(drawing.figure_data.lag_hours) == {-0.5, 0, 0.5}

def test_disabled_lag_remains_a_visible_outcome():
    data = saved_data()
    data['provenance']['question'] = {'enabled': False}
    data['lag_profiles'] = data['lag_profiles'].iloc[:0]
    data['lag_results'].loc[0, ['status', 'resolution_status']] = ['disabled', 'disabled']
    data['lag_results']['reason'] = 'Lag association was not requested'
    page = figures.pages(data, {'cells_per_page': 2, 'display_lag_range_hours': None})[0]
    values, stats = figures.values(data, page)
    assert len(values) == 1 and values.kind.iloc[0] == 'unavailable' and (stats.status.iloc[0] == 'disabled')
