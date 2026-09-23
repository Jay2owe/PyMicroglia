from pymicroglia.figure_tables.intervention_display import prepare as prepare_intervention_display
"""Saved timing events, unresolved windows and forbidden scientific rerenders."""
from tests.panel_helpers import panel_canvas
from types import SimpleNamespace
import numpy as np
from pymicroglia import workbench as circadian
import pymicroglia.pipelines.intervention.timing_figures as figures, pymicroglia.pipelines.intervention.timing as intervention_timing, pymicroglia.pipelines.intervention.rhythms as intervention_rhythms
from pymicroglia.pipelines._contracts import Settings
from pymicroglia.pipelines.intervention.options import run_request

def context(resolved, execution):
    return SimpleNamespace(request=resolved, dependencies=execution.results, saved=execution.results.__getitem__)

def test_unobserved_recovery_is_not_drawn_at_zero_or_at_the_endpoint(tmp_path, monkeypatch):
    from tests.test_intervention_timing import fixture
    _, _, paths, resolved = fixture(tmp_path)
    execution = run_request(resolved, paths, tmp_path / 'pipeline', only=['response-timing'])
    assert execution.successful

    def forbidden(*a, **k):
        raise AssertionError('Timing display recomputed events')
    monkeypatch.setattr(intervention_timing, 'scan', forbidden)
    monkeypatch.setattr(circadian, 'adjust_pvalues', forbidden)
    data = figures.collect(context(resolved, execution))
    settings, _ = figures.options(Settings({'timing_figures': {'views': ['response_delay', 'recovery']}}))
    values, statistics, pages = figures.pages(data, settings)
    assert len(values) == 10 and len(pages) == 2
    original = data['timing']['timing'].set_index('timing_id')
    assert set(values.timing_id) == set(original.index)
    assert np.allclose(values.recovery_delay_hours, original.loc[values.timing_id, 'recovery_delay_hours'], equal_nan=True)
    from pymicroglia.visualisation.figures import load as _figure_schema
    _figure_schema()
    from pymicroglia.visualisation.panels import intervention_timing as panel
    import matplotlib.pyplot as plt
    page = next((p for p in pages if p['view'] == 'recovery'))
    chosen = values.loc[values.entry_id.isin(page['entry_ids'])]
    figure, axes = panel.draw(prepare_intervention_display(chosen,page,'intervention_timing'), page, canvas=panel_canvas())
    observed = [line for line in axes['evidence'].lines if line.get_label() == 'Observed event']
    assert len(observed) == int(chosen.recovery_observed.sum()) == 2
    assert set(chosen.loc[chosen.recovery_observed, 'movie']) == {'recovering', 'gapped'}
    assert all((list(line.get_xdata()) == [10.0] for line in observed))
    lost = chosen.reset_index(drop=True).index[chosen.movie.eq('lost')][0]
    endpoint = [line for line in axes['evidence'].lines if line.get_label() == 'Last observation' and list(line.get_ydata()) == [lost]]
    assert len(endpoint) == 1 and list(endpoint[0].get_xdata()) == [9.0]
    assert all((list(line.get_ydata()) != [lost] for line in observed))
    plt.close(figure)

def test_window_significance_and_native_parameter_change_stay_separate(tmp_path, monkeypatch):
    from tests.test_intervention_rhythms import fixture
    _, _, paths, resolved = fixture(tmp_path)
    execution = run_request(resolved, paths, tmp_path / 'pipeline', only=['rhythm-changes'])
    assert execution.successful

    def forbidden(*a, **k):
        raise AssertionError('Saved rhythm figure called native analysis')
    monkeypatch.setattr(intervention_rhythms, 'analyse', forbidden)
    monkeypatch.setattr(circadian, 'estimate_grouped_rhythms', forbidden)
    monkeypatch.setattr(circadian, 'rhythm_window_comparison', forbidden)
    data = figures.collect(context(resolved, execution))
    settings, _ = figures.options(Settings())
    values, statistics, pages = figures.pages(data, settings)
    windows = values.loc[values.kind.eq('rhythm_window')]
    direct = values.loc[values.kind.eq('rhythm_change')]
    assert len(windows) == 12 and len(direct) == 18
    assert windows.loc[windows.movie.eq('short'), 'period_supported'].eq(False).all()
    assert direct.loc[direct.movie.eq('period') & direct.property.eq('phase'), 'effect'].isna().all()
    native = data['rhythms']['direct_comparisons'].set_index('direct_id')
    assert np.allclose(direct.q_value, native.loc[direct.direct_id, 'q_value'], equal_nan=True)
    assert windows.applied_recipe.map(lambda row: row['options']['fit_method'] == 'fft_nlls').all()
    assert all((p['total_requested'] == 6 for p in pages if p['view'] == 'direct_change'))
    settings['rows_per_page'] = 1
    changed, _, paginated = figures.pages(data, settings)
    assert set(changed.entry_id) == set(values.entry_id)
    assert len([p for p in paginated if p['view'] == 'direct_change']) == 18

def test_disabled_and_unavailable_questions_keep_original_population(tmp_path):
    from tests.test_intervention_evidence import fixture
    _, _, _, resolved = fixture(tmp_path)
    data = figures.collect(SimpleNamespace(request=resolved, dependencies={}))
    values, _, pages = figures.pages(data, figures.options(Settings())[0])
    assert len(values) == 2 and values.status.eq('disabled').all() and values.original_cells.eq(4).all()
    assert len(pages) == 1 and pages[0]['view'] == 'coverage'
