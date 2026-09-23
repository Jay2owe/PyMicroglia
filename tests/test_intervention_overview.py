from pymicroglia.figure_tables.intervention_display import prepare as prepare_intervention_display
"""Saved effects, full original populations and distinct missing-value encodings."""
from tests.panel_helpers import panel_canvas
from types import SimpleNamespace
import numpy as np
import pandas as pd
from pymicroglia import workbench as circadian
import pymicroglia.measure.sample_contrasts as sample_contrasts
import pymicroglia.pipelines.intervention.overview as overview
from pymicroglia.pipelines._contracts import Settings
from pymicroglia.pipelines.intervention.options import run_request
from tests.test_intervention_evidence import fixture

def test_complete_saved_pages_keep_exact_estimates_and_distinct_evidence(tmp_path, monkeypatch):
    _, _, paths, resolved = fixture(tmp_path)
    execution = run_request(resolved, paths, tmp_path / 'pipeline', only=['response-evidence'])
    assert execution.successful
    context = SimpleNamespace(request=resolved, dependencies=execution.results, saved=execution.results.__getitem__)

    def forbidden(*a, **k):
        raise AssertionError('Display invoked scientific analysis')
    monkeypatch.setattr(circadian, 'adjust_pvalues', forbidden)
    monkeypatch.setattr(circadian, 'estimate_grouped_rhythms', forbidden)
    monkeypatch.setattr(sample_contrasts, 'compare', forbidden)
    data = overview.collect(context)
    settings, _ = overview.options(Settings())
    values, statistics, pages = overview.pages(data, settings)
    rows = values.loc[values.kind.eq('cell_effect')]
    original = data['evidence']['effects'].set_index('effect_id')
    saved = rows.set_index('effect_id')
    assert len(rows) == 8 and rows.cell_id.nunique() == 4 and (set(saved.index) == set(original.index))
    for key in ['absolute_change', 'estimate', 'interval_low', 'interval_high', 'p_value', 'q_value']:
        assert np.allclose(saved.loc[original.index, key], original[key], equal_nan=True)
    assert {'increase', 'decrease', 'inconclusive', 'no_detected_change'} <= set(rows.outcome)
    assert set(statistics.loc[statistics.kind.eq('cell_effect'), 'effect_id']) == set(original.index)
    matrix = next((page for page in pages if page['view'] == 'status_matrix'))
    assert len(matrix['cell_ids']) == 4 and len(matrix['measurements']) == 2 and (len(matrix['entry_ids']) == 8)
    from pymicroglia.visualisation.figures import load as _figure_schema
    _figure_schema()
    from pymicroglia.visualisation.panels import intervention_overview as panel
    import matplotlib.pyplot as plt
    for page in pages:
        chosen = values.loc[values.entry_id.isin(page['entry_ids'])]
        fig, axes = panel.draw(prepare_intervention_display(chosen,page,'intervention_overview'), page, canvas=panel_canvas())
        if page['view'] == 'status_matrix':
            matrix = np.asarray(axes['evidence'].images[0].get_array())
            assert 3 in matrix and 2 in matrix
            assert all((label.get_text() for label in axes['evidence'].get_yticklabels()))
        plt.close(fig)

def test_entirely_unavailable_science_keeps_requested_cells_and_windows(tmp_path):
    _, _, _, resolved = fixture(tmp_path)
    context = SimpleNamespace(request=resolved, dependencies={})
    data = overview.collect(context)
    settings, _ = overview.options(Settings())
    values, statistics, pages = overview.pages(data, settings)
    rows = values.loc[values.kind.eq('cell_effect')]
    assert len(rows) == 8 and rows.cell_id.nunique() == 4 and rows.estimate.isna().all() and rows.outcome.eq('inconclusive').all()
    assert values.loc[values.kind.eq('coverage')].shape[0] == 6
    assert next((page for page in pages if page['view'] == 'status_matrix'))['entry_ids']

def test_pagination_keeps_full_membership_without_reselecting_responders(tmp_path):
    _, _, _, resolved = fixture(tmp_path)
    data = overview.collect(SimpleNamespace(request=resolved, dependencies={}))
    settings, _ = overview.options(Settings({'overview': {'views': ['status_matrix'], 'rows_per_page': 1, 'metrics_per_page': 1}}))
    values, statistics, pages = overview.pages(data, settings)
    assert len(pages) == 8 and len(values) == 8 and (len(statistics) == 8)
    ids = [entry for page in pages for entry in page['entry_ids']]
    assert len(ids) == len(set(ids)) == 8

def test_one_run_can_register_identical_presentations_at_distinct_saved_result_paths(tmp_path, monkeypatch):
    from pymicroglia.visualisation.figures import load as _figure_schema
    import pymicroglia.pipelines.intervention.display as display, pymicroglia.pipelines.rhythm.images as rhythm_images, pymicroglia.pipelines._saved_figures as saved_figures
    _figure_schema()
    run = tmp_path / 'original-run'
    run.mkdir()
    monkeypatch.setattr(rhythm_images, 'original_run', lambda context: run)
    from pymicroglia.figure_tables.prepared import PreparedPage, Drawing
    monkeypatch.setattr(saved_figures,'save_page',lambda *a,**k:None)
    monkeypatch.setattr(overview,'build',lambda source:PreparedPage(Drawing(None,()),pd.DataFrame(),producer_sources={}))
    values = pd.DataFrame([display.entry(kind='coverage', status='available')])
    contexts = []
    results = []
    for name in ['first', 'retry']:
        output = tmp_path / name
        context = SimpleNamespace(output=output, scientific_id='a' * 64, dependencies={})
        saved, _ = display.snapshot(context, values, values, {'pages': []})
        contexts.append(SimpleNamespace(output=output, dependencies={'intervention-display': saved}, presentation_id='b' * 64))

    def register(context):
        return saved_figures.draw_batch(context, slug=overview.SLUG, options=[{'evidence_page': 1}], aliases=display.ALIASES, data={}, cache_name='_intervention_display', sources=[], text={}, claim='Binding contract test', grammar='small-multiples')
    results = [register(context) for context in contexts]
    assert results[0]['masters'] != results[1]['masters']
    assert register(contexts[0])['masters'] == results[0]['masters']
