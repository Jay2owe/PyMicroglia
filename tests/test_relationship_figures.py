from tests.panel_helpers import reopen_page
from pymicroglia.figure_tables.relationship_matrix_display import prepare as prepare_display
"""Overview values remain saved science, including unavailable directed entries."""
from tests.panel_helpers import panel_canvas
import copy
import json
import numpy as np
import pandas as pd
import pytest
import pymicroglia.pipelines.relationships.figures as figures
from pymicroglia.pipelines._contracts import Settings, content_id
from pymicroglia.pipelines.relationships.options import run_request
from pymicroglia.pipelines._screening import read_table
from tests.test_relationship_between import inputs, declaration
from tests.test_relationship_options import lag
from tests.test_rhythm_relationship_figures import capture_figures

@pytest.fixture(autouse=True)
def restore_plot_style():
    import matplotlib as mpl
    with mpl.rc_context():
        yield

def data():
    pair = {'reference': 'signal', 'target': 'area'}
    pid = content_id(pair)
    base = {'pair_id': pid, **pair, 'source_scientific_id': 'within-source', 'requested_cells': 8, 'eligible_cells': 6, 'tested_cells': 5, 'supported_cells': 3, 'confirmed_samples': 3, 'members': [], 'effect_population': 'Saved mean coefficient', 'delay_supported_cells': 2, 'delay_unresolved_cells': 1}
    return {'preparation': {'resolved_request': {'inputs': {'source_run': 'source'}, 'measurements': [{'column': 'signal', 'label': 'Signal'}, {'column': 'area', 'label': 'Cell area'}], 'request': {'pairs': [pair], 'within_cell': {'enabled': True}, 'between_cells': {'enabled': True}, 'lag': {'enabled': True, 'range_hours': [-2, 2]}}}}, 'between_provenance': {'scientific_id': 'between-source'}, 'summary_provenance': {'scientific_id': 'summary-source'}, 'rel_summaries': pd.DataFrame([{**base, 'question': 'within_cell', 'level': 'all_cells', 'effect': 0.25, 'status': 'descriptive', 'reason': 'Saved all-cell mean'}, {**base, 'question': 'lag', 'level': 'all_cells', 'effect': 0.6, 'delay_summary_hours': -1.0, 'delay_summary_status': 'descriptive-compatible', 'delay_summary_reason': 'Compatible saved intervals'}]), 'rel_between': pd.DataFrame([{**base, 'effect': 0.75, 'status': 'no-detected-association', 'reason': 'Saved independent sample test', 'p_value': 0.2, 'q_value': 0.2, 'effect_interval': [-0.2, 0.9], 'cells_requested': 8, 'cells_eligible': 6}]), 'rel_inventory': pd.DataFrame([{'source_run': 'source', 'movie': 'a' if i < 4 else 'b', 'identity': i % 4 + 1, 'pair_id': pid} for i in range(8)])}

def test_separate_views_preserve_exact_values_counts_and_unrequested_direction():
    source = data()
    settings = figures.DEFAULTS
    expected = {'within_cell': 0.25, 'between_cells': 0.75, 'delay': -1.0}
    for page in figures.pages(source, settings):
        values, _ = figures.matrix_values(source, page)
        record = values.loc[values.requested].iloc[0]
        assert record.value == expected[page['view']] and record.requested_cells == 8 and (record.eligible_cells == 6) and (record.samples == 3)
        assert values.loc[~values.requested, 'value'].isna().all()
        assert record.reference == 'signal' and record.target == 'area' and (len(record.members) == 8)
        if page['view'] == 'delay':
            assert pd.isna(record.p_value) and record.delay_unresolved_cells == 1
    reordered = figures.pages(source, {**settings, 'measurement_order': ['area', 'signal'], 'matrix_block_size': 1})
    original = {figures.matrix_values(source, page)[0].loc[lambda frame: frame.requested, 'entry_id'].iloc[0] for page in figures.pages(source, settings)}
    assert {figures.matrix_values(source, page)[0].entry_id.iloc[0] for page in reordered} == original

def test_zero_unavailable_unresolved_disabled_and_non_detection_are_distinct():
    source = data()
    source['rel_summaries'].loc[0, 'effect'] = 0.0
    page = figures.pages(source, figures.DEFAULTS)[0]
    values, _ = figures.matrix_values(source, page)
    assert values.loc[values.requested, 'value'].iloc[0] == 0.0
    source['rel_summaries'].loc[1, ['delay_summary_hours', 'delay_summary_status']] = [None, 'incompatible']
    page = figures.pages(source, figures.DEFAULTS)[2]
    values, _ = figures.matrix_values(source, page)
    assert values.loc[values.requested, 'value'].isna().all() and values.loc[values.requested, 'status'].iloc[0] == 'incompatible'
    source['preparation']['resolved_request']['request']['lag']['enabled'] = False
    assert figures.matrix_values(source, page)[0].loc[lambda f: f.requested, 'status'].iloc[0] == 'disabled'
    page = figures.pages(source, figures.DEFAULTS)[1]
    assert figures.matrix_values(source, page)[0].loc[lambda f: f.requested, 'status'].iloc[0] == 'no-detected-association'
    source['rel_summaries'] = source['rel_summaries'].iloc[:0]
    page = figures.pages(source, figures.DEFAULTS)[0]
    assert figures.matrix_values(source, page)[0].loc[lambda f: f.requested, 'status'].iloc[0] == 'unavailable'

def test_registered_overview_and_cold_readback_make_no_scientific_calls(tmp_path, monkeypatch):
    from pymicroglia import workbench as circadian
    import pymicroglia.measure.relationship_statistics as relationship_statistics
    import pymicroglia.measure.relationship_lag_statistics as relationship_lag_statistics
    import pymicroglia.measure.relationship_population_statistics as relationship_population_statistics
    import pymicroglia.measure.relationship_consistency_statistics as relationship_consistency_statistics
    req = declaration(within_cell={'enabled': True, 'statistic': 'pearson', 'evidence': {'method': 'none'}}, lag=lag(), sample_summary={'enabled': True, 'aggregation': 'mean', 'evidence': {'method': 'none'}}, biological_samples={f'm{s:02d}_{movie}': f'sample-{s}' for s in range(12) for movie in range(2)})
    resolved, paths, _ = inputs(tmp_path, req)
    science = run_request(resolved, paths, tmp_path / 'pipeline', only=('sample-consistency',))
    assert science.successful, {name: saved.outcome.reason for name, saved in science.results.items()}
    captured = capture_figures(monkeypatch)

    def forbidden(*a, **k):
        raise AssertionError('Overview repeated science')
    for obj, name in ((circadian, 'estimate_one'), (circadian, 'detrend_trace'), (circadian, 'adjust_pvalues'), (relationship_statistics, 'same_time_evidence'), (relationship_lag_statistics, 'evaluate'), (relationship_population_statistics, 'sample_association'), (relationship_consistency_statistics, 'sign_evidence')):
        monkeypatch.setattr(obj, name, forbidden)
    rendered = run_request(resolved, paths, tmp_path / 'pipeline', only=('relationship-overview',))
    assert rendered.successful, {name: saved.outcome.reason for name, saved in rendered.results.items()}
    assert len(captured) == 3
    saved = rendered.results['relationship-overview']
    entries = read_table(saved.artifact('entries.json'))
    assert len(entries) == 3 and entries.entry_id.is_unique and (set(entries.view) == {'within_cell', 'between_cells', 'delay'})
    for ctx, drawing in captured:
        assert drawing.figure_data.loc[drawing.figure_data.requested, 'requested_cells'].iloc[0] == 48
        ctx.cached.clear()
        reopened = reopen_page(ctx)
        pd.testing.assert_frame_equal(reopened.figure_data, drawing.figure_data)
        import matplotlib.pyplot as plt

    reordered = run_request(resolved, paths, tmp_path / 'pipeline', only=('relationship-overview',), presentation={'measurement_relationships': {'measurement_order': ['area_px', 'corrected_mean'], 'matrix_block_size': 1}})
    assert reordered.successful and all((saved.outcome.status == 'reused' for name, saved in reordered.results.items() if name != 'relationship-overview'))
    newer = read_table(reordered.results['relationship-overview'].artifact('entries.json'))
    assert set(newer.entry_id) == set(entries.entry_id)

def test_rendered_colours_and_labels_match_the_frozen_values():
    from pymicroglia.visualisation.figures import load as _figure_schema
    _figure_schema()
    from pymicroglia.visualisation.panels.measurement_relationships import draw
    import matplotlib.pyplot as plt
    source = data()
    page = figures.pages(source, figures.DEFAULTS)[2]
    values, statistics = figures.matrix_values(source, page)
    figure, axes = draw(prepare_display(values,{**page, 'labels': {'signal': 'Signal', 'area': 'Cell area'}, 'lag_colour_limit_hours': 2.0, 'title': 'Known saved delay', 'evidence_note': 'Negative means reference leads', 'footnote': 'Controlled verification'}), {**page, 'labels': {'signal': 'Signal', 'area': 'Cell area'}, 'lag_colour_limit_hours': 2.0, 'title': 'Known saved delay', 'evidence_note': 'Negative means reference leads', 'footnote': 'Controlled verification'}, canvas=panel_canvas())
    assert axes['matrix'].images[0].get_array()[0, 1] == -1.0
    assert any(('-1 h' in text.get_text() and '1 unresolved' in text.get_text() for text in axes['matrix'].texts))
    plt.close(figure)

def test_render_stages_outside_synced_output_and_keeps_previous_on_failure(tmp_path):
    from pymicroglia.visualisation._delivery import render_destination
    output=tmp_path/'synced'/'figure.svg';output.parent.mkdir()
    output.write_text('preceding verified figure')
    with pytest.raises(ValueError,match='scientific table'):
        with render_destination(output) as staging:
            assert not staging.is_relative_to(output.parent)
            staging.write_text('unverified replacement')
            raise ValueError('scientific table differs')
    assert output.read_text()=='preceding verified figure'
    assert not list(output.parent.glob('.figure-*'))
    with render_destination(output) as staging:staging.write_text('verified replacement')
    assert output.read_text()=='verified replacement'


def test_delivery_uses_shared_atomic_retry_and_propagates_validation_errors(tmp_path,monkeypatch):
    from pymicroglia.visualisation._delivery import render_destination
    from auto_organotypic import io
    output=tmp_path/'figure.svg';output.write_text('preceding figure')
    called=[]
    def reject(candidate,destination):
        called.append((candidate,destination))
        assert candidate.parent==destination.parent
        raise ValueError('Scientific table hash mismatch')
    monkeypatch.setattr(io,'replace_with_retry',reject)
    with pytest.raises(ValueError,match='hash mismatch'):
        with render_destination(output) as staging:staging.write_text('new figure')
    assert len(called)==1 and output.read_text()=='preceding figure'
    assert not list(tmp_path.glob('.figure-*'))
