"""Timing figures preserve source membership, direction and analysis level."""
from pymicroglia._results import read_document
import json
import numpy as np
import pandas as pd
import pytest
from pymicroglia import workbench as circadian
import pymicroglia.pipelines.rhythm.relationship_figures as figures
from pymicroglia.pipelines.rhythm.discovery import run_request
from tests.test_rhythm_timing import fixture

def capture_figures(monkeypatch):
    from tests.panel_helpers import capture_pages
    return capture_pages(monkeypatch)


def test_complete_pair_pages_render_and_reopen_without_scientific_calls(tmp_path, monkeypatch):
    resolved, paths = fixture(tmp_path)
    science = run_request(resolved, paths, tmp_path / 'pipeline', only=('detection-agreement', 'timing-across-samples'))
    assert science.successful
    captured = capture_figures(monkeypatch)

    def forbidden(*a, **k):
        raise AssertionError('Saved figures attempted scientific analysis')
    for name in ('estimate_one', 'rhythm_pair_timing', 'rhythm_timing_summary', 'rhythm_detection_agreement', 'adjust_pvalues'):
        monkeypatch.setattr(circadian, name, forbidden)
    result = run_request(resolved, paths, tmp_path / 'pipeline', only=('timing-relationship-figures',), presentation={'timing_relationships': {'overview_rows': 2, 'grid_cells_per_page': 2}})
    assert result.successful, {key: value.outcome.reason for key, value in result.results.items()}
    saved = result.results['timing-relationship-figures']
    manifest = read_document(saved.artifact('relationship_manifest.json'))
    assert len(manifest['pages']) == len(captured) == 8
    assert len({page['master'] for page in manifest['pages']}) == 8
    trace_pages = [page for page in manifest['pages'] if page['kind'] == 'pair-traces']
    assert {(cell['movie'], cell['identity']) for page in trace_pages for cell in page['cells']} == {('a', 1), ('a', 2), ('b', 1)}
    assert sum((len(page['cells']) for page in trace_pages)) == 3
    levels = {page['level'] for page in manifest['pages'] if page['kind'] == 'matrix'}
    assert levels == set(figures.LEVELS)
    for ctx, drawing in captured:
        slug = ctx.spec.slug
        values = drawing.figure_data
        if slug == figures.MATRIX_SLUG:
            assert values[~values.requested].value.isna().all()
            assert values[values.requested][['reference', 'target']].values.tolist() == [['signal', 'other']]
        elif 'role' in values and values.role.eq('timing').any():
            assert values.loc[values.role.eq('timing'), 'hours'].min() > 50
            assert values.loc[values.role.eq('reference'), 'hours'].min() == 50
    ctx, drawing = captured[-1]
    from tests.panel_helpers import reopen_page
    reopened = reopen_page(ctx)
    pd.testing.assert_frame_equal(reopened.figure_data, drawing.figure_data)
    import matplotlib.pyplot as plt
    resumed = run_request(resolved, paths, tmp_path / 'pipeline', only=('timing-relationship-figures',), presentation={'timing_relationships': {'overview_rows': 2, 'grid_cells_per_page': 2}})
    assert resumed.successful and all((saved.outcome.status == 'reused' for saved in resumed.results.values()))

def test_zero_offset_is_preserved_and_unrequested_direction_remains_empty():
    summary = pd.DataFrame([{'pair_id': 'ordered', 'reference': 'a', 'target': 'b', 'stratum': 'eight-hour', 'period_hours': 8.0, 'mean_offset_hours': 0.0, 'within_cell_stable_fraction': 0.6, 'mean_within_unit_resultant': 0.7, 'resultant': 0.8, 'input_interval_region': {'direction_interval_hours': [-0.2, 0.2]}, 'sampling_region': {'resultant_lower': 0.3, 'resultant_upper': 1.0}, 'status': 'available'}])
    expected = {'offset': 0.0, 'within-cell': 0.6, 'within-sample': 0.7, 'across-samples': 0.8}
    for level, value in expected.items():
        values, _, _ = figures.matrix_values({'sample_summary': summary}, {'stratum': 'eight-hour', 'rows': ['a', 'b'], 'columns': ['a', 'b'], 'level': level})
        assert values[values.requested].value.iloc[0] == value
        assert values[~values.requested].value.isna().all()
    summary.loc[0, 'period_hours'] = np.nan
    for level in expected:
        values, _, _ = figures.matrix_values({'sample_summary': summary}, {'stratum': 'eight-hour', 'rows': ['a'], 'columns': ['b'], 'level': level})
        assert values.value.iloc[0] == 0.6 if level == 'within-cell' else values.value.isna().all()

def test_no_requested_pairs_does_not_draw_placeholder_figures(tmp_path, monkeypatch):
    resolved, paths = fixture(tmp_path, empty_pairs=True)
    captured = capture_figures(monkeypatch)
    result = run_request(resolved, paths, tmp_path / 'pipeline', only=('timing-relationship-figures',))
    assert result.successful, {key: value.outcome.reason for key, value in result.results.items()}
    assert not captured
    manifest = read_document(result.results['timing-relationship-figures'].artifact('relationship_manifest.json'))
    assert manifest['pages'] == []
