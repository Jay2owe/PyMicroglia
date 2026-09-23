from tests.panel_helpers import reopen_page
"""Supported unions and frozen observation reports preserve complete identities."""
import json
import numpy as np
import pandas as pd
import pytest
import pymicroglia.pipelines.relationships.report_figures as figures
from pymicroglia.pipelines.relationships.options import resolve_request, run_request
from pymicroglia.pipelines.relationships.inputs import PAIR_KEYS
from pymicroglia.pipelines._screening import file_hash, read_table
from tests.test_relationship_options import request, lag
from tests.test_relationship_inputs import table
from tests.test_rhythm_relationship_figures import capture_figures

@pytest.fixture(autouse=True)
def restore_style():
    import matplotlib as mpl
    with mpl.rc_context():
        yield

def inputs(tmp_path, descriptive=False):
    rng = np.random.default_rng(20260919)
    x = rng.normal(size=400)
    frames = []
    for movie, identity, y in (('a', 1, x + 0.1 * rng.normal(size=400)), ('a', 2, np.r_[rng.normal(), x[:-1]] + 0.1 * rng.normal(size=400)), ('b', 1, -np.r_[rng.normal(), x[:-1]] + 0.1 * rng.normal(size=400))):
        frames.append(table(50 + np.arange(400) * 0.5, x, y, movie=movie, identity=identity))
    frame = pd.concat(frames, ignore_index=True)
    path = tmp_path / 'cell_frame.csv'
    frame.to_csv(path, index=False)
    evidence = {'method': 'none'} if descriptive else {'method': 'truncated_time_shift', 'radius_hours': 60, 'stationary_series': 'target', 'stationarity_justification': 'Controlled stationary noise and finite shifted copies'}
    req = request(within_cell={'enabled': True, 'statistic': 'pearson', 'evidence': evidence}, lag=lag(range_hours=[-0.5, 0.5], evidence=evidence), inference={} if descriptive else {'alpha': 0.1, 'multiple_testing': 'bh', 'correction_scope': 'all'})
    return (resolve_request(req, source_run='report-source', tables={'cell_frame': frame}, input_hashes={'cell_frame': file_hash(path)}), {'cell_frame': path})

def keys(frame):
    return {tuple((row[name] for name in PAIR_KEYS)) for row in frame.to_dict('records')}

def test_union_reports_lag_only_unresolved_cells_and_reopens_without_science(tmp_path, monkeypatch):
    from pymicroglia import workbench as circadian
    import pymicroglia.measure.relationship_statistics as relationship_statistics
    import pymicroglia.measure.relationship_lag_statistics as relationship_lag_statistics
    resolved, paths = inputs(tmp_path)
    execution = run_request(resolved, paths, tmp_path / 'pipeline', only=('relationship-report-selection',))
    assert execution.successful, {k: v.outcome.reason for k, v in execution.results.items()}
    selection = execution.results['relationship-report-selection']
    members = read_table(selection.artifact('report_members'))
    within = read_table(execution.results['within-cell-association'].artifact('results'))
    lag_rows = read_table(execution.results['lag-association'].artifact('results'))
    assert keys(members) == keys(within.loc[within.significant]) | keys(lag_rows.loc[lag_rows.significant])
    assert len(members) == 3 and members.identity.eq(1).sum() == 2
    assert members.loc[members.identity.eq(2), 'supporting_questions'].iloc[0] == ['lag']
    assert not lag_rows.delay_supported.any()
    captured = capture_figures(monkeypatch)

    def forbidden(*a, **k):
        raise AssertionError('Report repeated scientific analysis')
    for obj, name in ((circadian, 'estimate_one'), (circadian, 'detrend_trace'), (circadian, 'adjust_pvalues'), (relationship_statistics, 'same_time_evidence'), (relationship_lag_statistics, 'evaluate')):
        monkeypatch.setattr(obj, name, forbidden)
    rendered = run_request(resolved, paths, tmp_path / 'pipeline', only=('relationship-reports',))
    assert rendered.successful, {k: v.outcome.reason for k, v in rendered.results.items()}
    assert len(captured) == 4
    saved = rendered.results['relationship-reports']
    entries = read_table(saved.artifact('entries.json'))
    assert len(entries) == 6 and entries.entry_id.is_unique
    for view in ('pair', 'cells'):
        assert keys(entries.loc[entries.view.eq(view)]) == keys(members)
    original_pairs = read_table(execution.results['paired-inputs'].artifact('same_time_pairs'))
    plotted = pd.concat([drawing.figure_data for ctx, drawing in captured])
    scatter = plotted.loc[plotted.kind.eq('scatter')]
    assert set(zip(scatter.reference_observation, scatter.target_observation)) == set(zip(original_pairs.reference_observation, original_pairs.target_observation))
    trace = plotted.loc[plotted.kind.eq('trace')]
    assert trace.hours.min() == 50 and trace.hours.max() == 249.5
    for ctx, drawing in captured:
        ctx.cached.clear()
        cold = reopen_page(ctx)
        pd.testing.assert_frame_equal(cold.figure_data, drawing.figure_data)
        import matplotlib.pyplot as plt
    reordered = run_request(resolved, paths, tmp_path / 'pipeline', only=('relationship-reports',), presentation={'relationship_reports': {'cells_per_page': 1, 'reverse_cells': True}})
    assert reordered.successful, {k: v.outcome.reason for k, v in reordered.results.items()}
    changed = read_table(reordered.results['relationship-reports'].artifact('entries.json'))
    assert set(changed.entry_id) == set(entries.entry_id)
    assert reordered.results['relationship-report-selection'].outcome.selections == selection.outcome.selections

def test_empty_supported_union_records_skipped_branch(tmp_path, monkeypatch):
    resolved, paths = inputs(tmp_path, descriptive=True)
    monkeypatch.setattr(figures, 'produce', lambda *a: pytest.fail('An empty report branch attempted drawing'))
    execution = run_request(resolved, paths, tmp_path / 'pipeline', only=('relationship-reports',))
    assert execution.successful, {k: v.outcome.reason for k, v in execution.results.items()}
    assert execution.results['relationship-reports'].outcome.status == 'skipped-empty'
    saved = execution.results['relationship-report-selection']
    assert read_table(saved.artifact('report_members')).empty
    assert saved.outcome.selections[0].rule['precise_delay_required'] is False

def test_trace_breaks_keep_masks_clocks_and_separate_values():
    from pymicroglia.visualisation.figures import load as _figure_schema
    _figure_schema()
    from pymicroglia.figure_tables.relationship_report_display import broken_trace
    frame = pd.DataFrame({'hours': [50, 51, 52, 57, 58], 'raw_value': [2, 3, np.nan, 4, 5], 'processed_value': [-1, 0, np.nan, 1, 2], 'raw_valid': [True, True, False, True, True], 'processed_valid': [True, True, False, True, True], 'within_range': [False, True, True, True, True]})
    x, y = broken_trace(frame, 'raw_value', 2)
    np.testing.assert_allclose(x, [50, 51, 52, np.nan, 57, 58], equal_nan=True)
    np.testing.assert_allclose(y, [np.nan, 3, np.nan, np.nan, 4, 5], equal_nan=True)
    _, processed = broken_trace(frame, 'processed_value', 2)
    np.testing.assert_allclose(processed, [np.nan, 0, np.nan, np.nan, 1, 2], equal_nan=True)

@pytest.mark.parametrize('settings', [{'cells_per_page': 0}, {'cells_per_page': True}, {'reverse_cells': 'yes'}, {'strongest_only': 3}])
def test_invalid_or_implicit_selection_options_are_refused(settings):
    from pymicroglia.pipelines._contracts import Settings
    with pytest.raises(ValueError):
        figures.options(Settings({'relationship_reports': settings}))
