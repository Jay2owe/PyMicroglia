"""Saved response union, complete membership and honest original-clock plots."""
from types import SimpleNamespace
import numpy as np
from pymicroglia import workbench as circadian
import pymicroglia.pipelines.intervention.reports as reports, pymicroglia.pipelines.intervention.windows as intervention_windows, pymicroglia.pipelines.intervention.evidence as intervention_evidence, pymicroglia.pipelines.intervention.rhythms as intervention_rhythms
from pymicroglia.pipelines._contracts import Settings
from pymicroglia.pipelines.intervention.options import run_request
from tests.test_intervention_evidence import fixture

def context(resolved, execution):
    return SimpleNamespace(request=resolved, dependencies=execution.results, saved=execution.results.__getitem__)

def test_every_saved_responder_and_original_observation_appears_without_retesting(tmp_path, monkeypatch):
    _, _, paths, resolved = fixture(tmp_path)
    saved = run_request(resolved, paths, tmp_path / 'pipeline', only=['response-evidence'])
    assert saved.successful

    def forbidden(*a, **k):
        raise AssertionError('Display reran science')
    for module, name in [(intervention_windows, 'prepare'), (intervention_evidence, 'analyse'), (circadian, 'adjust_pvalues'), (circadian, 'estimate_grouped_rhythms')]:
        monkeypatch.setattr(module, name, forbidden)
    data = reports.collect(context(resolved, saved))
    settings, _ = reports.options(Settings())
    values, statistics, pages = reports.pages(data, settings)
    assert len(data['selected']) == 2 and len([p for p in pages if p['view'] == 'cell_report']) == 3
    selected = data['evidence']['effects'].loc[data['evidence']['effects'].response_supported]
    actual = {(r['movie'], r['identity'], r['measurement']) for r in values.loc[values.kind.eq('effect')].to_dict('records')}
    assert actual == set(selected[['movie', 'identity', 'measurement']].itertuples(index=False, name=None))
    traces = values.loc[values.kind.eq('trace')]
    original = data['windows']['traces'].set_index('observation_id')
    assert not traces.observation_id.duplicated().any()
    assert np.allclose(traces.raw_value, original.loc[traces.observation_id, 'raw_value'], equal_nan=True)
    for row in traces.to_dict('records'):
        assert row['relative_hours'] == row['hours'] - resolved.recordings[row['movie']]['anchor']['hours']
    settings['cells_per_page'] = 1
    settings['shared_y'] = True
    settings['grid_clock'] = 'hours'
    changed, _, more = reports.pages(data, settings)
    assert set(changed.entry_id) == set(values.entry_id)
    grid = [p for p in more if p['view'] == 'trace_grid']
    assert len(grid) == 3
    assert len({(p['cell_ids'][0], p['measurement']) for p in grid}) == 3
    shape = [p for p in grid if p['measurement'] == 'custom_shape']
    assert len(shape) == 2 and shape[0]['y_limits'] == shape[1]['y_limits']
    assert shape[0]['y_limits'][0] < shape[0]['y_limits'][1]

def test_native_rhythm_only_cells_with_repeated_local_numbers_are_selected(tmp_path, monkeypatch):
    from tests.test_intervention_rhythms import fixture as rhythm_fixture
    _, _, paths, resolved = rhythm_fixture(tmp_path, cases=['amplitude-phase', 'period'])
    saved = run_request(resolved, paths, tmp_path / 'pipeline', only=['response-evidence', 'rhythm-changes'])
    assert saved.successful

    def forbidden(*a, **k):
        raise AssertionError('Rhythm report refitted native science')
    monkeypatch.setattr(intervention_rhythms, 'analyse', forbidden)
    monkeypatch.setattr(circadian, 'rhythm_window_comparison', forbidden)
    data = reports.collect(context(resolved, saved))
    settings, _ = reports.options(Settings())
    values, statistics, pages = reports.pages(data, settings)
    assert len(data['selected']) == 2 and {item['cell']['identity'] for item in data['selected'].values()} == {7}
    assert all((item['sources'] == ['rhythm-changes'] for item in data['selected'].values()))
    assert set(values.measurement) == {'custom_signal'}
    assert len([p for p in pages if p['view'] == 'cell_report']) == 2
    assert len([p for p in pages if p['view'] == 'trace_grid']) == 1
    assert statistics.loc[statistics.kind.eq('rhythm_change'), 'direct_id'].nunique() == 6

def test_empty_selection_has_no_report_pages_and_disabled_branches_remain_explicit(tmp_path):
    _, _, _, resolved = fixture(tmp_path)
    data = reports.collect(SimpleNamespace(request=resolved, dependencies={}))
    assert data['selected'] == {} and data['branches']['rhythm-changes']['status'] == 'disabled'
    assert reports.pages(data, reports.options(Settings())[0])[2] == []

def test_trace_geometry_never_connects_missing_values_frames_or_clock_resets():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path.home() / '.claude/skills/plot-that/scripts'))
    from pymicroglia.visualisation.figures import load as _figure_schema
    _figure_schema()
    from pymicroglia.visualisation.panels import intervention_reports as panel
    import matplotlib.pyplot as plt
    points = [{'kind': 'trace', 'sequence_index': i, 'frame_index': frame, 'hours': float(frame), 'relative_hours': float(frame) - 3.0, 'raw_value': value, 'raw_valid': np.isfinite(value), 'clock_valid': True} for i, (frame, value) in enumerate([(0, 1.0), (1, 2.0), (2, np.nan), (3, 3.0), (5, 4.0), (6, 5.0)])]
    fig, axis = plt.subplots()
    from pymicroglia.figure_tables.intervention_report_display import trace as prepare_trace
    anchor={'hours':3.,'kind':'control'}
    counts = panel.trace(axis, prepare_trace(points,'relative_hours',anchor,1.5), 'relative_hours', anchor, 'units')
    assert counts == {'original_points': 6, 'unplotted_invalid_points': 1, 'segments': 3}
    assert [list(line.get_xdata()) for line in axis.lines[:-1]] == [[-3.0, -2.0], [0.0], [2.0, 3.0]]
    assert 'control anchor' in axis.get_xlabel()
    plt.close(fig)
