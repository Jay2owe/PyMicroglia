"""Window membership and elapsed exposure are hand-checked on original clocks."""
import numpy as np
import pandas as pd
from tests.test_intervention_options import fixture, declaration
from pymicroglia.pipelines import parse
from pymicroglia.pipelines.intervention.options import resolve_request, run_request
from pymicroglia.pipelines.intervention.windows import prepare, read_windows
from pymicroglia.pipelines._screening import file_hash

def refresh(request, tables, paths):
    for name, frame in tables.items():
        if name not in paths:
            paths[name] = next(iter(paths.values())).parent / (name + '.csv')
        frame.to_csv(paths[name], index=False)
    return resolve_request(request, source_run='source', tables=tables, input_hashes={name: file_hash(path) for name, path in paths.items()})

def test_exact_half_open_event_windows_and_original_elapsed_support(tmp_path):
    _, tables, _, resolved = fixture(tmp_path, support={'max_gap_hours': 1.0, 'min_coverage': 0.5})
    result = prepare(resolved, tables)
    assert len(result['cells']) == 4 and len(result['windows']) == 8 and (len(result['comparisons']) == 4)
    assert len(result['traces']) == 32 and len(result['window_members']) == 32
    assert not result['window_members'].observation_id.duplicated().any()
    windows = result['windows']
    baseline = windows.loc[windows.window.eq('baseline')]
    response = windows.loc[windows.window.eq('response')]
    assert baseline.observations.eq(4).all() and baseline.observed_interval_hours.eq(1.5).all() and baseline.coverage_fraction.eq(0.75).all()
    assert response.observed_interval_hours.eq(1.5).all() and response.recording_truncated.all()
    assert result['comparisons'].eligible.all()
    assert (result['comparisons'].target_value - result['comparisons'].baseline_value).eq(4.0).all()
    zero = result['window_members'].loc[result['window_members'].relative_hours.eq(0)]
    assert len(zero) == 4 and zero.window.eq('response').all()
    for movie, start in [('treated', 50.0), ('control', 60.0)]:
        trace = result['traces'].loc[result['traces'].movie.eq(movie) & result['traces'].identity.eq(7)]
        assert trace.hours.tolist() == [start + i * 0.5 for i in range(8)] and trace.relative_hours.tolist() == [-2 + i * 0.5 for i in range(8)]

def test_late_lost_missing_and_truncated_cells_are_kept_without_filled_gaps(tmp_path):
    request, tables, paths, _ = fixture(tmp_path)
    frame = tables['cell_frame']
    frame = frame.loc[~(frame.stem.eq('treated') & frame.identity.eq(8) & frame.frame_index.ge(4))].copy()
    late = frame.loc[frame.stem.eq('control') & frame.identity.eq(7) & frame.frame_index.ge(4)].assign(identity=9)
    frame = pd.concat([frame, late], ignore_index=True)
    frame.loc[frame.stem.eq('control') & frame.identity.eq(7) & frame.frame_index.eq(1), 'custom_signal'] = np.nan
    frame = frame.loc[~(frame.stem.eq('treated') & frame.identity.eq(7) & frame.frame_index.eq(2))]
    tables['cell_frame'] = frame
    tables['cell_summary'] = pd.concat([tables['cell_summary'], pd.DataFrame([{'stem': 'control', 'identity': 9}])], ignore_index=True)
    resolved = refresh(request, tables, paths)
    result = prepare(resolved, tables)
    windows = result['windows']
    assert len(result['cells']) == 5 and len(windows) == 10 and (len(result['comparisons']) == 5)
    assert windows.loc[windows.movie.eq('control') & windows.identity.eq(9) & windows.window.eq('baseline'), 'observations'].iloc[0] == 0
    assert windows.loc[windows.movie.eq('treated') & windows.identity.eq(8) & windows.window.eq('response'), 'summary_value'].isna().all()
    gap = windows.loc[windows.movie.eq('treated') & windows.identity.eq(7) & windows.window.eq('baseline')].iloc[0]
    assert gap.observed_interval_hours == 0.5 and gap.gap_count == 1 and (not gap.eligible)
    invalid = result['intervals'].loc[~result['intervals'].valid]
    assert {'missing_original_frame', 'invalid_value_or_clock'} <= set(invalid.reason)
    assert invalid.observed_interval_hours.eq(0).all()

def test_original_recording_clock_retains_frames_after_tracks_disappear(tmp_path):
    request, tables, paths, _ = fixture(tmp_path)
    tables['frame_summary'] = tables['cell_frame'][['stem', 'frame_index', 'hours']].drop_duplicates()
    tables['cell_frame'] = tables['cell_frame'].loc[tables['cell_frame'].frame_index.lt(5)]
    resolved = refresh(request, tables, paths)
    assert 'frame_summary' in resolved.inputs.table_hashes
    result = prepare(resolved, tables)
    control = result['recordings'].loc[result['recordings'].movie.eq('control')].iloc[0]
    assert control.recorded_end_hours == 63.5 and control.clock_source == 'original_recording_frame_table'
    assert not result['windows'].loc[result['windows'].window.eq('response'), 'eligible'].any()
    response = result['windows'].loc[lambda x: x.movie.eq('control') & x.window.eq('response')].iloc[0]
    assert response.first_captured_hours == 62.0 and response.last_captured_hours == 63.5
    assert response.last_observed_hours == 62.0

def test_summary_only_windows_keep_values_but_do_not_invent_observations(tmp_path):
    _, tables, paths, _ = fixture(tmp_path)
    tables['windowed'] = pd.DataFrame([{'stem': movie, 'identity': identity, 'window': w['name'], 'window_coordinate': 'relative_hours', 'window_start': w['start'], 'window_end': w['end'], 'window_anchor_hours': 52.0 if movie == 'treated' else 62.0, 'summary_operation': 'mean', 'saved_metric': float(identity + (10 if w.get('baseline') else 0))} for movie in ['treated', 'control'] for identity in [7, 8] for w in declaration()['windows']])
    request = parse([declaration(measurements=[{'column': 'saved_metric', 'table': 'windowed'}], table_grains={'windowed': ['identity', 'window']})])[0]
    resolved = refresh(request, tables, paths)
    result = prepare(resolved, tables)
    assert result['traces'].empty and result['window_members'].empty and result['intervals'].empty
    assert len(result['windows']) == 8 and result['windows'].summary_value.notna().all()
    assert result['windows'].observations.isna().all() and (not result['windows'].eligible.any())
    assert result['windows'].reasons.map(lambda reasons: 'unavailable_observation_count' in reasons).all()

def test_saved_windows_reopen_without_preparing_again_and_validate_full_sources(tmp_path, monkeypatch):
    _, tables, paths, resolved = fixture(tmp_path)
    root = tmp_path / 'pipeline'
    first = run_request(resolved, paths, root, only=['aligned-windows'])
    assert first.successful
    saved = first.results['aligned-windows']
    data = read_windows(saved)
    assert len(data['windows']) == 8
    before = {str(saved.artifact(ref.name)): file_hash(saved.artifact(ref.name)) for ref in saved.outcome.artifacts}
    import pymicroglia.pipelines.intervention.windows as intervention_windows

    def forbidden(*args, **kwargs):
        raise AssertionError('Reopening recalculated original windows')
    monkeypatch.setattr(intervention_windows, 'prepare', forbidden)
    reopened = run_request(resolved, paths, root, only=['aligned-windows'], presentation={'report': {'title': 'New display'}})
    assert reopened.successful and reopened.results['aligned-windows'].outcome.status == 'reused'
    from pathlib import Path
    assert all((file_hash(Path(path)) == value for path, value in before.items()))
