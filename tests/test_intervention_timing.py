"""Measured episode bounds, original clock censoring and independent evidence."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy import ndimage
from pymicroglia.pipelines import parse
import pymicroglia.pipelines.intervention.timing as timing
import pymicroglia.pipelines.intervention.timing_rules as rules
from pymicroglia.pipelines.intervention.options import resolve_request, run_request
from pymicroglia.pipelines.intervention.windows import prepare
from pymicroglia.pipelines.intervention.evidence import analyse
from pymicroglia.pipelines._screening import file_hash

def settings():
    return {'enabled': True, 'method': 'observed_threshold_episodes', 'baseline_reference': 'saved_window_summary', 'thresholds': {'custom_signal': {'change': 3.0, 'direction': 'either', 'unit': 'software units'}}, 'persistence_hours': 2.0, 'immediate_hours': 1.0, 'recovery': {'enabled': True, 'reference': 'same_baseline', 'tolerances': {'custom_signal': {'value': 0.5, 'unit': 'software units'}}, 'persistence_hours': 2.0}}

def fixture(tmp_path):
    cases = ['immediate', 'delayed', 'recovering', 'gapped', 'lost', 'truncated', 'none', 'brief', 'alternating', 'brief-return']
    rows = []
    clocks = []
    anchors = {}
    for movie in cases:
        anchor = 100.0
        anchors[movie] = {'hours': anchor, 'kind': 'intervention', 'label': 'Software anchor'}
        for frame in range(32):
            t = frame - 8
            if movie == 'truncated' and t > 9:
                continue
            clocks.append({'stem': movie, 'frame_index': frame, 'hours': anchor + t})
            if movie == 'lost' and t > 9:
                continue
            active = t >= 0 if movie == 'immediate' else 3 <= t < 10 if movie in ['recovering', 'gapped'] else 3 <= t <= 4 if movie == 'brief' else t >= 3 if movie != 'none' else False
            delta = 4.0 * active
            if movie == 'alternating' and t >= 3:
                delta = 4.0 * (-1 if frame % 2 else 1)
            if movie == 'brief-return' and t == 23:
                delta = 0.0
            value = 10.0 + delta
            if movie == 'gapped' and 3 <= t <= 5:
                value = np.nan
            rows.append({'stem': movie, 'identity': 7, 'frame_index': frame, 'hours': anchor + t, 'custom_signal': value})
    frame = pd.DataFrame(rows)
    tables = {'cell_frame': frame, 'cell_summary': frame[['stem', 'identity']].drop_duplicates(), 'frame_summary': pd.DataFrame(clocks)}
    paths = {}
    for name, table in tables.items():
        paths[name] = tmp_path / (name + '.csv')
        table.to_csv(paths[name], index=False)
    request = {'pipeline': 'intervention-response', 'measurements': ['custom_signal'], 'summary': 'mean', 'anchors': anchors, 'windows': [{'name': 'baseline', 'coordinate': 'relative_hours', 'start': -8, 'end': 0}, {'name': 'followup', 'coordinate': 'relative_hours', 'start': 0, 'end': 24, 'baseline': 'baseline'}], 'support': {'max_gap_hours': 1.5}, 'timing': settings()}
    return (request, tables, paths, resolve(request, tables, paths))

def resolve(request, tables, paths):
    return resolve_request(parse([request])[0], source_run='timing-software', tables=tables, input_hashes={key: file_hash(path) for key, path in paths.items()})

def calculate(resolved, tables):
    prepared = prepare(resolved, tables)
    evidence = analyse(prepared, resolved)
    return timing.scan(prepared, evidence, resolved, timing.policy(resolved))

def test_known_immediate_delayed_and_recovering_observation_bounds(tmp_path):
    _, tables, _, resolved = fixture(tmp_path)
    data = calculate(resolved, tables)
    rows = data['timing'].set_index('movie')
    immediate = rows.loc['immediate']
    assert immediate.response_delay_hours == 0.0 and immediate.response_lower_hours == -1.0 and (immediate.response_upper_hours == 0.0)
    assert immediate.response_label == 'observed_within_declared_immediate_window'
    delayed = rows.loc['delayed']
    assert delayed.response_delay_hours == 3.0 and delayed.response_lower_hours == 2.0 and (delayed.response_upper_hours == 3.0)
    assert delayed.response_label == 'observed_after_declared_immediate_window'
    recovered = rows.loc['recovering']
    assert recovered.recovery_observed and recovered.recovery_delay_hours == 10.0
    assert recovered.recovery_lower_hours == 9.0 and recovered.recovery_upper_hours == 10.0
    assert recovered.recovery_status == 'observed_sustained_return' and (not recovered.within_cell_supported)
    assert len(rows) == 10 and rows.probability.isna().all()
    assert set(data['episode_members'].observation_id) <= set(data['criteria'].observation_id)

def test_gap_does_not_become_an_exact_onset_or_persistent_observation(tmp_path):
    _, tables, _, resolved = fixture(tmp_path)
    data = calculate(resolved, tables)
    row = data['timing'].loc[lambda x: x.movie.eq('gapped')].iloc[0]
    assert row.response_observed and row.response_delay_hours == 6.0 and pd.isna(row.response_lower_hours)
    assert row.response_timing_status == 'left_unresolved' and row.response_label == 'timing_unresolved' and (row.gap_count > 0)
    selected = data['episodes'].loc[data['episodes'].episode_id.eq(row.response_episode_id)].iloc[0]
    assert selected.qualified_relative_hours == 8.0 and selected.observed_duration_hours == 3.0
    assert not data['support_intervals'].loc[lambda x: ~x.valid, 'observed_interval_hours'].any()
    assert data['criteria'].loc[lambda x: x.movie.eq('gapped') & x.relative_hours.between(3, 5), 'increase'].isna().all()

def test_lost_tracks_and_recording_end_are_distinct_from_recovery(tmp_path):
    _, tables, _, resolved = fixture(tmp_path)
    rows = calculate(resolved, tables)['timing'].set_index('movie')
    lost = rows.loc['lost']
    truncated = rows.loc['truncated']
    assert lost.recovery_status == 'recovery_unobserved_after_lost_observation' and lost.observed_endpoint_hours == 109.0 and (lost.expected_endpoint_hours == 123.0)
    assert truncated.recovery_status == 'response_observed_at_recording_end' and truncated.expected_endpoint_hours == 109.0
    assert not lost.recovery_observed and (not truncated.recovery_observed) and lost.recovery_censored and truncated.recovery_censored
    assert pd.isna(lost.recovery_delay_hours) and pd.isna(truncated.recovery_delay_hours)
    assert rows.loc['delayed', 'recovery_status'] == 'response_observed_at_recording_end'

def test_return_requires_persistence_and_non_significance_is_not_recovery(tmp_path):
    _, tables, _, resolved = fixture(tmp_path)
    rows = calculate(resolved, tables)['timing'].set_index('movie')
    assert rows.loc['brief-return', 'recovery_status'] == 'return_observed_without_required_persistence'
    assert not rows.loc['brief-return', 'recovery_observed'] and pd.isna(rows.loc['brief-return', 'recovery_delay_hours'])
    assert not rows.within_cell_supported.any() and rows.loc['recovering', 'recovery_observed'] and (not rows.loc['delayed', 'recovery_observed'])
    assert not rows.loc['none', 'response_observed'] and rows.loc['none', 'status'] == 'no_observed_qualifying_response'

def test_brief_or_alternating_sign_changes_cannot_supply_persistence(tmp_path):
    _, tables, _, resolved = fixture(tmp_path)
    rows = calculate(resolved, tables)['timing'].set_index('movie')
    for movie in ['brief', 'alternating']:
        assert not rows.loc[movie, 'response_observed'] and rows.loc[movie, 'status'] == 'response_persistence_unresolved'
    assert not rows.loc['alternating', 'recovery_observed']

def test_native_components_preserve_irregular_clock_and_actual_elapsed_support():
    times = [0.0, 0.4, 1.1, 1.6, 2.2, 3.0, 3.7]
    rows = [{'observation_id': str(i), 'hours': t, 'relative_hours': t, 'sequence_index': i, 'frame_index': i, 'raw_valid': True, 'clock_valid': True} for i, t in enumerate(times)]
    marks = [False, True, True, True, False, True, True]
    result = rules.episodes(rows, marks, 1.2, 1.0, kind='response', direction='increase')
    labels, _ = ndimage.label(marks)
    slices = ndimage.find_objects(labels)
    assert [r['observation_ids'] for r in result] == [[str(i) for i in range(s[0].start, s[0].stop)] for s in slices]
    assert result[0]['qualified'] and result[0]['qualified_hours'] == 1.6 and (result[0]['observed_duration_hours'] == pytest.approx(1.2))
    assert result[0]['lower_hours'] == 0.0 and result[0]['upper_hours'] == 0.4
    assert not result[1]['qualified'] and result[1]['observed_duration_hours'] == pytest.approx(0.7)
    rows[2]['frame_index'] = 4
    split = rules.episodes(rows, marks, 1.2, 1.0, kind='response', direction='increase')
    assert not any((item['qualified'] for item in split))

@pytest.mark.parametrize('update', [{'method': 'first_significant_frame'}, {'evidence': {'method': 'frame_t_tests'}}, {'persistence_hours': 0}, {'baseline_reference': 'whole_recording_fit'}, {'window_hours': 2.0}, {'recovery': {'enabled': True, 'reference': 'same_baseline', 'tolerances': {'custom_signal': {'value': 3.0, 'unit': 'software units'}}, 'persistence_hours': 2.0}}])
def test_timing_needs_declared_supported_meaning_and_nonoverlapping_criteria(update):
    with pytest.raises(ValueError):
        rules.policy({**settings(), **update}, ['custom_signal'])

def test_missing_baseline_remains_a_full_unavailable_timing_result(tmp_path):
    request, tables, paths, _ = fixture(tmp_path)
    tables['cell_frame'].loc[tables['cell_frame'].stem.eq('delayed') & tables['cell_frame'].hours.lt(100), 'custom_signal'] = np.nan
    tables['cell_frame'].to_csv(paths['cell_frame'], index=False)
    data = calculate(resolve(request, tables, paths), tables)
    row = data['timing'].loc[lambda x: x.movie.eq('delayed')].iloc[0]
    assert len(data['timing']) == 10 and row.status == 'insufficient_original_baseline' and (not row.response_observed)

def test_disabled_timing_never_scans_episodes(tmp_path, monkeypatch):
    request, tables, paths, _ = fixture(tmp_path)
    request['timing'] = {'enabled': False}
    resolved = resolve(request, tables, paths)

    def forbidden(*a, **k):
        raise AssertionError('Disabled timing performed a scan')
    monkeypatch.setattr(timing, 'scan', forbidden)
    monkeypatch.setattr(rules, 'episodes', forbidden)
    result = run_request(resolved, paths, tmp_path / 'pipeline', only=['response-timing'])
    assert result.successful and result.results['response-timing'].outcome.status == 'skipped-empty'

def test_original_results_reopen_and_timing_settings_invalidate_only_timing(tmp_path, monkeypatch):
    request, tables, paths, resolved = fixture(tmp_path)
    root = tmp_path / 'pipeline'
    first = run_request(resolved, paths, root, only=['response-timing'])
    assert first.successful
    saved = first.results['response-timing']
    data = timing.read_timing(saved, first.results['response-evidence'].outcome.scientific_id)
    assert len(data['timing']) == 10
    before = {str(item.artifact(ref.name)): file_hash(item.artifact(ref.name)) for item in first.results.values() for ref in item.outcome.artifacts}
    originals = {str(path): file_hash(path) for path in paths.values()}
    import pymicroglia.pipelines.intervention.windows as intervention_windows, pymicroglia.pipelines.intervention.evidence as intervention_evidence

    def forbidden(*a, **k):
        raise AssertionError('Saved timing reopening repeated science')
    monkeypatch.setattr(intervention_windows, 'prepare', forbidden)
    monkeypatch.setattr(intervention_evidence, 'analyse', forbidden)
    with monkeypatch.context() as local:
        local.setattr(timing, 'scan', forbidden)
        local.setattr(rules, 'episodes', forbidden)
        repeated = run_request(resolved, paths, root, only=['response-timing'], presentation={'report': {'title': 'New timing presentation'}})
        assert repeated.successful and repeated.results['response-timing'].outcome.status == 'reused'
    request['timing']['persistence_hours'] = 3.0
    changed = resolve(request, tables, paths)
    repeated = run_request(changed, paths, root, only=['response-timing'])
    assert repeated.successful
    assert repeated.results['aligned-windows'].outcome.status == 'reused' and repeated.results['response-evidence'].outcome.status == 'reused'
    assert repeated.results['response-timing'].outcome.scientific_id != saved.outcome.scientific_id
    assert all((file_hash(Path(path)) == value for path, value in {**before, **originals}.items()))
