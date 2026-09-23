"""Hand-verifiable physical-time examples, never biological validation data."""
import numpy as np
import pandas as pd
import pytest
import pymicroglia.pipelines.behaviour.durations as duration, pymicroglia.pipelines._screening as screening
from pymicroglia.pipelines.behaviour.options import ObservationKey, StateKey, run_request
from pymicroglia.pipelines._contracts import CellKey
from tests.test_behaviour_validation import fixture
SETTINGS = {'interval_rule': 'adjacent_midpoint', 'max_gap_hours': 2.0, 'transition_interval_hours': [0.4, 0.6], 'sample_aggregation': 'mean', 'comparison': {'method': 'none'}}

def sequence(labels=(0, 0, 1, 1, 0), hours=(0.0, 0.5, 1.0, 2.0, 2.5), frames=None, movie='a', time_range=None):
    key = CellKey('controlled-source', movie, 7)
    meta = {'source_run': key.source_run, 'movie': movie, 'identity': 7, 'model_id': 'controlled-model', 'sample': movie, 'sample_confirmed': True, 'condition': 'one', 'role': 'assignment_only'}
    rows = []
    for index, (label, hour) in enumerate(zip(labels, hours)):
        frame = index if frames is None else frames[index]
        inside = time_range is None or (hour is not None and time_range[0] <= hour < time_range[1])
        valid = hour is not None
        status = 'outside_range' if not inside else 'invalid_time' if not valid else 'missing_features' if label is None else 'assigned'
        rows.append({**meta, 'observation_id': ObservationKey(key, frame, hour).record_id if valid else 'missing-clock-' + movie + str(frame), 'frame_index': frame, 'hours': hour, 'time_status': 'recorded' if valid else 'missing', 'within_range': inside, 'status': status, 'state_id': StateKey(meta['model_id'], label).record_id if status == 'assigned' else None, 'component': label if status == 'assigned' else None})
    states = pd.DataFrame([{'model_id': meta['model_id'], 'state_id': StateKey(meta['model_id'], component).record_id, 'component': component} for component in [0, 1]])
    return (pd.DataFrame(rows, columns=[*meta, 'observation_id', 'frame_index', 'hours', 'time_status', 'within_range', 'status', 'state_id', 'component']), states, pd.DataFrame([meta]))

def test_hand_counted_physical_time_bouts_diagonal_and_switches():
    rows, states, cells = sequence()
    result = duration.statistics(rows, states, cells, SETTINGS)
    summary = result['cell_statistics'].iloc[0]
    assert summary.observed_hours == 2.5 and summary.assigned_hours == 2.5 and (summary.unknown_hours == 0)
    assert summary.valid_transition_opportunities == 3 and summary.valid_transition_hours == 1.5
    assert summary.switches == 2 and summary.switches_per_valid_transition_hour == pytest.approx(4 / 3)
    assert summary.bouts == 3 and summary.complete_bouts == 1 and (summary.incomplete_bouts == 2)
    occupancy = result['occupancy'].sort_values('component')
    assert occupancy.state_hours.tolist() == [1.0, 1.5] and occupancy.fraction_observed.tolist() == [0.4, 0.6]
    bouts = result['bouts']
    assert bouts.bout_id.is_unique and sorted(sum(bouts.observation_ids.tolist(), [])) == sorted(rows.observation_id)
    complete = bouts.loc[bouts.complete].iloc[0]
    assert complete.allocated_start_hours == 0.75 and complete.allocated_end_hours == 2.25
    assert complete.onset_bounds_hours == [0.5, 1.0] and complete.ending_bounds_hours == [2.0, 2.5]
    transitions = result['transitions']
    a, b = states.state_id
    assert transitions.loc[transitions.source_state_id.eq(a)].sort_values('target_state_id').probability.sum() == 1.0
    assert transitions.loc[transitions.source_state_id.eq(a) & transitions.target_state_id.eq(a), 'probability'].iloc[0] == 0.5
    assert transitions.loc[transitions.source_state_id.eq(b) & transitions.target_state_id.eq(b), 'probability'].iloc[0] == 0.0
    assert result['steps'].reason.tolist() == ['supported', 'supported', 'outside_transition_interval', 'supported']

def test_unknown_halves_remain_observed_but_never_bridge_switches():
    result = duration.statistics(*sequence([0, 0, None, 1, 1], [0.0, 0.5, 1.0, 1.5, 2.0]), SETTINGS)
    summary = result['cell_statistics'].iloc[0]
    assert summary.observed_hours == 2.0 and summary.unknown_hours == 0.5 and (summary.unknown_fraction_observed == 0.25)
    assert summary.switches == 0 and summary.valid_transition_hours == 1.0 and (summary.switches_per_valid_transition_hour == 0.0)
    assert result['occupancy'].state_hours.tolist() == [0.75, 0.75]
    assert result['occupancy'].fraction_assigned.tolist() == [0.5, 0.5]
    assert not result['bouts'].complete.any() and len(result['bouts']) == 2
    assert result['bouts'].ending_reason.iloc[0] == 'unknown_assignment'
    assert result['bouts'].onset_reason.iloc[1] == 'unknown_assignment'

@pytest.mark.parametrize('frames,hours,reason', [([0, 1, 3, 4], [0.0, 0.5, 1.5, 2.0], 'missing_original_frame'), ([0, 1, 2, 3], [0.0, 0.5, 3.5, 4.0], 'excess_time_gap')])
def test_gaps_do_not_extend_bouts_or_time(frames, hours, reason):
    result = duration.statistics(*sequence([0, 0, 1, 1], hours, frames), SETTINGS)
    summary = result['cell_statistics'].iloc[0]
    assert summary.observed_hours == 1.0 and summary.unobserved_hours == hours[-1] - 1.0
    assert summary.switches == 0 and summary.bouts == 2 and (summary.complete_bouts == 0)
    assert result['steps'].reason.iloc[1] == reason
    assert result['bouts'].allocated_hours.tolist() == [0.5, 0.5]
    assert result['exposures'].duration_hours.sum() == summary.reference_hours

def test_missing_clock_and_half_open_window_leave_unsupported_time_visible():
    result = duration.statistics(*sequence([0] * 5, [0.0, 0.5, None, 1.5, 2.0]), SETTINGS)
    summary = result['cell_statistics'].iloc[0]
    assert summary.observed_hours == 1.0 and summary.unobserved_hours == 1.0
    bounds = (0.25, 1.75)
    result = duration.statistics(*sequence([0] * 5, [0.0, 0.5, 1.0, 1.5, 2.0], time_range=bounds), SETTINGS, bounds)
    summary = result['cell_statistics'].iloc[0]
    assert summary.reference_hours == 1.5 and summary.observed_hours == 1.0 and (summary.unobserved_hours == 0.5)
    assert result['bouts'].observation_ids.map(len).tolist() == [3]
    assert not result['bouts'].complete.any()

@pytest.mark.parametrize('hours', [[0.0, 0.5, 0.5, 1.0], [0.0, 1.0, 0.5, 1.5]])
def test_invalid_global_clock_withholds_overlapping_time_inference(hours):
    result = duration.statistics(*sequence([0, 0, 1, 1], hours), SETTINGS)
    summary = result['cell_statistics'].iloc[0]
    assert summary.status == 'invalid_clock_order' and summary.observed_hours == 0
    assert pd.isna(summary.reference_hours) and pd.isna(summary.unobserved_hours)
    assert result['exposures'].empty and (not result['steps'].valid_transition_opportunity.any())
    assert result['occupancy'].fraction_observed.isna().all()
    assert result['bouts'].observed_duration_hours.isna().all() and (not result['bouts'].complete.any())

def test_single_points_empty_cells_and_zero_switches_are_distinct():
    rows, states, cells = sequence([0], [0.0])
    extra = cells.copy()
    extra['movie'] = 'missing'
    result = duration.statistics(rows, states, pd.concat([cells, extra]), SETTINGS)
    assert len(result['cell_statistics']) == 2 and len(result['occupancy']) == 4 and (len(result['transitions']) == 8)
    assert result['cell_statistics'].switches_per_valid_transition_hour.isna().all()
    assert result['transitions'].probability.isna().all()
    assert result['bouts'].duration_status.tolist() == ['point_only'] and result['bouts'].observed_duration_hours.isna().all()
    empty = duration.statistics(rows.iloc[:0], states, cells.iloc[:0], SETTINGS)
    assert all((table.empty and 'model_id' in table for table in empty.values()))

def test_frame_rates_and_irregular_intervals_remain_separate_strata():
    rows, states, cells = sequence([0, 1, 1], [50.0, 50.25, 50.5], movie='a')
    other, _, other_cells = sequence([0, 1, 1], [50.0, 50.5, 51.0], movie='b')
    result = duration.statistics(pd.concat([rows, other]), states, pd.concat([cells, other_cells]), {**SETTINGS, 'transition_interval_hours': [0.0, 1.0]})
    assert set(result['transitions'].interval_stratum_hours) == {0.25, 0.5}
    assert result['cell_statistics'].observed_hours.tolist() == [0.5, 1.0]
    assert result['cell_statistics'].switches_per_valid_transition_hour.tolist() == [2.0, 1.0]
    assert result['cell_statistics'].transition_interval_profile.tolist() == [[{'hours': 0.25, 'opportunities': 2}], [{'hours': 0.5, 'opportunities': 2}]]
    irregular = duration.statistics(*sequence(), {**SETTINGS, 'transition_interval_hours': [0.0, 2.0]})
    assert set(irregular['transitions'].interval_stratum_hours) == {0.5, 1.0}
    assert len(irregular['transitions']) == 8

def test_time_units_change_with_clock_scale_and_model_mismatch_is_refused():
    rows, states, cells = sequence()
    result = duration.statistics(rows, states, cells, SETTINGS)
    scaled = rows.copy()
    scaled['hours'] *= 2
    altered = duration.statistics(scaled, states, cells, {**SETTINGS, 'max_gap_hours': 4.0, 'transition_interval_hours': [0.8, 1.2]})
    assert altered['cell_statistics'].observed_hours.iloc[0] == 2 * result['cell_statistics'].observed_hours.iloc[0]
    assert altered['cell_statistics'].switches_per_valid_transition_hour.iloc[0] == result['cell_statistics'].switches_per_valid_transition_hour.iloc[0] / 2
    pd.testing.assert_series_equal(altered['occupancy'].fraction_observed, result['occupancy'].fraction_observed)
    with pytest.raises(ValueError, match='different model'):
        duration.statistics(rows.assign(model_id='other'), states, cells, SETTINGS)
    with pytest.raises(ValueError, match='repeat'):
        duration.statistics(pd.concat([rows, rows.iloc[:1]]), states, cells, SETTINGS)

def test_actual_pipeline_reconciles_exposure_and_reopens_without_calculation(tmp_path, monkeypatch):
    monkeypatch.setattr(screening, 'producer_identity', lambda: {'implementation': 'controlled-duration-test'})
    resolved, paths, source = fixture(tmp_path)
    first = run_request(resolved, paths, tmp_path / 'pipeline', only=('durations-and-switches',))
    assert first.successful, {key: value.outcome.reason for key, value in first.results.items()}
    saved = first.results['durations-and-switches']
    tables, provenance = duration.read_statistics(saved)
    assert len(tables['cell_statistics']) == 124 and tables['cell_statistics'].observations.sum() == len(source)
    assert tables['bouts'].bout_id.is_unique
    assert tables['exposures'].duration_hours.sum() == pytest.approx(tables['cell_statistics'].reference_hours.sum())
    for key, cell in tables['cell_statistics'].set_index(duration.KEYS).iterrows():
        exposure = tables['exposures'].set_index(duration.KEYS).loc[[key]]
        assert exposure.loc[exposure.support.ne('unobserved'), 'duration_hours'].sum() == pytest.approx(cell.observed_hours)
        assert cell.assigned_hours + cell.unknown_hours == pytest.approx(cell.observed_hours)
        assert cell.observed_hours + cell.unobserved_hours == pytest.approx(cell.reference_hours)
    before = {ref.name: screening.file_hash(saved.artifact(ref.name)) for ref in saved.outcome.artifacts}

    def forbidden(*a, **k):
        raise AssertionError('Reopen recalculated physical-time statistics')
    monkeypatch.setattr(duration, 'statistics', forbidden)
    second = run_request(resolved, paths, tmp_path / 'pipeline', only=('durations-and-switches',), presentation={'state_order': [1, 0]})
    assert second.successful and all((result.outcome.status == 'reused' for result in second.results.values()))
    assert all((screening.file_hash(saved.artifact(name)) == fingerprint for name, fingerprint in before.items()))
    assert not provenance['inference_performed'] and (not provenance['assignments_changed'])
