"""Cross-cell lags have physical direction and retain the whole searched grid."""
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
import pymicroglia.measure.relationship_lag_statistics as native
import pymicroglia.pipelines.coordination.delay as delay, pymicroglia.pipelines.coordination.simultaneous as simultaneous, pymicroglia.pipelines.coordination.inputs as inputs
from pymicroglia.pipelines.coordination.temporal_inputs import ADJUSTMENT_MEANING
from pymicroglia.pipelines.coordination.options import resolve_request, run_request
from pymicroglia.pipelines._screening import file_hash, read_table
from tests.test_coordination_simultaneous import fixture, prepared
from tests.test_coordination_options import resolve, request

def question(formal=True, uncertainty=True):
    return {'enabled': True, 'statistic': 'pearson', 'range_hours': [-2.0, 2.0], 'resolution_hours': 0.5, 'evidence': {'method': 'truncated_time_shift', 'radius_hours': 120.0, 'stationary_series': 'target', 'stationarity_justification': 'Controlled stationary delayed noise software process'} if formal else {'method': 'none'}, 'peak_resolution': {'method': 'stationary_bootstrap', 'block_hours': 5.0, 'resamples': 2000, 'seed': 27, 'confidence': 0.95, 'max_width_hours': 0.5, 'stationarity_justification': 'Known jointly stationary finite-dependence software process'} if uncertainty else {'method': 'none'}}

def temporal_fixture(sign=1.0, reverse=False, values=None, formal=True, uncertainty=True, **changes):
    rng = np.random.default_rng(9269)
    x = rng.normal(size=900)
    y = sign * np.r_[rng.normal(size=2), x[:-2]] + 0.1 * rng.normal(size=len(x))
    if reverse:
        x, y = (y, x)
    _, tables, _ = fixture(values=[x, y] if values is None else values)
    req = request(target_measurements=['custom_signal'], questions={'delay': question(formal, uncertainty)}, **{'inference': {'alpha': 0.05, 'multiple_testing': 'bonferroni', 'correction_scope': 'question'}} if formal else {}, **changes)
    resolved = resolve(tables, req)
    output, _ = simultaneous.analyse(resolved, prepared(resolved, tables), 'temporal')
    output['provenance'] = {'adjustment_interpretation': ADJUSTMENT_MEANING}
    assert output['pair_effects'].empty
    return (resolved, tables, output)

@pytest.mark.parametrize('sign,reverse', [(1, False), (-1, False), (1, True)])
def test_known_delays_use_full_search_and_actual_clock_direction(sign, reverse):
    resolved, _, temporal = temporal_fixture(sign=sign, reverse=reverse)
    output, details = delay.analyse(resolved, temporal, 'delay')
    row = output['pair_effects'].iloc[0]
    profiles = output['profiles']
    assert row.reference == 'custom_signal' and row.target == 'custom_signal'
    assert row.status == 'tested' and row.p_value <= 0.05 and (row.evidence_level == 'pair')
    assert row.resolution_status == 'resolved' and row.estimated_delay_hours == (1 if reverse else -1)
    assert row.delay_hours is None and (not row.delay_supported)
    assert np.sign(row.effect) == sign
    assert abs(profiles.loc[profiles.lag_hours.eq(0), 'effect'].iloc[0]) < 0.15
    assert profiles.tested_observations.nunique() == 1 and profiles.tested_observations.iloc[0] == 412
    assert profiles.coefficient_lower.notna().all() and len(profiles) == 9
    assert len(output['shift_reference']) == 481
    assert len(output['matches']) == 9 * 900 - 20
    shifted = output['matches'].loc[output['matches'].lag_hours.eq(-1)]
    assert np.allclose(shifted.target_hours - shifted.reference_hours, 1.0)
    assert row.uncertainty_scope.startswith('Simultaneous within this curve')
    assert details[0]['result']['search_definition'].startswith('Maximum absolute')

@pytest.mark.parametrize('problem', ['missing_value', 'missing_frame', 'staggered_clock', 'short', 'constant'])
def test_unsupported_search_keeps_every_lag_and_its_original_support(problem):
    resolved, tables, _ = temporal_fixture(uncertainty=False)
    frame = tables['cell_frame']
    selected = frame.identity.eq(2)
    if problem == 'missing_value':
        frame.loc[selected & frame.frame_index.eq(300), 'custom_signal'] = np.nan
    elif problem == 'missing_frame':
        frame.loc[selected, 'frame_index'] *= 2
    elif problem == 'staggered_clock':
        frame.loc[selected, 'hours'] += 0.1
    elif problem == 'short':
        tables['cell_frame'] = frame.loc[frame.frame_index.lt(200)]
    else:
        frame.loc[selected, 'custom_signal'] = 1.0
    temporal, _ = simultaneous.analyse(resolved, prepared(resolved, tables), 'temporal')
    temporal['provenance'] = {'adjustment_interpretation': ADJUSTMENT_MEANING}
    output, _ = delay.analyse(resolved, temporal, 'delay')
    row = output['pair_effects'].iloc[0]
    assert row.status == 'untestable' and row.p_value is None and (row.estimated_delay_hours is None)
    assert len(output['profiles']) == 9
    if problem == 'missing_frame':
        assert not output['matching_intervals'].valid.any()
    if problem == 'staggered_clock':
        assert output['matches'].empty

def test_repeating_and_boundary_peaks_are_explicit_without_delay_certification():
    x = np.tile([0.0, 1.0, -0.2, -1.0], 75)
    for y, expected in [(np.roll(x, 1), {'search-boundary', 'multiple-candidates'}), (x, {'search-boundary'})]:
        resolved, _, temporal = temporal_fixture(values=[x, y], formal=False, uncertainty=False)
        output, _ = delay.analyse(resolved, temporal, 'delay')
        row = output['pair_effects'].iloc[0]
        assert row.resolution_status in expected and row.estimated_delay_hours is None
        assert len(row.empirical_peak_lags_hours) > 1 and (not row.delay_supported)
    rng = np.random.default_rng(77)
    x = rng.normal(size=200)
    y = np.r_[rng.normal(size=4), x[:-4]]
    resolved, _, temporal = temporal_fixture(values=[x, y], formal=False, uncertainty=False)
    output, _ = delay.analyse(resolved, temporal, 'delay')
    assert output['pair_effects'].iloc[0].resolution_status == 'search-boundary'

def test_reference_and_increment_views_have_distinct_unscreened_profiles():
    values = np.random.default_rng(89).normal(size=(3, 40))
    resolved, _, temporal = temporal_fixture(values=values, formal=False, uncertainty=False, representation='changes', shared_reference={'method': 'leave_pair_out_mean', 'min_cells': 1, 'adjustment': 'subtract'})
    output, _ = delay.analyse(resolved, temporal, 'delay')
    assert len(output['pair_effects']) == 9 and len(output['profiles']) == 81
    assert output['pair_effects'].view_id.nunique() == 9
    assert output['pair_effects'].is_hypothesis.sum() == 6
    assert output['pair_effects'].p_value.isna().all()
    assert output['pair_effects'].loc[output['pair_effects'].adjustment.eq('shared_reference'), 'shared_reference_interpretation'].notna().all()

def test_empty_population_has_complete_delay_schema():
    resolved, tables, _ = temporal_fixture(values=np.zeros((1, 8)), formal=False, uncertainty=False)
    output, _ = simultaneous.analyse(resolved, prepared(resolved, tables), 'temporal')
    output['provenance'] = {'adjustment_interpretation': ADJUSTMENT_MEANING}
    result, _ = delay.analyse(resolved, output, 'delay')
    assert result['pair_effects'].empty and result['profiles'].empty
    assert {'lag_hours', 'effect', 'tested_effect', 'paired_observations'} <= set(result['profiles'])

def test_real_delay_producer_reopens_all_preparation_without_refitting(tmp_path):
    resolved, tables, _ = temporal_fixture(values=np.random.default_rng(1).normal(size=(2, 30)), formal=False, uncertainty=False)
    paths = {}
    for name, frame in tables.items():
        paths[name] = tmp_path / (name + '.csv')
        frame.to_csv(paths[name], index=False)
    resolved = resolve_request(resolved.request, source_run='source', tables=tables, input_hashes={name: file_hash(path) for name, path in paths.items()})
    first = run_request(resolved, paths, tmp_path / 'pipeline', only=['lagged-coordination'])
    assert first.successful, {key: value.outcome.reason for key, value in first.results.items()}
    assert len(read_table(first.results['lagged-coordination'].artifact('profiles'))) == 9
    with patch.object(native, 'evaluate', side_effect=AssertionError('No repeated search')), patch.object(simultaneous, 'analyse', side_effect=AssertionError('No repeated representation')), patch.object(inputs, 'prepare', side_effect=AssertionError('No repeated inputs')):
        second = run_request(resolved, paths, tmp_path / 'pipeline', only=['lagged-coordination'])
    assert second.successful and all((row.outcome.status == 'reused' for row in second.results.values()))
