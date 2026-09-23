"""Full-search reference/calibration, physical delays and complete saved outcomes."""
from pymicroglia._results import read_document
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy import stats
import pymicroglia.measure.relationship_lag_statistics as statistics
from pymicroglia.pipelines.relationships.options import resolve_request, run_request
from pymicroglia.pipelines._screening import file_hash, read_table
from tests.test_relationship_options import request, lag
from tests.test_relationship_inputs import table
DATA = Path(__file__).parent / 'test_data'
REFERENCE = read_document(DATA / 'relationship_lag_reference.json')
INPUTS = {case['name']: case for case in read_document(DATA / 'relationship_tts_reference.json')['cases']}

def question(dt=0.5, **changes):
    return lag(range_hours=[-4 * dt, 4 * dt], resolution_hours=dt, evidence={'method': 'truncated_time_shift', 'radius_hours': 300 * dt, 'stationary_series': 'target', 'stationarity_justification': 'Controlled stationary delayed noise process'}, peak_resolution={'method': 'stationary_bootstrap', 'block_hours': 10 * dt, 'resamples': 2000, 'seed': 27, 'confidence': 0.95, 'max_width_hours': dt, 'stationarity_justification': 'Controlled jointly stationary process with finite dependence and finite moments'}, **changes)

def delayed(seed=20260910, sign=1.0, delay=2, n=1200):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = sign * np.r_[rng.normal(size=delay), x[:-delay]] + 0.2 * rng.normal(size=n)
    return (x, y)

def trace(hours, values):
    return pd.DataFrame({'hours': hours, 'processed_value': values, 'within_range': True, 'processed_valid': np.isfinite(values)})

@pytest.mark.parametrize('case', REFERENCE['cases'], ids=lambda case: case['name'])
def test_maximum_search_matches_published_general_statistic_reference(case):
    source = INPUTS[case['name']]
    result = statistics.complete_search(source['x'], source['y'], np.array(case['lags']), case['radius'])
    expected = dict(zip(case['shifts'], case['statistics']))
    np.testing.assert_allclose(result['shift_statistics'], [expected[k] for k in result['shifts_observations']], atol=1e-12)
    assert result['exceedances'] == case['exceedances'] and result['unclipped_bound'] == case['bound']

def test_complete_search_calibration_under_temporal_dependence():
    rng, trials, alpha = (np.random.default_rng(20260911), 256, 0.05)
    limit = stats.binom.ppf(1 - 0.001 / 2, trials, alpha)
    counts = []
    for statistic in ('pearson', 'spearman'):
        count = 0
        for _ in range(trials):
            noise = rng.normal(size=(360, 2))
            for i in range(1, len(noise)):
                noise[i] += [0.8, 0.65] * noise[i - 1]
            x, y = noise[200:].T
            count += statistics.complete_search(x, y, np.arange(-2, 3), 39, statistic)['p_value'] <= alpha
        counts.append(count)
    assert all((n <= limit for n in counts)), {'false_positives': counts, 'limit': limit}

@pytest.mark.parametrize('dt,sign,reverse', [(0.5, 1, False), (0.25, -1, False), (0.5, 1, True)])
def test_nonperiodic_delay_is_independent_of_same_time_gate_and_preserves_direction(dt, sign, reverse):
    x, y = delayed(sign=sign)
    if reverse:
        x, y = (y, x)
    assert abs(stats.pearsonr(x, y).statistic) < 0.15
    hours = 50 + dt * np.arange(len(x))
    q = question(dt)
    result = statistics.evaluate(trace(hours, x), trace(hours, y), q, request().support, np.arange(-4, 5) * dt)
    assert result['p_value'] <= 0.05 and result['resolution_status'] == 'resolved'
    assert result['delay_hours'] == (2 if reverse else -2) * dt
    assert np.sign(result['effect']) == sign
    assert result['tested_start_hours'] == 50 + 304 * dt
    assert result['uncertainty']['coverage_scope'].startswith('Simultaneous within this curve')

def test_native_bootstrap_intervals_are_reproduced_without_invented_lag_pairs():
    from arch.bootstrap import StationaryBootstrap
    x, y = delayed()
    lags = np.arange(-4, 5)
    data, _, _ = statistics.lag_embedding(x, y, lags, 300)
    settings = question()['peak_resolution']
    result = statistics.delay_uncertainty(data, lags * 0.5, 0.5, settings, 'pearson')
    native = StationaryBootstrap(10, data, seed=27).conf_int(lambda values: np.corrcoef(values.T)[0, 1:], reps=2000, method='percentile', size=1 - 0.05 / 9)
    np.testing.assert_allclose(result['coefficient_bounds'], native, atol=1e-12)
    np.testing.assert_array_equal(data[:, 3], y[306:898])

def test_bootstrap_curve_coverage_for_controlled_stationary_population():
    rng, trials, failed = (np.random.default_rng(20260912), 80, 0)
    settings = dict(method='stationary_bootstrap', block_hours=8, resamples=400, seed=41, confidence=0.9, max_width_hours=1, stationarity_justification='Known stationary finite moving-average process')
    truth = np.array([1 / np.sqrt(2), 0.0, 0.0])
    for trial in range(trials):
        x = rng.normal(size=602)
        error = rng.normal(size=602)
        y = np.r_[rng.normal(), x[:-1]] + error
        data, _, _ = statistics.lag_embedding(x, y, np.array([-1, 0, 1]))
        result = statistics.delay_uncertainty(data, [-1, 0, 1], 1.0, {**settings, 'seed': trial}, 'pearson')
        low, high = np.array(result['coefficient_bounds'])
        failed += not np.all((low <= truth) & (truth <= high))
    assert failed <= stats.binom.ppf(0.999, trials, 0.1), {'uncovered_curves': failed, 'trials': trials}

@pytest.mark.parametrize('kind', ['gap', 'irregular', 'misaligned_radius', 'short', 'constant'])
def test_unavailable_inference_keeps_reason_and_never_imputes(kind):
    x, y = delayed()
    hours = 50 + 0.5 * np.arange(len(x))
    q = question()
    if kind == 'gap':
        x[600] = np.nan
    if kind == 'irregular':
        hours[600] += 0.1
    if kind == 'misaligned_radius':
        q['evidence']['radius_hours'] += 0.1
    if kind == 'short':
        x, y, hours = (x[:100], y[:100], hours[:100])
    if kind == 'constant':
        y[:] = 1
    result = statistics.evaluate(trace(hours, x), trace(hours, y), q, request().support, np.arange(-4, 5) * 0.5)
    assert result['status'] == 'untestable' and result['p_value'] is None and result['reason']
    assert result['delay_hours'] is None
    assert 'Requested delay uncertainty' in result['resolution_reason']

@pytest.mark.parametrize('profile,bounds,width,expected', [([0.9, 0.5, 0.1], [[0.8, 0.4, 0], [1, 0.6, 0.2]], 1, 'search-boundary'), ([0.1, 0.8, 0.2, 0.8, 0.1], [[0, 0.7, 0.1, 0.7, 0], [0.2, 0.9, 0.3, 0.9, 0.2]], 1, 'multiple-candidates'), ([0.1, 0.8, 0.85, 0.8, 0.1], [[0, 0.6, 0.6, 0.6, 0], [0.2, 0.95, 0.95, 0.95, 0.2]], 1, 'broad')])
def test_boundary_multiple_and_broad_regions_never_become_precise(profile, bounds, width, expected):
    result = statistics.classify_delay(np.arange(len(profile)), profile, bounds, width)
    assert result['resolution_status'] == expected and result['delay_hours'] is None

def test_periodically_repeated_pattern_preserves_competing_delay_candidates():
    x = np.tile([0.0, 1.0, -0.2, -1.0], 300)
    y = np.roll(x, 1)
    q = question()
    q['range_hours'] = [-3.0, 3.0]
    q['peak_resolution']['resamples'] = 3000
    result = statistics.evaluate(trace(np.arange(1200) * 0.5, x), trace(np.arange(1200) * 0.5, y), q, request().support, np.arange(-6, 7) * 0.5)
    assert result['resolution_status'] in {'multiple-candidates', 'search-boundary'}
    assert len(result['candidate_lags_hours']) > 1 and result['delay_hours'] is None

def test_complete_family_and_profiles_are_reused_without_same_time_or_rhythm_fits(tmp_path, monkeypatch):
    from pymicroglia import workbench as circadian
    monkeypatch.setattr(circadian, 'estimate_one', lambda *a, **k: pytest.fail('Unexpected period fit'))
    x, y = delayed()
    frame = pd.concat([table(50 + 0.5 * np.arange(1200), x, y), table(50 + 0.5 * np.arange(1200), x, np.ones(1200), identity=2)], ignore_index=True)
    path = tmp_path / 'cell_frame.csv'
    frame.to_csv(path, index=False)
    req = request(within_cell={'enabled': False}, lag=question(), inference={'alpha': 0.1, 'multiple_testing': 'bh', 'correction_scope': 'all'})
    resolved = resolve_request(req, source_run='source-one', tables={'cell_frame': frame}, input_hashes={'cell_frame': file_hash(path)})
    execution = run_request(resolved, {'cell_frame': path}, tmp_path / 'pipeline', only=('lag-association',))
    assert execution.successful, {k: v.outcome.reason for k, v in execution.results.items()}
    assert 'within-cell-association' not in execution.results
    saved = execution.results['lag-association']
    rows, profiles = [read_table(saved.artifact(name)) for name in ('results', 'profiles')]
    assert len(rows) == 2 and len(profiles) == 18 and (set(rows.family_requested) == {2}) and (set(rows.family_tested) == {1})
    assert rows.iloc[0].delay_hours == -1 and rows.iloc[0].delay_supported
    assert set(profiles.loc[profiles.coefficient_lower.notna(), 'interval_kind']) == {'Approximate simultaneous within-curve percentile coefficient confidence bounds'}
    assert set(rows.reference) == set(profiles.reference) == {req.pairs[0].reference}
    selected = next((item for item in saved.outcome.selections if item.name == 'lag-association-supported'))
    assert selected.members[0]['reference'] == req.pairs[0].reference
    assert all((tuple((row[name] for name in ('source_run', 'movie', 'identity', 'pair_id', 'reference', 'target'))) in {tuple((p[name] for name in ('source_run', 'movie', 'identity', 'pair_id', 'reference', 'target'))) for p in profiles.to_dict('records')} for row in rows.to_dict('records')))
    assert rows.iloc[1].status == 'untestable' and pd.isna(rows.iloc[1].p_value)
    monkeypatch.setattr(statistics, 'evaluate', lambda *a, **k: pytest.fail('Saved-only display repeated search'))
    monkeypatch.setattr(circadian, 'adjust_pvalues', lambda *a, **k: pytest.fail('Display recorrected selected family'))
    reopened = run_request(resolved, {'cell_frame': path}, tmp_path / 'pipeline', only=('lag-association',), presentation={'lag_range_hours': [-1, 1], 'report_cells': [1]})
    assert reopened.successful and reopened.results['lag-association'].outcome.status == 'reused'
    assert reopened.results['lag-association'].outcome.selections == saved.outcome.selections

def test_uncertainty_configuration_does_not_accept_unsupported_or_underresolved_methods():
    q = question()
    q['peak_resolution']['resamples'] = 100
    with pytest.raises(ValueError, match='resamples'):
        statistics.validate_lag_question(q)
    q = question()
    q['peak_resolution']['method'] = 'pointwise_surrogate_band'
    with pytest.raises(ValueError, match='stationary_bootstrap'):
        statistics.validate_lag_question(q)

def test_descriptive_uncertainty_and_complete_grid_contract_are_independent():
    x, y = delayed()
    hours = 50 + 0.5 * np.arange(len(x))
    q = question()
    q['evidence'] = {'method': 'none'}
    result = statistics.evaluate(trace(hours, x), trace(hours, y), q, request().support, np.arange(-4, 5) * 0.5)
    assert result['status'] == 'descriptive' and result['p_value'] is None
    assert result['resolution_status'] == 'resolved' and result['tested_observations'] == 1192
    with pytest.raises(ValueError, match='complete declared'):
        statistics.evaluate(trace(hours, x), trace(hours, y), q, request().support, [-1.0, 0.0, 1.0])
