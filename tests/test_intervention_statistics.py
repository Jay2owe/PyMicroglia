"""Mean-model authority, original clock and estimand checks."""
import numpy as np
import pytest
from pymicroglia.pipelines.intervention.statistics import compare, settings

def options(**changes):
    return {'method': 'segmented_glsar', 'estimand': 'baseline_trend_adjusted_change', 'baseline_trend': 'linear', 'response_model': 'level_and_slope', 'ar_order': 1, 'model_justification': 'Constructed linear baseline and level shift', 'error_model_justification': 'Constructed stationary autoregressive errors', **changes}

def observations():
    rng = np.random.default_rng(83)
    hours = (np.arange(128) - 48) * 0.5
    from scipy.signal import lfilter
    noise = lfilter([1], [1, -0.4], rng.normal(size=728))[600:]
    values = 7 + 0.12 * hours + 3 * (hours >= 0) + noise
    rows = [{'observation_id': 'original-' + str(i), 'relative_hours': float(t), 'raw_value': float(y), 'sequence_index': i, 'frame_index': i, 'raw_valid': True, 'clock_valid': True} for i, (t, y) in enumerate(zip(hours, values))]
    return (rows[:48], rows[48:])

def test_public_native_reference_and_distinct_raw_change():
    import statsmodels.api as sm
    base, target = observations()
    rows = base + target
    t = np.array([r['relative_hours'] for r in rows])
    y = np.array([r['raw_value'] for r in rows])
    post = (t >= 0).astype(float)
    x = np.column_stack([np.ones(len(y)), t, post, post * t])
    fit = sm.GLSAR(y, x, rho=1, missing='raise').iterative_fit(maxiter=100, rtol=1e-08)
    reference = fit.t_test([0, 0, 1, t[48:].mean()])
    actual = compare(base, target, operation='mean', options=options(), alpha=0.05)
    assert actual['status'] == 'tested'
    assert actual['estimate'] == pytest.approx(float(np.asarray(reference.effect).item()))
    assert actual['p_value'] == pytest.approx(float(np.asarray(reference.pvalue).item()))
    assert [actual['interval_low'], actual['interval_high']] == pytest.approx(reference.conf_int().reshape(-1))
    raw = compare(base, target, operation='mean', options=options(estimand='modelled_window_mean_change'), alpha=0.05)
    assert raw['estimate'] == pytest.approx(float(np.asarray(fit.t_test(x[48:].mean(axis=0) - x[:48].mean(axis=0)).effect).item()))
    assert raw['estimate'] - actual['estimate'] > 3
    assert actual['biological_replicates'] is None and actual['values_transformed'] is False
    assert actual['observation_ids'] == [r['observation_id'] for r in rows]

@pytest.mark.parametrize('failure', ['missing_value', 'missing_frame', 'separated_windows', 'irregular_time', 'bad_clock', 'nonmean', 'short'])
def test_refusals_preserve_original_observations(failure):
    base, target = observations()
    operation = 'mean'
    if failure == 'missing_value':
        target[10]['raw_valid'] = False
    if failure == 'missing_frame':
        target[10]['frame_index'] += 1
    if failure == 'separated_windows':
        target = target[4:]
    if failure == 'irregular_time':
        target[10]['relative_hours'] += 0.1
    if failure == 'bad_clock':
        target[10]['clock_valid'] = False
    if failure == 'nonmean':
        operation = 'median'
    if failure == 'short':
        base = base[-8:]
    result = compare(base, target, operation=operation, options=options(), alpha=0.05)
    assert result['status'] == 'untestable' and result['p_value'] is None and (result['estimate'] is None)
    assert result['observation_ids'] == [r['observation_id'] for r in base + target]

def test_no_probability_for_numerically_deterministic_model():
    base, target = observations()
    for row in base + target:
        row['raw_value'] = 4
    result = compare(base, target, operation='mean', options=options(), alpha=0.05)
    assert result['p_value'] is None and 'degenerate' in result['reason']

def test_assumptions_and_lag_are_explicit():
    for changes in [{'error_model_justification': ''}, {'model_justification': ''}, {'ar_order': 0}, {'min_observations_per_window': 8}, {'estimand': 'significant_change'}, {'baseline_trend': 'automatic'}]:
        with pytest.raises(ValueError):
            settings(options(**changes))
