"""Native independent-sample reference values and controlled calibration."""
import numpy as np
import pytest
from scipy import stats
import pymicroglia.measure.behaviour_sample_statistics as backend
OPTIONS = {'method': 'independent_sample_permutation', 'contrasts': [{'reference': 'control', 'target': 'treated'}], 'independent_samples': 'Independent exchangeable controlled samples unused in choosing the state definition', 'resamples': 999, 'seed': 3, 'min_samples': 4, 'alpha': 0.05, 'multiple_testing': 'bonferroni'}

def test_exact_native_permutation_values_and_signed_target_minus_reference():
    a, b = (np.arange(4) / 10, 0.7 + np.arange(4) / 10)
    result = backend.compare(a, b, OPTIONS)
    native = stats.permutation_test((a, b), backend.difference, permutation_type='independent', alternative='two-sided', vectorized=True, n_resamples=999, random_state=np.random.default_rng(3))
    assert result['effect'] == pytest.approx(0.7) and result['p_value'] == pytest.approx(2 / 70)
    assert result['p_value'] == native.pvalue and result['exact'] and (result['permutation_count'] == 70)
    np.testing.assert_array_equal(result['null_distribution'], native.null_distribution)
    opposite = backend.compare(b, a, OPTIONS)
    assert opposite['effect'] == pytest.approx(-0.7) and opposite['p_value'] == result['p_value']

def test_native_bootstrap_interval_matches_independent_sample_resampling():
    a, b = (np.arange(12) / 20, 0.4 + np.arange(10) / 20)
    interval = {'method': 'independent_bootstrap', 'confidence': 0.95, 'resamples': 1000, 'seed': 12}
    result = backend.compare(a, b, {**OPTIONS, 'interval': interval})
    native = stats.bootstrap((a, b), backend.difference, paired=False, vectorized=True, method='percentile', confidence_level=0.95, n_resamples=1000, batch=256, random_state=np.random.default_rng(12))
    np.testing.assert_allclose(result['effect_interval'], [native.confidence_interval.low, native.confidence_interval.high])
    assert result['interval_status'] == 'approximate'
    flat = backend.compare(np.zeros(4), np.ones(4), {**OPTIONS, 'interval': interval})
    assert flat['p_value'] is not None and flat['effect_interval'] is None and (flat['interval_status'] == 'unavailable')

def test_missing_independence_small_samples_and_unknown_methods_do_not_create_tests():
    assert backend.compare(np.zeros(4), np.ones(4), {**OPTIONS, 'independent_samples': None})['status'] == 'untestable'
    assert backend.compare(np.zeros(3), np.ones(4), OPTIONS)['p_value'] is None
    assert backend.compare(np.zeros(4), np.ones(4), {'method': 'none'})['status'] == 'descriptive'
    with pytest.raises(backend.UnsupportedComparison):
        backend.policy({'method': 'frame_shuffle'})
    with pytest.raises(ValueError, match='finite'):
        backend.compare([0.0, np.nan], [1.0, 2.0], OPTIONS)
    with pytest.raises(ValueError, match='repeat'):
        backend.policy({**OPTIONS, 'contrasts': OPTIONS['contrasts'] * 2})
    with pytest.raises(ValueError, match='five expected'):
        backend.policy({**OPTIONS, 'interval': {'method': 'independent_bootstrap', 'confidence': 0.99, 'resamples': 100, 'seed': 2}})

def test_controlled_independent_null_power_and_interval_coverage():
    rng = np.random.default_rng(20260917)
    rejections = sum((backend.compare(rng.normal(size=12), rng.normal(size=12), {**OPTIONS, 'resamples': 499, 'seed': trial})['p_value'] <= 0.05 for trial in range(256)))
    assert rejections <= 22, rejections
    power = sum((backend.compare(rng.normal(0, 0.15, 20), rng.normal(0.6, 0.15, 20), {**OPTIONS, 'seed': trial})['p_value'] <= 0.05 for trial in range(16)))
    assert power == 16
    covered = 0
    for trial in range(128):
        result = backend.compare(rng.normal(0, 0.15, 24), rng.normal(0.4, 0.15, 24), {**OPTIONS, 'resamples': 499, 'interval': {'method': 'independent_bootstrap', 'confidence': 0.95, 'resamples': 1000, 'seed': trial}})
        lower, upper = result['effect_interval']
        covered += lower <= 0.4 <= upper
    assert covered >= 112, covered
    print({'independent_null_rejections': rejections, 'null_trials': 256, 'power': power, 'power_trials': 16, 'approximate_interval_coverage': covered, 'interval_trials': 128})
