"""Frozen native state candidates reproduce public fits without held-out learning."""
import copy
import json
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler, StandardScaler
import pymicroglia.states.behaviour_models as model
from tests.test_behaviour_options import declaration

@pytest.fixture(autouse=True)
def bounded_native_threads():
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        yield

def inputs(scaling='standard', missing=False):
    rng = np.random.default_rng(731)
    values = np.concatenate([rng.normal([-3, -1], [0.3, 0.2], (90, 2)), rng.normal([3, 2], [0.3, 0.2], (90, 2))])
    if missing:
        values[::9, 1] = np.nan
    frame = pd.DataFrame(values, columns=['signal', 'shape'])
    definitions = [{'column': key, 'table': 'cell_frame', 'grain': ['identity', 'frame_index'], 'unit': 'arbitrary'} for key in frame]
    block = declaration()
    learning = block['learning']
    learning['scaling']['method'] = scaling
    if missing:
        learning['missing'] = {'method': 'median', 'max_fraction': 0.5}
        block['assignment']['min_observed_fraction'] = 0.5
    membership = [{'observation_id': 'original-' + str(i), 'learning_weight': 1.0} for i in range(len(frame))]
    return (frame, definitions, learning, block['candidates'][1], block['assignment'], membership)

@pytest.mark.parametrize('covariance', ['full', 'tied', 'diag', 'spherical'])
@pytest.mark.parametrize('scaling,missing', [('standard', False), ('robust', True), ('none', False)])
def test_saved_model_matches_public_native_fit_and_cold_replay(covariance, scaling, missing, monkeypatch):
    frame, definitions, learning, candidate, assignment, membership = inputs(scaling, missing)
    candidate['covariance_type'] = covariance
    result = model.fit_candidate(frame, definitions, learning, candidate, assignment, membership, {'kind': 'raw'})
    assert result['status'] == 'fitted'
    frozen = json.loads(json.dumps(result['model'], allow_nan=False))
    values = frame.to_numpy()
    if missing:
        values = SimpleImputer(strategy='median').fit_transform(values)
    if scaling != 'none':
        values = (StandardScaler() if scaling == 'standard' else RobustScaler()).fit_transform(values)
    native = GaussianMixture(**model.candidate_parameters(candidate)).fit(values)
    np.testing.assert_allclose(frozen['native_state']['means_']['array'], native.means_, rtol=1e-12, atol=1e-12)

    def forbidden(*a, **k):
        raise AssertionError('Replay fitted a transform or model')
    for cls in (GaussianMixture, StandardScaler, RobustScaler, SimpleImputer):
        monkeypatch.setattr(cls, 'fit', forbidden)
    replay = model.predict_candidate(frozen, frame, definitions)
    np.testing.assert_allclose(replay['values'], values, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(replay['probabilities'], native.predict_proba(values)[:, frozen['component_order']], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(replay['log_density'], native.score_samples(values), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(replay['component'][replay['status'] == 'assigned'], np.argmax(replay['probabilities'][replay['status'] == 'assigned'], axis=1))
    assert len(frozen['learning_membership']) == len(frame)

def test_new_values_missingness_and_domain_flags_do_not_redefine_states():
    frame, definitions, learning, candidate, assignment, membership = inputs()
    frozen = model.fit_candidate(frame, definitions, learning, candidate, assignment, membership, 'raw')['model']
    original = copy.deepcopy(frozen)
    values = pd.DataFrame([[-3, -1], [3, 2], [100000000.0, 100000000.0], [np.nan, 2]], columns=frame.columns)
    result = model.predict_candidate(frozen, values, definitions)
    assert result['status'].tolist() == ['assigned', 'assigned', 'outside_training_distribution', 'missing_features']
    assert result['component'].tolist() == [0, 1, -1, -1]
    assert np.isnan(result['probabilities'][-1]).all() and np.isfinite(result['probabilities'][-2]).all()
    assert frozen == original
    assert model.predict_candidate(frozen, values.iloc[:0], definitions)['probabilities'].shape == (0, 2)

def test_saved_feature_definition_hash_and_backend_version_are_verified():
    frame, definitions, learning, candidate, assignment, membership = inputs()
    frozen = model.fit_candidate(frame, definitions, learning, candidate, assignment, membership, 'raw')['model']
    for modified in (list(reversed(definitions)), [{**f, 'unit': 'new-unit'} for f in definitions]):
        with pytest.raises(ValueError, match='feature order'):
            model.predict_candidate(frozen, frame, modified)
    with pytest.raises(ValueError, match='exactly'):
        model.predict_candidate(frozen, frame.assign(condition=1), definitions)
    broken = copy.deepcopy(frozen)
    broken['native_state']['means_']['array'][0][0] += 1
    with pytest.raises(ValueError, match='changed'):
        model.verify_model(broken)
    broken = copy.deepcopy(frozen)
    broken['implementation']['scikit_learn'] = 'unknown'
    broken['model_id'] = model.content_id({k: v for k, v in broken.items() if k != 'model_id'})
    with pytest.raises(model.UnsupportedModel, match='version'):
        model.verify_model(broken)

def test_unsupported_weighted_degenerate_missing_and_failed_candidates_are_explicit(monkeypatch):
    frame, definitions, learning, candidate, assignment, membership = inputs()

    def fit(values=frame, config=candidate, members=membership):
        return model.fit_candidate(values, definitions, learning, config, assignment, members, 'raw')
    with pytest.raises(model.UnsupportedModel):
        fit(config={'method': 'training_only_clustering'})
    with pytest.raises(ValueError, match='sample weights'):
        fit(members=[{**m, 'learning_weight': 2} for m in membership])
    for values in (frame * 0, frame.assign(signal=np.nan)):
        assert fit(values)['status'] == 'ineligible' and fit(values)['model'] is None
    assert fit(config={**candidate, 'max_iter': 1})['status'] == 'not_converged'
    assert fit(frame.iloc[:1], members=membership[:1])['status'] == 'ineligible'

    def fail(*a, **k):
        raise np.linalg.LinAlgError('controlled singular model')
    monkeypatch.setattr(GaussianMixture, 'fit', fail)
    assert fit()['status'] == 'fit_failed' and 'controlled singular' in fit()['reason']

def test_learning_imputation_uses_only_learning_median():
    frame, definitions, learning, candidate, assignment, membership = inputs(missing=True)
    frozen = model.fit_candidate(frame, definitions, learning, candidate, assignment, membership, 'raw')['model']
    np.testing.assert_allclose(frozen['transformation']['imputer']['statistics_']['array'], np.nanmedian(frame, axis=0))
    future = pd.DataFrame([[1000000.0, np.nan], [np.nan, np.nan]], columns=frame.columns)
    result = model.predict_candidate(frozen, future, definitions)
    assert np.isfinite(result['values'][0]).all() and np.isnan(result['values'][1]).all()
    assert result['status'].tolist() == ['outside_training_distribution', 'missing_features']
