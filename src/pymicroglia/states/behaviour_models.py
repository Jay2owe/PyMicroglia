"""Public scikit-learn state candidates with explicit, data-only fitted records.

These are candidate measurement-density models. Fitting does not establish
distinct states, and their membership probabilities are not biological labels.
"""
from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from pymicroglia.pipelines._contracts import content_id
from pymicroglia.pipelines._screening import _json_value, file_hash


MODEL_FIELDS = ('weights_', 'means_', 'covariances_', 'precisions_', 'precisions_cholesky_',
    'n_features_in_', 'converged_', 'n_iter_', 'lower_bound_')
SCALER_FIELDS = {'standard': ('mean_', 'scale_', 'var_', 'n_features_in_', 'n_samples_seen_'),
    'robust': ('center_', 'scale_', 'n_features_in_')}
IMPUTER_FIELDS = ('statistics_', 'n_features_in_', 'indicator_', '_fit_dtype', '_fill_dtype')


class UnsupportedModel(ValueError):
    """The requested public model has no implemented compatible replay adapter."""


def implementation_version():
    return {'adapter': file_hash(Path(__file__)), 'scikit_learn': version('scikit-learn'),
        'numpy': version('numpy'), 'scipy': version('scipy')}


def candidate_parameters(candidate):
    from pymicroglia.pipelines.behaviour.options import _number, _seed
    required = {'method', 'components', 'covariance_type', 'reg_covar', 'n_init', 'max_iter', 'seed'}
    optional = {'tol', 'init_params'}
    if candidate.get('method') != 'gaussian_mixture':
        raise UnsupportedModel('A saved new-observation adapter is available for gaussian_mixture; requested ' + str(candidate.get('method')))
    if required - candidate.keys() or set(candidate) - required - optional:
        raise ValueError('Gaussian mixture candidates require ' + ', '.join(sorted(required)) + '; optional tol and init_params')
    count = _number(candidate['components'], 'candidate.components', minimum=1, integer=True)
    if candidate['covariance_type'] not in {'full', 'tied', 'diag', 'spherical'}: raise ValueError('Unknown covariance_type')
    initial = candidate.get('init_params', 'kmeans')
    if initial not in {'kmeans', 'k-means++', 'random', 'random_from_data'}: raise ValueError('Unknown Gaussian mixture initialization')
    return {'n_components': count, 'covariance_type': candidate['covariance_type'],
        'reg_covar': _number(candidate['reg_covar'], 'candidate.reg_covar', positive=True),
        'n_init': _number(candidate['n_init'], 'candidate.n_init', minimum=1, integer=True),
        'max_iter': _number(candidate['max_iter'], 'candidate.max_iter', minimum=1, integer=True),
        'random_state': _seed(candidate['seed'], 'candidate.seed'),
        'tol': _number(candidate.get('tol', .001), 'candidate.tol', positive=True), 'init_params': initial,
        'warm_start': False, 'verbose': 0}


def feature_signature(features):
    return [{name: feature[name] for name in ('column', 'table', 'grain', 'unit')} for feature in features]


def _state(estimator, fields):
    result = {}
    for name in fields:
        if not hasattr(estimator, name): continue
        value = getattr(estimator, name)
        if isinstance(value, np.dtype): result[name] = {'dtype': str(value)}
        elif isinstance(value, np.ndarray): result[name] = {'array': value.tolist(), 'dtype': str(value.dtype)}
        else: result[name] = _json_value(value)
    return result


def _restore(estimator, state, allowed):
    if set(state) - set(allowed): raise ValueError('Unknown serialized native fitted attribute')
    for name, value in state.items():
        if isinstance(value, dict):
            if set(value) == {'dtype'}: value = np.dtype(value['dtype'])
            elif set(value) == {'array', 'dtype'}: value = np.asarray(value['array'], dtype=value['dtype'])
            else: raise ValueError('Malformed native fitted attribute')
        setattr(estimator, name, value)
    return estimator


def _scaler(method):
    from sklearn.preprocessing import RobustScaler, StandardScaler
    if method == 'standard': return StandardScaler(copy=True, with_mean=True, with_std=True)
    if method == 'robust': return RobustScaler(copy=True, with_centering=True, with_scaling=True, quantile_range=(25., 75.), unit_variance=False)
    if method == 'none': return None
    raise ValueError('Unsupported learned feature scaling')


def _values(frame, definitions):
    columns = [feature['column'] for feature in definitions]
    if frame.columns.duplicated().any() or set(frame.columns) != set(columns):
        raise ValueError('Candidate data must contain exactly the declared feature columns')
    values = frame[columns].to_numpy(dtype=float, copy=True)
    values[~np.isfinite(values)] = np.nan
    return values


@threadpool_limits.wrap(limits=1)
def fit_candidate(frame, definitions, learning, candidate, assignment, membership, representation):
    """Fit only the explicit learning rows; no held-out values are accepted here."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.impute import SimpleImputer
    from sklearn.mixture import GaussianMixture
    parameters = candidate_parameters(candidate)
    raw = _values(frame, definitions)
    if len(raw) != len(membership): raise ValueError('Every learning row needs its exact observation membership')
    if any(float(row.get('learning_weight', 1.)) != 1. for row in membership):
        raise ValueError('GaussianMixture does not support sample weights; provide the declared balanced observation sample')
    result = {'status': 'ineligible', 'reason': '', 'model': None, 'learning_observations': len(raw)}
    if len(raw) < max(2, parameters['n_components']):
        return {**result, 'reason': 'Too few eligible learning observations for the requested components'}
    if np.any(np.sum(np.isfinite(raw), axis=0) == 0):
        return {**result, 'reason': 'A requested feature has no observed learning values; no median or model definition is available'}
    if np.any(np.mean(~np.isfinite(raw), axis=1) > learning['missing']['max_fraction']):
        raise ValueError('Learning membership violates the declared missing-feature policy')
    if not np.any(np.nanmax(raw, axis=0) > np.nanmin(raw, axis=0)):
        return {**result, 'reason': 'All requested learning features are constant; distinct measurement states are unresolvable'}
    imputer = None
    if learning['missing']['method'] == 'median':
        imputer = SimpleImputer(strategy='median', copy=True, add_indicator=False, keep_empty_features=False).fit(raw)
        values = imputer.transform(raw)
    elif learning['missing']['method'] == 'complete_case':
        if not np.isfinite(raw).all(): raise ValueError('Complete-case learning includes missing values')
        values = raw.copy()
    else: raise ValueError('Unknown missing-feature policy')
    scaler = _scaler(learning['scaling']['method'])
    if scaler is not None: values = scaler.fit_transform(values)
    if not np.isfinite(values).all(): return {**result, 'reason': 'Learning transformation produced nonfinite values'}
    unique = len(np.unique(values, axis=0))
    if unique < parameters['n_components']:
        return {**result, 'reason': 'Too few distinct learning vectors for the requested components'}
    native = GaussianMixture(**parameters)
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always', ConvergenceWarning)
            native.fit(values)
    except (ValueError, np.linalg.LinAlgError) as error:
        return {**result, 'status': 'fit_failed', 'reason': str(error)}
    warning_text = [str(item.message) for item in captured]
    if not native.converged_:
        return {**result, 'status': 'not_converged', 'reason': 'Native Gaussian mixture did not converge under the declared settings',
            'native_warnings': warning_text}
    density = native.score_samples(values)
    probabilities = native.predict_proba(values)
    if not np.isfinite(density).all() or not np.isfinite(probabilities).all():
        return {**result, 'status': 'fit_failed', 'reason': 'Native fitted density or membership is nonfinite'}
    order = sorted(range(native.n_components), key=lambda index: tuple(native.means_[index]))
    threshold = float(np.quantile(density, assignment['outlier_quantile'])) if assignment['outlier_quantile'] else None
    body = {'schema_version': 1, 'method': 'gaussian_mixture', 'implementation': implementation_version(),
        'features': feature_signature(definitions), 'candidate': dict(candidate), 'native_parameters': _json_value(native.get_params()),
        'native_state': _state(native, MODEL_FIELDS), 'component_order': order,
        'transformation': {'learning': dict(learning), 'representation': representation,
            'imputer': _state(imputer, IMPUTER_FIELDS) if imputer is not None else None,
            'scaler_method': learning['scaling']['method'],
            'scaler': _state(scaler, SCALER_FIELDS[learning['scaling']['method']]) if scaler is not None else None},
        'assignment': dict(assignment), 'density_threshold': threshold,
        'learning_membership': _json_value(membership), 'learning_values_id': content_id(_json_value(raw.tolist())),
        'training_log_density': float(np.mean(density)), 'unique_learning_vectors': unique,
        'native_warnings': warning_text, 'native_thread_limit': 1, 'meaning': 'Unvalidated candidate components; no accepted behavioural state claim',
        'probability_meaning': 'Native Gaussian mixture component membership conditional on this fitted model; not calibrated biological confidence'}
    model = {**body, 'model_id': content_id(body)}
    return {**result, 'status': 'fitted', 'reason': 'Fitted public native mixture and transformation on the declared learning sample', 'model': model}


def verify_model(model):
    if model.get('schema_version') != 1 or model.get('method') != 'gaussian_mixture': raise ValueError('Unsupported saved candidate model')
    body = {key: value for key, value in model.items() if key != 'model_id'}
    if content_id(body) != model.get('model_id'): raise ValueError('Saved model definition or fitted parameters changed')
    if model['implementation']['scikit_learn'] != version('scikit-learn'):
        raise UnsupportedModel('Saved native model requires its recorded scikit-learn version')


def native_model(model):
    from sklearn.mixture import GaussianMixture
    verify_model(model)
    return _restore(GaussianMixture(**model['native_parameters']), model['native_state'], MODEL_FIELDS)


def transform(model, frame, definitions):
    """Apply restored native transformations without fitting, filtering or clipping."""
    from sklearn.impute import SimpleImputer
    verify_model(model)
    if feature_signature(definitions) != model['features']:
        raise ValueError('Saved feature order, source grain or units do not match the requested model inputs')
    values = _values(frame, definitions)
    observed = np.isfinite(values).mean(axis=1)
    eligible = observed >= model['assignment']['min_observed_fraction']
    processing = model['transformation']
    if processing['imputer'] is None: eligible &= np.isfinite(values).all(axis=1)
    out = np.full_like(values, np.nan)
    if eligible.any():
        prepared = values[eligible]
        if processing['imputer'] is not None:
            imputer = _restore(SimpleImputer(strategy='median', copy=True, add_indicator=False, keep_empty_features=False),
                processing['imputer'], IMPUTER_FIELDS)
            prepared = imputer.transform(prepared)
        scaler = _scaler(processing['scaler_method'])
        if scaler is not None: prepared = _restore(scaler, processing['scaler'], SCALER_FIELDS[processing['scaler_method']]).transform(prepared)
        out[eligible] = prepared
    eligible &= np.isfinite(out).all(axis=1)
    return out, observed, eligible


@threadpool_limits.wrap(limits=1)
def predict_candidate(model, frame, definitions):
    values, observed, eligible = transform(model, frame, definitions)
    native = native_model(model)
    probability = np.full((len(frame), len(model['component_order'])), np.nan)
    density = np.full(len(frame), np.nan)
    labels = np.full(len(frame), -1, dtype=int)
    status = np.full(len(frame), 'missing_features', dtype=object)
    if eligible.any():
        probability[eligible] = native.predict_proba(values[eligible])[:, model['component_order']]
        density[eligible] = native.score_samples(values[eligible])
        finite = eligible & np.isfinite(density) & np.isfinite(probability).all(axis=1)
        status[eligible & ~finite] = 'invalid_prediction'
        best = np.max(np.where(np.isfinite(probability), probability, -np.inf), axis=1)
        confident = finite & (best >= model['assignment']['min_probability'])
        outside = finite & (density < model['density_threshold']) if model['density_threshold'] is not None else np.zeros(len(frame), dtype=bool)
        labels[confident & ~outside] = np.argmax(probability[confident & ~outside], axis=1)
        status[finite] = 'ambiguous'; status[confident] = 'assigned'; status[outside] = 'outside_training_distribution'
    return {'values': values, 'observed_fraction': observed, 'eligible': eligible,
        'probabilities': probability, 'log_density': density, 'component': labels, 'status': status}
