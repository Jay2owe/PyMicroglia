"""Structural validation before loading the scientific engine."""
from copy import deepcopy
import numpy as np
METHOD="segmented_glsar"

def relative_settings(value):
    if not isinstance(value, dict) or set(value) != {'denominator', 'minimum_absolute_baseline', 'justification'}:
        raise ValueError('Relative effects require denominator, minimum_absolute_baseline and justification')
    if value['denominator'] not in {'baseline', 'absolute_baseline'}:
        raise ValueError('Relative denominator must be baseline or absolute_baseline')
    floor = value['minimum_absolute_baseline']
    if isinstance(floor, (bool, np.bool_)) or not isinstance(floor, (int, float, np.number)) or (not np.isfinite(floor)) or (floor <= 0):
        raise ValueError('Declare a strictly positive minimum absolute baseline for relative effects')
    if not isinstance(value['justification'], str) or not value['justification'].strip():
        raise ValueError('Explain why the measurement supports the chosen relative denominator')
    return dict(value)

def settings(value):
    value = deepcopy(value)
    if value == {'method': 'none'}:
        return value
    keys = {'method', 'estimand', 'baseline_trend', 'response_model', 'ar_order', 'min_observations_per_window', 'model_justification', 'error_model_justification'}
    if set(value) - keys:
        raise ValueError('Unknown within-cell evidence setting: ' + ', '.join(sorted(set(value) - keys)))
    if value.get('method') != METHOD:
        raise ValueError('Within-cell evidence method must be none or ' + METHOD)
    if value.get('estimand') not in {'modelled_window_mean_change', 'baseline_trend_adjusted_change'}:
        raise ValueError('Choose modelled_window_mean_change or baseline_trend_adjusted_change as the evidence estimand')
    if value.get('baseline_trend') not in {'constant', 'linear'}:
        raise ValueError('Declare a constant or linear baseline mean model')
    if value.get('response_model') not in {'level', 'level_and_slope'}:
        raise ValueError('Declare a level or level_and_slope response model')
    for key in ['model_justification', 'error_model_justification']:
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError(key + ' requires an explicit assumption justification')
    lag = value.get('ar_order')
    if isinstance(lag, bool) or not isinstance(lag, int) or lag < 1:
        raise ValueError('Declare a positive autoregressive error order in original-observation lags')
    minimum = value.get('min_observations_per_window', max(48, 10 * (lag + 1)))
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < max(24, 5 * (lag + 1)):
        raise ValueError('Minimum per-window support must be at least 24 and five times ar_order plus one')
    value['min_observations_per_window'] = minimum
    return value
