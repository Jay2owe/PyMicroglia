"""Evaluate saved rhythm models for display; never estimate or test a rhythm."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from .exploration_data import _observations, number, read_rows, recording
from .exploration_rhythms import _fingerprint, load_attached_rhythms


@lru_cache(maxsize=12)
def _curves(root, movie_json, source_signature, evidence_signature, metric):
    from pymicroglia import workbench
    with workbench.analysis_session():
        return _evaluate_curves(root, movie_json, evidence_signature, metric)


def _evaluate_curves(root, movie_json, evidence_signature, metric):
    from pymicroglia import workbench
    from pymicroglia.figure_tables.fft_component_grid import three_frame_mean

    _, observations, frames, _ = _observations(Path(root), json.loads(movie_json))
    rows = []
    for evidence in read_rows(Path(evidence_signature[0])):
        if (evidence.get('metric') != metric or evidence.get('estimate_status') != 'ok'
                or evidence.get('estimation_method') != 'fft_nlls'):
            continue
        identity = int(evidence['identity'])
        observed = observations.get(identity, {})
        indices = sorted(observed)
        hours = np.array([frames[i]['hours'] for i in indices], dtype=float)
        values = np.array([number(observed[i].get(metric)) for i in indices], dtype=float)
        filtering = json.loads(evidence.get('trace_filter_json') or 'null')
        if filtering is not None:
            values = np.asarray(workbench.filter_rhythm_trace(hours, values, filtering)['values'], float)
        elif metric == 'integrated_density':
            values = three_frame_mean(values, np.array(indices))
        usable = np.isfinite(hours) & np.isfinite(values)
        if usable.sum() < 2:
            continue
        try:
            model = {'method': 'fft_nlls',
                     'diagnostics': json.loads(evidence.get('estimate_diagnostics_json') or '{}'),
                     'components': json.loads(evidence.get('estimate_components_json') or '[]')}
            native = workbench.fft_nlls_fitted_values(hours[usable], model)
            if not np.isfinite(native).all():
                continue
            settings = json.loads(evidence['applied_settings_json'])
            fitted = np.full(len(indices), np.nan)
            # Restore the baseline removed by the saved analysis, in original units.
            fitted[usable] = workbench.restore_rhythm_fit(hours[usable], values[usable], native, settings)
        except (ValueError, TypeError, KeyError):
            continue
        rows.append({'identity': identity, 'metric': metric, 'indices': indices,
                     'values': [number(value) for value in fitted],
                     'kind': 'Saved summed FFT-NLLS model + restored baseline'})
    return rows


def load_rhythm_fits(run, stem, metric, analysis_id=None):
    """Read the exact saved model, retaining its clock, units and missing samples."""
    root, _, movie = recording(run, stem)
    source, _, _, metrics = _observations(root, movie)
    if metric not in {row['key'] for row in metrics}:
        raise ValueError('The fit metric is not available in this recording')
    result = load_attached_rhythms(root, movie, analysis_id)
    payload = {'stem': stem, 'metric': metric, 'rows': [], 'message': '',
               'meaning': 'Dashed curves use the saved model and restored baseline; display smoothing does not refit it.'}
    if not result or metric not in result.get('metrics', []):
        return {**payload, 'message': 'No saved fitted rhythm for this measurement. Run rhythm analysis first.'}
    path = Path(result['source']).with_name('statistics.csv').resolve()
    if not path.is_relative_to(root) or not path.is_file():
        return {**payload, 'message': 'The saved rhythm model is unavailable.'}
    if path.stat().st_size > 16 * 1024**2:
        raise ValueError('The saved rhythm evidence exceeds the display budget')
    if result.get('statistics_sha256') and _fingerprint(path) != result['statistics_sha256']:
        raise ValueError('Saved rhythm model evidence changed after analysis')
    signature = lambda p: (str(p), p.stat().st_size, p.stat().st_mtime_ns)
    rows = _curves(str(root), json.dumps(movie, sort_keys=True), signature(source), signature(path), metric)
    return {**payload, 'rows': rows, 'analysis_id': result.get('analysis_id'), 'source': str(path),
            'message': '' if rows else 'This saved analysis has no fitted model curve for this measurement.'}
