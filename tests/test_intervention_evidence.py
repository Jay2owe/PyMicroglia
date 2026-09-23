"""Complete temporal families, original effects, denominator and reuse checks."""
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from scipy.signal import lfilter
from pymicroglia.pipelines import parse
from pymicroglia.pipelines.intervention.options import resolve_request, run_request
from pymicroglia.pipelines.intervention.windows import prepare
from pymicroglia.pipelines.intervention.evidence import analyse, adjust, read_evidence, relative, relative_settings
from pymicroglia.pipelines._screening import file_hash
from tests.test_intervention_statistics import options

def fixture(tmp_path, **changes):
    rng = np.random.default_rng(73801)
    rows = []
    for movie, anchor in [('treated', 100.0), ('control', 200.0)]:
        for identity in [7, 8]:
            hours = (np.arange(192) - 96) * 0.5
            noise = lfilter([1], [1, -0.4], rng.normal(size=792))[600:]
            signal = 10 + 0.01 * hours + noise + (5 * (hours >= 0) if movie == 'treated' else 0)
            shape = 8 + 0.02 * hours + noise + (-5 * (hours >= 0) if movie == 'treated' else 0)
            for i, t in enumerate(hours):
                if identity == 8 and movie == 'control' and (i < 90):
                    continue
                rows.append({'stem': movie, 'identity': identity, 'frame_index': i, 'hours': anchor + t, 'custom_signal': float(signal[i]) if not (movie == 'treated' and identity == 8 and (i == 90)) else np.nan, 'custom_shape': float(shape[i])})
    frame = pd.DataFrame(rows)
    tables = {'cell_frame': frame, 'cell_summary': frame[['stem', 'identity']].drop_duplicates()}
    paths = {}
    for name, table in tables.items():
        paths[name] = tmp_path / (name + '.csv')
        table.to_csv(paths[name], index=False)
    request = {'pipeline': 'intervention-response', 'measurements': ['custom_signal', 'custom_shape'], 'summary': 'mean', 'anchors': {movie: {'hours': anchor, 'kind': 'intervention' if movie == 'treated' else 'control', 'label': 'Software event'} for movie, anchor in [('treated', 100.0), ('control', 200.0)]}, 'windows': [{'name': 'baseline', 'coordinate': 'relative_hours', 'start': -48, 'end': 0}, {'name': 'followup', 'coordinate': 'relative_hours', 'start': 0, 'end': 48, 'baseline': 'baseline'}], 'support': {'max_gap_hours': 0.6}, 'evidence': options(response_model='level'), 'inference': {'alpha': 0.05, 'multiple_testing': 'bonferroni', 'correction_scope': 'all'}, **changes}
    resolved = resolve_request(parse([request])[0], source_run='software-source', tables=tables, input_hashes={key: file_hash(path) for key, path in paths.items()})
    return (request, tables, paths, resolved)

def test_full_families_keep_missing_windows_and_increase_decrease(tmp_path):
    _, tables, _, resolved = fixture(tmp_path)
    prepared = prepare(resolved, tables)
    data = analyse(prepared, resolved)
    effects = data['effects']
    assert len(effects) == 8 and len(data['cell_decisions']) == 4
    family = data['families'].iloc[0]
    assert family.requested == 8 and family.tested == 5 and (len(family.members) == 8)
    one = effects.loc[effects.movie.eq('treated') & effects.identity.eq(7)].set_index('measurement')
    assert one.loc['custom_signal', 'outcome'] == 'increase' and one.loc['custom_shape', 'outcome'] == 'decrease'
    assert effects.loc[effects.movie.eq('treated') & effects.identity.eq(8) & effects.measurement.eq('custom_signal'), 'p_value'].isna().all()
    assert effects.loc[effects.movie.eq('control') & effects.identity.eq(8), 'outcome'].eq('inconclusive').all()
    assert np.allclose(effects.absolute_change, effects.target_value - effects.baseline_value, equal_nan=True)
    tested = effects.loc[effects.p_value.notna()]
    assert np.allclose(tested.q_value, np.minimum(tested.p_value * 8, 1))
    assert len(set(effects.effect_id)) == 8

def test_meaningful_magnitude_is_separate_from_significance(tmp_path):
    _, tables, _, resolved = fixture(tmp_path, meaningful_effects={'custom_signal': 100.0})
    data = analyse(prepare(resolved, tables), resolved)
    row = data['effects'].loc[lambda x: x.movie.eq('treated') & x.identity.eq(7) & x.measurement.eq('custom_signal')].iloc[0]
    assert row.statistically_supported and (not row.response_supported)
    assert row.outcome == 'detected_below_meaningful_threshold' and row.q_value < 0.05
    assert not row.absolute_change_exceeds_threshold and (not row.model_estimate_exceeds_threshold)

def test_zero_negative_and_unknown_relative_denominators():
    setting = {'denominator': 'baseline', 'minimum_absolute_baseline': 0.01, 'justification': 'Known positive ratio-scale measurement'}
    assert relative(2, 0, setting) == (None, 'baseline_too_close_to_zero')
    assert relative(2, -1, setting) == (None, 'nonpositive_ratio_baseline')
    assert relative(None, 1, setting) == (None, 'unavailable_original_values')
    assert relative(2, 4, setting) == (0.5, 'descriptive_fractional_change')
    assert relative(2, -4, {**setting, 'denominator': 'absolute_baseline'})[0] == 0.5
    with pytest.raises(ValueError):
        relative_settings({**setting, 'minimum_absolute_baseline': 0})

def test_original_summaries_remain_when_inference_is_disabled(tmp_path):
    _, tables, _, resolved = fixture(tmp_path, evidence={'method': 'none'}, inference={})
    data = analyse(prepare(resolved, tables), resolved)
    assert len(data['effects']) == 8 and data['effects'].absolute_change.notna().all()
    assert data['effects'].p_value.isna().all() and data['families'].empty and data['native_details'].empty
    assert not data['effects'].response_supported.any()

def test_actual_runner_preserves_sources_selections_and_selective_window_reuse(tmp_path, monkeypatch):
    request, tables, paths, resolved = fixture(tmp_path)
    root = tmp_path / 'pipeline'
    first = run_request(resolved, paths, root, only=['response-evidence'])
    assert first.successful
    source = first.results['aligned-windows']
    saved = first.results['response-evidence']
    data = read_evidence(saved, source.outcome.scientific_id)
    assert len(data['effects']) == 8 and any((selection.name == 'responding-cells' and len(selection.members) > 0 for selection in saved.outcome.selections))
    originals = {str(path): file_hash(path) for path in paths.values()}
    before = {str(item.artifact(ref.name)): file_hash(item.artifact(ref.name)) for item in first.results.values() for ref in item.outcome.artifacts}
    import pymicroglia.pipelines.intervention.windows as intervention_windows, pymicroglia.pipelines.intervention.evidence as intervention_evidence

    def forbidden(*a, **k):
        raise AssertionError('Scientific work repeated during saved-result reuse')
    with monkeypatch.context() as local:
        local.setattr(intervention_windows, 'prepare', forbidden)
        local.setattr(intervention_evidence, 'analyse', forbidden)
        reopened = run_request(resolved, paths, root, only=['response-evidence'], presentation={'report': {'title': 'New presentation'}})
        assert reopened.successful and reopened.results['response-evidence'].outcome.status == 'reused'
    changed = deepcopy(request)
    changed['meaningful_effects'] = {'custom_signal': 100.0}
    new = resolve_request(parse([changed])[0], source_run='software-source', tables=tables, input_hashes={key: file_hash(path) for key, path in paths.items()})
    monkeypatch.setattr(intervention_windows, 'prepare', forbidden)
    rerun = run_request(new, paths, root, only=['response-evidence'])
    assert rerun.successful
    assert rerun.results['aligned-windows'].outcome.status == 'reused'
    assert rerun.results['response-evidence'].outcome.scientific_id != saved.outcome.scientific_id
    assert all((file_hash(Path(path)) == value for path, value in {**before, **originals}.items()))
