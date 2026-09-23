"""Spatial mark evidence randomises cells, never shared-cell pair rows."""
from itertools import combinations
import numpy as np
import pandas as pd
import pytest
import pymicroglia.measure.coordination_statistics as native
import pymicroglia.pipelines.coordination.inputs as inputs, pymicroglia.pipelines.coordination.characteristics as characteristics
from pymicroglia.pipelines.coordination.options import resolve_request, run_request
from pymicroglia.pipelines._screening import file_hash
from pymicroglia.pipelines._runner import Unavailable
from tests.test_coordination_options import request, resolve, declaration

def question(**changes):
    return {'enabled': True, 'summary': 'mean', 'statistic': 'absolute_difference', 'evidence': {'method': 'mark_location_permutation', 'null_model': native.SPATIAL_NULL, 'exchangeability_justification': 'Controlled independent marks assigned to fixed software-test positions', 'min_cells': 4, 'n_resamples': 199, 'seed': 29, 'alternative': 'two-sided'}, **changes}

def fixture(n=9, marks=None, **changes):
    values = np.arange(n, dtype=float) if marks is None else np.asarray(marks, float)
    frame = pd.DataFrame([{'stem': 'movie', 'identity': cell + 1, 'frame_index': i, 'hours': 50.0 + 0.5 * i, 'custom_signal': values[cell], 'centroid_x': float(cell * 4), 'centroid_y': 0.0} for cell in range(n) for i in range(8)])
    tables = {'cell_frame': frame, 'cell_summary': frame[['stem', 'identity']].drop_duplicates()}
    req = request(target_measurements=['custom_signal'], questions={'characteristics': question()}, inference={'alpha': 0.05, 'multiple_testing': 'bonferroni', 'correction_scope': 'question'}, **changes)
    resolved = resolve(tables, req)
    prepared, definitions = inputs.prepare(resolved, tables)
    prepared.update(pair_definitions=definitions, provenance={'scientific_id': 'input-id'})
    return (resolved, tables, prepared)

def native_fixture(n, values, q=None):
    a, b = np.asarray(list(combinations(range(n), 2))).T
    distance = (b - a).astype(float)
    return native.spatial_test(np.column_stack([values, values]), a, b, distance, distance <= 2, q or question())

def test_known_gradient_saves_descriptive_pairs_and_one_recording_probability():
    resolved, _, prepared = fixture()
    outputs = characteristics.analyse(resolved, prepared, 'science')
    pairs = outputs['pair_characteristics']
    recordings = outputs['recording_effects']
    assert len(pairs) == 36 and pairs.p_value.isna().all()
    assert pairs.evidence_level.eq('pair').all()
    result = recordings.iloc[0]
    assert result.effect == pytest.approx(1.0) and result.p_value <= 0.05
    assert result.evidence_level == 'recording' and result.cells == 9 and (result.pairs == 36)
    assert len(outputs['null_reference']) == 199
    assert len(outputs['cell_characteristics']) == 9
    assert outputs['cell_characteristics'].observed_interval_hours.eq(3.5).all()
    assert outputs['distance_profiles'].pairs.sum() == 36
    assert set(outputs['distance_profiles'].distance) == set(np.arange(1, 9) * 4)
    assert outputs['spatial_membership'].complete_for_spatial_reference.all()

def test_exact_native_reference_permutates_cells_as_complete_units():
    result = native_fixture(4, [0, 1, 3, 8])
    assert result['actual_resamples'] == 24
    assert result['cells'] == 4 and result['pairs'] == 6
    assert len(result['null_values']) == 24
    assert result['unit_of_randomisation'] == 'complete_cell_mark_vector'
    assert result['confidence_low'] is None and result['uncertainty_status'] == 'not_estimated'

def test_constant_marks_distances_and_too_few_cells_do_not_invent_evidence():
    assert native_fixture(9, np.ones(9))['status'] == 'unresolvable'
    result = native_fixture(4, [0, 1, 3, 8], question(evidence={**question()['evidence'], 'min_cells': 6}))
    assert result['status'] == 'insufficient' and result['p_value'] is None
    a, b = np.asarray(list(combinations(range(4), 2))).T
    result = native.spatial_test(np.column_stack([np.arange(4), np.arange(4)]), a, b, np.ones(len(a)), a == 0, question())
    assert result['status'] == 'unresolvable' and result['p_value'] is None

def test_a_two_spatial_field_null_cannot_use_the_random_label_reference():
    with pytest.raises(Unavailable, match='two spatially structured fields'):
        native.validate_spatial(question(evidence={**question()['evidence'], 'null_model': 'independent_spatial_fields'}))
    with pytest.raises(ValueError, match='justification'):
        native.validate_spatial(question(evidence={**question()['evidence'], 'exchangeability_justification': ''}))

def test_declared_near_minus_far_effect_and_squared_quantity():
    result = native_fixture(9, np.arange(9), question(distance_effect='near_minus_far', statistic='squared_difference'))
    assert result['effect'] < 0 and result['p_value'] <= 0.05
    assert result['distance_effect'] == 'near_minus_far'
    descriptive = native_fixture(9, np.arange(9), question(evidence={'method': 'none'}))
    assert descriptive['status'] == 'descriptive' and descriptive['p_value'] is None

def test_partial_and_unobserved_cells_stay_in_complete_population_tables():
    resolved, tables, _ = fixture()
    frame = tables['cell_frame']
    frame.loc[frame.identity.eq(9), 'custom_signal'] = np.nan
    prepared, definitions = inputs.prepare(resolved, tables)
    prepared.update(pair_definitions=definitions, provenance={'scientific_id': 'input-id'})
    output = characteristics.analyse(resolved, prepared, 'science')
    assert len(output['pair_characteristics']) == 36
    assert output['pair_characteristics'].eligible.sum() == 28
    assert output['recording_effects'].iloc[0].cells == 8
    assert len(output['spatial_membership']) == 9
    assert not output['spatial_membership'].loc[output['spatial_membership'].identity.eq(9), 'complete_for_spatial_reference'].any()

def test_unknown_backend_is_unavailable_not_a_successful_negative_result():
    with pytest.raises(Unavailable, match='unsupported'):
        native.validate_spatial(question(evidence={'method': 'invented_method'}))

def test_saved_scalar_traits_and_half_open_trace_windows_keep_their_meaning():
    _, tables, _ = fixture()
    tables['cell_summary']['trait'] = tables['cell_summary'].identity.astype(float)
    req = request(reference_measurements=['trait'], target_measurements=['trait'], questions={'characteristics': question()}, inference={'alpha': 0.05, 'multiple_testing': 'bonferroni', 'correction_scope': 'question'})
    resolved = resolve(tables, req)
    prepared, definitions = inputs.prepare(resolved, tables)
    prepared.update(pair_definitions=definitions, provenance={'scientific_id': 'scalar-input'})
    output = characteristics.analyse(resolved, prepared, 'scalar-science')
    assert output['cell_characteristics'].summary.eq('saved_scalar').all()
    assert output['cell_characteristics'].observations.eq(1).all()
    assert output['recording_effects'].iloc[0].effect == pytest.approx(1.0)
    frame = tables['cell_frame']
    frame['custom_signal'] = frame.identity + frame.hours
    support = {**declaration()['support'], 'min_observations': 3, 'min_span_hours': 1.0}
    req = request(target_measurements=['custom_signal'], questions={'characteristics': question()}, support=support, time_range_hours=[50.5, 52.0], inference={'alpha': 0.05, 'multiple_testing': 'bonferroni', 'correction_scope': 'question'})
    resolved = resolve(tables, req)
    prepared, definitions = inputs.prepare(resolved, tables)
    prepared.update(pair_definitions=definitions, provenance={'scientific_id': 'window-input'})
    output = characteristics.analyse(resolved, prepared, 'window-science')
    traits = output['cell_characteristics']
    np.testing.assert_allclose(traits.value, traits.identity + 51.0)
    assert traits.observations.eq(3).all() and traits.observed_interval_hours.eq(1.0).all()

def test_empty_population_and_single_cell_leave_explicit_recording_outcomes():
    _, tables, _ = fixture(n=1)
    req = request(target_measurements=['custom_signal'], questions={'characteristics': question()}, inference={'alpha': 0.05, 'multiple_testing': 'bonferroni', 'correction_scope': 'question'})
    resolved = resolve(tables, req)
    prepared, definitions = inputs.prepare(resolved, tables)
    prepared.update(pair_definitions=definitions, provenance={'scientific_id': 'single-input'})
    output = characteristics.analyse(resolved, prepared, 'single-science')
    assert output['pair_characteristics'].empty
    assert len(output['recording_effects']) == 1 and output['recording_effects'].iloc[0].p_value is None
    empty = resolve(tables, request(target_measurements=['custom_signal'], questions={'characteristics': question()}, cells=[], inference={'alpha': 0.05, 'multiple_testing': 'bonferroni', 'correction_scope': 'question'}))
    prepared, definitions = inputs.prepare(empty, tables)
    prepared.update(pair_definitions=definitions, provenance={'scientific_id': 'empty-input'})
    output = characteristics.analyse(empty, prepared, 'empty-science')
    assert output['recording_effects'].empty and output['pair_characteristics'].empty

def test_real_characteristic_producer_keeps_native_result_and_reuses_it(tmp_path, monkeypatch):
    resolved, tables, _ = fixture()
    paths = {}
    for name, frame in tables.items():
        paths[name] = tmp_path / (name + '.csv')
        frame.to_csv(paths[name], index=False)
    resolved = resolve_request(resolved.request, source_run='source', tables=tables, input_hashes={k: file_hash(v) for k, v in paths.items()})
    result = run_request(resolved, paths, tmp_path / 'pipeline', only=('characteristic-similarity',))
    assert result.successful
    saved = result.results['characteristic-similarity']
    before = {ref.name: file_hash(saved.artifact(ref.name)) for ref in saved.outcome.artifacts}

    def forbidden(*a, **k):
        raise AssertionError('Presentation invoked spatial inference')
    monkeypatch.setattr(native, 'spatial_test', forbidden)
    second = run_request(resolved, paths, tmp_path / 'pipeline', only=('characteristic-similarity',), presentation={'overview': {'columns': 1}})
    assert second.results['characteristic-similarity'].outcome.status == 'reused'
    assert all((file_hash(saved.artifact(name)) == fingerprint for name, fingerprint in before.items()))
