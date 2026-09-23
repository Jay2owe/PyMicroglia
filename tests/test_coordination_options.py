"""Scientific geometry, endpoint orientation and optional inputs stay explicit."""
from pymicroglia._results import read_document
import copy
import json
from types import SimpleNamespace
import pandas as pd
import pytest
from pymicroglia.pipelines import parse
from pymicroglia.pipelines._contracts import CellKey, content_id
from pymicroglia.pipelines.coordination.options import CoordinationRequest, CoordinationPairKey, EndpointKey, RECIPE, resolve_request, run_request, _pending
from pymicroglia.pipelines._screening import file_hash

def declaration(**changes):
    return {'pipeline': 'spatial-coordination', 'reference_measurements': ['custom_signal'], 'target_measurements': ['custom_signal', 'custom_shape'], 'pairs': {'mode': 'cartesian'}, 'questions': {'simultaneous': {'enabled': True, 'statistic': 'pearson', 'evidence': {'method': 'none'}}}, 'geometry': {'x': 'centroid_x', 'y': 'centroid_y', 'definition': 'centroid', 'input_unit': 'px', 'unit': 'px', 'scale': [1, 1], 'neighbourhood': {'method': 'all'}}, 'support': {'min_observations': 8, 'min_span_hours': 3, 'max_gap_hours': 1, 'matching': 'exact', 'matching_tolerance_hours': 0}, 'representation': 'raw', 'shared_reference': {'method': 'none'}, **changes}

def request(**changes):
    return CoordinationRequest.from_dict(declaration(**changes), {})

@pytest.fixture
def tables():
    frame = pd.DataFrame([{'stem': movie, 'identity': cell, 'frame_index': frame, 'hours': 50.0 + frame * 0.5, 'centroid_x': float(cell * 10 + frame), 'centroid_y': float(cell + frame), 'custom_signal': float(cell + frame), 'custom_shape': float(cell - frame)} for movie in ['one', 'two'] for cell in [7, 8] for frame in range(8)])
    return {'cell_frame': frame, 'cell_summary': pd.DataFrame({'stem': ['one', 'one', 'two', 'two'], 'identity': [7, 8, 7, 8], 'trait': [1.0, 2.0, 3.0, 4.0]})}

def resolve(tables, req=None):
    return resolve_request(req or request(), source_run='source', tables=tables, input_hashes={name: content_id(name) for name in tables})

def test_independent_free_measurement_lists_and_groups(tables):
    from pymicroglia.measure.metric_groups import build
    tables['cell_frame']['area_px'] = 10.0
    req = CoordinationRequest.from_dict(declaration(reference_measurements=['custom_signal', '@known'], target_measurements=['custom_shape']), build({'known': ['area_px']}))
    result = resolve(tables, req)
    assert [m.column for m in result.reference_measurements] == ['custom_signal', 'area_px']
    assert [m.column for m in result.target_measurements] == ['custom_shape']
    assert len(result.inputs.cells) == 4
    assert result.processing_version is None
    assert all((not q['enabled'] for name, q in result.request.questions.items() if name != 'simultaneous'))

def test_symmetric_full_keys_collapse_but_orientations_do_not(tables):
    result = resolve(tables)
    cells = result.inputs.cells
    a = EndpointKey(cells[0], result.reference_measurements[0])
    b = EndpointKey(cells[1], result.target_measurements[0])
    assert CoordinationPairKey(a, b, False) == CoordinationPairKey(b, a, False)
    assert CoordinationPairKey(a, b, True).record_id != CoordinationPairKey(b, a, True).record_id
    cross = EndpointKey(cells[1], result.target_measurements[1])
    assert CoordinationPairKey(a, cross, True).record_id != CoordinationPairKey(a, b, True).record_id
    with pytest.raises(ValueError, match='distinct cells'):
        CoordinationPairKey(a, a, True)
    for cell in [cells[2], CellKey('other-source', 'one', 8)]:
        with pytest.raises(ValueError, match='cross recordings'):
            CoordinationPairKey(a, EndpointKey(cell, b.measurement), False)

def test_explicit_measurement_directions_survive_and_exact_duplicates_collapse():
    req = request(reference_measurements=['custom_signal', 'custom_shape'], pairs={'mode': 'explicit', 'pairs': [['custom_signal', 'custom_shape'], ['custom_shape', 'custom_signal'], ['custom_signal', 'custom_shape']]})
    assert len(req.measurement_pairs) == 2
    assert req.measurement_pairs[0]['reference'] == 'custom_signal'
    assert req.measurement_pairs[1]['reference'] == 'custom_shape'
    with pytest.raises(ValueError, match='reference and target'):
        request(pairs={'mode': 'explicit', 'pairs': [['custom_shape', 'custom_signal']]})

def test_static_scalars_and_explicit_trace_summaries_keep_their_units(tables):
    question = {'characteristics': {'enabled': True, 'summary': 'median', 'statistic': 'absolute_difference', 'evidence': {'method': 'none'}}}
    req = request(reference_measurements=['trait'], target_measurements=['trait'], questions=question)
    result = resolve(tables, req)
    assert result.reference_measurements[0].table == 'cell_summary'
    assert result.reference_measurements[0].summary is None
    result = resolve(tables, request(target_measurements=['custom_signal'], questions=question))
    assert result.reference_measurements[0].summary == 'median'
    with pytest.raises(ValueError, match='matching recorded units'):
        resolve(tables, request(questions=question))
    with pytest.raises(ValueError, match='saved scalar'):
        resolve(tables, request(reference_measurements=['trait'], target_measurements=['trait'], questions=question, time_range_hours=[50, 60]))

def test_ambiguous_tables_metadata_and_temporal_scalars_are_rejected(tables):
    tables['other'] = tables['cell_frame'].copy()
    grain = {'other': ['identity', 'frame_index']}
    with pytest.raises(ValueError, match='ambiguous'):
        resolve(tables, request(table_grains=grain))
    req = request(reference_measurements=[{'column': 'custom_signal', 'table': 'cell_frame'}], target_measurements=[{'column': 'custom_signal', 'table': 'cell_frame'}], geometry={**declaration()['geometry'], 'x': {'column': 'centroid_x', 'table': 'cell_frame'}, 'y': {'column': 'centroid_y', 'table': 'cell_frame'}}, table_grains=grain)
    assert resolve(tables, req).reference_measurements[0].table == 'cell_frame'
    tables.pop('other')
    with pytest.raises(ValueError, match='unsuitable'):
        resolve(tables, request(reference_measurements=['trait'], target_measurements=['trait']))
    with pytest.raises(ValueError, match='unsuitable'):
        resolve(tables, request(reference_measurements=['identity'], target_measurements=['identity']))

def test_geometry_units_neighbour_population_and_samples_are_scientific(tables):
    baseline = resolve(tables)
    geometry = declaration()['geometry']
    for changed in [{**geometry, 'unit': 'um', 'scale': [0.5, 0.5]}, {**geometry, 'neighbourhood': {'method': 'nearest', 'neighbours': 1, 'comparison': 'retain_distant'}}]:
        assert resolve(tables, request(geometry=changed)).scientific_id != baseline.scientific_id
    with pytest.raises(ValueError, match='coordinate unit'):
        resolve(tables, request(geometry={**geometry, 'input_unit': 'um'}))
    with pytest.raises(ValueError, match='unit scale'):
        resolve(tables, request(geometry={**geometry, 'scale': [0.5, 0.5]}))
    req = request(biological_samples={'one': 'same', 'two': 'same'}, conditions={'one': 'control', 'two': 'control'})
    assert all((s.confirmed and s.sample == 'same' for s in resolve(tables, req).inputs.samples))
    with pytest.raises(ValueError, match='consistent declared conditions'):
        resolve(tables, request(biological_samples={'one': 'same', 'two': 'same'}, conditions={'one': 'control', 'two': 'treated'}))
    selected = request(cells=[{'movie': 'one', 'identity': 7}, {'movie': 'one', 'identity': 8}], conditions={'one': 'control', 'two': 'treated'}, geometry={**geometry, 'unit': 'um', 'scale': [0.5, 0.5], 'recording_scales': {'two': [0.4, 0.4]}})
    assert len(resolve(tables, selected).inputs.cells) == 2

def test_optional_sources_are_declarations_until_enabled_adapter_runs(tables, monkeypatch):
    from pathlib import Path

    def forbidden(*a, **k):
        raise AssertionError('An optional source was opened during parsing/resolution')
    with monkeypatch.context() as m:
        m.setattr(Path, 'read_text', forbidden)
        result = resolve(tables)
        question = {'rhythm': {'enabled': True, 'statistic': 'compatible_timing', 'evidence': {'method': 'none'}, 'source': {'execution_record': 'does-not-exist.json', 'scientific_id': 'a' * 64}}}
        assert resolve(tables, request(questions=question)).request.questions['rhythm']['enabled']
    for step in ['rhythm-coordination', 'state-coordination']:
        outcome = _pending(SimpleNamespace(step=SimpleNamespace(name=step), request=result, scientific_id='b' * 64))
        assert outcome.status == 'skipped-empty' and (not outcome.artifacts)
    with pytest.raises(ValueError, match='unknown setting'):
        request(questions={'rhythm': {'enabled': False, 'source': {}}})

def test_recipe_has_independent_delay_and_unconditional_evidence_overview():
    steps = {s.name: s for s in RECIPE.steps}
    assert steps['lagged-coordination'].selection is None
    assert steps['coordination-overview'].selection is None
    assert steps['connection-maps'].selection == 'coordination-evidence:supported-pairs'
    assert steps['pair-report-cards'].requires_selected_rows
    assert 'rhythm-coordination' not in steps['simultaneous-coordination'].prerequisites
    assert 'state-coordination' not in steps['simultaneous-coordination'].prerequisites

def test_names_and_presentation_cannot_change_frozen_science(tables, tmp_path):
    original = resolve(tables)
    assert resolve(tables, request(name='different')).scientific_id == original.scientific_id
    paths = {}
    for name, frame in tables.items():
        paths[name] = tmp_path / (name + '.csv')
        frame.to_csv(paths[name], index=False)
    resolved = resolve_request(request(), source_run='source', tables=tables, input_hashes={k: file_hash(v) for k, v in paths.items()})
    first = run_request(resolved, paths, tmp_path / 'pipeline', only=('coordination-design',))
    second = run_request(resolved, paths, tmp_path / 'pipeline', only=('coordination-design',), presentation={'maps': {'limit': 1}})
    assert first.successful and second.successful
    assert second.results['coordination-design'].outcome.status == 'reused'
    assert not read_document(first.results['coordination-design'].artifact('coordination_design'))['optional_sources_opened']
    with pytest.raises(ValueError, match='unknown setting'):
        request(edge_limit=1)

@pytest.mark.parametrize('change', [{'geometry': {**declaration()['geometry'], 'neighbourhood': {'method': 'radius', 'distance': 10}}}, {'representation': 'cosinor'}, {'shared_reference': {'method': 'leave_pair_out_mean', 'min_cells': 2}}, {'questions': {'delay': {'enabled': True, 'statistic': 'pearson', 'evidence': {'method': 'none'}, 'range_hours': [-2, 2], 'resolution_hours': 0.3}}}, {'questions': {'proximity': {'enabled': True, 'statistic': 'pearson', 'evidence': {'method': 'none'}, 'window_hours': 4, 'step_hours': 1, 'coordination': 'delay'}}}, {'questions': {'states': {'enabled': True, 'statistic': 'agreement', 'evidence': {'method': 'none'}, 'source': {'execution_record': 'x', 'scientific_id': 'a' * 64}}}}])
def test_ambiguous_scientific_choices_fail_early(change):
    with pytest.raises(ValueError):
        request(**copy.deepcopy(change))

def test_registered_parser_keeps_all_existing_requests():
    assert parse([declaration()])[0].pipeline == 'spatial-coordination'
    from tests.test_behaviour_options import declaration as states
    assert [r.pipeline for r in parse([declaration(), states()])] == ['spatial-coordination', 'cell-behaviour-states']
