"""Cross-cell rhythm timing uses corrected saved endpoints and public Workbench."""
import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from pymicroglia import workbench as circadian
from pymicroglia.pipelines import parse
import pymicroglia.pipelines.coordination.inputs as inputs
import pymicroglia.pipelines.coordination.rhythm as rhythm
import pymicroglia.pipelines.rhythm.discovery as rhythm_discovery
from pymicroglia.pipelines.coordination.options import resolve_request, run_request
from pymicroglia.pipelines.coordination.sources import load_source
from pymicroglia.pipelines._contracts import Settings
from pymicroglia.pipelines._screening import file_hash, read_table
from tests.test_coordination_options import request

@pytest.fixture(scope='module')
def saved_screen(tmp_path_factory):
    root = tmp_path_factory.mktemp('cross_cell_rhythm')
    hours = np.arange(50.0, 210.0, 0.25)
    parts = []
    for identity, period, offset in [(1, 8.0, 0.0), (2, 8.0, 2.0), (3, None, 0.0), (4, 12.0, 0.0)]:
        values = np.zeros(len(hours)) if period is None else np.cos(2 * np.pi * (hours - offset) / period)
        parts.append(pd.DataFrame({'stem': 'movie', 'identity': identity, 'frame_index': np.arange(len(hours)), 'hours': hours, 'custom_signal': values, 'centroid_x': float(identity * 4), 'centroid_y': 0.0}))
    frame = pd.concat(parts, ignore_index=True)
    tables = {'cell_frame': frame, 'cell_summary': frame[['stem', 'identity']].drop_duplicates()}
    paths = {name: root / (name + '.csv') for name in tables}
    for name, table in tables.items():
        table.to_csv(paths[name], index=False)
    block = {'pipeline': 'rhythm-discovery', 'test_measurements': ['custom_signal'], 'analysis_options': {'fit_method': 'spectrum_resampling', 'significance_method': 'lomb', 'detrend': 'none', 'period_min_hours': 2.0, 'period_max_hours': 20.0, 'period_config': {'sr_iterations': 100, 'sr_seed': 7}}}
    resolved = rhythm_discovery.resolve_request(parse([block])[0], source_run='cross-cell-controls', tables=tables, input_hashes={name: file_hash(path) for name, path in paths.items()})
    execution = rhythm_discovery.run_request(resolved, paths, root / 'screen', only=['rhythm-screen'])
    assert execution.successful
    source = {'execution_record': str(execution.record_path), 'scientific_id': execution.results['rhythm-screen'].outcome.scientific_id}
    return (source, tables, paths)

def fixture(saved_screen, **changes):
    source, tables, paths = saved_screen
    q = {'enabled': True, 'statistic': 'phase_offset_hours', 'evidence': {'method': 'none'}, 'source': source, 'settings': {}}
    req = request(target_measurements=['custom_signal'], questions={'rhythm': q}, **changes)
    resolved = resolve_request(req, source_run='cross-cell-controls', tables=tables, input_hashes={name: file_hash(path) for name, path in paths.items()})
    prepared, definitions = inputs.prepare(resolved, tables)
    prepared.update(pair_definitions=definitions, provenance={'scientific_id': 'inputs'})
    return (resolved, prepared, paths)

def test_corrected_eight_hour_rhythms_compare_across_cells_without_refitting(saved_screen):
    resolved, prepared, _ = fixture(saved_screen)
    source, provenance, binding = rhythm.source_data(resolved)
    with patch.object(circadian, 'estimate_one', side_effect=AssertionError('Do not repeat endpoint screening')), patch.object(circadian, 'rhythm_pair_timing', wraps=circadian.rhythm_pair_timing) as gateway:
        output, details, options = rhythm.analyse(resolved, prepared, source, provenance, rhythm.validate_question(resolved.request.questions['rhythm'], ['custom_signal']), 'timing')
    effects = output['pair_effects']
    assert len(effects) == 6 and gateway.call_count == 6
    selected = effects.query('reference_identity == 1 and target_identity == 2').iloc[0]
    assert selected.status == 'eligible' and selected.period_hours == pytest.approx(8.0, abs=0.15)
    assert selected.effect == pytest.approx(2.0, abs=0.1) and selected.both_significant
    assert effects.p_value.isna().all() and effects.reference.eq('custom_signal').all()
    assert len(output['endpoint_evidence']) == 12
    assert output['endpoint_evidence'].estimator.dropna().eq('spectrum_resampling').all()
    assert output['endpoint_evidence'].significance_method.dropna().eq('lomb').all()
    assert effects.loc[effects.reference_identity.eq(3) | effects.target_identity.eq(3), 'status'].eq('ineligible').all()
    assert effects.loc[effects.target_identity.eq(4), 'status'].eq('ineligible').all()
    assert options['min_observations'] >= 24 and options['min_cycles'] >= 3
    assert output['timecourse'].hours.min() > 50.0
    assert all(('workbench_run_record' in item['result'] for item in details))
    assert binding['scientific_id'] == source.outcome.scientific_id

def test_saved_rhythm_source_identity_window_and_original_table_must_match(saved_screen):
    resolved, _, _ = fixture(saved_screen)
    changed = replace(resolved, inputs=replace(resolved.inputs, source_run='different'))
    with pytest.raises(ValueError, match='source run'):
        rhythm.source_data(changed)
    changed = replace(resolved, request=replace(resolved.request, time_range_hours=(60.0, 100.0)))
    with pytest.raises(ValueError, match='window'):
        rhythm.source_data(changed)
    hashes = resolved.inputs.table_hashes.as_dict()
    hashes['cell_frame'] = 'f' * 64
    changed = replace(resolved, inputs=replace(resolved.inputs, table_hashes=Settings(hashes)))
    with pytest.raises(ValueError, match='changed'):
        rhythm.source_data(changed)
    source = saved_screen[0]
    with pytest.raises(ValueError, match='scientific identity'):
        load_source({**source, 'scientific_id': 'f' * 64}, recipe='rhythm-discovery', step='rhythm-screen')

def test_disabled_rhythm_never_opens_optional_sources_or_calls_workbench():
    context = SimpleNamespace(request=SimpleNamespace(request=SimpleNamespace(questions={'rhythm': {'enabled': False}})), step=SimpleNamespace(name='rhythm-coordination'), scientific_id='disabled')
    with patch.object(rhythm, 'load_source', side_effect=AssertionError('No optional I/O')), patch.object(circadian, 'rhythm_pair_timing', side_effect=AssertionError('No timing')):
        assert rhythm.identity(context)
        assert rhythm.produce(context).status == 'skipped-empty'

def test_period_test_probabilities_cannot_become_phase_pair_probabilities(saved_screen):
    resolved, _, _ = fixture(saved_screen)
    q = resolved.request.questions['rhythm']
    q = {**q, 'evidence': {'method': 'reuse_trace_p'}}
    from pymicroglia.pipelines._runner import Unavailable
    with pytest.raises(Unavailable, match='no pair significance'):
        rhythm.validate_question(q, ['custom_signal'])

def test_missing_saved_endpoints_and_invalid_original_clocks_remain_visible(saved_screen):
    resolved, prepared, _ = fixture(saved_screen)
    screen, provenance, _ = rhythm.source_data(resolved)
    screen = replace(screen, results=screen.results.loc[~screen.results.identity.eq(2)], traces=screen.traces.loc[~screen.traces.identity.eq(2)])
    prepared['endpoints'].loc[prepared['endpoints'].identity.eq(4), 'status'] = 'invalid_clock'
    outputs, _, _ = rhythm.analyse(resolved, prepared, screen, provenance, rhythm.validate_question(resolved.request.questions['rhythm'], ['custom_signal']), 'timing')
    effects = outputs['pair_effects']
    assert len(effects) == 6
    assert effects.loc[effects.reference_identity.eq(2) | effects.target_identity.eq(2), 'status'].eq('ineligible').all()
    assert outputs['endpoint_evidence'].loc[outputs['endpoint_evidence'].identity.eq(2), 'screen_status'].eq('not_screened').all()
    assert effects.loc[effects.target_identity.eq(4), 'reason'].str.contains('original acquisition clock').all()

def test_real_rhythm_adapter_reopens_with_screening_and_phase_calls_blocked(saved_screen, tmp_path):
    resolved, _, paths = fixture(saved_screen)
    first = run_request(resolved, paths, tmp_path / 'pipeline', only=['rhythm-coordination'])
    assert first.successful, {key: value.outcome.reason for key, value in first.results.items()}
    assert len(read_table(first.results['rhythm-coordination'].artifact('pair_effects'))) == 6
    with patch.object(circadian, 'estimate_one', side_effect=AssertionError('No refit')), patch.object(circadian, 'rhythm_pair_timing', side_effect=AssertionError('No repeat phase')), patch.object(inputs, 'prepare', side_effect=AssertionError('No repeated inputs')):
        second = run_request(resolved, paths, tmp_path / 'pipeline', only=['rhythm-coordination'])
    assert second.successful and all((value.outcome.status == 'reused' for value in second.results.values()))
