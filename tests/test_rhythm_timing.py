"""Whole saved screens feed timing without changing the original detections."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from pymicroglia import workbench as circadian
from pymicroglia.pipelines import parse
from pymicroglia.pipelines.rhythm.discovery import resolve_request, run_request
from pymicroglia.pipelines._screening import file_hash, read_table

def fixture(tmp_path, *, empty_pairs=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    parts = []
    hours = np.arange(50, 210, 0.25)
    for movie, identity, offset in (('a', 1, 2.0), ('b', 1, 4.0), ('a', 2, None)):
        parts.append(pd.DataFrame({'stem': movie, 'identity': identity, 'hours': hours, 'frame_index': np.arange(len(hours)), 'signal': np.cos(2 * np.pi * hours / 8), 'other': np.cos(2 * np.pi * (hours - offset) / 8) if offset is not None else np.zeros(len(hours))}))
    frame = pd.concat(parts, ignore_index=True)
    tables = {'cell_frame': frame, 'cell_summary': frame[['stem', 'identity']].drop_duplicates()}
    paths = {key: tmp_path / (key + '.csv') for key in tables}
    for key, table in tables.items():
        table.to_csv(paths[key], index=False)
    block = {'pipeline': 'rhythm-discovery', 'test_measurements': ['signal', 'other'], 'biological_samples': {'a': 'sample1', 'b': 'sample2'}, 'analysis_options': {'fit_method': 'spectrum_resampling', 'significance_method': 'lomb', 'detrend': 'none', 'period_min_hours': 2, 'period_max_hours': 20, 'period_config': {'sr_iterations': 100, 'sr_seed': 7}}}
    if empty_pairs:
        block['pairs'] = {'mode': 'explicit', 'pairs': []}
    resolved = resolve_request(parse([block])[0], source_run='non-daily-timing-fixture', tables=tables, input_hashes={key: file_hash(path) for key, path in paths.items()})
    return (resolved, paths)

def test_actual_workbench_timing_keeps_every_cell_and_resumes_without_analysis(tmp_path, monkeypatch):
    resolved, paths = fixture(tmp_path)
    execution = run_request(resolved, paths, tmp_path / 'pipeline', only=('within-cell-timing',))
    assert execution.successful, {key: result.outcome.reason for key, result in execution.results.items()}
    saved = execution.results['within-cell-timing']
    pairs = read_table(saved.artifact('pairs'))
    assert len(pairs) == 3 and set(pairs.status) == {'eligible', 'ineligible'}
    selected = pairs[pairs.status.eq('eligible')]
    assert len(selected) == 2
    assert selected[selected.movie.eq('a')].offset_hours.iloc[0] == pytest.approx(2.0, abs=0.05)
    assert abs(selected[selected.movie.eq('b')].offset_hours.iloc[0]) == pytest.approx(4.0, abs=0.05)
    assert sum((len(selection.members) for selection in saved.outcome.selections)) == 3
    assert read_table(saved.artifact('timecourse')).hours.min() > 50
    before = execution.results['rhythm-screen'].outcome.scientific_id

    def forbidden(*a, **k):
        raise AssertionError('Reused timing attempted analysis')
    monkeypatch.setattr(circadian, 'estimate_one', forbidden)
    monkeypatch.setattr(circadian, 'rhythm_pair_timing', forbidden)
    resumed = run_request(resolved, paths, tmp_path / 'pipeline', only=('within-cell-timing',))
    assert resumed.successful and all((result.outcome.status == 'reused' for result in resumed.results.values()))
    assert resumed.results['rhythm-screen'].outcome.scientific_id == before

def test_empty_requested_pairs_complete_without_a_timing_call(tmp_path, monkeypatch):
    resolved, paths = fixture(tmp_path, empty_pairs=True)

    def forbidden(*a, **k):
        raise AssertionError('No timing pair requested')
    monkeypatch.setattr(circadian, 'rhythm_pair_timing', forbidden)
    execution = run_request(resolved, paths, tmp_path / 'pipeline', only=('within-cell-timing',))
    assert execution.successful
    assert read_table(execution.results['within-cell-timing'].artifact('pairs')).empty

def test_public_gateway_preserves_evidence_options_and_native_run_record(monkeypatch):
    called = {}

    def run(reference, target, **options):
        called.update(reference=reference, target=target, **options)
        return SimpleNamespace(data={'status': 'ineligible', 'reason': 'missing uncertainty'}, run_record={'method': 'public'})
    monkeypatch.setattr(circadian.cw, 'rhythm_pair_timing', run)
    result = circadian.rhythm_pair_timing({'source': 1}, {'source': 2}, settings={'min_cycles': 7})
    assert called == {'reference': {'source': 1}, 'target': {'source': 2}, 'settings': {'min_cycles': 7}}
    assert result['workbench_run_record'] == {'method': 'public'}

@pytest.mark.parametrize('timing', [{'trace_representation': 'fitted'}, {'component_bands': {'missing': [7, 9]}}, {'component_bands': {'signal': [9, 7]}}, {'phase_options': []}])
def test_invalid_timing_choices_are_rejected_before_execution(timing):
    with pytest.raises(ValueError):
        parse([{'pipeline': 'rhythm-discovery', 'test_measurements': ['signal'], 'timing': timing}])
