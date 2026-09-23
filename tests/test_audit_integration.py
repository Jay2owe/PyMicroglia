"""Boundary checks for the reproducible actual-engine audit exercise.

The full registered-figure exercise is runnable with audit_demo --verify; exact
decision outcomes are tested separately without relying on stochastic luck.
"""
from pymicroglia._results import read_document
import json
import pandas as pd
import pytest
from pymicroglia.pipelines import parse
from tests.pipelines.audit.demo import prepare, check_links
from pymicroglia.pipelines.audit.options import resolve_request
from pymicroglia.pipelines._screening import file_hash

def test_prepared_run_keeps_independent_movies_measurements_gaps_and_truth(tmp_path):
    output = prepare(tmp_path / 'fixture')
    path = output / 'run' / 'pooled' / 'tables' / 'cell_frame.csv'
    frame = pd.read_csv(path)
    assert len(frame) == 320 and set(frame.identity) == {1}
    assert set(frame.stem) == {'a', 'b'}
    assert frame.loc[frame.hours.isin([20.0, 21.0]), ['signal', 'other']].isna().all().all()
    assert not frame[frame.stem.eq('a')].signal.reset_index(drop=True).equals(frame[frame.stem.eq('b')].signal.reset_index(drop=True))
    request = parse([read_document(output / 'audit.json')])[0]
    resolved = resolve_request(request, source_run=file_hash(output / 'run' / 'manifest.json'), tables={'cell_frame': frame}, input_hashes={'cell_frame': file_hash(path)})
    assert len(resolved.profiles) == 4 and len(resolved.candidates) == 2
    assert {p['metadata']['sample_assignment']['sample'] for p in resolved.profiles} == {'synthetic-A', 'synthetic-B'}
    generated = read_document(output / 'source_generation.json')
    assert {case['scenario'] for case in generated['cases']} == {'triangle-12h', 'square-36h'}
    before = file_hash(path)
    with pytest.raises(ValueError, match='new empty folder'):
        prepare(output)
    assert file_hash(path) == before

def test_portability_check_detects_missing_anchors_and_escaping_files(tmp_path):
    (tmp_path / 'index.html').write_text('<a href="#missing">Missing</a>', encoding='utf-8')
    with pytest.raises(ValueError, match='anchor'):
        check_links(tmp_path)
    (tmp_path / 'index.html').write_text('<a href="../outside.csv">Escaping</a>', encoding='utf-8')
    with pytest.raises(ValueError, match='escaping'):
        check_links(tmp_path)

@pytest.mark.parametrize('unlock_at', [3, 10])
def test_execution_record_survives_a_temporary_sync_lock(tmp_path, monkeypatch, unlock_at):
    import pymicroglia.pipelines._runner as runner
    destination = tmp_path / 'execution.json'
    runner._write(destination, {'state': 'previous'})
    replace = runner.os.replace
    attempts = []

    def locked(source, target):
        attempts.append(source)
        assert read_document(destination)['state'] == 'previous'
        if len(attempts) < unlock_at:
            raise PermissionError('Temporary verification lock')
        return replace(source, target)
    monkeypatch.setattr(runner.os, 'replace', locked)
    monkeypatch.setattr('time.sleep', lambda seconds: None)
    runner._write(destination, {'state': 'completed'})
    assert len(attempts) == unlock_at and len(set(attempts)) == 1
    assert read_document(destination)['state'] == 'completed'
    assert not list(tmp_path.glob('*.partial'))

def test_persistent_sync_lock_keeps_the_previous_durable_record(tmp_path, monkeypatch):
    import pymicroglia.pipelines._runner as runner
    destination = tmp_path / 'execution.json'
    runner._write(destination, {'state': 'previous'})

    def locked(*args):
        raise PermissionError('Persistent verification lock')
    monkeypatch.setattr(runner.os, 'replace', locked)
    monkeypatch.setattr('time.sleep', lambda seconds: None)
    with pytest.raises(PermissionError):
        runner._write(destination, {'state': 'completed'})
    assert read_document(destination)['state'] == 'previous'
    assert not list(tmp_path.glob('*.partial'))
