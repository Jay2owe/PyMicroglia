"""The recorded CLI must deliver tracking files even if its audit omits them."""
import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from pymicroglia.tracking.contract import TrackingResult


@pytest.fixture
def runner(monkeypatch):
    path = Path.home() / '.claude/skills/pymicroglia/scripts/pymicroglia_runner.py'
    if not path.is_file():
        pytest.skip('Optional local skill runner is not installed')
    spec = importlib.util.spec_from_file_location('tracking_result_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class Registry:
        def __contains__(self, name): return True
        def spec(self, name): return SimpleNamespace(params=())
    monkeypatch.setattr(module, 'build_registry', Registry)
    return module


@pytest.mark.parametrize('record', [{}, {'result': None}, {'result': {'labels': 'stale'}}])
def test_actual_tracking_contract_survives_audit_summary(runner, monkeypatch, tmp_path, record):
    labels, raw = tmp_path/'labels.tif', tmp_path/'raw.tif'
    labels.write_bytes(b'labels'); raw.write_bytes(b'raw')
    actual = TrackingResult(stem='sample', labels=labels, raw=raw, source_frame_offset=0)
    monkeypatch.setattr(importlib.import_module('pymicroglia.run'), 'run_recorded',
                        lambda *a, **kw: dict(result=actual, record=record))
    response = runner.execute({'action': 'track', 'root': str(tmp_path)})
    assert response['ok']
    assert response['result'] == actual.as_dict()


def test_empty_tracking_result_is_not_cached_as_success(runner, monkeypatch, tmp_path):
    monkeypatch.setattr(importlib.import_module('pymicroglia.run'), 'run_recorded',
                        lambda *a, **kw: dict(result=None, record={'result': None}))
    response = runner.execute({'action': 'track', 'root': str(tmp_path)})
    assert not response['ok']


def test_other_actions_keep_bounded_summary(runner, monkeypatch, tmp_path):
    monkeypatch.setattr(importlib.import_module('pymicroglia.run'), 'run_recorded',
                        lambda *a, **kw: dict(result=object(), record={'result': {'shape': [9, 4]}}))
    response = runner.execute({'action': 'segment', 'root': str(tmp_path)})
    assert response['result'] == {'shape': [9, 4]}
