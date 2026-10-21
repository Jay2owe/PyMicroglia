"""A stale verification receipt must not disable the actual batch checkpoints."""
from pathlib import Path
import importlib.util
import json
import sys
import types

import pytest


@pytest.mark.parametrize('receipt', [None, {'status': 'passed', 'code_sha256': {'changed.py': 'old'}}])
def test_batch_always_requests_verified_stage_resume(tmp_path, monkeypatch, receipt):
    task = tmp_path/'task'
    automation = tmp_path/'automation'
    source = automation/'rigid_native/A1'
    source.mkdir(parents=True)
    red = tmp_path/'red.tif'; red.write_bytes(b'red fixture')
    green = tmp_path/'green.tif'; green.write_bytes(b'green fixture')
    (source/'manifest.json').write_text(json.dumps({'outputs': [
        {'role': 'red_original', 'path': str(red)},
        {'role': 'green_original', 'path': str(green)}]}))
    (automation/'original_downloads.json').write_text('{"status":"complete"}')
    if receipt is not None:
        (automation/'tracking_checkpoint_check.json').write_text(json.dumps(receipt))
    requests = []
    class ReachedTracking(Exception):
        pass
    def recorded(target, name, request):
        requests.append(request)
        raise ReachedTracking
    stub = types.ModuleType('track_measure')
    stub.ROOT, stub.AUTO, stub.recorded = task, automation, recorded
    monkeypatch.setitem(sys.modules, 'track_measure', stub)
    producer = Path(__file__).resolve().parents[1]/'development/cd68_full_red_green/track_rigid.py'
    spec = importlib.util.spec_from_file_location('full_recording_checkpoint_test', producer)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    recovery_calls = []
    monkeypatch.setattr(loaded.subprocess, 'run', lambda args, **kw: recovery_calls.append((args, kw)))
    monkeypatch.setattr(sys, 'argv', [str(producer), 'A1', str(tmp_path/'output')])
    with pytest.raises(ReachedTracking):
        loaded.main()
    assert len(recovery_calls) == 1 and recovery_calls[0][1]['check'] is True
    assert Path(recovery_calls[0][0][1]) == automation/'checkpoint_recovery.py'
    options = requests[0]['params']['tracker_options']
    assert options == dict(stem='95_full_A1', bounded_memory=True,
                           working_array_min_bytes=1024**2, resume=True)
