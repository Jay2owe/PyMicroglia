"""Parallel discovery must preserve the frozen serial proposals exactly."""
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from pymicroglia.tracking import bracketed_handoff_reuse as reuse
from pymicroglia.tracking.execution import ExecutionPolicy, activate, current


@pytest.fixture
def handoff():
    code = Path(reuse.__file__).parent/'data/motion_engine/code'
    sys.path.insert(0, str(code))
    spec = importlib.util.spec_from_file_location('_execution_test_handoff', code/'bracketed_handoff_recovery.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        sys.path.remove(str(code))


def movie():
    labels = np.zeros((13, 16, 16), np.uint16)
    labels[:4, 3, 3] = 1
    labels[7:, 3, 3] = 1
    labels[:, 10, 10] = 2
    rows = []
    for track, frames, owners, x, y in (
            (1, range(4), [1]*4, 3, 3),
            (2, range(4, 13), [2]*3+[1]*6, 3, 3),
            (3, range(13), [2]*13, 10, 10)):
        for frame, owner in zip(frames, owners):
            rows.append(dict(track_id=track, frame=frame, accepted_owner=owner,
                             physically_visible=True, in_encounter=False,
                             x=float(x), y=float(y), radius_px=1.0))
    return pd.DataFrame(rows), labels


def test_policy_validation_and_scope():
    assert current() is None
    with activate(dict(workers=2, stage_workers={'detection': 1})) as policy:
        assert policy.count('detection') == 1
        assert current() is policy
        with activate(dict(workers=1)):
            assert current().workers == 1
        assert current() is policy
    assert current() is None
    for invalid in [dict(workers=True), dict(workers=0), dict(cpu_budget=-1),
                    dict(memory_gb=float('nan')), dict(stage_workers={'unknown': 2})]:
        with pytest.raises((ValueError, TypeError)):
            ExecutionPolicy(**invalid)


@pytest.mark.parametrize('variant', ['eligible', 'shuffled', 'invisible', 'empty', 'ties'])
def test_spawned_discovery_exact(handoff, variant):
    scored, labels = movie()
    if variant == 'shuffled':
        scored = scored.sample(frac=1, random_state=4).set_axis(np.arange(len(scored))*3+5)
    elif variant == 'invisible':
        scored['physically_visible'] = False
    elif variant == 'empty':
        scored = scored.iloc[:0]
    elif variant == 'ties':
        scored = pd.concat([scored, scored[scored.track_id.eq(1)].assign(track_id=4)])
    params = {'maximum_foreign_run_fraction': .25}
    before = labels.copy()
    expected = handoff.discover(scored, labels, params)
    if variant == 'eligible':
        assert expected.discovery_status.eq('eligible').any()
    original = handoff.discover
    with activate(dict(workers=2, memory_gb=.25)):
        with reuse.install(handoff) as stats:
            assert stats['supported'], hashlib.sha256(Path(handoff.__file__).read_bytes()).hexdigest()
            actual = handoff.discover(scored, labels, params)
            if variant in ('eligible', 'shuffled', 'ties'):
                assert stats['parallel_calls'] == 1
    assert handoff.discover is original
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    np.testing.assert_array_equal(labels, before)


def test_memory_budget_falls_back_to_serial(handoff):
    scored, labels = movie()
    expected = handoff.discover(scored, labels, {})
    with activate(dict(workers=4, memory_gb=.001)):
        with reuse.install(handoff) as stats:
            actual = handoff.discover(scored, labels, {})
            assert stats['workers'] == 1
            assert stats['parallel_calls'] == 0
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_legacy_bridge_restores_adapters(handoff):
    from pymicroglia.tracking import legacy_execution
    scored, labels = movie()
    expected = handoff.discover(scored, labels, {})
    original = reuse.install
    with legacy_execution.install(dict(workers=2, memory_gb=.25)):
        with reuse.install(handoff):
            actual = handoff.discover(scored, labels, {})
    assert reuse.install is original
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_parallel_implementation_change_invalidates_checkpoint(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from pymicroglia.tracking import checkpoints, shipped_engines
    config = tmp_path/'config.json'; config.write_text('{}')
    pipeline = SimpleNamespace(ROOT=tmp_path/'engine',_verify_pins=lambda *_:None)
    cfg = SimpleNamespace(path=config)
    monkeypatch.setattr(shipped_engines,'fingerprint',lambda _: 'frozen-test-engine')
    first = checkpoints.Stages(pipeline,cfg,'sample','run',lambda _:None)
    for name in ('execution.py','parallel_handoff.py','legacy_execution.py'):
        assert name in first.signature['adapter']
    unchanged = checkpoints.Stages(pipeline,cfg,'sample','run',lambda _:None)
    assert not unchanged.fresh
    original_sha = checkpoints.sha
    monkeypatch.setattr(checkpoints,'sha',lambda path:
        'changed-parallel-implementation' if Path(path).name=='parallel_handoff.py' else original_sha(path))
    changed = checkpoints.Stages(pipeline,cfg,'sample','run',lambda _:None)
    assert changed.fresh
    assert list((tmp_path/'engine').glob('tracking_resume_superseded_*.json'))


def test_resume_timing_log_preserves_earlier_evidence(tmp_path):
    from pymicroglia.tracking.engine import _history_timing_path
    first = _history_timing_path(tmp_path)
    first.write_text('earlier timing evidence')
    resumed = _history_timing_path(tmp_path)
    assert resumed != first and not resumed.exists()
    assert first.read_text() == 'earlier timing evidence'
    assert _history_timing_path(tmp_path) != resumed


def test_degraded_completed_checkpoint_is_recomputed(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from pymicroglia.tracking import completed_feedback,completed_evidence
    output=tmp_path/'m23_completed_feedback/sample'
    (output/'out').mkdir(parents=True)
    (output/'run.json').write_text('{"status":"complete","degraded":true}')
    labels=output/'out/labels.tif';labels.write_bytes(b'old labels')
    files={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in output.rglob('*') if p.is_file()}
    (output/'completion_files.json').write_text(json.dumps(files))
    monkeypatch.setitem(sys.modules,'pipeline',SimpleNamespace())
    monkeypatch.setitem(sys.modules,'common',SimpleNamespace(
        Config=SimpleNamespace(load=lambda _:SimpleNamespace()),save_stack=lambda *_:None))
    def fresh(*args,**kwargs):raise RuntimeError('fresh calculation reached')
    monkeypatch.setattr(completed_evidence,'generate',fresh)
    with pytest.raises(RuntimeError,match='fresh calculation reached'):
        completed_feedback.run_current(tmp_path,'sample','sample',tmp_path/'config.json',tmp_path,resume=True)
    kept=list((tmp_path/'m23_completed_feedback').glob('sample_degraded_*'))
    assert len(kept)==1 and (kept[0]/'out/labels.tif').read_bytes()==b'old labels'
