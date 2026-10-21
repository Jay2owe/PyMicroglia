"""Interrupt every boundary: reuse earlier stages and preserve exact evidence."""
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest
import tifffile

from pymicroglia.tracking.checkpoints import STAGES, exclusive, sha
from pymicroglia.tracking.engine_bounded import run


def fixture(folder, fail=None):
    folder.mkdir(parents=True, exist_ok=True)
    raw = folder/'raw.tif'
    labels = np.zeros((3, 7, 9), np.uint16)
    labels[:, 1:4, 2:5] = 1
    labels[:, 4:6, 6:8] = 2
    flag = labels == 2
    tifffile.imwrite(raw, labels, photometric='minisblack')
    cfg_path = folder/'config.json'
    cfg_path.write_text(json.dumps(dict(stems=['sample'], raw_sha256=sha(raw))))
    cfg = SimpleNamespace(path=cfg_path, stems=['sample'])
    common = SimpleNamespace(Config=SimpleNamespace(load=lambda _: cfg))
    table = pd.DataFrame({'x': [np.nextafter(1., 2.), np.pi],
                          'identity': pd.Series([1, 2], dtype='int32'),
                          'kind': ['observed', 'inferred']})
    calls = []
    broken = [False]
    pipeline = SimpleNamespace(ROOT=folder/'engine',
        _pinned_paths=lambda *_: {'registered_raw': raw},
        AcceptedHistoryInputs=lambda: SimpleNamespace())

    def verify(*_):
        if sha(raw) != json.loads(cfg_path.read_text())['raw_sha256']:
            raise ValueError('Input changed')
    pipeline._verify_pins = verify

    def stage(method):
        def compute(*args, **kwargs):
            calls.append(method)
            directory = pipeline.ROOT/STAGES[method]/'run'
            directory.mkdir(parents=True)
            (directory/'run.json').write_text(json.dumps(dict(status='running', outputs={})))
            outputs = {}

            def array(role, data):
                path = directory/'out'/(role+'.tif')
                path.parent.mkdir(exist_ok=True)
                tifffile.imwrite(path, data.astype(np.uint8) if data.dtype == bool else data,
                                 photometric='minisblack')
                outputs[role] = dict(path=str(path.resolve()), sha256=sha(path))

            def csv(role, value):
                path = directory/'out'/(role+'.csv')
                path.parent.mkdir(exist_ok=True)
                value.to_csv(path, index=False)
                outputs[role] = dict(path=str(path.resolve()), sha256=sha(path))

            def compare_history(history):
                np.testing.assert_array_equal(history.reservation_labels, labels)
                np.testing.assert_array_equal(history.pre_reservation_labels, labels + 1)
                pd.testing.assert_frame_equal(history.host_audit, table)
                pd.testing.assert_frame_equal(history.motion_reservation_events, table)
                assert history.motion_reserved == {1, 2}

            if method == 'stage_m22_accepted_history':
                work = directory/'mid/accepted_history'
                work.mkdir(parents=True)
                (work/'saved_issue.txt').write_text('accepted issue work')
            if fail == method and not broken[0]:
                broken[0] = True
                raise RuntimeError('simulated interruption')
            if method == 'stage_m0':
                path = directory/'out/sample_sources.json'
                path.parent.mkdir()
                path.write_text(json.dumps(dict(config_sha256=sha(cfg_path))))
                outputs['sources'] = dict(path=str(path.resolve()), sha256=sha(path))
                result = {'registered_raw': raw}
            elif method == 'stage_m1':
                result = None
            elif method == 'stage_m2':
                array('observations', labels)
                csv('observation_table', table)
                result = (labels.copy(), table.copy(), 1, flag.copy(), flag.copy())
            elif method == 'stage_m3':
                np.testing.assert_array_equal(args[4], labels)
                pd.testing.assert_frame_equal(args[5], table)
                assert args[6] == 1
                array('initial_identities', labels)
                result = (labels.copy(), table.copy())
            elif method == 'stage_m4':
                array('event_identities', labels)
                array('inferred_pixels', flag)
                array('unresolved_pixels', ~flag)
                result = (labels.copy(), table.copy(), flag.copy(), (~flag).copy())
            elif method == 'stage_m5':
                np.testing.assert_array_equal(args[6], flag)
                pd.testing.assert_frame_equal(args[5], table)
                result = (table.copy(), table.copy())
            elif method == 'stage_m7_accepted':
                array('labels', labels)
                result = (labels.copy(), table.copy())
            elif method == 'stage_m22_accepted_history':
                history = kwargs['history_inputs']
                compare_history(history)
                np.testing.assert_array_equal(history.physical_somas, labels + 2)
                np.testing.assert_array_equal(kwargs['upstream_inferred'], flag)
                np.testing.assert_array_equal(kwargs['unresolved'], ~flag)
                assert kwargs['m20_result_future'].done()
                if fail == method:
                    upstream = pipeline.ROOT/STAGES[method]/kwargs['checkpoint_run_name']/'mid/accepted_history'
                    assert (upstream/'saved_issue.txt').read_text() == 'accepted issue work'
                array('labels', labels)
                array('unclaimed', labels * 0)
                array('inferred', flag)
                result = (labels.copy(), labels * 0, flag.copy(), table.copy())
            else:
                array('labels', labels)
                array('inferred', flag)
                if method == 'stage_m12_accepted':
                    history = kwargs['history_inputs']
                    history.reservation_labels = labels.copy()
                    history.pre_reservation_labels = labels + 1
                    history.host_audit = table.copy()
                    history.motion_reserved = {1, 2}
                    history.motion_reservation_events = table.copy()
                    array('pre_reservation_labels', history.pre_reservation_labels)
                if method == 'stage_m19_accepted':
                    compare_history(kwargs['history_inputs'])
                    kwargs['history_inputs'].physical_somas = labels + 2
                    array('physical_somas', labels + 2)
                result = (labels.copy(), flag.copy(), table.copy())
            (directory/'run.json').write_text(json.dumps(dict(status='done', outputs=outputs)))
            return result
        return compute
    for method in STAGES:
        setattr(pipeline, method, stage(method))
    return pipeline, common, cfg, calls


@pytest.mark.parametrize('interrupted', list(STAGES))
def test_resume_at_each_completed_boundary_matches_cold_outputs(tmp_path, interrupted):
    cold, common, cfg, _ = fixture(tmp_path/'cold')
    run(cold, common, 'run', 'sample', cfg.path)
    resumed, common, cfg, calls = fixture(tmp_path/'resumed', fail=interrupted)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        run(resumed, common, 'run', 'sample', cfg.path, resume=True)
    run(resumed, common, 'run', 'sample', cfg.path, resume=True)
    assert calls.count(interrupted) == 2
    assert all(calls.count(method) == 1 for method in STAGES if method != interrupted)
    assert not list(resumed.ROOT.rglob('checkpoint*.tif'))
    # Compare saved masks and decision tables, excluding execution receipts.
    for path in cold.ROOT.rglob('*'):
        if path.suffix in ('.tif', '.csv'):
            assert sha(path) == sha(resumed.ROOT/path.relative_to(cold.ROOT))
    state = json.loads((resumed.ROOT/'tracking_resume.json').read_text())
    assert state['reused'] == list(STAGES.values())[:list(STAGES).index(interrupted)]


def test_resume_rejects_changed_input_settings_and_stage_bytes(tmp_path):
    pipeline, common, cfg, _ = fixture(tmp_path)
    run(pipeline, common, 'run', 'sample', cfg.path, resume=True)
    original = cfg.path.read_text()
    cfg.path.write_text(original + ' ')
    with pytest.raises(ValueError, match='settings.*changed'):
        run(pipeline, common, 'run', 'sample', cfg.path, resume=True)
    cfg.path.write_text(original)
    stage = pipeline.ROOT/'m2_anchor/run/out/observations.tif'
    stage.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checkpoint file changed'):
        run(pipeline, common, 'run', 'sample', cfg.path, resume=True)
    # A changed input is refused before any cache is used.
    (tmp_path/'raw.tif').write_bytes(b'changed input')
    with pytest.raises(ValueError, match='Input changed'):
        run(pipeline, common, 'run', 'sample', cfg.path, resume=True)


def test_resume_requires_bounded_memory(tmp_path):
    from pymicroglia.tracking.engine import run as engine_run
    with pytest.raises(ValueError, match='bounded_memory=True'):
        engine_run(tmp_path/'missing.json', tmp_path/'output', resume=True)


def test_unfinished_stage_with_truncated_metadata_can_restart(tmp_path):
    pipeline, common, cfg, calls = fixture(tmp_path, fail='stage_m2')
    with pytest.raises(RuntimeError, match='simulated interruption'):
        run(pipeline, common, 'run', 'sample', cfg.path, resume=True)
    (pipeline.ROOT/'m2_anchor/run/run.json').write_text('{"status":')
    run(pipeline, common, 'run', 'sample', cfg.path, resume=True)
    assert calls.count('stage_m0') == calls.count('stage_m1') == 1
    assert calls.count('stage_m2') == 2


def test_legacy_metadata_stages_can_be_reused_without_old_numeric_tables(tmp_path):
    pipeline, common, cfg, calls = fixture(tmp_path, fail='stage_m2')
    with pytest.raises(RuntimeError, match='simulated interruption'):
        run(pipeline, common, 'run', 'sample', cfg.path)
    run(pipeline, common, 'run', 'sample', cfg.path, resume=True)
    assert calls.count('stage_m0') == calls.count('stage_m1') == 1
    assert calls.count('stage_m2') == 2


def test_complete_checkpoint_skips_all_stages_and_rejects_library_change(tmp_path, monkeypatch):
    pipeline, common, cfg, calls = fixture(tmp_path)
    run(pipeline, common, 'run', 'sample', cfg.path, resume=True)
    calls.clear()
    run(pipeline, common, 'run', 'sample', cfg.path, resume=True)
    assert calls == []
    monkeypatch.setattr(pd, '__version__', 'different-version')
    with pytest.raises(ValueError, match='software or code changed'):
        run(pipeline, common, 'run', 'sample', cfg.path, resume=True)


def test_corrupt_table_is_rejected_before_unpickling(tmp_path, monkeypatch):
    pipeline, common, cfg, _ = fixture(tmp_path, fail='stage_m3')
    with pytest.raises(RuntimeError, match='simulated interruption'):
        run(pipeline, common, 'run', 'sample', cfg.path, resume=True)
    table = pipeline.ROOT/'m2_anchor/run/checkpoint_table_0.pkl'
    table.write_bytes(b'changed table')
    def forbidden(*args, **kwargs):
        raise AssertionError('Unverified table must not be loaded')
    monkeypatch.setattr(pd, 'read_pickle', forbidden)
    with pytest.raises(ValueError, match='checkpoint file changed'):
        run(pipeline, common, 'run', 'sample', cfg.path, resume=True)


def test_tracking_lock_releases_after_failure(tmp_path):
    path = tmp_path/'tracking.lock'
    with pytest.raises(RuntimeError, match='interrupted'):
        with exclusive(path):
            with pytest.raises(FileExistsError, match='Another process'):
                with exclusive(path):
                    pytest.fail('Two trackers acquired the same run')
            raise RuntimeError('interrupted')
    with exclusive(path):
        pass


def test_completed_correction_checkpoint_reuse_and_corruption(tmp_path, monkeypatch):
    import sys
    from pymicroglia.tracking.completed_feedback import run_current
    monkeypatch.setitem(sys.modules, 'pipeline', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'common', SimpleNamespace(Config=object, save_stack=lambda *_: None))
    output = tmp_path/'m23_completed_feedback/run'
    output.mkdir(parents=True)
    (output/'run.json').write_text(json.dumps(dict(status='complete')))
    labels = output/'labels.tif'; labels.write_bytes(b'accepted label bytes')
    (output/'completion_files.json').write_text(json.dumps({
        'run.json': sha(output/'run.json'), 'labels.tif': sha(labels)}))
    assert run_current(tmp_path, 'sample', 'run', tmp_path/'config', tmp_path, resume=True) == output
    with pytest.raises(FileExistsError, match='Immutable'):
        run_current(tmp_path, 'sample', 'run', tmp_path/'config', tmp_path)
    labels.write_bytes(b'changed label bytes')
    with pytest.raises(ValueError, match='checkpoint changed'):
        run_current(tmp_path, 'sample', 'run', tmp_path/'config', tmp_path, resume=True)
