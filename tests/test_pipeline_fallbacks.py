"""A whole tracking stage that cannot handle a recording passes its input on; a run always publishes labels."""
from concurrent.futures import Future
from importlib import util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import json
import sys

import numpy as np
import pandas as pd
import pytest
import tifffile

from pymicroglia.tracking import engine
from pymicroglia.tracking import pipeline_fallbacks as fallbacks
from pymicroglia.tracking import shipped_engines

ENGINE = shipped_engines.folder(shipped_engines.MOTION)
STEM = 'well'


@pytest.fixture
def motion(tmp_path, monkeypatch):
    """The engine's own ``common`` module, rooted in tmp_path, and a small stand-in pipeline."""
    shipped_engines.unpack(ENGINE, tmp_path, 'code/common.py')
    spec = util.spec_from_file_location('motion_common_under_test', tmp_path/'code/common.py')
    common = util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, common)
    spec.loader.exec_module(common)
    raw = np.arange(4*6*6, dtype=np.uint16).reshape(4, 6, 6)
    tifffile.imwrite(tmp_path/'raw.tif', raw)
    pipeline = ModuleType('pipeline_under_test'); pipeline.ROOT = tmp_path

    def identity_tables(labels, raw, inferred):
        rows = [dict(t=t, identity=int(i)) for t in range(len(labels)) for i in np.unique(labels[t]) if i]
        frames = pd.DataFrame(rows)
        return frames, frames.groupby('identity').size().rename('frames').reset_index()
    pipeline._identity_tables = identity_tables
    cfg = SimpleNamespace(values=dict(frame_interval_min=30.0), source_frame_offset=0)
    sources = dict(registered_raw=tmp_path/'raw.tif')
    return SimpleNamespace(root=tmp_path, common=common, pipeline=pipeline, cfg=cfg, sources=sources)


def movie():
    labels = np.zeros((4, 6, 6), np.uint16)
    labels[:, 1:3, 1:3] = 4
    labels[:, 3:5, 3:5] = 7
    return labels


def install(motion, **stages):
    for name, function in stages.items():
        setattr(motion.pipeline, name, function)
    return fallbacks.install(motion.pipeline, motion.common, motion.root)


def stage_raising(folder, error, returns=None):
    def stage(cfg, stem, run_name, sources, labels, history_inputs=None, m20_result_future=None):
        run = Path(stage.root)/folder/run_name
        (run/'out').mkdir(parents=True)
        (run/'run.json').write_text('{"status": "running"}')
        if error is not None:
            raise error
        return returns
    return stage


def test_failing_stage_publishes_its_input_in_its_own_folder(motion):
    labels = movie()
    stage = stage_raising('m8_merge', AssertionError('accepted delayed-merge gates failed'))
    stage.root = motion.root
    with install(motion, stage_m8_accepted=stage) as audit:
        out, inferred, tracks = motion.pipeline.stage_m8_accepted(
            motion.cfg, STEM, 'run', motion.sources, labels)
    assert out is not None and np.array_equal(out, labels) and not inferred.any()
    assert sorted(tracks.identity) == [4, 7]
    folder = motion.root/'m8_merge/run'
    manifest = json.loads((folder/'run.json').read_text())
    assert manifest['status'] == 'done' and {'labels', 'inferred'} <= set(manifest['outputs'])
    assert np.array_equal(tifffile.imread(manifest['outputs']['labels']['path']), labels)
    assert json.loads((folder/fallbacks.MARKER).read_text())['error_type'] == 'AssertionError'
    kept = [p.name for p in (motion.root/'m8_merge').iterdir() if p.name.startswith('run_failed_')]
    assert len(kept) == 1, 'the failed attempt is kept beside the published one'
    assert audit.skipped == ['m8_merge']
    assert json.loads((motion.root/fallbacks.AUDIT).read_text())[0]['action'] == 'input_carried_forward'


def test_successful_stage_is_untouched(motion):
    expected = (movie(), np.zeros((4, 6, 6), bool), pd.DataFrame())
    stage = stage_raising('m9_alias', None, expected); stage.root = motion.root
    with install(motion, stage_m9_accepted=stage) as audit:
        assert motion.pipeline.stage_m9_accepted(motion.cfg, STEM, 'run', motion.sources, movie()) is expected
    assert audit.skipped == [] and not (motion.root/fallbacks.AUDIT).exists()


@pytest.mark.parametrize('error', [MemoryError(), OSError('disk full'), ImportError('missing')])
def test_memory_disk_and_missing_code_still_stop(motion, error):
    stage = stage_raising('m9_alias', error); stage.root = motion.root
    with install(motion, stage_m9_accepted=stage):
        with pytest.raises(type(error)):
            motion.pipeline.stage_m9_accepted(motion.cfg, STEM, 'run', motion.sources, movie())


def test_install_restores_every_stage(motion):
    original = stage_raising('m9_alias', None)
    motion.pipeline.stage_m9_accepted = original
    with fallbacks.install(motion.pipeline, motion.common, motion.root):
        assert motion.pipeline.stage_m9_accepted is not original
    assert motion.pipeline.stage_m9_accepted is original


def test_skipped_reservation_hands_over_nothing_reserved(motion):
    stage = stage_raising('m12_identity_reservation', AssertionError(
        'accepted identity reservation assigned one identity twice')); stage.root = motion.root
    history = SimpleNamespace(reservation_labels=None, pre_reservation_labels=None, host_audit=None,
                              motion_reserved=None, motion_reservation_events=None)
    labels = movie()
    with install(motion, stage_m12_accepted=stage):
        motion.pipeline.stage_m12_accepted(motion.cfg, STEM, 'run', motion.sources, labels,
                                           history_inputs=history)
    assert np.array_equal(history.reservation_labels, labels) and history.motion_reserved == set()
    outputs = json.loads((motion.root/'m12_identity_reservation/run/run.json').read_text())['outputs']
    assert 'pre_reservation_labels' in outputs, 'checkpoints read this role back'


def test_skipped_lineage_also_skips_accepted_history_and_final_corrections(motion):
    lineage = stage_raising('m19_layered_lineage', AssertionError(
        'layered-lineage output duplicated a soma identity')); lineage.root = motion.root

    def history_stage(cfg, stem, run_name, sources, m20_labels, history_inputs=None, m20_result_future=None):
        if history_inputs.physical_somas is None:
            raise ValueError('accepted history needs physical_somas')
    history = SimpleNamespace(physical_somas=np.ones(3))
    labels = movie()
    oscillatory = labels.copy(); oscillatory[oscillatory == 7] = 9
    future = Future(); future.set_result((oscillatory, np.zeros(labels.shape, bool), pd.DataFrame()))
    motion.cfg.source_frame_offset = 1
    with install(motion, stage_m19_accepted=lineage, stage_m22_accepted_history=history_stage):
        motion.pipeline.stage_m19_accepted(motion.cfg, STEM, 'run', motion.sources, labels,
                                           history_inputs=history)
        assert history.physical_somas is None
        final, unclaimed, inferred, tracks = motion.pipeline.stage_m22_accepted_history(
            motion.cfg, STEM, 'run', motion.sources, labels, history_inputs=history,
            m20_result_future=future)
    assert np.array_equal(final, oscillatory[1:]), 'starts from the oscillatory labels, trimmed'
    assert not unclaimed.any() and sorted(tracks.identity) == [4, 9]

    def never(): raise AssertionError('final corrections must not run without accepted history')
    output = fallbacks.final_corrections(never, motion.root, STEM, 'run', motion.cfg)
    record = json.loads((output/'run.json').read_text())
    assert record['degraded'] and Path(record['carried_forward_from']).parent.name == 'm22_accepted_history'
    assert np.array_equal(tifffile.imread(output/'out'/f'{STEM}.tif'), oscillatory[1:])


def finished_stage(motion, folder, labels, role='labels'):
    run = motion.common.Run(folder, 'run', {})
    path = run.dir/'out'/f'{STEM}.tif'
    motion.common.save_stack(path, labels, 30.0)
    run.record(role, path); run.finish({})


def test_final_corrections_publish_accepted_history_when_they_fail(motion):
    labels = movie()
    finished_stage(motion, 'm22_accepted_history', labels)
    def failing(): raise RuntimeError('Incomplete accepted correction sequence; no final labels published')
    output = fallbacks.final_corrections(failing, motion.root, STEM, 'run', motion.cfg, resume=True)
    assert np.array_equal(tifffile.imread(output/'out'/f'{STEM}.tif'), labels)
    assert json.loads((output/'run.json').read_text())['status'] == 'complete'
    assert (output/'completion_files.json').is_file(), 'a resumed run can reuse the published result'
    assert fallbacks.final_corrections(lambda: 'done', motion.root, STEM, 'other', motion.cfg) == 'done'
    def out_of_memory(): raise MemoryError()
    with pytest.raises(MemoryError):
        fallbacks.final_corrections(out_of_memory, motion.root, STEM, 'run', motion.cfg)


def test_last_resort_trims_untrimmed_stage_labels_to_the_analysed_frames(motion):
    labels = movie()
    finished_stage(motion, 'm9_alias', labels)
    finished_stage(motion, 'm4_events', labels[::-1].copy(), role='event_identities')
    output = fallbacks.publish_without_corrections(motion.root, STEM, 'run', 'test',
                                                   source_frame_offset=1)
    assert np.array_equal(tifffile.imread(output/'out'/f'{STEM}.tif'), labels[1:])
    assert json.loads((output/'run.json').read_text())['carried_forward_from'].endswith('run')
    with pytest.raises(RuntimeError, match='No tracking stage finished'):
        fallbacks.publish_without_corrections(motion.root, STEM, 'missing', 'test')


def test_run_retries_memory_stops_in_the_resumable_route(tmp_path, monkeypatch):
    calls = []
    def run_frozen(root, stem, run_name, config, log, **options):
        calls.append(options)
        if len(calls) == 1:
            raise MemoryError()
    monkeypatch.setattr(engine, '_run_frozen', run_frozen)
    monkeypatch.setattr(engine, 'RETRY_WAITS', (0, 0))
    engine._run_with_retries(tmp_path, STEM, 'run', tmp_path/'c.json', {}, tmp_path/'m.log',
                             bounded_memory=False, working_array_min_bytes=32*1024**2, resume=False)
    assert calls == [{}, {'bounded_memory': True, 'resume': True}]
    assert 'RETRY 1: MemoryError' in (tmp_path/'m.log').read_text()


def test_run_publishes_finished_labels_when_every_attempt_stops(motion, monkeypatch):
    finished_stage(motion, 'm20_oscillatory_lineage', movie())
    def run_frozen(*args, **options): raise MemoryError()
    monkeypatch.setattr(engine, '_run_frozen', run_frozen)
    monkeypatch.setattr(engine, 'RETRY_WAITS', (0,))
    engine._run_with_retries(motion.root, STEM, 'run', motion.root/'c.json', {}, motion.root/'m.log',
                             bounded_memory=True, working_array_min_bytes=32*1024**2, resume=False)
    record = json.loads((motion.root/fallbacks.FINAL/'run/run.json').read_text())
    assert record['degraded'] and 'MemoryError' in record['reason']


def test_input_errors_before_any_stage_are_reported_as_themselves(tmp_path, monkeypatch):
    def run_frozen(*args, **options): raise ValueError('pinned input changed')
    monkeypatch.setattr(engine, '_run_frozen', run_frozen)
    with pytest.raises(ValueError, match='pinned input changed'):
        engine._run_with_retries(tmp_path, STEM, 'run', tmp_path/'c.json', {}, tmp_path/'m.log',
                                 bounded_memory=False, working_array_min_bytes=1, resume=False)


def test_half_unpacked_engine_is_unpacked_again(tmp_path):
    (tmp_path/'engine/code').mkdir(parents=True)
    root = engine._engine(tmp_path)
    assert (root/'code/pipeline.py').is_file() and (root/'archive.sha256').is_file()
    assert any(p.name.startswith('engine_incomplete_') for p in tmp_path.iterdir())


def test_one_failed_recording_does_not_stop_the_others(tmp_path, monkeypatch):
    from pymicroglia.pipelines import motion_handoff
    inputs = tmp_path/'motion_inputs.json'
    inputs.write_text(json.dumps(dict(still_missing=[], stems=['a', 'b'])))
    accepted = tmp_path/'b/out'; accepted.mkdir(parents=True)
    (tmp_path/'b/run.json').write_text(json.dumps(dict(degraded=True, reason='accepted history was skipped')))
    def run(path, folder, tracker_options):
        if tracker_options['stem'] == 'a':
            raise RuntimeError('Motion failed for a before any identity stage finished')
        return SimpleNamespace(labels=accepted/'b.tif', as_dict=lambda: {'labels': 'b.tif'})
    monkeypatch.setattr(motion_handoff._tracking, 'run', run)
    entry = {}
    outputs = motion_handoff.track(dict(outputs=dict(motion_inputs=str(inputs))), tmp_path, entry)
    assert list(outputs['tracking']) == ['b'] and entry['status'] == 'partial'
    assert 'a' in entry['failed_recordings'] and entry['degraded_recordings'] == {'b': 'accepted history was skipped'}
