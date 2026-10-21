"""Accepted native engine packaging and actual embedded execution boundary."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
import tifffile
from pymicroglia.tracking import engine

ARCHIVE = Path(engine.__file__).parent / 'data/motion_engine.zip'

def test_all_packaged_sources_match_manifest():
    with zipfile.ZipFile(ARCHIVE) as archive:
        manifest = json.loads(archive.read('source_hashes.json'))
        assert manifest['accepted_engine'] == engine.NATIVE_METHOD_VERSION
        for name, digest in manifest['files_sha256'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name


@pytest.mark.parametrize('bounded', [False, True])
def test_both_runner_modes_install_accepted_execution_contexts(tmp_path, monkeypatch, bounded):
    root = engine._engine(tmp_path)
    config = tmp_path / 'recording.json'
    with zipfile.ZipFile(ARCHIVE) as archive:
        values = json.loads(archive.read('config.json'))
    values.update(frame_interval_min=7.5, source_frame_offset=3)
    config.write_text(json.dumps(values))
    previous_argv = sys.argv
    observed = []
    def execute(pipeline, code, passed_root, run_name, stem, passed_config, bounded_memory, minimum_bytes):
        common = sys.modules['common']
        assert common.Config.load(config).source_frame_offset == 3
        assert common.Config.load(config).values['frame_interval_min'] == 7.5
        assert common.Run.finish.__module__ == 'optimized_pipeline'
        assert common.Config.load.__module__ == 'optimized_pipeline'
        assert pipeline.run_accepted_history is sys.modules['accepted_history'].run_accepted_history
        assert bounded_memory is bounded
        assert passed_root == root and passed_config == config
        observed.append(True)
    monkeypatch.setattr(engine, '_execute_pipeline', execute)
    monkeypatch.setattr(engine, '_execute_corrections', lambda *args: None)
    engine._run_frozen(root, 'fixture', 'test', config, tmp_path/'run.log', bounded_memory=bounded)
    assert observed == [True]
    assert sys.argv is previous_argv
    assert not any(engine._inside(getattr(module, '__file__', None), root) for module in sys.modules.values())


@pytest.mark.parametrize('offset', [0, 2, 3])
def test_result_alignment_follows_recording_setting(tmp_path, monkeypatch, offset):
    raw = tmp_path/'raw.tif'
    tifffile.imwrite(raw, np.zeros((7,8,8),np.uint16), photometric='minisblack')
    config = tmp_path/'config.json'
    config.write_text(json.dumps(dict(registered_input_dir='.',motion_input_dir='.',source_frame_offset=offset)))
    handoff = tmp_path/'inputs.json'
    handoff.write_text(json.dumps(dict(stems=['fixture'],prepared=dict(fixture=dict(config=str(config),pins=dict(
        registered_raw=dict(relative_to_registered_input_dir='raw.tif'),
        motion_composite=dict(relative_to_motion_input_dir='evidence.tif')))))))
    root = tmp_path/'engine'
    monkeypatch.setattr(engine, '_engine', lambda folder: root)
    def run(*args, **kwargs):
        out = root/'m23_completed_feedback/pymicroglia_fixture/out'
        out.mkdir(parents=True)
        tifffile.imwrite(out/'fixture.tif',np.zeros((7-offset,8,8),np.uint16),photometric='minisblack')
    monkeypatch.setattr(engine, '_run_frozen', run)
    monkeypatch.setattr(engine.TrackingResult, 'from_folder', lambda *args, **kwargs: SimpleNamespace(**kwargs))
    assert engine.run(handoff,tmp_path).source_frame_offset == offset
