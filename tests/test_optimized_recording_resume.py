"""Reject unsafe native-parent reuse without modifying its saved provenance."""
import importlib.util
import json
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "tools/resume_optimized_recording.py"
spec = importlib.util.spec_from_file_location("optimized_recording_resume", SOURCE)
resume = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resume)


@pytest.fixture
def parent(tmp_path):
    root = tmp_path / "engine"
    snapshot = tmp_path / "candidate"
    root.mkdir()
    archive = snapshot / "pymicroglia/tracking/data/motion_engine.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"frozen archive")
    (root / "archive.sha256").write_text(resume.sha(archive))
    config = tmp_path / "config.json"
    config.write_text('{"source_frame_offset": 0}')
    code = root / "code/pipeline.py"
    code.parent.mkdir()
    code.write_text("# frozen native implementation")
    signature = {"config_sha256": resume.sha(config),
                 "code": {"code/pipeline.py": resume.sha(code)}}
    (root / "tracking_resume.json").write_text(json.dumps({"signature": signature}))
    for stage in resume.STAGES:
        directory = root / stage / "fixture"
        directory.mkdir(parents=True)
        data = directory / "labels.bin"
        data.write_bytes(stage.encode())
        (directory / "checkpoint.json").write_text(json.dumps({
            "signature": signature, "files": {
                str(data.relative_to(root)): resume.sha(data)}}))
        (directory / "run.json").write_text(json.dumps({
            "status": "done", "outputs": {
                "labels": {"path": str(data), "sha256": resume.sha(data)}}}))
    return root, "fixture", config, snapshot


def test_verified_parent_is_read_only(parent):
    root, run, *_ = parent
    before = {str(p): resume.sha(p) for p in root.rglob("*") if p.is_file()}
    assert resume.completed_native(root, run)
    assert resume.verify_native(*parent)["native_stages"] == 13
    assert before == {str(p): resume.sha(p) for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("changed", ["code/pipeline.py", "m22_accepted_history/fixture/labels.bin"])
def test_changed_native_bytes_rejected(parent, changed):
    (parent[0] / changed).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        resume.verify_native(*parent)


def test_changed_config_rejected(parent):
    parent[2].write_text('{"source_frame_offset": 2}')
    with pytest.raises(ValueError, match="configuration changed"):
        resume.verify_native(*parent)


def test_changed_engine_archive_rejected(parent):
    (parent[3] / "pymicroglia/tracking/data/motion_engine.zip").write_bytes(b"new science")
    with pytest.raises(ValueError, match="native engine archive"):
        resume.verify_native(*parent)


def test_partial_native_is_not_complete(parent):
    root, run, *_ = parent
    (root / "m22_accepted_history" / run / "checkpoint.json").unlink()
    assert not resume.completed_native(root, run)


def test_repaired_history_parent_requires_matching_repair_record(parent):
    root, run, *_ = parent
    path = root/'m22_accepted_history'/run/'checkpoint.json'
    receipt = json.loads(path.read_text())
    repair = {'version': 'fixture', 'repair_sha256': 'verified separately against authority'}
    receipt['signature']['stage_repair'] = repair
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='different parent signature'):
        resume.verify_native(*parent)
    (root/'residency_frame_repair.json').write_text(json.dumps(repair))
    assert resume.verify_native(*parent)['native_stages'] == 13


def test_changed_stage_signature_rejected(parent):
    p = parent[0] / "m8_merge/fixture/checkpoint.json"
    d = json.loads(p.read_text())
    d["signature"]["extra"] = "different run"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="different parent signature"):
        resume.verify_native(*parent)


def test_escaped_record_rejected(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        resume.contained(tmp_path / "engine", "../outside")


def test_candidate_source_change_rejected(tmp_path):
    source = tmp_path / "worker.py"
    source.write_text("# original")
    (tmp_path / "composition.json").write_text(json.dumps({
        "sources": {"worker.py": resume.sha(source)}}))
    resume.verify_snapshot(tmp_path)
    source.write_text("# changed")
    with pytest.raises(ValueError, match="candidate changed"):
        resume.verify_snapshot(tmp_path)
