"""The engines ship as plain folders, byte for byte, and old snapshot zips still read the same."""
import hashlib
import importlib.util
import json
import shutil
import zipfile
from pathlib import Path

import pytest

from pymicroglia.tracking import shipped_engines

TOOLS = Path(__file__).resolve().parents[1] / 'tools'


def small_engine(root):
    files = {'code/pipeline.py': b'print(1)\r\n', 'config.json': b'{}\n'}
    manifest = dict(files_sha256={n: hashlib.sha256(d).hexdigest() for n, d in files.items()})
    files['source_hashes.json'] = json.dumps(manifest).encode()
    return shipped_engines.write(root, files), files


@pytest.mark.parametrize('name', [shipped_engines.MOTION, shipped_engines.COMPLETED_FEEDBACK])
def test_packaged_engines_match_their_manifests(name):
    shipped_engines.verify(shipped_engines.folder(name))


def test_copy_refuses_a_changed_file_and_keeps_windows_line_endings(tmp_path):
    root, _ = small_engine(tmp_path / 'engine')
    shipped_engines.copy(root, tmp_path / 'run')
    assert (tmp_path / 'run/code/pipeline.py').read_bytes() == b'print(1)\r\n'
    (root / 'code/pipeline.py').write_bytes(b'print(1)\n')
    with pytest.raises(ValueError, match='code/pipeline.py'):
        shipped_engines.copy(root, tmp_path / 'other')


def test_fingerprint_ignores_bytecode_and_follows_content(tmp_path):
    root, _ = small_engine(tmp_path / 'engine')
    before = shipped_engines.fingerprint(root)
    (root / 'code/__pycache__').mkdir()
    (root / 'code/__pycache__/pipeline.cpython-312.pyc').write_bytes(b'x')
    assert shipped_engines.fingerprint(root) == before
    (root / 'config.json').write_bytes(b'{"a": 1}\n')
    assert shipped_engines.fingerprint(root) != before


def test_snapshot_zips_from_before_release_1b_read_like_folders(tmp_path):
    root, files = small_engine(tmp_path / 'engine')
    archive = tmp_path / 'old/engine.zip'
    archive.parent.mkdir()
    with zipfile.ZipFile(archive, 'w') as z:
        for name, data in files.items():
            z.writestr(name, data)
    old = tmp_path / 'old/engine'  # the folder path; only its zip exists
    assert shipped_engines.files(old) == shipped_engines.files(root) == files
    assert shipped_engines.read(old, 'config.json') == b'{}\n'
    assert shipped_engines.digest(old) == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert shipped_engines.digest(root) == shipped_engines.fingerprint(root)
    for source, target in ((old, tmp_path / 'from_zip'), (root, tmp_path / 'from_folder')):
        assert shipped_engines.unpack(source, target, 'code/') == ['code/pipeline.py']
        assert (target / 'code/pipeline.py').read_bytes() == b'print(1)\r\n'
        assert not (target / 'config.json').exists()


def test_replay_runner_fingerprint_matches_the_package(tmp_path):
    spec = importlib.util.spec_from_file_location('replay_bundle_under_test', TOOLS / 'full_replay_bundle.py')
    bundle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bundle)
    root, _ = small_engine(tmp_path / 'engine')
    assert bundle._fingerprint(root) == shipped_engines.fingerprint(root)
    snapshot = tmp_path / 'snapshot'
    shutil.copytree(root, snapshot / 'pymicroglia/tracking/data/motion_engine')
    a1 = tmp_path / 'a1'
    (a1 / 'm0_inputs').mkdir(parents=True)
    (a1 / 'archive.sha256').write_text('older engine\n')
    target = bundle.snapshot_engine(a1, snapshot, tmp_path / 'outputs', 'x')
    assert (target / 'archive.sha256').read_text().strip() == shipped_engines.fingerprint(root)
    assert (target / 'code/pipeline.py').read_bytes() == b'print(1)\r\n' and (target / 'm0_inputs').is_dir()
