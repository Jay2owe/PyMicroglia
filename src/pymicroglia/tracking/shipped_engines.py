"""The two frozen engines PyMicroglia ships as plain folders under tracking/data.

Each folder holds an engine's code, its settings and ``source_hashes.json``, the
SHA-256 of every other file in it. A run copies the folder into its own run
folder and records the folder's fingerprint, so a changed engine refuses to
resume an earlier run. The folders are stored byte for byte (``-text`` in
.gitattributes): some files keep Windows line endings, and the speed patches
check exact file digests.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from importlib import resources
from pathlib import Path

MOTION = 'motion_engine'
COMPLETED_FEEDBACK = 'completed_feedback_engine'
MANIFEST = 'source_hashes.json'


def folder(name: str) -> Path:
    """The packaged folder of one engine."""
    return Path(str(resources.files('pymicroglia.tracking').joinpath('data', name)))


def _kept(path: Path, root: Path) -> bool:
    return '__pycache__' not in path.relative_to(root).parts and path.suffix != '.pyc'


def names(root: Path) -> list[str]:
    """Every engine file, as a sorted relative path; Python's bytecode caches are ignored."""
    root = Path(root)
    return sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and _kept(p, root))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fingerprint(root: Path) -> str:
    """SHA-256 over every file's relative path and content."""
    root = Path(root)
    digest = hashlib.sha256()
    for name in names(root):
        digest.update(f'{name}\0{_sha((root / name).read_bytes())}\n'.encode())
    return digest.hexdigest()


def verify(root: Path) -> None:
    """Refuse an engine folder whose files differ from its own source_hashes.json."""
    root = Path(root)
    expected = json.loads((root / MANIFEST).read_text(encoding='utf-8'))['files_sha256']
    actual = {name: _sha((root / name).read_bytes()) for name in names(root) if name != MANIFEST}
    if actual != expected:
        changed = sorted(n for n in set(actual) | set(expected) if actual.get(n) != expected.get(n))
        raise ValueError(f'{root.name} differs from its {MANIFEST}: ' + ', '.join(changed[:5])
                         + (f' and {len(changed) - 5} more' if len(changed) > 5 else ''))


def copy(root: Path, target: Path) -> None:
    """Copy a verified engine folder into ``target`` (created if needed)."""
    verify(root)
    shutil.copytree(root, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))


# Replay snapshots made before the engines shipped as folders hold them as zips;
# the two readers below accept either, given the folder path or the zip path.
def _zip(source: Path) -> Path | None:
    if source.suffix == '.zip':
        return source
    return source.with_suffix('.zip') if not source.exists() and source.with_suffix('.zip').is_file() else None


def digest(source: Path) -> str:
    """What a run's archive.sha256 marker holds: the folder's fingerprint, or a snapshot zip's SHA-256."""
    source = Path(source)
    if archive := _zip(source):
        return _sha(archive.read_bytes())
    return fingerprint(source)


def read(source: Path, name: str) -> bytes:
    """One engine file from an engine folder (or a snapshot's zip)."""
    source = Path(source)
    if archive := _zip(source):
        with zipfile.ZipFile(archive) as z:
            return z.read(name)
    return (source / name).read_bytes()


def files(source: Path) -> dict[str, bytes]:
    """Every engine file by relative path, from an engine folder (or a snapshot's zip)."""
    source = Path(source)
    if archive := _zip(source):
        with zipfile.ZipFile(archive) as z:
            return {n: z.read(n) for n in z.namelist() if not n.endswith('/')}
    return {n: (source / n).read_bytes() for n in names(source)}


def write(target: Path, contents: dict[str, bytes]) -> Path:
    """Build step: replace the engine folder ``target`` with exactly ``contents``, then verify it."""
    target = Path(target)
    if target.exists():
        shutil.rmtree(target)
    for name, data in sorted(contents.items()):
        (target / name).parent.mkdir(parents=True, exist_ok=True)
        (target / name).write_bytes(data)
    verify(target)
    return target


def unpack(source: Path, target: Path, prefix: str = '') -> list[str]:
    """Copy the engine files under ``prefix`` into ``target``, keeping their relative paths."""
    source, target = Path(source), Path(target)
    if archive := _zip(source):
        with zipfile.ZipFile(archive) as z:
            chosen = [n for n in z.namelist() if n.startswith(prefix) and not n.endswith('/')]
            if any(n.startswith('/') or '..' in Path(n).parts for n in chosen):
                raise ValueError(f'{archive.name} has unsafe paths')
            z.extractall(target, chosen)
        return chosen
    chosen = [n for n in names(source) if n.startswith(prefix)]
    for name in chosen:
        (target / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target / name)
    return chosen
