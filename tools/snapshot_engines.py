"""The engines inside a replay snapshot: folders now, zips in snapshots made before release 1b.

shipped_engines.py is loaded from this checkout by file path, so a caller can
still put the snapshot's own pymicroglia first on sys.path afterwards.
"""
import importlib.util
from pathlib import Path

_SOURCE = Path(__file__).resolve().parents[1] / 'src/pymicroglia/tracking/shipped_engines.py'
_spec = importlib.util.spec_from_file_location('_snapshot_shipped_engines', _SOURCE)
shipped_engines = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shipped_engines)

MOTION = shipped_engines.MOTION
COMPLETED_FEEDBACK = shipped_engines.COMPLETED_FEEDBACK


def engine(snapshot, name):
    """The engine's folder path in the snapshot; readers fall back to its zip in older snapshots."""
    return Path(snapshot) / 'pymicroglia/tracking/data' / name


def unpack(snapshot, name, target, prefix=''):
    """Copy the snapshot engine's files under ``prefix`` into ``target``."""
    return shipped_engines.unpack(engine(snapshot, name), target, prefix)


def read(snapshot, name, member):
    return shipped_engines.read(engine(snapshot, name), member)


def digest(snapshot, name):
    """What a run folder's archive.sha256 marker holds for this engine."""
    return shipped_engines.digest(engine(snapshot, name))
