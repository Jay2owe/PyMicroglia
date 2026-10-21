"""Versioned frame-coordinate repair for already-frozen native archives.

Only the general residency event adapter changes. Earlier stage checkpoints
retain their original signatures; stage 22 records this extra producer hash.
The canonical Motion source contains the same correction for future archives.
"""
from contextlib import contextmanager
import ast
import hashlib
import json
from pathlib import Path

VERSION = "2026-10-21-residency-source-frames"
FUNCTION = "_general_blob_residency"


def digest(value):
    return hashlib.sha256(value).hexdigest()


def repaired_function(source):
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == FUNCTION)
    original = ast.get_source_segment(source, node)
    if ('"source_frame": source_frame,' in original
            and '"residency_activation_source_frame": source_frame,' in original
            and 'source_frame = int(row.imagej_frame) + int(source_frame_offset)' in original):
        return original, original
    before = '"residency_activation_source_frame": int(row.imagej_frame) + 2,'
    if original.count(before) != 1:
        raise ValueError("Unrecognised residency implementation; refusing an implicit repair")
    patched = original.replace(before, '"residency_activation_source_frame": int(row.imagej_frame) + int(source_frame_offset),')
    patched = patched.replace('"source_frame": int(row.imagej_frame),',
                              '"source_frame": int(row.imagej_frame) + int(source_frame_offset),')
    patched = patched.replace('int(row.imagej_frame), source_frame_offset)',
                              'int(row.imagej_frame) + int(source_frame_offset), source_frame_offset)')
    patched = patched.replace('"source_imagej_frame": int(row.imagej_frame),',
                              '"source_imagej_frame": int(row.imagej_frame) + int(source_frame_offset),')
    return original, patched


@contextmanager
def install(history, checkpoints, root):
    """Install only while running this engine, including explicit provenance."""
    source = Path(history.__file__).read_text(encoding="utf8")
    original_source, patched = repaired_function(source)
    if original_source == patched:
        yield None  # A future archive already contains the canonical correction.
        return
    signature = dict(version=VERSION, repair_sha256=digest(Path(__file__).read_bytes()),
                     original_function_sha256=digest(original_source.encode()),
                     patched_function_sha256=digest(patched.encode()))
    root = Path(root)
    record = root / "residency_frame_repair.json"
    if not record.is_file() and any((root / 'm22_accepted_history').glob(
            '*/mid/accepted_history/11_issue017_rebase.tif')):
        raise ValueError("Unrepaired downstream history exists; retain it and resume from pre-residency inputs")
    if record.is_file() and json.loads(record.read_text()) != signature:
        raise ValueError("Residency repair changed; cannot reuse repaired history")
    record.write_text(json.dumps(signature, indent=2), encoding="utf8")
    namespace = {}
    exec(compile(patched, str(Path(__file__)), "exec"), history.__dict__, namespace)
    old_function = getattr(history, FUNCTION)
    old_call = checkpoints.Stages.call

    def call(self, method, *args, **kwargs):
        if method != "stage_m22_accepted_history":
            return old_call(self, method, *args, **kwargs)
        base = self.signature
        self.signature = {**base, "stage_repair": signature}
        try:
            return old_call(self, method, *args, **kwargs)
        finally:
            self.signature = base

    setattr(history, FUNCTION, namespace[FUNCTION])
    checkpoints.Stages.call = call
    try:
        yield signature
    finally:
        setattr(history, FUNCTION, old_function)
        checkpoints.Stages.call = old_call
