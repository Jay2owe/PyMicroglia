"""Keep the accepted history's first two Issue 005 tracking passes for correction evidence.

Correction evidence starts from a gap reference: the first two complete tracking
passes of Issue 005 on the same inputs, of which only the label movies are used.
run_accepted_history already computes those passes, so while it runs Issue 005
each pass's labels are written beside 01_issue005.tif with a manifest. The
manifest holds one SHA-256 over every Issue 005 input (arrays with dtype and
shape, the observation table, anchor frame, reserved names, the accepted
post-processing settings), every engine source file, the NumPy version and the
calling thread's scalar-promotion state, plus each pass's own digest.

load() returns the passes only when the evidence inputs give the same input
digest and each loaded movie its recorded digest; otherwise the caller runs the
passes itself. Issue 005's result and every other output are unchanged.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

FOLDER = '01_issue005_tracking_passes'
VERSION = 1
_output_root = ContextVar('tracking_pass_output_root', default=None)
stats = dict(recorded=0, loaded=0, mismatched=0, absent=0)


def _array(h, value):
    a = np.ascontiguousarray(value)
    h.update(a.dtype.str.encode()); h.update(repr(a.shape).encode())
    flat = a.reshape(-1)
    step = max(1, (64 * 1024**2) // max(1, a.itemsize))
    for start in range(0, len(flat), step):
        h.update(memoryview(np.ascontiguousarray(flat[start:start + step])).cast('B'))


def array_digest(value):
    h = hashlib.sha256()
    _array(h, value)
    return h.hexdigest()


def _table(h, table):
    import pandas as pd
    h.update(json.dumps([str(c) for c in table.columns]).encode())
    h.update(json.dumps([str(d) for d in table.dtypes]).encode())
    h.update(json.dumps([str(table.index.dtype), len(table)]).encode())
    h.update(pd.util.hash_pandas_object(table, index=True).to_numpy().tobytes())
    for column in table.columns:
        if table[column].dtype == object:
            h.update(json.dumps(sorted({type(v).__name__ for v in table[column]})).encode())


def inputs_digest(history, config, physical, reservation, raw, lag, observations,
                  observation_table, anchor_t, motion_reserved):
    """One digest of everything the first two Issue 005 passes read."""
    h = hashlib.sha256()
    h.update(f'tracking-passes-v{VERSION}'.encode())
    for value in (physical, reservation, raw, lag, observations):
        _array(h, value)
    _table(h, observation_table)
    h.update(repr(int(anchor_t)).encode())
    h.update(json.dumps(sorted(int(v) for v in motion_reserved)).encode())
    h.update(json.dumps(config.values['accepted_postprocessing'], sort_keys=True, default=repr).encode())
    code = Path(history.__file__).resolve().parent
    for path in sorted(code.glob('*.py')):
        h.update(path.name.encode()); h.update(hashlib.sha256(path.read_bytes()).digest())
    h.update(np.__version__.encode())
    h.update(repr(getattr(np, '_get_promotion_state', lambda: None)()).encode())
    return h.hexdigest()


@contextmanager
def install(history, pipeline=None):
    """Record the passes whenever run_accepted_history computes Issue 005."""
    import inspect
    original_run = history.run_accepted_history
    original_issue = history._confidence_layered_retracking
    original_track = history.track_physical_somas
    signature = inspect.signature(original_run)

    def run_accepted_history(*args, **kwargs):
        output_root = signature.bind(*args, **kwargs).arguments['output_root']
        token = _output_root.set(Path(output_root))
        try:
            return original_run(*args, **kwargs)
        finally:
            _output_root.reset(token)
    run_accepted_history.__wrapped__ = original_run

    def _confidence_layered_retracking(config, physical, reservation, raw, lag, observations,
                  observation_table, anchor_t, motion_reserved):
        root = _output_root.get()
        # Only run_accepted_history's own call records; the evidence gap
        # reference replaces track_physical_somas with its own capture.
        if root is None or history.track_physical_somas is not original_track:
            return original_issue(config, physical, reservation, raw, lag, observations,
                                  observation_table, anchor_t, motion_reserved)
        digest = inputs_digest(history, config, physical, reservation, raw, lag, observations,
                               observation_table, anchor_t, motion_reserved)
        staging = root / (FOLDER + '.partial')
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        passes = []

        def track(*args, **kwargs):
            result = original_track(*args, **kwargs)
            if len(passes) < 2:
                labels = result[0]
                name = f'pass{len(passes) + 1}_labels.tif'
                _write(staging / name, labels)
                passes.append(dict(file=name, sha256=array_digest(labels),
                                   dtype=labels.dtype.str, shape=list(labels.shape)))
            return result
        history.track_physical_somas = track
        try:
            labels = original_issue(config, physical, reservation, raw, lag, observations,
                                    observation_table, anchor_t, motion_reserved)
        finally:
            history.track_physical_somas = original_track
        if len(passes) == 2:
            (staging / 'manifest.json').write_text(json.dumps(dict(
                version=VERSION, inputs_sha256=digest, passes=passes,
                note='First two Issue 005 tracking passes; used by correction evidence only on an exact input digest match.'),
                indent=1) + '\n')
            target = root / FOLDER
            shutil.rmtree(target, ignore_errors=True)
            from .checkpoints import retry
            try:
                retry(os.replace, staging, target)
                stats['recorded'] += 1
            except OSError:
                # The record only speeds up later evidence; without it that
                # evidence recomputes the same two passes.
                shutil.rmtree(staging, ignore_errors=True)
        else:
            shutil.rmtree(staging, ignore_errors=True)
        return labels

    aliases = []
    history.run_accepted_history = run_accepted_history
    history._confidence_layered_retracking = _confidence_layered_retracking
    for module in [pipeline] if pipeline is not None else []:
        if getattr(module, 'run_accepted_history', None) is original_run:
            aliases.append(module)
            module.run_accepted_history = run_accepted_history
    try:
        yield stats
    finally:
        for module in aliases:
            module.run_accepted_history = original_run
        history.run_accepted_history = original_run
        history._confidence_layered_retracking = original_issue


def _write(path, labels):
    import tifffile
    with tifffile.TiffWriter(path, bigtiff=True) as writer:
        for frame in labels:
            writer.write(np.asarray(frame), compression='zlib', compressionargs=dict(level=1),
                         contiguous=False, metadata=None)


def load(history_root, history, config, physical, reservation, raw, lag, observations,
         observation_table, anchor_t, motion_reserved):
    """The two recorded pass label movies when every input matches, else None."""
    folder = Path(history_root) / FOLDER
    manifest_path = folder / 'manifest.json'
    if not manifest_path.is_file():
        stats['absent'] += 1
        return None
    started = time.perf_counter()
    manifest = json.loads(manifest_path.read_text())
    digest = inputs_digest(history, config, physical, reservation, raw, lag, observations,
                           observation_table, anchor_t, motion_reserved)
    if manifest.get('version') != VERSION or manifest.get('inputs_sha256') != digest:
        stats['mismatched'] += 1
        return None
    import tifffile
    movies = []
    for entry in manifest['passes']:
        with tifffile.TiffFile(folder / entry['file']) as opened:
            pages = opened.pages
            # The tracker allocates its output the same way, so the active
            # working-array storage decides where the movie lives.
            movie = np.zeros(tuple(entry['shape']), np.dtype(entry['dtype']))
            if len(pages) != len(movie):
                stats['mismatched'] += 1
                return None
            for index, page in enumerate(pages):
                movie[index] = page.asarray()
        if array_digest(movie) != entry['sha256']:
            stats['mismatched'] += 1
            return None
        movies.append(movie)
    stats['loaded'] += 1
    seconds = time.perf_counter() - started
    return ([(movie, None) for movie in movies],
            [dict(operation='recorded_tracking_passes', reused=True, seconds=seconds)])
