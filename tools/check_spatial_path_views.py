"""Compare every frozen constrain_spatial_soma_paths version with and without plain views, exactly.

Each frozen revision's spatial_soma_paths.py is imported with its own folder
first on the path. Its constrain_spatial_soma_paths runs on real frames: the
accepted labels as proposals, those labels where an earlier history stage holds
foreground as the measured movie, the registered raw movie and tracker points,
with default flags and with the production flag sets each version accepts. The
inputs are given as plain arrays, as read-only disk-backed memmaps and as an
altered copy (one outline removed from every other frame, after the memo has
seen the unaltered frames) and over a raw movie brightened inside the outlines
of every third frame (stored raw tests must not be reused there), and under
the working-array pool with no RAM allowance (the function's own movie-sized
arrays are then disk-backed). One outline-part memo is shared by every adapted
call, so later calls reuse parts measured under other versions and flags.
Labels and audit tables (values, dtypes, order) must equal the original's.
"""
import argparse
import importlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
import time

from check_reference_area_boxes import same

FLAGS = (
    dict(),
    dict(resolve_thin_enclosure=True),
    dict(resolve_thin_enclosure=True, coherent_measured_core=True, enclose_measured_core=True,
         require_tracker_peak=True, tracker_peak_measured_only=True, tracker_peak_quantisation=True),
    dict(resolve_thin_enclosure=True, coherent_measured_core=True, enclose_measured_core=True,
         require_tracker_peak=True, tracker_peak_measured_only=False, tracker_peak_quantisation=True,
         reciprocal_overlap=True, exclusive_tracker_core=True, qualify_inferred_paths=True,
         tracker_foreground_majority=True, retain_named_centre=True, inferred_only=True),
    dict(resolve_thin_enclosure=True, coherent_measured_core=True, enclose_measured_core=True,
         require_tracker_peak=True, tracker_peak_quantisation=True, reciprocal_overlap=True,
         require_shared_tracker_core=True, exclusive_tracker_core=True, require_soma_profile=True),
    dict(resolve_thin_enclosure=True, coherent_measured_core=True, enclose_measured_core=True,
         require_tracker_peak=True, tracker_peak_quantisation=True, reciprocal_overlap=True,
         exclusive_tracker_core=True, qualify_inferred_paths=True, tracker_foreground_majority=True,
         retain_named_centre=True, repeated_thin_measured_only=True, repeated_template_gap_copies=True,
         prefer_centred_owner=True),
)


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'code', 'frozen', 'config', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--stem', default='95_full_A1')
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--start', type=int, default=200)
    p.add_argument('--frames', type=int, default=40)
    # 0 keeps no frame copies, so every stored part is found by its full digest.
    p.add_argument('--copy-budget', type=int, default=None)
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    import tifffile
    sys.path[:0] = [str((a.engine / 'code').resolve())]
    import common
    import pipeline
    cfg = common.Config.load(a.config)
    window = slice(a.start, a.start + a.frames)
    raw = tifffile.imread(pipeline._pinned_paths(cfg, a.stem)['registered_raw'])[window]
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    labels = tifffile.imread(folder / '99_well_b3_issue013_anchored_projection_owner_relay.tif')[window]
    earlier = tifffile.imread(folder / '08_issue012.tif')[window]
    measured = np.where(earlier > 0, labels, 0).astype(labels.dtype)
    points = pd.read_csv(folder / '26_latent_body_tracks/out/latent_track_points.csv', low_memory=False)
    points = points[(points.frame >= a.start) & (points.frame < a.start + a.frames)].copy()
    points['frame'] -= a.start
    points = points.reset_index(drop=True)
    scratch = Path(tempfile.mkdtemp(prefix='spatial-views-'))

    def mapped(name, array):
        path = scratch / (name + '.dat')
        array.tofile(path)
        return np.memmap(path, dtype=array.dtype, mode='r', shape=array.shape)
    altered = labels.copy()
    for t in range(0, len(altered), 2):
        present = np.unique(altered[t])
        if len(present) > 1:
            altered[t][altered[t] == present[len(present) // 2]] = 0
    brightened = raw.copy()
    rng = np.random.default_rng(11)
    for t in range(1, len(brightened), 3):
        mask = labels[t] > 0
        values = brightened[t][mask].astype(np.float64) * (1.2 + rng.random(int(mask.sum()))) + 1
        if brightened.dtype.kind in 'iu':
            values = np.minimum(values, np.iinfo(brightened.dtype).max)
        brightened[t][mask] = values.astype(brightened.dtype)
    assert all(not np.array_equal(brightened[t], raw[t]) for t in range(1, len(raw), 3) if np.any(labels[t] > 0))
    # One measured pixel outside every proposal, in the last frame only: the
    # fixed-proposal test must raise the same error after reading every frame.
    unfixed = measured.copy()
    outside = np.argwhere(labels[-1] == 0)[0]
    unfixed[-1][tuple(outside)] = max(1, int(labels.max()))
    inputs = dict(plain=(measured, labels, raw),
                  unfixed=(unfixed, labels, raw),
                  memmap=(mapped('measured', measured), mapped('labels', labels), mapped('raw', raw)),
                  altered=(np.where(altered > 0, measured, 0).astype(measured.dtype), altered, raw),
                  # Same outlines over a changed raw movie: stored raw tests must not be reused.
                  raw_changed=(measured, labels, brightened),
                  # Under the working-array pool with no RAM allowance, so the
                  # function's own movie-sized arrays (nodes) are disk-backed.
                  pooled=(measured, labels, raw))
    sys.path[:0] = [str(a.code.resolve()), str(a.adapter.resolve())]
    import spatial_path_views as adapter
    import array_storage
    from contextlib import ExitStack

    def storage(kind):
        stack = ExitStack()
        if kind == 'pooled':
            folder = stack.enter_context(tempfile.TemporaryDirectory(ignore_cleanup_errors=True))
            pool = array_storage.Pool(folder, minimum_bytes=1024**2, temporary_ram_bytes=0)
            stack.callback(pool.close)
            stack.enter_context(array_storage.install(pool))
        return stack
    versions = sorted(a.frozen.glob('*/*/spatial_soma_paths.py'))
    report = dict(status='pass', frames=a.frames, start=a.start, versions={}, calls=0)
    memo = adapter.PartsMemo() if a.copy_budget is None else adapter.PartsMemo(copy_budget=a.copy_budget)
    totals = {}
    timings = dict(original=0.0, adapted=0.0)
    for path in versions:
        local = {p.stem for p in path.parent.glob('*.py')}
        for name in local:
            sys.modules.pop(name, None)
        sys.path.insert(0, str(path.parent))
        try:
            module = importlib.import_module('spatial_soma_paths')
        finally:
            sys.path.remove(str(path.parent))
        unique = 'spatial_soma_paths__' + path.parent.name
        sys.modules[unique] = sys.modules.pop('spatial_soma_paths')
        for name in local - {'spatial_soma_paths'}:
            if name in sys.modules:
                sys.modules[name + '__' + path.parent.name] = sys.modules.pop(name)
        accepted = set(inspect.signature(module.constrain_spatial_soma_paths).parameters)
        original = module.constrain_spatial_soma_paths
        results = []
        for flags in FLAGS:
            if set(flags) - accepted:
                continue
            for kind, (observed, proposals, movie) in inputs.items():
                tick = time.perf_counter()
                with storage(kind):
                    try:
                        expected = original(observed, proposals, movie, points, **flags)
                    except Exception as error:  # the same failure must occur adapted
                        expected = ('raised', type(error).__name__, str(error))
                    expected = tuple(v.view(np.ndarray).copy() if isinstance(v, np.ndarray) else v for v in expected)
                middle = time.perf_counter()
                stats = {}
                with storage(kind), adapter.install(stats, memo):
                    assert module.constrain_spatial_soma_paths.__wrapped__ is original
                    try:
                        actual = module.constrain_spatial_soma_paths(observed, proposals, movie, points, **flags)
                    except Exception as error:
                        actual = ('raised', type(error).__name__, str(error))
                    actual = tuple(v.view(np.ndarray).copy() if isinstance(v, np.ndarray) else v for v in actual)
                timings['original'] += middle - tick
                timings['adapted'] += time.perf_counter() - middle
                assert module.constrain_spatial_soma_paths is original
                assert same(list(expected), list(actual)), (path, flags, kind)
                raised = isinstance(expected[0], str)
                rows = 0 if raised else len(expected[1])
                results.append(dict(flags=sorted(flags), input=kind, audit_rows=rows,
                                    raised=raised, stats=stats))
                for key, value in stats.items():
                    totals[key] = totals.get(key, 0) + value
                report['calls'] += 1
        report['versions'][str(path.relative_to(a.frozen))] = results
        print(path.parent.name, len(results), flush=True)
    assert totals.get('values_reused', 0) > 0 and totals.get('pairs_keyed', 0) > 0
    assert totals.get('fixed_frames', 0) > 0 and totals.get('clip_bounds', 0) > 0, totals
    report['seconds'] = timings
    report['stats'] = totals
    report['memo_bytes'] = memo.bytes
    for array in inputs['memmap']:
        array._mmap.close()
    del inputs
    import shutil
    shutil.rmtree(scratch, ignore_errors=True)
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], calls=report['calls'], seconds=timings, stats=totals)))


if __name__ == '__main__':
    main()
