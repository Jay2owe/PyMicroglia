"""Compare owner_pixel_boxes with the original distance and centre functions, exactly.

transiently_misowned_body_recovery.recover and recording_boundary_owner_cycle.recover
run on their accepted A1 stage inputs (plain and memory-pool label movies). During
the original run every _distance_to_owner and _center call is also answered by the
adapter (the distance one with its own copy of the discover cache), and the answers
must have the same type and float bits; then recover runs with the adapter installed
and every returned array and table must equal the original's. Synthetic calls add
zero, negative, absent and above-maximum owners, signed and float frames, empty and
one-pixel masks, edges and corners, and non-Boolean masks.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'


def numpy_crash(stack, owner):
    """numpy 2.1 crashes comparing a strided integer array with an out-of-range int."""
    array = np.asarray(stack)
    if array.dtype.kind not in 'iu' or array.flags.c_contiguous or array.flags.f_contiguous:
        return False
    try:
        value = int(owner)
    except (TypeError, ValueError, OverflowError):
        return False
    limits = np.iinfo(array.dtype)
    return not limits.min <= value <= limits.max


def outcome(function, *args):
    try:
        return ('ok', function(*args))
    except Exception as error:
        return ('error', (type(error).__name__, str(error)))


def same(a, b):
    if type(a) is not type(b):
        return f'type {type(a).__name__} != {type(b).__name__}'
    if isinstance(a, np.ndarray):
        if a.dtype != b.dtype or a.shape != b.shape:
            return 'array dtype or shape'
        return None if np.array_equal(np.asarray(a), np.asarray(b), equal_nan=a.dtype.kind in 'fc') else 'array values'
    if isinstance(a, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(a, b, check_exact=True)
        except AssertionError as error:
            return 'frame ' + str(error)[:200]
        return None
    if isinstance(a, dict):
        if list(a) != list(b):
            return 'dict keys'
        for key in a:
            problem = same(a[key], b[key])
            if problem:
                return f'{key}: {problem}'
        return None
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return 'length'
        for index, (x, y) in enumerate(zip(a, b)):
            problem = same(x, y)
            if problem:
                return f'[{index}] {problem}'
        return None
    if isinstance(a, float):
        return None if np.float64(a).tobytes() == np.float64(b).tobytes() else f'float {a!r} != {b!r}'
    return None if a == b else f'{a!r} != {b!r}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--part', choices=('all', 'stages', 'functions'), default='all',
                        help='run the stage comparisons and the single-function cases in separate processes')
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import tifffile
    import array_storage
    import transiently_misowned_body_recovery as misowned
    import recording_boundary_owner_cycle as boundary
    import owner_pixel_boxes as adapter
    config = json.loads((ENGINE / 'config.json').read_text(encoding='utf-8'))
    accepted = config['accepted_postprocessing']
    a.scratch.mkdir(parents=True, exist_ok=True)
    pool = array_storage.Pool(a.scratch / 'pool')
    serial = [0]
    kept = []

    def pooled(stack):
        path = a.scratch / f'stack_{serial[0]}.dat'
        serial[0] += 1
        mapped = np.memmap(path, dtype=stack.dtype, mode='w+', shape=stack.shape)
        mapped[:] = stack
        mapped.flush()
        working = pool.adopt(mapped, path)
        # Held to the end: an adopted file is closed as soon as its array goes.
        kept.append(working)
        return working

    raw = tifffile.imread(next((HISTORY / '07_trimmed_evidence/out').glob('*_registered_raw.tif')))
    points = pd.read_csv(HISTORY / '26_latent_body_tracks/out/latent_track_points.csv')
    thresholds = pd.read_csv(HISTORY / '25_raw_physical_hypotheses/out/frame_evidence_thresholds.csv')

    def parameters(key):
        params = copy.deepcopy(accepted.get(key, {}).get('parameters', {}))
        params['targeting_mode'] = 'field_wide_discovery'
        return params
    misowned_params = parameters('field_wide_transient_misownership_recovery')
    boundary_params = parameters('field_wide_recording_boundary_owner_cycle')
    report = dict(failures=[], stages={}, calls={}, synthetic=0)

    def fail(**item):
        if len(report['failures']) < 30:
            report['failures'].append(item)

    stages = (
        ('transiently_misowned_body_recovery', misowned, '_distance_to_owner', '33_issue063_post_split_excursion_recovery',
         lambda labels, unclaimed: misowned.recover(labels, unclaimed, raw, points.copy(), thresholds.copy(), misowned_params)),
        ('recording_boundary_owner_cycle', boundary, '_center', '41_issue089_cross_reference_owner_relay',
         lambda labels, unclaimed: boundary.recover(labels, unclaimed, raw, points.copy(), boundary_params)),
    )
    if a.part == 'functions':
        stages = ()
    for name, module, function, stem, call in stages:
        labels = tifffile.imread(HISTORY / (stem + '.tif'))
        unclaimed = tifffile.imread(HISTORY / (stem + '_unclaimed.tif'))
        for form in ('plain', 'pool'):
            def inputs():
                if form == 'plain':
                    return labels.copy(), unclaimed.copy()
                return pooled(labels), pooled(unclaimed)
            original = getattr(module, function)
            with adapter.install(misowned, boundary) as stats:
                assert name in stats['supported'], stats
                replacement = getattr(module, function)
                setattr(module, function, original)
                counted = [0]
                shadow = {}

                def recorded(*args):
                    if function == '_distance_to_owner':
                        frame, owner, x, y, cache, index = args
                        theirs = shadow.setdefault(id(cache), (cache, {}))[1]
                        expected = outcome(original, *args)
                        actual = outcome(replacement, frame, owner, x, y, theirs, index)
                    else:
                        expected = outcome(original, *args)
                        actual = outcome(replacement, *args)
                    counted[0] += 1
                    problem = same(expected, actual)
                    if problem:
                        fail(stage=name, form=form, call=counted[0], problem=problem)
                    if expected[0] == 'error':
                        raise RuntimeError(expected[1])
                    return expected[1]
                setattr(module, function, recorded)
                t0 = time.perf_counter()
                expected = outcome(call, *inputs())
                setattr(module, function, replacement)
                t1 = time.perf_counter()
                actual = outcome(call, *inputs())
                t2 = time.perf_counter()
                stage_stats = dict(stats)
            restored = getattr(module, function) is original
            t3 = time.perf_counter()
            baseline = outcome(call, *inputs())
            t4 = time.perf_counter()
            report['stages'][f'{name}:{form}'] = dict(status=expected[0], calls=counted[0], restored=restored,
                                                      original_seconds=round(t4 - t3, 2),
                                                      adapted_seconds=round(t2 - t1, 2), stats=stage_stats)
            for label, left in (('recorded', expected), ('baseline', baseline)):
                problem = same(left, actual)
                if problem:
                    fail(stage=name, form=form, case='recover ' + label, problem=problem)
        del labels, unclaimed
    if a.part == 'stages':
        report['status'] = 'pass' if not report['failures'] and len(report['stages']) == 4 else 'fail'
        report['part'] = a.part
        a.output.write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
        print(json.dumps(report, indent=1, default=str))
        return 0 if report['status'] == 'pass' else 1
    # Synthetic calls.
    rng = np.random.default_rng(91)
    frame = np.zeros((40, 50), np.uint16)
    frame[0:3, 0:2] = 4
    frame[37:40, 46:50] = 4
    frame[10:20, 10:15] = 7
    frame[39, 0] = 9
    frame[25, 49] = 2
    frames = {'shaped': frame, 'random': rng.integers(0, 6, (30, 31)).astype(np.uint16),
              'uint8': rng.integers(0, 4, (12, 9)).astype(np.uint8), 'int32': frame.astype(np.int32),
              'float': frame.astype(np.float32), 'big': (frame.astype(np.uint32) * 30000),
              'noncontiguous': np.asfortranarray(frame)[::1, ::-1]}
    owners = [0, -1, 1, 2, 3, 4, 7, 9, 10, 70000, np.int64(4), 4.0, '7']
    spots = [(0.0, 0.0), (49.4, 39.6), (12.5, 15.5), (-5.0, 100.0), (25.0, 25.0)]
    with adapter.install(misowned, boundary) as stats:
        for fname, frame in frames.items():
            for owner in owners:
                if numpy_crash(frame, owner):
                    report.setdefault('skipped_numpy_crash', []).append((fname, repr(owner)))
                    continue
                for x, y in spots:
                    for index in (0, 3):
                        report['synthetic'] += 1
                        problem = same(outcome(misowned._distance_to_owner.__wrapped__, frame, owner, x, y, {}, index),
                                       outcome(misowned._distance_to_owner, frame, owner, x, y, {}, index))
                        if problem:
                            fail(case='distance', frame=fname, owner=repr(owner), spot=(x, y), problem=problem)
        masks = {'empty': np.zeros((20, 20), bool), 'one': np.pad(np.ones((1, 1), bool), ((19, 0), (0, 19))),
                 'corners': frame > 0, 'random': rng.random((33, 47)) > 0.7, 'uint8': (frame > 0).astype(np.uint8),
                 'three_dimensional': rng.random((2, 5, 5)) > 0.5, 'strided': (rng.random((40, 40)) > 0.6)[::2, 1::3]}
        for mname, mask in masks.items():
            report['synthetic'] += 1
            problem = same(outcome(boundary._center.__wrapped__, mask), outcome(boundary._center, mask))
            if problem:
                fail(case='center', mask=mname, problem=problem)
        synthetic_stats = dict(stats)
    report['synthetic_stats'] = synthetic_stats
    report['part'] = a.part
    report['status'] = 'pass' if (not report['failures'] and all(s['restored'] for s in report['stages'].values())
                                  and all(s['calls'] for s in report['stages'].values())) else 'fail'
    a.output.write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
    print(json.dumps(report, indent=1, default=str))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
