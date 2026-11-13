"""Compare the handoff-episode owner runs with and without the array version.

Both frozen handoff-episode modules are loaded from the packaged correction
engine. _runs is compared on every track's owned rows of the real track points
(as correct() selects them, and unsorted), on the frame/owner tables correct()
builds from sets (including empty ones), and on edge cases: duplicate frames,
float and boolean columns, NaN and infinite values (original path), uint64,
missing columns. Each module's whole correct() is then run on the cropped final
labels with and without the adapter; labels and every table must be identical.
"""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

import snapshot_engines

NAMES = ('complete_owner_handoff_episodes_r10_a002', 'complete_owner_handoff_episodes_a003')


def outcome(function, *args):
    try:
        return ('value', function(*args))
    except Exception as error:  # the same exception type and message must be raised
        return ('error', type(error).__name__, str(error))


def same(x, y):
    if type(x) is not type(y):
        return False
    if isinstance(x, tuple) or isinstance(x, list):
        return len(x) == len(y) and all(same(a, b) for a, b in zip(x, y))
    if isinstance(x, dict):
        return list(x) == list(y) and all(same(x[k], y[k]) and type(x[k]) is type(y[k]) for k in x)
    if isinstance(x, np.ndarray):
        return x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y)
    if isinstance(x, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(x, y, check_exact=True)
            return True
        except AssertionError:
            return False
    return x == y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True, help='folder of cropped final.tif, raw.tif, points.csv')
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    snapshot_engines.unpack(a.snapshot, snapshot_engines.COMPLETED_FEEDBACK, a.scratch / 'engine')
    sys.path[:0] = [str(a.scratch / 'engine/code'), str(a.adapters.resolve())]
    modules = [importlib.import_module(name) for name in NAMES]
    import handoff_runs
    labels = tifffile.imread(a.inputs / 'final.tif')
    raw = tifffile.imread(a.inputs / 'raw.tif')
    points = pd.read_csv(a.inputs / 'points.csv', low_memory=False)
    scored = modules[0].physical.attach_owners(points, labels)
    for column in ('strong', 'physically_visible', 'in_encounter'):
        scored[column] = scored[column].astype(bool)
    tables = []
    for _, group in scored[scored.physically_visible].groupby('track_id', sort=True):
        ordered = group.sort_values('frame')
        stable = ordered[(ordered.state.astype(str) == 'observed') & ordered.strong & ~ordered.in_encounter
                         & ordered.accepted_owner.astype(int).gt(0)]
        tables += [stable, group, group.iloc[::-1]]
    rng = np.random.default_rng(5)
    base = pd.DataFrame({'frame': [3, 1, 2, 2, 5, 6, 7, 7, 9], 'accepted_owner': [4, 4, 4, 5, 5, 5, 5, 6, 6]})
    edge = [base, base.iloc[0:0], pd.DataFrame(columns=['frame', 'accepted_owner']),
            pd.DataFrame([{'frame': t, 'accepted_owner': 3} for t in {4, 5, 6, 9}], columns=['frame', 'accepted_owner']),
            base.astype({'frame': float}), base.astype({'accepted_owner': float}),
            base.assign(frame=base.frame + 0.5), base.assign(accepted_owner=base.accepted_owner > 4),
            base.astype({'frame': np.uint64}), base.astype({'frame': np.int8}),
            base.assign(frame=[3, 1, 2, np.nan, 5, 6, 7, 7, 9]), base.assign(frame=[3, 1, 2, np.inf, 5, 6, 7, 7, 9]),
            base.astype({'frame': 'Int64'}), base.assign(frame=base.frame.astype(object)), base[['frame']],
            pd.DataFrame({'frame': rng.integers(0, 40, 300), 'accepted_owner': rng.integers(1, 4, 300)}),
            # Already in frame order (the sort is skipped), and a repeated frame column (it is not).
            base.sort_values('frame').drop_duplicates('frame'), base.astype({'frame': np.uint16}).iloc[[1, 2, 4, 5]],
            pd.concat([base, base[['frame']]], axis=1)]
    failures, cases = [], {}
    for module in modules:
        original = module._runs
        name = module.__name__
        expected = [outcome(original, t) for t in tables + edge]
        t0 = time.perf_counter()
        with handoff_runs.install() as stats:
            assert stats['supported'] == len(NAMES), stats
            t1 = time.perf_counter()
            actual = [outcome(module._runs, t) for t in tables + edge]
            t2 = time.perf_counter()
        t3 = time.perf_counter()
        for _ in range(1):
            [outcome(original, t) for t in tables]
        t4 = time.perf_counter()
        bad = [i for i, (x, y) in enumerate(zip(expected, actual)) if not same(x, y)]
        cases[name + ':runs'] = dict(tables=len(tables), edge=len(edge), arrays=stats['arrays'], original=stats['original'],
                                     in_order=stats['in_order'], runs=sum(len(x[1]) for x in expected if x[0] == 'value'),
                                     array_seconds=round(t2 - t1, 3), original_seconds=round(t4 - t3, 3), mismatches=bad[:10])
        if bad or stats['in_order'] == 0:
            failures.append(name + ':runs')
        if module._runs is not original:
            failures.append(name + ':restore')
        t0 = time.perf_counter()
        expected = outcome(module.correct, labels, raw, points)
        t1 = time.perf_counter()
        with handoff_runs.install() as stats:
            actual = outcome(module.correct, labels, raw, points)
        t2 = time.perf_counter()
        cases[name + ':correct'] = dict(kind=expected[0], arrays=stats['arrays'], original=stats['original'],
                                        applications=len(expected[1][3]) if expected[0] == 'value' else None,
                                        original_seconds=round(t1 - t0, 2), adapted_seconds=round(t2 - t1, 2))
        if not same(expected, actual) or stats['arrays'] == 0:
            failures.append(name + ':correct')
    report = dict(status='pass' if not failures else 'fail', frames=len(labels), cases=cases, failures=failures)
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
