"""Compare recurrent cyclic alias track handoffs with and without one sorted pass.

The frozen modules are loaded from the packaged correction engine. On cropped
native and final label movies with the matching track points,
_track_handoffs is compared at several gap limits, on reordered points and on
edge cases (duplicate track frames, missing or float track ids, missing strong
column, empty points); then recurrent_cyclic_alias.discover and correct and
high_confidence_lineage_union.discover and correct are compared whole. Tables,
labels and raised errors must be identical.
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


def outcome(function, *args):
    try:
        return ('value', function(*args))
    except Exception as error:
        return ('error', type(error).__name__, str(error))


def same(x, y):
    if type(x) is not type(y):
        return False
    if isinstance(x, (tuple, list)):
        return len(x) == len(y) and all(same(a, b) for a, b in zip(x, y))
    if isinstance(x, dict):
        return list(x) == list(y) and all(same(x[k], y[k]) for k in x)
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
    parser.add_argument('--inputs', type=Path, required=True, help='folder of cropped native.tif, final.tif, points.csv')
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    snapshot_engines.unpack(a.snapshot, snapshot_engines.COMPLETED_FEEDBACK, a.scratch / 'engine')
    sys.path[:0] = [str(a.scratch / 'engine/code'), str(a.adapters.resolve())]
    cyclic = importlib.import_module('recurrent_cyclic_alias')
    union = importlib.import_module('high_confidence_lineage_union')
    import cyclic_track_handoffs
    points = pd.read_csv(a.inputs / 'points.csv', low_memory=False)
    rng = np.random.default_rng(9)
    failures, cases = [], {}
    for movie in ('native', 'final'):
        labels = tifffile.imread(a.inputs / f'{movie}.tif')
        variants = {'recorded': points, 'shuffled': points.sample(frac=1, random_state=3),
                    'duplicate': pd.concat([points, points.iloc[:50]]),
                    'float_track': points.assign(track_id=points.track_id.astype(float)),
                    'missing_track': points.assign(track_id=points.track_id.where(rng.random(len(points)) > .01)),
                    'no_strong': points.drop(columns=['strong']) if 'strong' in points else points,
                    'empty': points.iloc[0:0]}
        for name, table in variants.items():
            for gap in (1, 3, 12, 10**6):
                label = f'{movie}:{name}:gap{gap}'
                t0 = time.perf_counter()
                expected = outcome(cyclic._track_handoffs, table, labels, gap)
                t1 = time.perf_counter()
                with cyclic_track_handoffs.install() as stats:
                    assert stats['supported'] == 1
                    actual = outcome(cyclic._track_handoffs, table, labels, gap)
                t2 = time.perf_counter()
                cases[label] = dict(kind=expected[0], rows=len(expected[1]) if expected[0] == 'value' else None,
                                    sorted=stats['sorted'], original=stats['original'],
                                    original_seconds=round(t1 - t0, 3), adapted_seconds=round(t2 - t1, 3))
                if not same(expected, actual):
                    failures.append(label)
        for module, function in ((cyclic, 'discover'), (cyclic, 'correct'), (union, 'discover'), (union, 'correct')):
            label = f'{movie}:{module.__name__}.{function}'
            t0 = time.perf_counter()
            expected = outcome(getattr(module, function), labels, points)
            t1 = time.perf_counter()
            with cyclic_track_handoffs.install() as stats:
                actual = outcome(getattr(module, function), labels, points)
            t2 = time.perf_counter()
            cases[label] = dict(kind=expected[0], sorted=stats['sorted'], original=stats['original'],
                                original_seconds=round(t1 - t0, 2), adapted_seconds=round(t2 - t1, 2))
            if not same(expected, actual) or (expected[0] == 'value' and stats['sorted'] == 0):
                failures.append(label)
    if getattr(cyclic._track_handoffs, '__wrapped__', None) is not None:
        failures.append('restore')
    report = dict(status='pass' if not failures else 'fail', cases=cases, failures=failures)
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(dict(status=report['status'], failures=failures,
                          whole={k: v for k, v in cases.items() if '.' in k})))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
