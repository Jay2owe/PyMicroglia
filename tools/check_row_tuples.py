"""Compare DataFrame.itertuples with and without the remembered row classes, exactly.

Frames cover the tracker points table (whole, per-track groups, filtered and
reindexed), every column kind the corrections use (integers, floats with NaN,
booleans, strings and objects, categories, nullable integers, datetimes,
timedeltas), duplicate, numeric, keyword and tuple column names, MultiIndex
rows and columns, empty frames, a single column and more than 255 columns,
with index on and off and named, unnamed and plain tuples. Each row must have
the same type name, fields, module, values and value types; rows made in
separate calls must still be equal and hash alike.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time


def same_value(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, float) and math.isnan(a):
        return math.isnan(b)
    try:
        result = a == b
        if result is True or result is False:
            return result or (a is b)
        return bool(result)
    except (TypeError, ValueError):
        return a is b


def same_rows(left, right):
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if type(a).__name__ != type(b).__name__ or getattr(type(a), '_fields', None) != getattr(type(b), '_fields', None):
            return False
        if type(a).__module__ != type(b).__module__ or len(a) != len(b):
            return False
        if not all(same_value(x, y) for x, y in zip(a, b)):
            return False
    return True


def frames(points):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(3)
    yield 'points', points
    for i, (_, group) in enumerate(points.groupby('track_id', sort=False)):
        if i % 50 == 0:
            yield 'track', group.sort_values('frame', kind='stable')
    yield 'filtered', points[points.frame % 3 == 0]
    yield 'reindexed', points.reset_index(drop=True).iloc[::-1]
    n = 60
    mixed = pd.DataFrame({
        'i': rng.integers(-5, 5, n), 'u8': rng.integers(0, 255, n).astype('uint8'),
        'f': np.where(rng.random(n) < .2, np.nan, rng.random(n)), 'f32': rng.random(n).astype('float32'),
        'b': rng.random(n) < .5, 's': rng.choice(['a', 'b', None], n), 'o': [(k, None) if k % 2 else k for k in range(n)],
        'c': pd.Categorical(rng.choice(['x', 'y'], n)), 'n': pd.array([None if k % 7 == 0 else k for k in range(n)], dtype='Int64'),
        'd': pd.date_range('2026-01-01', periods=n, freq='h'), 'td': pd.to_timedelta(np.arange(n), unit='s'),
        'bn': pd.array([None if k % 5 == 0 else bool(k % 2) for k in range(n)], dtype='boolean')})
    yield 'mixed', mixed
    yield 'mixed_index', mixed.set_index(pd.Index(rng.permutation(n) * 3 + 1, name='row'))
    duplicate = pd.DataFrame(rng.random((5, 3)), columns=['a', 'a', 'b'])
    yield 'duplicate', duplicate
    yield 'numeric_names', pd.DataFrame(rng.random((4, 3)), columns=[0, 1.5, None])
    yield 'keyword_names', pd.DataFrame(rng.random((4, 4)), columns=['class', 'def', '_x', 'Index'])
    yield 'multi_columns', pd.DataFrame(rng.random((3, 2)), columns=pd.MultiIndex.from_tuples([('a', 1), ('b', 2)]))
    yield 'multi_rows', pd.DataFrame({'v': [1, 2, 3]}, index=pd.MultiIndex.from_tuples([(1, 'a'), (1, 'b'), (2, 'a')]))
    yield 'empty', pd.DataFrame({'a': pd.Series([], dtype=int), 'b': pd.Series([], dtype=object)})
    yield 'no_columns', pd.DataFrame(index=[1, 2])
    yield 'single', pd.DataFrame({'only': [1.0, 2.0]})
    yield 'wide', pd.DataFrame(rng.integers(0, 9, (3, 300)))
    view = points.copy()
    view['added'] = 1
    yield 'unconsolidated', view


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--points', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import pandas as pd
    sys.path.insert(0, str(a.adapter.resolve()))
    import row_tuples as adapter
    points = pd.read_csv(a.points, low_memory=False)
    original = pd.DataFrame.itertuples
    report = dict(status='pass', cases=[], pandas=pd.__version__)
    timing = dict(original=0.0, adapted=0.0)
    stats = {}
    for label, frame in frames(points):
        for kwargs in (dict(), dict(index=False), dict(name=None), dict(name='Row'), dict(index=False, name=None)):
            tick = time.perf_counter()
            expected = [list(frame.itertuples(**kwargs)) for _ in range(2)]
            middle = time.perf_counter()
            with adapter.install(stats):
                assert pd.DataFrame.itertuples.__wrapped__ is original
                actual = [list(frame.itertuples(**kwargs)) for _ in range(2)]
            timing['original'] += middle - tick
            timing['adapted'] += time.perf_counter() - middle
            assert pd.DataFrame.itertuples is original
            assert same_rows(expected[0], actual[0]) and same_rows(expected[1], actual[1]), (label, kwargs)
            assert same_rows(actual[0], actual[1]), (label, kwargs)
            if actual[0] and all(isinstance(v, (int, float, str, bool, type(None))) for v in actual[0][0]):
                assert actual[0][0] == actual[1][0] and hash(actual[0][0]) == hash(actual[1][0])
            report['cases'].append(dict(frame=label, kwargs=kwargs, rows=len(actual[0])))
    # Timing on the per-track pattern the corrections use.
    groups = [g for _, g in points.groupby('track_id', sort=False)][:2000]
    tick = time.perf_counter()
    for g in groups:
        list(g.itertuples())
    middle = time.perf_counter()
    with adapter.install(stats):
        for g in groups:
            list(g.itertuples())
    report['per_track'] = dict(groups=len(groups), original=middle - tick, adapted=time.perf_counter() - middle)
    assert 'versions_unmatched' not in stats and stats['calls'] > 0
    report.update(seconds=timing, stats=stats)
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}))


if __name__ == '__main__':
    main()
