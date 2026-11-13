"""Compare DataFrame.itertuples with and without row_tuples, row for row, before installing it for corrections.

Tables: recorded A1 latent track points (whole, grouped by track and by frame),
and constructed tables covering every column kind pandas stores (int, float,
bool, object, float32, nullable Int64/Float64/boolean, string, category,
naive and time-zone datetimes, timedeltas), duplicate and non-identifier and
integer column labels, MultiIndex columns and index, an empty table and a table
without columns. Each is iterated with index True and False and names "Pandas",
None and a custom name, plus an invalid name that must raise the same error.
Rows must match in type name, module, fields, values and value types. A table
changed in place after its iterator is made but before it is consumed must give
the same rows as the original does.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def described(value):
    if isinstance(value, tuple) and hasattr(value, '_fields'):
        return ('row', type(value).__name__, type(value).__module__, value._fields,
                [described(v) for v in value])
    if isinstance(value, tuple):
        return ('tuple', [described(v) for v in value])
    if isinstance(value, float) and np.isnan(value):
        return ('nan', type(value).__name__)
    if value is pd.NA or value is pd.NaT:
        return ('missing', repr(value))
    return (type(value).__name__, repr(value))


def rows(table, **options):
    try:
        return ('rows', [described(row) for row in table.itertuples(**options)])
    except Exception as error:
        return ('error', type(error).__name__, str(error))


def tables(points):
    n = 7
    kinds = pd.DataFrame({
        'i': np.arange(n), 'f': np.linspace(0, 1, n), 'b': np.arange(n) % 2 == 0,
        'o': ['a', None, 3, 4.5, 'e', (1, 2), 'g'], 'f32': np.arange(n, dtype=np.float32) / 3,
        'u8': np.arange(n, dtype=np.uint8), 'nan': [np.nan, 1.0, np.nan, 2, 3, 4, 5],
        'I': pd.array([1, None, 3, 4, 5, 6, 7], dtype='Int64'),
        'F': pd.array([1.5, None, 3, 4, 5, 6, 7], dtype='Float64'),
        'B': pd.array([True, None, False, True, True, False, None], dtype='boolean'),
        's': pd.array(['x', None, 'z', 'w', 'v', 'u', 't'], dtype='string'),
        'c': pd.Categorical(list('abcabca')),
        'dt': pd.date_range('2026-01-01', periods=n), 'tz': pd.date_range('2026-01-01', periods=n, tz='UTC'),
        'td': pd.to_timedelta(np.arange(n), unit='h')})
    duplicate = pd.DataFrame(np.arange(12).reshape(4, 3), columns=['a', 'a', 'b'])
    awkward = pd.DataFrame({'class': [1, 2], '1x': [3, 4], 'two words': [5, 6], '_hidden': [7, 8], 0: [9, 10]})
    multi = pd.DataFrame(np.arange(8).reshape(2, 4), columns=pd.MultiIndex.from_product([['p', 'q'], [1, 2]]),
                         index=pd.MultiIndex.from_tuples([('r', 1), ('s', 2)]))
    labelled = kinds.set_index(pd.Index(list('ABCDEFG'), name='key'))
    found = dict(points=points, kinds=kinds, labelled=labelled, duplicate=duplicate, awkward=awkward,
                 multi=multi, empty=points.iloc[0:0], no_columns=pd.DataFrame(index=range(3)),
                 sliced=points.iloc[5:40:3], transposed=pd.DataFrame(np.arange(12.).reshape(3, 4)).T)
    for number, (track, group) in enumerate(points.groupby('track_id')):
        if number % 40 == 0:
            found[f'track_{track}'] = group.sort_values('frame')
    for number, (frame, group) in enumerate(points.groupby('frame')):
        if number % 60 == 0:
            found[f'frame_{frame}'] = group
    return found


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--adapters', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path[:0] = [str(a.adapters.resolve())]
    import row_tuples as adapter
    history = Path('development/cd68_full_red_green/rigid_analysis/A1/tracking/engine/'
                   'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history')
    points = pd.read_csv(history / '26_latent_body_tracks/out/latent_track_points.csv', low_memory=False)
    original = pd.DataFrame.itertuples
    failures, cases = [], {}
    options = [dict(), dict(index=False), dict(name=None), dict(index=False, name=None),
               dict(name='Point'), dict(index=False, name='not valid')]
    for label, table in tables(points).items():
        for option in options:
            key = f'{label}:{option}'
            t0 = time.perf_counter()
            expected = rows(table, **option)
            t1 = time.perf_counter()
            with adapter.install() as stats:
                assert 'versions_unmatched' not in stats
                actual = rows(table, **option)
                again = rows(table, **option)
            t2 = time.perf_counter()
            if label in ('points', 'kinds'):
                cases[key] = dict(rows=len(expected[1]) if expected[0] == 'rows' else expected,
                                  original_seconds=round(t1 - t0, 3), adapted_seconds=round((t2 - t1) / 2, 3))
            if expected != actual or expected != again:
                failures.append(key)
    # Changed in place between making the iterator and consuming it.
    for use_adapter in (False, True):
        table = pd.DataFrame({'x': np.arange(5.), 'y': np.arange(5), 's': list('abcde')})
        if use_adapter:
            with adapter.install():
                iterator = table.itertuples()
                table.loc[3, 'x'] = 99.
                table.iloc[4, 1] = -1
                result = [described(row) for row in iterator]
        else:
            iterator = table.itertuples()
            table.loc[3, 'x'] = 99.
            table.iloc[4, 1] = -1
            result = [described(row) for row in iterator]
        cases['changed_in_place' + (':adapted' if use_adapter else ':original')] = result
    if cases['changed_in_place:adapted'] != cases['changed_in_place:original']:
        failures.append('changed_in_place')
    if pd.DataFrame.itertuples is not original:
        failures.append('restore')
    report = dict(status='pass' if not failures else 'fail', pandas=pd.__version__, failures=failures,
                  cases={k: v for k, v in cases.items() if not k.startswith('changed')},
                  sources={str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                           for f in [Path(__file__), Path(adapter.__file__)]})
    a.output.write_text(json.dumps(report, indent=1, default=str) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], failures=failures,
                          timing={k: v for k, v in report['cases'].items() if k.startswith('points')}), default=str))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
