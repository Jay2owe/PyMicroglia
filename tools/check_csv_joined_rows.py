"""Compare DataFrame.to_csv bytes with and without csv_joined_rows.

Real accepted-history tables (the Issue 089 reference audits with their tuple
and list columns rebuilt, the physical owner points) and synthetic tables
holding every value kind the writer must reproduce or hand back: separators,
quotes, line breaks, empty text, None, NaN and pd.NA, infinities, signed zero,
NumPy floats, non-ASCII text, categoricals, datetimes, nullable integers,
indexes and non-default writer options. Each table is written to a file and to
a text buffer both ways; the bytes must be equal.
"""
import argparse, ast, io, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / ('development/cd68_full_red_green/rigid_analysis/A1/tracking/engine/'
                  'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history')
RELAY = HISTORY / '52_cross_reference_owner_relay/out'


def synthetic(rng, rows):
    text = np.array(['plain', 'a,b', 'say "hi"', 'line\nbreak', 'cr\rhere', '', ' lead', 'trail ',
                     '""', ',', 'x\r\ny', 'tab\tin', "it's"], dtype=object)
    frame = pd.DataFrame({
        'id': np.arange(rows, dtype=np.int64) * 7919 - 10 ** 12,
        'x': rng.normal(0, 1e3, rows),
        'flag': rng.random(rows) < 0.5,
        'label': text[rng.integers(len(text), size=rows)],
        'pair': [(int(a), int(b)) for a, b in rng.integers(0, 500, size=(rows, 2))],
        'owners': [list(map(int, rng.integers(0, 9, size=k))) for k in rng.integers(0, 4, size=rows)],
        'small': rng.integers(-5, 5, rows).astype(np.int32),
        'f32': rng.random(rows).astype(np.float32),
    })
    special = np.array([np.nan, np.inf, -np.inf, -0.0, 0.0, 1e-300, 1e16, 0.1, 5e-324, 123456789.125])
    frame.loc[frame.index[::17], 'x'] = special[rng.integers(len(special), size=len(frame.index[::17]))]
    mixed = frame['label'].astype(object).copy()
    mixed.iloc[::11] = None
    mixed.iloc[5::13] = np.nan
    mixed.iloc[3::19] = 2.5
    mixed.iloc[7::23] = True
    mixed.iloc[9::29] = 42
    frame['mixed'] = mixed
    return frame


def variants(rng, rows):
    for name, frame, options in _variants(rng, rows):
        # The engine writes without its index; index=True is checked as its own case.
        yield name, frame, {'index': False, **options}


def _variants(rng, rows):
    base = synthetic(rng, rows)
    yield 'synthetic', base, {}
    yield 'synthetic_index', base, dict(index=True)
    yield 'synthetic_lf', base, dict(lineterminator='\n')
    yield 'synthetic_sep', base, dict(sep=';')
    yield 'synthetic_quote_all', base, dict(quoting=1)
    yield 'synthetic_float_format', base, dict(float_format='%.3f')
    yield 'synthetic_na_rep', base, dict(na_rep='NA')
    yield 'synthetic_columns', base, dict(columns=['label', 'x', 'pair'])
    yield 'synthetic_no_header', base, dict(header=False)
    yield 'single_column', base[['label']], {}
    yield 'small_table', base.iloc[:500], {}
    numpy_floats = base.copy()
    numpy_floats['mixed'] = [np.float64(v) for v in rng.random(rows)]
    yield 'numpy_float_objects', numpy_floats, {}
    unicode = base.copy()
    unicode.loc[unicode.index[::101], 'label'] = 'µm²'
    yield 'non_ascii', unicode, {}
    typed = base.copy()
    typed['category'] = pd.Categorical(base['label'])
    typed['when'] = pd.Timestamp('2026-10-24') + pd.to_timedelta(np.arange(rows), unit='s')
    typed['nullable'] = pd.array(np.where(np.arange(rows) % 7 == 0, None, np.arange(rows)), dtype='Int64')
    typed['na'] = pd.Series([pd.NA if k % 5 == 0 else 'v' for k in range(rows)], dtype=object)
    yield 'extension_types', typed, {}
    yield 'plain_numbers', base[['id', 'x', 'flag', 'small', 'f32']], {}


def real(sample):
    points = pd.read_csv(RELAY / 'physical_owner_points.csv')
    yield 'physical_owner_points', points, dict(index=False)
    del points
    for name in ('reference_handoff_audit.csv', 'reference_relay_proposals.csv'):
        table = pd.read_csv(RELAY / name)
        yield name + ':read', table, dict(index=False)
        rebuilt = table.iloc[:sample].copy()
        rebuilt['track_pair'] = [ast.literal_eval(v) for v in rebuilt['track_pair']]
        rebuilt['competing_owners'] = [ast.literal_eval(v) for v in rebuilt['competing_owners']]
        yield name + ':objects', rebuilt, dict(index=False)
        del table, rebuilt


def write(frame, options, target, path):
    if target == 'file':
        frame.to_csv(path, **options)
        return path.read_bytes()
    buffer = io.StringIO()
    frame.to_csv(buffer, **options)
    return buffer.getvalue().encode('utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--rows', type=int, default=60000)
    parser.add_argument('--sample', type=int, default=120000)
    a = parser.parse_args()
    sys.path.insert(0, str(a.adapters.resolve()))
    import csv_joined_rows
    a.scratch.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261024)
    results, failures = [], []
    seconds = dict(original=0.0, adapted=0.0)
    totals = dict(chunks=0, rows=0, original_chunks=0)
    for source in (variants(rng, a.rows), real(a.sample)):
        for name, frame, options in source:
            for target in ('file', 'buffer'):
                t0 = time.perf_counter()
                expected = write(frame, options, target, a.scratch / 'expected.csv')
                t1 = time.perf_counter()
                with csv_joined_rows.install() as stats:
                    assert stats['supported']
                    actual = write(frame, options, target, a.scratch / 'actual.csv')
                t2 = time.perf_counter()
                seconds['original'] += t1 - t0
                seconds['adapted'] += t2 - t1
                for key in totals:
                    totals[key] += stats[key]
                row = dict(table=name, target=target, rows=len(frame), bytes=len(expected),
                           equal=expected == actual, joined_rows=stats['rows'],
                           original_seconds=round(t1 - t0, 3), adapted_seconds=round(t2 - t1, 3))
                results.append(row)
                print(json.dumps(row), file=sys.stderr, flush=True)
                if not row['equal']:
                    failures.append(row)
    for path in (a.scratch / 'expected.csv', a.scratch / 'actual.csv'):
        path.unlink(missing_ok=True)
    report = dict(status='pass' if not failures and totals['rows'] else 'fail', cases=len(results),
                  seconds=seconds, totals=totals, failures=failures[:10], results=results)
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'results'}))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
