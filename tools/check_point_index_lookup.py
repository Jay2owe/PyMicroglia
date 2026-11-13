"""Compare point_index_lookup with the reciprocal exchanges' own _point, exactly.

Both discover functions run on their accepted A1 stage inputs; the gap-tolerant
one runs with gap_bracket_memo installed, as the pipeline runs it (its bare
bracket search makes tens of millions of _point calls). During the original
run every _point call is also answered by the lookup on the same table, and the two answers must match: None for None, otherwise the same row
Series (values, dtype, index, name) with the same warnings. Then discover runs
with the lookup installed and its audit and internal tables must equal the
original's. Synthetic tables add repeated keys, float and missing levels, a
single-level index, absent keys and keys that int() rejects.
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'


def answer(function, *args):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            result = ('ok', function(*args))
        except Exception as error:
            result = ('error', (type(error).__name__, str(error)))
    return result, [(w.category.__name__, str(w.message)) for w in caught]


def same(left, right):
    (a, aw), (b, bw) = left, right
    if aw != bw:
        return f'warnings {aw} != {bw}'
    if a[0] != b[0]:
        return 'outcome'
    x, y = a[1], b[1]
    if a[0] == 'error':
        return None if x == y else 'error'
    if x is None or y is None:
        return None if x is y else 'None'
    if type(x) is not type(y):
        return 'type'
    if isinstance(x, pd.Series):
        try:
            pd.testing.assert_series_equal(x, y, check_exact=True)
        except AssertionError as error:
            return 'series ' + str(error)[:200]
        if x.name != y.name or type(x.name) is not type(y.name) or x.dtype != y.dtype:
            return 'series name or dtype'
        if any(type(p) is not type(q) for p, q in zip(x.name, y.name)) if isinstance(x.name, tuple) else False:
            return 'name element types'
        return None
    if isinstance(x, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(x, y, check_exact=True)
        except AssertionError as error:
            return 'frame ' + str(error)[:200]
        return None
    if isinstance(x, tuple):
        for p, q in zip(x, y):
            problem = same((('ok', p), []), (('ok', q), []))
            if problem:
                return problem
        return None if len(x) == len(y) else 'length'
    return None if x == y else 'value'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import tifffile
    import reciprocal_two_seat_exchange as reciprocal
    import gap_tolerant_reciprocal_seat_exchange as gap
    import gap_tolerant_reciprocal_seat_exchange_integration as gap_integration
    import point_index_lookup as adapter
    import gap_bracket_memo
    config = json.loads((ENGINE / 'config.json').read_text(encoding='utf-8'))
    points = pd.read_csv(HISTORY / '26_latent_body_tracks/out/latent_track_points.csv')
    configured = config['accepted_postprocessing'].get('field_wide_reciprocal_two_seat_exchange', {})
    reciprocal_params = dict(configured.get('parameters', {}), targeting_mode='field_wide_discovery')
    _, gap_params = gap_integration._configured(config)
    report = dict(failures=[], calls={}, stages={}, synthetic=0)

    def fail(**item):
        report['failures'].append(item)

    for name, module, stem, params in (
            ('reciprocal_two_seat_exchange', reciprocal, '46_well_a2_issue001_exclusive_pair_seat_recovery', reciprocal_params),
            ('gap_tolerant_reciprocal_seat_exchange', gap, '87_well_b2_issue018_recording_start_two_seat_inheritance', gap_params)):
        labels = tifffile.imread(HISTORY / (stem + '.tif'))
        original = module._point

        def run(module=module, labels=labels, params=params):
            with ExitStack() as scope:
                if module is gap:
                    memo = scope.enter_context(gap_bracket_memo.install(gap))
                    assert memo['supported'], memo
                return answer(module.discover, labels, points.copy(), params)
        with adapter.install(module) as stats:
            assert stats['supported'] == [name], stats
            replacement = module._point
            module._point = original
            calls = [0, 0]

            def recorded(index, track, frame):
                expected = answer(original, index, track, frame)
                actual = answer(replacement, index, track, frame)
                calls[0] += 1
                calls[1] += expected[0][0] == 'ok' and expected[0][1] is not None
                problem = same(expected, actual)
                if problem and len(report['failures']) < 30:
                    fail(module=name, track=repr(track), frame=repr(frame), problem=problem)
                if expected[0][0] == 'error':
                    raise RuntimeError(expected[0][1])
                return expected[0][1]
            module._point = recorded
            t0 = time.perf_counter()
            expected = run()
            t1 = time.perf_counter()
            module._point = replacement
            actual = run()
            t2 = time.perf_counter()
            stage_stats = dict(stats)
        restored = module._point is original
        # Untimed baseline: the original discover alone.
        t3 = time.perf_counter()
        baseline = run()
        t4 = time.perf_counter()
        report['calls'][name] = dict(calls=calls[0], rows=calls[1])
        report['stages'][name] = dict(status=expected[0][0], warnings=len(expected[1]), restored=restored,
                                      original_seconds=round(t4 - t3, 2), adapted_seconds=round(t2 - t1, 2),
                                      stats=stage_stats)
        for label, left in (('recorded', expected), ('baseline', baseline)):
            problem = same(left, actual)
            if problem:
                fail(module=name, case='discover ' + label, problem=problem)
    # Synthetic tables.
    frames = {
        'repeated': pd.DataFrame({'track_id': [1, 1, 2, 2, 2], 'frame': [3, 3, 4, 5, 5], 'v': [0.5, 1.5, 2, 3, 4]}),
        'float_levels': pd.DataFrame({'track_id': [1.0, 2.0], 'frame': [3.0, 4.0], 'v': [1, 2]}),
        'missing': pd.DataFrame({'track_id': [1, np.nan], 'frame': [3, 4], 'v': [1, 2]}),
        'mixed_columns': pd.DataFrame({'track_id': [1, 2], 'frame': [3, 4], 'v': ['a', 2], 'b': [True, False]}),
        'empty': pd.DataFrame({'track_id': pd.Series([], dtype=int), 'frame': pd.Series([], dtype=int)}),
    }
    keys = [(1, 3), (2, 4), (2, 5), (9, 9), (1.0, 3.0), (np.int64(2), np.int32(5)), ('2', '4'), (None, 3), (np.nan, 3), (1.7, 3.2)]
    for module in (reciprocal, gap):
        original = module._point
        with adapter.install(module):
            for name, table in frames.items():
                indexes = {'multi': table.set_index(['track_id', 'frame'], drop=False),
                           'single': table.set_index('track_id', drop=False)}
                for kind, index in indexes.items():
                    for track, frame in keys:
                        report['synthetic'] += 1
                        problem = same(answer(original, index, track, frame), answer(module._point, index, track, frame))
                        if problem:
                            fail(module=module.__name__, case=f'{name}/{kind}', key=repr((track, frame)), problem=problem)
    report['status'] = 'pass' if not report['failures'] and all(
        s['restored'] and s['stats']['lookups'] for s in report['stages'].values()) else 'fail'
    a.output.write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
    print(json.dumps(report, indent=1, default=str))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
