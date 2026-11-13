"""Check indexed frame lookups in mixed-owner flash and owner-relay discovery.

python check_track_frame_lookups.py --adapter <identity_candidate dir> --engine <A1 engine dir> --output <report.json>

Runs both discover functions on A1's saved stage inputs and latent track points,
as they are and with the lookup adapter, and compares the three returned tables
(values, dtypes, index, columns and CSV text). Also compares the two lookups with
the original expressions on small tables: repeated, missing and non-integer
frames, one-row and empty groups, a table whose rows all match, and NaN frames
(the same error).
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def outcome(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            value = call()
        except Exception as error:  # the error itself is compared
            return ('error', type(error).__name__, str(error)), [str(w.message) for w in caught]
    return ('value', value), [str(w.message) for w in caught]


def same(left, right):
    if isinstance(left, pd.DataFrame) or isinstance(right, pd.DataFrame):
        return (type(left) is type(right) and left.equals(right) and list(left.columns) == list(right.columns)
                and left.dtypes.equals(right.dtypes) and left.index.equals(right.index)
                and left.to_csv() == right.to_csv())
    if isinstance(left, pd.Series) or isinstance(right, pd.Series):
        return (type(left) is type(right) and left.equals(right) and left.name == right.name
                and left.dtype == right.dtype and left.index.equals(right.index))
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--engine', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    engine = Path(args.engine)
    sys.path.insert(0, str(engine / 'code'))
    sys.path.insert(0, args.adapter)
    import bracketed_mixed_owner_flash as flash
    import anchored_projection_owner_relay as relay
    import bracketed_mixed_owner_flash_integration as flash_integration
    import anchored_projection_owner_relay_integration as relay_integration
    from common import load_stack
    import track_frame_lookups as lookups

    failures, counts = [], dict(rows=0, points=0, discover=0)

    def compare(case, left, right):
        if not same(left[0], right[0]) or left[1] != right[1]:
            failures.append(dict(case=case, original=repr(left)[:300], adapted=repr(right)[:300]))

    # The two lookups against the original expressions.
    stats = dict(rows=0, fallback_rows=0, points=0, fallback_points=0)
    rng = np.random.default_rng(5)
    table = pd.DataFrame(dict(frame=rng.integers(0, 9, 60), track_id=rng.integers(0, 4, 60),
                              x=rng.random(60), candidate_owner=rng.integers(0, 5, 60),
                              physically_visible=rng.random(60) > 0.3))
    tables = {
        'mixed': table,
        'float frames': table.assign(frame=table.frame.astype(float)),
        'shifted index': table.set_index(np.arange(100, 160)),
        'one row': table.iloc[:1],
        'one frame': table.assign(frame=3),
        'empty': table.iloc[:0],
        'nan frame': table.assign(frame=table.frame.astype(float).where(table.index != 7)),
        'object frames': table.assign(frame=table.frame.astype(object)),
    }
    for name, value in tables.items():
        for frame in list(range(-1, 11)) + [np.int64(3), 3.0, np.float64(2.0), True]:
            memo = {}
            compare('rows %s %r' % (name, frame),
                    outcome(lambda: value[value.frame.astype(int).eq(frame)]),
                    outcome(lambda: lookups.frame_rows(stats, memo, value, frame)))
            memo = {}
            compare('point %s %r' % (name, frame), outcome(lambda: flash._point(value, frame)),
                    outcome(lambda: lookups.point(stats, memo, value, frame)))
            counts['rows'] += 1
            counts['points'] += 1
        for track, group in value.groupby('track_id'):
            memo = {}
            for frame in range(10):
                compare('group %s %s %d' % (name, track, frame), outcome(lambda: relay._point(group, frame)),
                        outcome(lambda: lookups.point(stats, memo, group, frame)))
                counts['points'] += 1

    history = engine / 'm22_accepted_history' / 'pymicroglia_95_full_A1' / 'mid' / 'accepted_history'
    values = json.loads((engine / 'config.json').read_text())
    points = pd.read_csv(history / '26_latent_body_tracks' / 'out' / 'latent_track_points.csv')
    cases = [('flash', flash, flash_integration, '60_well_b2_issue004_terminal_two_seat_assimilation.tif'),
             ('relay', relay, relay_integration, '97_well_b3_issue012_delayed_owner_projection_flash.tif')]
    seconds, run_stats = {}, None
    for name, module, integration, stem in cases:
        _, params = integration._configured(values)
        labels = load_stack(history / stem)
        start = time.perf_counter()
        before = outcome(lambda: module.discover(labels, points.copy(), params))
        seconds[name + '_original'] = round(time.perf_counter() - start, 2)
        with lookups.install(flash, relay) as run_stats:
            start = time.perf_counter()
            after = outcome(lambda: module.discover(labels, points.copy(), params))
            seconds[name + '_adapted'] = round(time.perf_counter() - start, 2)
        if module.__name__ not in run_stats['supported']:
            failures.append(dict(case=name, problem='adapter not installed', stats=dict(run_stats)))
        compare('A1 ' + name, before, after)
        counts['discover'] += 1
        seconds[name + '_rows'] = int(len(before[0][1][0])) if before[0][0] == 'value' else None
        seconds[name + '_lookups'] = dict(run_stats)
    report = dict(status='pass' if not failures else 'fail', counts=counts, unit_stats=stats, a1=seconds,
                  failures=failures[:40])
    Path(args.output).write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
    print(json.dumps(dict(status=report['status'], counts=counts, failures=len(failures), a1=seconds), default=str))


if __name__ == '__main__':
    main()
