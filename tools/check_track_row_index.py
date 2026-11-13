"""Compare transient weak seats and persistent duplicate sites with their grouped-row rewrites, exactly.

python tools/check_track_row_index.py --code <completed_feedback_engine/code> --adapters <identity_candidate> --output <report.json>

Both corrections (and _measure_tracks and discover themselves) run on real
label movies (the final accepted labels and an earlier stage, first 160
frames), the registered raw movie and the latent track points, with no
protection and with every third frame protected, then on altered points
(shuffled rows, a non-range index, float frames, an object x column, and
points without owners in the early frames). Every returned array and table
must be identical, dtypes included, or the same error raised.
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

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'
FINAL = ENGINE / 'm23_completed_feedback/pymicroglia_95_full_A1/out'
FRAMES = 160


def outcome(function, *args):
    try:
        return 'value', function(*args)
    except Exception as error:
        return 'error', (type(error).__name__, str(error))


def same(expected, actual):
    if expected[0] != actual[0]:
        return False
    if expected[0] == 'error':
        return expected[1] == actual[1]
    x, y = expected[1], actual[1]
    if isinstance(x, tuple):
        return isinstance(y, tuple) and len(x) == len(y) and all(same(('value', a), ('value', b)) for a, b in zip(x, y))
    if isinstance(x, np.ndarray):
        return isinstance(y, np.ndarray) and x.dtype == y.dtype and np.array_equal(x, y)
    if isinstance(x, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(x, y, check_exact=True)
            return True
        except AssertionError:
            return False
    if isinstance(x, list) and x and isinstance(x[0], dict):
        if len(x) != len(y):
            return False
        for a, b in zip(x, y):
            if list(a) != list(b):
                return False
            for key in a:
                if not same(('value', a[key]), ('value', b[key])):
                    return False
        return True
    if isinstance(x, pd.Series):
        try:
            pd.testing.assert_series_equal(x, y, check_exact=True)
            return True
        except AssertionError:
            return False
    return type(x) is type(y) and x == y


def main():
    parser = argparse.ArgumentParser()
    for name in ('code', 'adapters', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    sys.path[:0] = [str(a.code.resolve()), str(a.adapters.resolve())]
    seats = importlib.import_module('remove_transient_weak_seats')
    sites = importlib.import_module('recover_persistent_duplicate_soma_sites')
    import track_row_index
    raw = tifffile.imread(HISTORY / '07_trimmed_evidence/out/95_full_A1_registered_raw.tif')[:FRAMES]
    movies = [('final', tifffile.imread(FINAL / '95_full_A1.tif')[:FRAMES]),
              ('earlier', tifffile.imread(HISTORY / '44_well_issue001_isolated_pair_encounter_recovery.tif')[:FRAMES])]
    points = pd.read_csv(HISTORY / '26_latent_body_tracks/out/latent_track_points.csv', low_memory=False)
    points = points[points.frame < FRAMES].reset_index(drop=True)
    variants = [('recorded', points), ('shuffled', points.sample(frac=1, random_state=4)),
                ('offset-index', points.set_axis(np.arange(len(points)) * 3 + 11)),
                ('float-frames', points.assign(frame=points.frame.astype(float))),
                ('object-x', points.assign(x=points.x.astype(object)))]
    cases, failures, seconds = [], [], dict(original=0.0, adapted=0.0)
    totals = {}
    for movie_name, labels in movies:
        protections = [('none', np.zeros(labels.shape, bool)),
                       ('every-third', np.zeros(labels.shape, bool))]
        protections[1][1][::3] = labels[::3] > 0
        for point_name, table in variants:
            for protect_name, protected in protections:
                if point_name != 'recorded' and protect_name != 'none':
                    continue
                calls = [('weak seats correct', 'seats', 'correct', (labels, raw, table, protected)),
                         ('measure tracks', 'seats', '_measure_tracks', (labels, raw, table)),
                         ('duplicate sites correct', 'sites', 'correct', (labels, raw, table, protected)),
                         ('discover', 'sites', 'discover', (labels, table))]
                for call, module_name, function, args in calls:
                    if protect_name != 'none' and call in ('measure tracks', 'discover'):
                        continue
                    module = seats if module_name == 'seats' else sites
                    t0 = time.perf_counter()
                    expected = outcome(getattr(module, function), *args)
                    t1 = time.perf_counter()
                    with track_row_index.install() as stats:
                        assert stats['versions'] == 2, stats
                        actual = outcome(getattr(module, function), *args)
                    t2 = time.perf_counter()
                    seconds['original'] += t1 - t0
                    seconds['adapted'] += t2 - t1
                    for key, value in stats.items():
                        totals[key] = totals.get(key, 0) + value
                    label = f'{movie_name} {point_name} {protect_name} {call}'
                    ok = same(expected, actual)
                    cases.append(dict(case=label, result=expected[0], equal=ok,
                                      original_seconds=round(t1 - t0, 2), adapted_seconds=round(t2 - t1, 2)))
                    if not ok:
                        failures.append(label)
    assert seats._measure_tracks.__name__ == '_measure_tracks' and not hasattr(seats._measure_tracks, '__wrapped__')
    # Recorded points skip the per-track frame sort; shuffled and float frames sort.
    passed = (not failures and totals.get('tracks_grouped', 0) > 0 and totals.get('owners_grouped', 0) > 0
              and totals.get('in_order', 0) > 0 and totals.get('sorted', 0) > 0
              and any(c['result'] == 'value' for c in cases))
    report = dict(status='pass' if passed else 'fail', cases=len(cases), failures=failures, stats=totals,
                  seconds={k: round(v, 1) for k, v in seconds.items()}, detail=cases)
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'detail'}))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
