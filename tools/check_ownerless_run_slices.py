"""Compare ownerless-soma run slices with the frozen run builder, exactly.

python tools/check_ownerless_run_slices.py --snapshot <composition> --scratch <dir> --adapters <identity_candidate> --output <report.json>

_consecutive_runs is compared directly on tables of each index kind (range,
shuffled, named, unsigned, duplicate, string, float), frame dtype (integer,
float, NaN, unsorted, repeated) and size, including empty tables: the same
number of runs, each frame equal with index type and name, or the same error.
The ownerless correction then runs on the final accepted labels, ledger and
raw with the latent track points (as recorded, with owned cells blanked so
recoveries apply, and with looser rules), alone and beside the boundary-frame
adapter: labels, ledger, audit and application tables must be identical.
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

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'
FINAL = ENGINE / 'm23_completed_feedback/pymicroglia_95_full_A1/out'


def outcome(function, rows):
    try:
        return 'value', function(rows)
    except Exception as error:
        return 'error', (type(error).__name__, str(error))


def same_runs(expected, actual):
    if expected[0] != actual[0]:
        return False
    if expected[0] == 'error':
        return expected[1] == actual[1]
    if len(expected[1]) != len(actual[1]):
        return False
    for x, y in zip(expected[1], actual[1]):
        try:
            pd.testing.assert_frame_equal(x, y, check_exact=True, check_index_type=True,
                                          check_column_type=True, check_names=True)
        except AssertionError:
            return False
        if type(x.index) is not type(y.index) or x.index.name != y.index.name:
            return False
    return True


def same(expected, actual):
    try:
        for x, y in zip(expected[:2], actual[:2]):
            assert x.dtype == y.dtype and np.array_equal(x, y)
        for x, y in zip(expected[2:], actual[2:]):
            pd.testing.assert_frame_equal(x, y, check_exact=True)
        return len(expected) == len(actual)
    except AssertionError:
        return False


def tables(rng):
    for trial in range(300):
        n = int(rng.integers(0, 60))
        frames = np.sort(rng.choice(np.arange(200), n, replace=False)) if rng.random() < .7 else rng.integers(0, 30, n)
        rows = pd.DataFrame(dict(frame=frames, x=rng.uniform(0, 50, n), strong=rng.random(n) < .5,
                                 state=rng.choice(['observed', 'predicted'], n)))
        kind = trial % 7
        if kind == 1:
            rows.index = rng.permutation(n) * 3 + 7
        elif kind == 2:
            rows.index = pd.Index(rng.permutation(n), name='row')
        elif kind == 3:
            rows.index = rng.permutation(n).astype(np.uint32)
        elif kind == 4 and n > 1:
            rows.index = list(range(n - 1)) + [0]
        elif kind == 5:
            rows.index = [f'r{i}' for i in range(n)]
        elif kind == 6:
            rows.index = np.arange(n) + .5
        if trial % 11 == 0:
            rows = rows.sample(frac=1, random_state=trial)
        if trial % 13 == 0:
            rows['frame'] = rows.frame.astype(float)
        if trial % 17 == 0 and n:
            rows.loc[rows.index[0], 'frame'] = np.nan
        yield rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    snapshot_engines.unpack(a.snapshot, snapshot_engines.COMPLETED_FEEDBACK, a.scratch / 'engine')
    sys.path[:0] = [str(a.scratch / 'engine/code'), str(a.adapters.resolve())]
    ownerless = importlib.import_module('recover_ownerless_soma_runs')
    import boundary_frame_checks
    import ownerless_run_slices
    original = ownerless._consecutive_runs
    stats = dict(sliced=0, original=0)
    fast = ownerless_run_slices.runs(original, stats)
    direct = equal = 0
    for rows in tables(np.random.default_rng(7)):
        direct += 1
        equal += same_runs(outcome(original, rows), outcome(fast, rows))
    labels = tifffile.imread(FINAL / '95_full_A1.tif')
    ledger = tifffile.imread(FINAL / '95_full_A1_unclaimed_original_ids.tif')
    raw = tifffile.imread(HISTORY / '07_trimmed_evidence/out/95_full_A1_registered_raw.tif')
    points = pd.read_csv(HISTORY / '26_latent_body_tracks/out/latent_track_points.csv', low_memory=False)
    blanked = labels.copy()
    scored = ownerless.physical.attach_owners(points, labels)
    owned = scored[scored.accepted_owner.astype(int) > 0]
    rng = np.random.default_rng(43)
    chosen = 0
    for track, group in owned.groupby('track_id'):
        frames = np.sort(group.frame.astype(int).to_numpy())
        if len(frames) < 40 or chosen >= 12 or rng.random() > .3:
            continue
        start = int(frames[30])
        owner = int(group.accepted_owner.iloc[30])
        for t in range(start, min(start + 4, len(labels))):
            blanked[t][blanked[t] == owner] = 0
        chosen += 1
    # The run builder alone on every real track's ownerless and owned rows, as correct asks.
    real_runs = dict(groups=0, equal=0, original_seconds=0.0, sliced_seconds=0.0)
    for score in (scored, ownerless.physical.attach_owners(points, blanked)):
        for field in ('strong', 'physically_visible', 'in_encounter'):
            score[field] = score[field].astype(bool)
        for _, group in score.groupby('track_id', sort=True):
            visible = group[group.physically_visible]
            for rows in (visible[visible.accepted_owner.astype(int).eq(0)],
                         group[group.accepted_owner.astype(int).gt(0)]):
                t0 = time.perf_counter()
                expected = outcome(original, rows)
                t1 = time.perf_counter()
                actual = outcome(fast, rows)
                t2 = time.perf_counter()
                real_runs['groups'] += 1
                real_runs['equal'] += same_runs(expected, actual)
                real_runs['original_seconds'] += t1 - t0
                real_runs['sliced_seconds'] += t2 - t1
    real_runs = {k: round(v, 2) if isinstance(v, float) else v for k, v in real_runs.items()}
    loose = ownerless.Params(minimum_ownerless_run_frames=1, minimum_strong_fraction=0.0)
    failures, cases = [], {}
    for name, movie, params in (('recorded', labels, ownerless.Params()), ('blanked', blanked, ownerless.Params()),
                                ('loose', blanked, loose)):
        t0 = time.perf_counter()
        expected = ownerless.correct(movie, ledger, raw, points, params)
        t1 = time.perf_counter()
        with ownerless_run_slices.install() as alone:
            assert alone['versions'] == 1, alone
            actual = ownerless.correct(movie, ledger, raw, points, params)
        t2 = time.perf_counter()
        with boundary_frame_checks.install(), ownerless_run_slices.install() as both:
            combined = ownerless.correct(movie, ledger, raw, points, params)
        t3 = time.perf_counter()
        assert ownerless._consecutive_runs is original
        cases[name] = dict(applications=len(expected[3]), audits=len(expected[2]), sliced=alone['sliced'],
                           original_seconds=round(t1 - t0, 2), sliced_seconds=round(t2 - t1, 2),
                           with_boundary_seconds=round(t3 - t2, 2))
        for label, result in (('alone', actual), ('with boundary', combined)):
            if not same(expected, result):
                failures.append(f'{name} {label}')
    passed = (not failures and equal == direct and stats['sliced'] > 0 and stats['original'] > 0
              and real_runs['equal'] == real_runs['groups'])
    report = dict(status='pass' if passed else 'fail', direct_tables=direct, direct_equal=int(equal),
                  direct_stats=stats, real_runs=real_runs, cases=cases, failures=failures)
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
