"""Compare short ownerless-soma recovery with and without per-run tables.

The frozen correction module is loaded from the packaged correction engine
and run on the final accepted labels, ledger and registered raw with the
latent track points the corrections receive: as recorded, with cells blanked
for a few frames after long named runs (so recoveries are applied), with no
duration limit and a one-frame minimum (every run and reference is built),
and with a duplicated row index (the original steps must be taken). Labels,
ledger, audit and application tables must be identical, dtypes included.
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


def same_frames(a, b):
    try:
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        return True
    except AssertionError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    snapshot_engines.unpack(a.snapshot, snapshot_engines.COMPLETED_FEEDBACK, a.scratch)
    sys.path[:0] = [str(a.scratch / 'code'), str(a.adapters.resolve())]
    module = importlib.import_module('recover_short_ownerless_soma_runs')
    import short_ownerless_runs
    labels = tifffile.imread(FINAL / '95_full_A1.tif')
    ledger = tifffile.imread(FINAL / '95_full_A1_unclaimed_original_ids.tif')
    raw = tifffile.imread(HISTORY / '07_trimmed_evidence/out/95_full_A1_registered_raw.tif')
    points = pd.read_csv(HISTORY / '26_latent_body_tracks/out/latent_track_points.csv', low_memory=False)
    blanked = labels.copy()
    scored = module.physical.attach_owners(points, labels)
    owned = scored[scored.accepted_owner.astype(int) > 0]
    rng = np.random.default_rng(41)
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
    duplicated = points.copy()
    duplicated.index = np.repeat(np.arange(len(points) // 2 + 1), 2)[:len(points)]
    loose = module.Params(minimum_ownerless_run_frames=1, maximum_ownerless_run_frames=None)
    cases = [('recorded', labels, points, module.Params()), ('blanked', blanked, points, module.Params()),
             ('loose', blanked, points, loose), ('duplicated-index', blanked, duplicated, module.Params())]
    failures, seconds, report_cases = [], dict(original=0.0, fast=0.0), {}
    # Episodes alone: every kept run against rows.loc of its labels (the
    # original build), index type included, on recorded, shuffled and
    # offset-index points with the default and the loosest limits.
    episodes = dict(tracks=0, failures=0, sliced=0)
    for name, table in (('recorded', points), ('shuffled', points.sample(frac=1, random_state=5)),
                        ('offset-index', points.set_axis(np.arange(len(points)) * 3 + 11))):
        table_scored = module.physical.attach_owners(table, labels)
        runs = short_ownerless_runs.Runs(module, dict(original=0, runs=0, built=0, references=0))
        for params in (module.Params(), loose):
            for track, group in table_scored.groupby('track_id'):
                actual = runs.episodes(table_scored, group, track, params)
                zero = runs._zero_rows(table_scored).get(track)
                expected = []
                if zero is not None and not zero.empty:
                    rows = zero.sort_values('frame')
                    cuts = np.flatnonzero(np.diff(rows.frame.astype(int).to_numpy()) != 1) + 1
                    expected = [rows.loc[i] for i in np.split(rows.index.to_numpy(), cuts)
                                if len(i) and len(i) >= params.minimum_ownerless_run_frames
                                and not (params.maximum_ownerless_run_frames is not None
                                         and len(i) > params.maximum_ownerless_run_frames)]
                episodes['tracks'] += 1
                episodes['failures'] += not (len(expected) == len(actual) and all(
                    same_frames(x, y) and type(x.index) is type(y.index) and x.index.dtype == y.index.dtype
                    for x, y in zip(expected, actual)))
        episodes['sliced'] += runs.stats.get('sliced', 0)
        episodes['in_order'] = episodes.get('in_order', 0) + runs.stats.get('in_order', 0)
        # References alone: every track at a spread of onsets, the original
        # _reference against the adapter's, index type included.
        reference = runs.reference(module._reference)
        for track, group in table_scored.groupby('track_id'):
            frames = group.frame.astype(int)
            for onset in {int(frames.min()), int(frames.median()), int(frames.max()) + 1, np.int64(frames.min())}:
                owner, rows = module._reference(group, onset)
                fast_owner, fast_rows = reference(group, onset)
                episodes['references'] = episodes.get('references', 0) + 1
                episodes['failures'] += not (owner == fast_owner and type(owner) is type(fast_owner)
                                             and same_frames(rows, fast_rows) and type(rows.index) is type(fast_rows.index)
                                             and rows.index.dtype == fast_rows.index.dtype)
        episodes['empty'] = episodes.get('empty', 0) + runs.stats.get('empty', 0)
    if episodes['failures'] or not episodes['sliced'] or not episodes['empty'] or not episodes['in_order']:
        failures.append('episodes')
    for name, movie, table, params in cases:
        t0 = time.perf_counter()
        expected = module.correct(movie, ledger, raw, table, params)
        t1 = time.perf_counter()
        with short_ownerless_runs.install() as stats:
            assert stats['supported'], 'adapter did not install'
            actual = module.correct(movie, ledger, raw, table, params)
        seconds['original'] += t1 - t0; seconds['fast'] += time.perf_counter() - t1
        ok = (all(isinstance(x, np.ndarray) and x.dtype == y.dtype and np.array_equal(x, y)
                  for x, y in zip(expected[:2], actual[:2]))
              and same_frames(expected[2], actual[2]) and same_frames(expected[3], actual[3]))
        report_cases[name] = dict(applications=len(expected[3]), audits=len(expected[2]),
                                  stats=dict(stats), seconds=[round(t1 - t0, 2), round(time.perf_counter() - t1, 2)])
        if not ok:
            failures.append(name)
    assert not hasattr(module.correct, '__wrapped__'), 'adapter not restored'
    report = dict(status='pass' if not failures else 'fail', cases=report_cases, episodes=episodes,
                  seconds=seconds, failures=failures)
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
