"""Compare leading-soma and ownerless-soma recovery with frame-limited boundary checks.

Both frozen correction modules are loaded from the packaged correction engine
and run on the final accepted labels, ledger and registered raw with the latent
track points: as recorded, with cells blanked in frame 0 and for a few frames
after long named runs (so recoveries are applied), and with looser rules.
Labels, ledger, audit and application tables must be identical, dtypes included.
The boundary test itself is compared with the whole-movie expression on edits
confined to a few frames: inside the ledger and labels rules, changing a
labelled pixel, changing the ledger outside a change, adding and removing names,
for plain and disk-backed working arrays.
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


def same(expected, actual):
    try:
        assert len(expected) == len(actual)
        for x, y in zip(expected[:2], actual[:2]):
            assert x.dtype == y.dtype and np.array_equal(x, y)
        for x, y in zip(expected[2:], actual[2:]):
            pd.testing.assert_frame_equal(x, y, check_exact=True)
        return True
    except AssertionError:
        return False


def whole(candidate, labels, candidate_ledger, ledger):
    changed = candidate != labels
    return bool(np.any(changed & (labels != 0))
                or np.any((candidate_ledger != ledger) & ~changed)
                or not np.array_equal(np.unique(candidate), np.unique(labels)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    snapshot_engines.unpack(a.snapshot, snapshot_engines.COMPLETED_FEEDBACK, a.scratch / 'engine')
    sys.path[:0] = [str(a.scratch / 'engine/code'), str(a.adapters.resolve())]
    leading = importlib.import_module('recover_leading_soma_from_future_run')
    ownerless = importlib.import_module('recover_ownerless_soma_runs')
    import boundary_frame_checks
    import array_storage
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
    first = scored[scored.frame.astype(int).eq(0) & (scored.accepted_owner.astype(int) > 0)]
    for owner in first.accepted_owner.astype(int).unique()[::3]:
        blanked[0][blanked[0] == owner] = 0
    cases = []
    for module, loose in ((leading, leading.Params(minimum_reference_frames=3, minimum_reference_movie_fraction=0.0,
                                                    minimum_reference_independent_fraction=0.0)),
                          (ownerless, ownerless.Params(minimum_ownerless_run_frames=1, minimum_strong_fraction=0.0))):
        for name, movie, params in (('recorded', labels, module.Params()), ('blanked', blanked, module.Params()),
                                    ('loose', blanked, loose)):
            cases.append((module, name, movie, params))
    failures, seconds, report_cases = [], dict(original=0.0, fast=0.0), {}
    framed = 0
    for module, name, movie, params in cases:
        label = f'{module.__name__}:{name}'
        t0 = time.perf_counter()
        expected = module.correct(movie, ledger, raw, points, params)
        t1 = time.perf_counter()
        with boundary_frame_checks.install() as stats:
            assert stats['supported'] == 2, stats
            actual = module.correct(movie, ledger, raw, points, params)
        t2 = time.perf_counter()
        seconds['original'] += t1 - t0
        seconds['fast'] += t2 - t1
        framed += stats['framed']
        report_cases[label] = dict(applications=len(expected[3]), audits=len(expected[2]),
                                   original_seconds=round(t1 - t0, 2), fast_seconds=round(t2 - t1, 2))
        if not same(expected, actual):
            failures.append(label)
    # The boundary test itself, on edits confined to a few frames.
    with boundary_frame_checks.install() as stats:
        crossed = vars(leading)['_boundary_crossed']
        tests = 0
        small, small_ledger = labels[:40].copy(), ledger[:40].copy()
        unused = int(small.max()) + 1
        for storage in ('plain', 'working'):
            pool = None
            if storage == 'working':
                pool = array_storage.Pool(a.scratch / 'pool', minimum_bytes=1 << 10, temporary_ram_bytes=1 << 20)
                scope = array_storage.install(pool); scope.__enter__()
            try:
                base, base_ledger = np.array(small), np.array(small_ledger)
                for kind in ('none', 'background', 'labelled', 'ledger', 'new-name', 'lost-name', 'background-ledger'):
                    for t in (0, 7, 39):
                        candidate, candidate_ledger = base.copy(), base_ledger.copy()
                        frame = candidate[t]
                        zeros = np.argwhere(np.asarray(frame) == 0)[:5]
                        names = np.unique(np.asarray(frame))[1:]
                        if kind == 'background':
                            owner = int(names[0])
                            for y, x in zeros:
                                candidate[t, y, x] = owner; candidate_ledger[t, y, x] = 0
                        elif kind == 'labelled':
                            y, x = np.argwhere(np.asarray(frame) == names[0])[0]
                            candidate[t, y, x] = names[1]
                        elif kind == 'ledger':
                            y, x = zeros[0]
                            candidate_ledger[t, y, x] = candidate_ledger[t, y, x] + 1
                        elif kind == 'new-name':
                            for y, x in zeros:
                                candidate[t, y, x] = unused
                        elif kind == 'lost-name':
                            # Remove every pixel of a name that appears only in this frame.
                            counts = {int(v): int(np.count_nonzero(np.asarray(base) == v)) for v in names[:30]}
                            local = [v for v in names[:30] if counts[int(v)] == np.count_nonzero(np.asarray(frame) == v)]
                            if local:
                                candidate[t][np.asarray(candidate[t]) == local[0]] = 0
                        elif kind == 'background-ledger':
                            y, x = zeros[0]
                            candidate[t, y, x] = int(names[0]); candidate_ledger[t, y, x] = candidate_ledger[t, y, x] + 1
                        for frames in ({t}, {t, 3}, set() if kind == 'none' else {t}):
                            tests += 1
                            if bool(crossed(candidate, base, candidate_ledger, base_ledger, frames)) != whole(
                                    candidate, base, candidate_ledger, base_ledger):
                                failures.append(f'boundary:{storage}:{kind}:{t}:{sorted(frames)}')
            finally:
                if pool is not None:
                    scope.__exit__(None, None, None)
    report = dict(status='pass' if not failures and framed else 'fail', cases=report_cases,
                  framed_checks=framed, boundary_tests=tests, seconds=seconds, failures=failures)
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
