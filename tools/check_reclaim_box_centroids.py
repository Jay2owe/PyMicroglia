"""Compare the Issue 010 gap reclaimer with and without box centroids on real frames.

Windows of the accepted Issue 009 labels are profiled, short-lived names are
dropped and the gaps reclaimed exactly as _short_lived_reclaim does, once with the frozen
reclaim and once with the adapter; the labels, the unclaimed ledger and the
action table (whose distances and costs read the centroids) must be identical.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'snapshot', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--stem', required=True)
    p.add_argument('--windows', default='0:160,300:476')
    a = p.parse_args()
    sys.path[:0] = [str(a.snapshot.resolve()), str(a.adapter.resolve()), str((a.engine / 'code').resolve())]
    from pymicroglia.tracking.frozen_io import LabelMask
    import accepted_history as history
    import persistence_profile_reuse as reuse
    source = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    authority = pd.read_csv(source / '12_movie_authority/out/movie_wide_authority.csv')
    lag_path = source / '07_trimmed_evidence/out' / f'{a.stem}_lag01_float.tif'
    with tifffile.TiffFile(source / '05_issue009.tif') as handle:
        frames = len(handle.pages)
    with tifffile.TiffFile(lag_path) as handle:
        lag_frames = len(handle.pages)
    profiler = history._issue_module(history.SHORT_LIVED_OPS, 'profile_persistence')
    dropper = history._issue_module(history.SHORT_LIVED_OPS, 'drop_short_lived')
    reclaimer = history._issue_module(history.SHORT_LIVED_OPS, 'reclaim_gaps')
    original = reclaimer.reclaim
    failures, cases = [], {}
    for window in a.windows.split(','):
        start, stop = map(int, window.split(':'))
        labels = tifffile.imread(source / '05_issue009.tif', key=range(start, stop))
        lag = tifffile.imread(lag_path, key=range(start, min(stop - (frames - lag_frames), lag_frames)))
        profiles, _ = profiler.profile(labels, authority, history._params('issue010', 'persistence_profile', start))
        drop_params = history._params('issue010', 'drop_short_lived')
        dropped = dropper.choose_dropped(profiles, int(drop_params['maximum_short_frames']),
                                         bool(drop_params['preserve_edge_censored']))
        ownership, unclaimed, _ = dropper.drop(labels, dropped)
        params = history._params('issue010', 'reclaim_gaps', start)
        tick = time.perf_counter()
        expected = original(labels, ownership.copy(), unclaimed.copy(), profiles, lag, params)
        before = time.perf_counter() - tick
        with reuse.install_reclaimer(reclaimer, LabelMask) as stats:
            assert stats['supported']
            tick = time.perf_counter()
            actual = reclaimer.reclaim(labels, ownership.copy(), unclaimed.copy(), profiles, lag, params)
            after = time.perf_counter() - tick
        ok = (np.array_equal(expected[0], actual[0]) and expected[0].dtype == actual[0].dtype
              and np.array_equal(expected[1], actual[1]) and expected[1].dtype == actual[1].dtype)
        try:
            pd.testing.assert_frame_equal(expected[2], actual[2], check_exact=True)
            ok = ok and expected[2].to_csv(index=False) == actual[2].to_csv(index=False)
        except AssertionError:
            ok = False
        cases[window] = dict(actions=len(expected[2]), original_seconds=round(before, 2),
                             adapted_seconds=round(after, 2), stats=dict(stats), equal=ok)
        if not ok or len(expected[2]) == 0 or stats['seeded_reads'] == 0 or stats['box_centroids'] == 0:
            failures.append(window)
    if reclaimer.reclaim is not original:
        failures.append('restore')
    report = dict(status='pass' if not failures else 'fail', cases=cases, failures=failures,
                  sources={str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                           for f in [Path(__file__), Path(reuse.__file__), Path(reclaimer.__file__)]})
    a.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], failures=failures, cases=cases)))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
