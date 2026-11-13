"""Compare bracketed consensus memory's per-track named rows with the one-sort helper.

python tools/check_bracketed_consensus_tracks.py --snapshot <combined snapshot> --scratch <empty folder> --adapters <identity_candidate> --output <report.json>

The recorded A1 latent track points get a proposal name column (zeros and
names, as proposals[t, y, x] gives); every track's g[g.native.gt(0)]
.sort_values('frame') is compared with the helper's group (tracks with no
named rows are skipped by the function either way), as recorded, shuffled,
with an offset index, with repeated (track, frame) rows, float frames, a
nullable name column and an empty table. Groups must match in key, rows,
index type, columns and dtypes. The adapter is installed on the frozen
module from the packaged engine and must rewrite and restore it.
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import snapshot_engines

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / ('development/cd68_full_red_green/rigid_analysis/A1/tracking/engine/'
                  'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history')


def same(a, b):
    try:
        pd.testing.assert_frame_equal(a, b, check_exact=True)
    except AssertionError:
        return False
    return type(a.index) is type(b.index) and a.index.dtype == b.index.dtype


def main():
    p = argparse.ArgumentParser()
    for name in ('snapshot', 'scratch', 'adapters', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    snapshot_engines.unpack(a.snapshot, snapshot_engines.COMPLETED_FEEDBACK, a.scratch)
    frozen = a.scratch / 'frozen/current/r202_a007'
    sys.path[:0] = [str(frozen), str(a.scratch / 'code'), str(a.adapters.resolve())]
    import bracketed_consensus_tracks as adapter
    points = pd.read_csv(HISTORY / '26_latent_body_tracks/out/latent_track_points.csv', low_memory=False)
    rng = np.random.default_rng(13)
    pts = points[points.strong.astype(bool) & points.state.eq('observed')].copy()
    pts['native'] = np.where(rng.random(len(pts)) < .4, 0, rng.integers(1, 60, len(pts))).astype(np.uint16)
    variants = dict(
        recorded=pts,
        shuffled=pts.sample(frac=1, random_state=2),
        offset_index=pts.set_axis(np.arange(len(pts)) * 5 + 7),
        repeated=pd.concat([pts, pts.iloc[::9]]),
        float_frames=pts.assign(frame=pts.frame.astype(float)),
        nullable_native=pts.assign(native=pts.native.astype('Int64')),
        all_zero=pts.assign(native=np.zeros(len(pts), np.uint16)),
        empty=pts.iloc[0:0])
    stats = dict(sorted_once=0, per_track=0)
    helper = adapter.tracks(stats)
    cases, failures = {}, []
    for name, table in variants.items():
        expected = [(tr, g) for tr, g in ((tr, g[g.native.gt(0)].sort_values('frame'))
                                          for tr, g in table.groupby('track_id')) if not g.empty]
        actual = [(tr, g) for tr, g in helper(table) if not g.empty]
        ok = (len(expected) == len(actual)
              and all(x[0] == y[0] and type(x[0]) is type(y[0]) and same(x[1], y[1])
                      for x, y in zip(expected, actual)))
        cases[name] = dict(tracks=len(expected), equal=ok)
        if not ok:
            failures.append(name)
    if not stats['sorted_once'] or stats['per_track'] < 3:
        failures.append('routes')
    module = importlib.import_module(adapter.NAME)
    original = getattr(module, adapter.FUNCTION)
    with adapter.install() as install_stats:
        rewritten = getattr(module, adapter.FUNCTION)
        installed = install_stats['versions'] == 1 and rewritten.__wrapped__ is original
    restored = getattr(module, adapter.FUNCTION) is original and '_bracketed_tracks' not in vars(module)
    if not installed:
        failures.append('install')
    if not restored:
        failures.append('restore')
    report = dict(status='pass' if not failures else 'fail', cases=cases, stats=stats, failures=failures)
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
