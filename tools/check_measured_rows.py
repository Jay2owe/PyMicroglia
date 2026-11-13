"""Compare four soma-memory corrections with their NumPy measured-row rewrite, exactly.

python tools/check_measured_rows.py --code <completed_feedback_engine/code> --adapters <identity_candidate> --output <report.json>

dual_soma_ancestry_memory_a001, observed_soma_gap_memory_a003,
recording_start_soma_memory_a001 and successive_detector_soma_memory_a004 run
on real label movies (the final accepted labels and an earlier stage, first
160 frames), the registered raw movie and the latent track points, then on
altered points (shuffled rows, a non-range index, missing state values, an
object strong column and a nullable owner column, the last two through the
original expression). Every returned array and table must be identical,
dtypes included, or the same error raised. The helper alone is also compared
with the expression on every track of every variant.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_track_row_index import FINAL, FRAMES, HISTORY, outcome, same  # noqa: E402

NAMES = ('dual_soma_ancestry_memory_a001', 'observed_soma_gap_memory_a003',
         'recording_start_soma_memory_a001', 'successive_detector_soma_memory_a004')


def main():
    parser = argparse.ArgumentParser()
    for name in ('code', 'adapters', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    sys.path[:0] = [str(a.code.resolve()), str(a.adapters.resolve())]
    modules = {name: importlib.import_module(name) for name in NAMES}
    import measured_rows
    raw = tifffile.imread(HISTORY / '07_trimmed_evidence/out/95_full_A1_registered_raw.tif')[:FRAMES]
    movies = [('final', tifffile.imread(FINAL / '95_full_A1.tif')[:FRAMES]),
              ('earlier', tifffile.imread(HISTORY / '44_well_issue001_isolated_pair_encounter_recovery.tif')[:FRAMES])]
    points = pd.read_csv(HISTORY / '26_latent_body_tracks/out/latent_track_points.csv', low_memory=False)
    points = points[points.frame < FRAMES].reset_index(drop=True)
    missing = points.copy()
    missing.loc[missing.index[::17], 'state'] = np.nan
    variants = [('recorded', points), ('shuffled', points.sample(frac=1, random_state=4)),
                ('offset-index', points.set_axis(np.arange(len(points)) * 3 + 11)),
                ('missing-state', missing),
                ('object-strong', points.assign(strong=points.strong.astype(object))),
                ('nullable-owner', points.assign(**{'candidate_owner': points.candidate_owner.astype('Int64')})
                 if 'candidate_owner' in points.columns else points)]
    cases, failures, seconds, totals = [], [], dict(original=0.0, adapted=0.0), {}
    # The helper on every track group, as correct builds them.
    helper_stats = dict(numpy=0, original=0)
    helper = measured_rows.measured_rows(helper_stats)
    owner = importlib.import_module('resident_takeover')
    helper_checked = helper_failures = 0
    groups_helper = measured_rows.track_groups(helper_stats)
    helper_stats.update(grouped_once=0, groups_original=0)
    repeated = pd.concat([points, points[points.track_id == points.track_id.iloc[0]].iloc[:3]])
    group_variants = variants + [('repeated-frame', repeated),
                                 ('float-frames', points.assign(frame=points.frame.astype(float)))]
    groups_checked = groups_failures = 0
    for point_name, table in group_variants:
        scored = owner.attach_owners(table, movies[0][1])
        expected = {int(tr): g.sort_values('frame').set_index('frame', drop=False)
                    for tr, g in scored.groupby('track_id')}
        actual = groups_helper(scored)
        groups_checked += 1
        ok = list(expected) == list(actual)
        for key in expected:
            try:
                pd.testing.assert_frame_equal(expected[key], actual[key], check_exact=True)
                ok = ok and list(expected[key].dtypes) == list(actual[key].dtypes)
            except (AssertionError, KeyError):
                ok = False
        groups_failures += not ok
    for point_name, table in variants:
        scored = owner.attach_owners(table, movies[0][1])
        for tr, g in scored.groupby('track_id'):
            g = g.sort_values('frame').set_index('frame', drop=False)
            expected = g[g.strong.astype(bool) & g.state.eq('observed') & g.candidate_owner.gt(0)]
            actual = g[helper(g)]
            helper_checked += 1
            try:
                pd.testing.assert_frame_equal(expected, actual, check_exact=True)
                ok = expected.index.equals(actual.index) and list(expected.dtypes) == list(actual.dtypes)
            except AssertionError:
                ok = False
            helper_failures += not ok
    for movie_name, labels in movies:
        protected = np.zeros(labels.shape, bool)
        for point_name, table in variants:
            if movie_name == 'earlier' and point_name not in ('recorded', 'shuffled'):
                continue
            for name, module in modules.items():
                t0 = time.perf_counter()
                expected = outcome(module.correct, labels, raw, table, protected)
                t1 = time.perf_counter()
                with measured_rows.install() as stats:
                    assert stats['versions'] == 4 and module.correct.__wrapped__ is not None, stats
                    actual = outcome(module.correct, labels, raw, table, protected)
                t2 = time.perf_counter()
                seconds['original'] += t1 - t0
                seconds['adapted'] += t2 - t1
                for key, value in stats.items():
                    totals[key] = totals.get(key, 0) + value
                label = f'{movie_name} {point_name} {name}'
                ok = same(expected, actual)
                cases.append(dict(case=label, result=expected[0], equal=ok,
                                  original_seconds=round(t1 - t0, 2), adapted_seconds=round(t2 - t1, 2)))
                if not ok:
                    failures.append(label)
    restored = all(not hasattr(m.correct, '__wrapped__') and '_measured_rows' not in vars(m)
                   and '_track_groups' not in vars(m) and '_end_values' not in vars(m) for m in modules.values())
    # End values: a row's own values, or the row itself when one is missing.
    ends = measured_rows.end_values(dict(end_values=0, end_original=0))
    row = pd.Series({'frame': 3, 'x': 1.5, 'y': 2.0, 'radius_px': 4.0}, dtype=object)
    short = row.drop('y')
    got = ends(row)
    ends_ok = ((got.x, got.y, got.radius_px) == (row.x, row.y, row.radius_px)
               and all(type(a) is type(b) for a, b in zip(got, (row.x, row.y, row.radius_px)))
               and ends(short) is short)
    passed = (not failures and restored and helper_failures == 0 and helper_stats['numpy'] > 0
              and helper_stats['original'] > 0 and totals.get('numpy', 0) > 0
              and groups_failures == 0 and helper_stats['grouped_once'] > 0 and helper_stats['groups_original'] >= 2
              and totals.get('grouped_once', 0) > 0 and ends_ok
              and totals.get('single_mask_edits', 0) > 0 and totals.get('end_values_edits', 0) > 0
              and totals.get('end_values', 0) > 0
              and any(c['result'] == 'value' for c in cases))
    report = dict(status='pass' if passed else 'fail', cases=len(cases), failures=failures, restored=restored,
                  helper=dict(tracks=helper_checked, failures=helper_failures, stats=helper_stats),
                  groups=dict(tables=groups_checked, failures=groups_failures), stats=totals,
                  seconds={k: round(v, 1) for k, v in seconds.items()}, detail=cases)
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'detail'}))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
