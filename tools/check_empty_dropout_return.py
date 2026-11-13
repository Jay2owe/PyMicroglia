"""Compare short-dropout core expansion with and without the empty-input return.

The frozen module is loaded from the packaged correction engine and run on the
final accepted labels, ledger and registered raw with the latent track points:
with no prior applications (as recorded, an empty table with and without
columns) and with prior applications naming owned tracks at later frames, on
the recorded movie and on one with those cells partly eroded so expansions
apply. Labels, ledger, audit and application tables must be identical, dtypes
included; on the empty path the patched call must return its inputs.
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
from scipy import ndimage

import snapshot_engines

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'
FINAL = ENGINE / 'm23_completed_feedback/pymicroglia_95_full_A1/out'


def same(expected, actual):
    try:
        for x, y in zip(expected[:2], actual[:2]):
            assert x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y)
        for x, y in zip(expected[2:], actual[2:]):
            pd.testing.assert_frame_equal(x, y, check_exact=True)
        return len(expected) == len(actual)
    except AssertionError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, help='folder of cropped final.tif, final_ledger.tif, raw.tif, points.csv')
    a = parser.parse_args()
    snapshot_engines.unpack(a.snapshot, snapshot_engines.COMPLETED_FEEDBACK, a.scratch / 'engine')
    sys.path[:0] = [str(a.scratch / 'engine/code'), str(a.adapters.resolve())]
    module = importlib.import_module('expand_short_dropout_soma_cores')
    import empty_dropout_return
    if a.inputs:
        labels = tifffile.imread(a.inputs / 'final.tif')
        ledger = tifffile.imread(a.inputs / 'final_ledger.tif')
        raw = tifffile.imread(a.inputs / 'raw.tif')
        points = pd.read_csv(a.inputs / 'points.csv', low_memory=False)
    else:
        labels = tifffile.imread(FINAL / '95_full_A1.tif')
        ledger = tifffile.imread(FINAL / '95_full_A1_unclaimed_original_ids.tif')
        raw = tifffile.imread(HISTORY / '07_trimmed_evidence/out/95_full_A1_registered_raw.tif')
        points = pd.read_csv(HISTORY / '26_latent_body_tracks/out/latent_track_points.csv', low_memory=False)
    scored = module.physical.attach_owners(points, labels)
    owned = scored[scored.accepted_owner.astype(int) > 0]
    rng = np.random.default_rng(11)
    rows, eroded = [], labels.copy()
    for track, group in owned.groupby('track_id'):
        group = group.sort_values('frame')
        if len(group) < 50 or len(rows) >= 60 or rng.random() > .4:
            continue
        for _, point in group.iloc[40:43].iterrows():
            t, owner = int(point.frame), int(point.accepted_owner)
            rows.append(dict(track_id=int(track), reference_owner=owner, frame=t + 1))
            cell = eroded[t] == owner
            eroded[t][cell & ~ndimage.binary_erosion(cell, iterations=2)] = 0
    prior = pd.DataFrame(rows)
    cases = [('empty-no-columns', labels, pd.DataFrame()),
             ('empty-with-columns', labels, prior.iloc[0:0]),
             ('prior-recorded', labels, prior),
             ('prior-eroded', eroded, prior)]
    failures, report_cases = [], {}
    for name, movie, table in cases:
        t0 = time.perf_counter()
        expected = module.correct(movie, ledger, raw, points, table)
        t1 = time.perf_counter()
        with empty_dropout_return.install() as stats:
            assert stats['supported'] == 1, stats
            actual = module.correct(movie, ledger, raw, points, table)
        t2 = time.perf_counter()
        returned_inputs = actual[0] is movie and actual[1] is ledger
        report_cases[name] = dict(applications=len(expected[3]), empty_returns=stats['empty'],
                                  returned_inputs=returned_inputs,
                                  original_seconds=round(t1 - t0, 2), patched_seconds=round(t2 - t1, 2))
        if not same(expected, actual) or returned_inputs != table.empty:
            failures.append(name)
    # The non-empty path is the original code after the same copies; its cases only confirm that.
    report = dict(status='pass' if not failures else 'fail',
                  cases=report_cases, prior_rows=len(prior), frames=len(labels), failures=failures)
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
