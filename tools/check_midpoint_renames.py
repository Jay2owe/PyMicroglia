"""Compare midpoint reconciliation renames with the frame-limited renames.

Real accepted labels (windows, several dtypes, a transposed copy) with a
stand-in candidate table so many pairs are accepted, including chains where a
renamed-into name is itself renamed later and pairs whose names never share a
frame. Returned labels, decision table and candidate table must be equal.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import midpoint_reconciliation as module
    import midpoint_bookend_reuse as adapter
    original_candidates = module.midpoint_bookend_candidates
    original_reconcile = module.reconcile_midpoint_bookends
    stack = tifffile.imread(HISTORY / '01_issue005.tif')
    rng = np.random.default_rng(20261020)
    failures, decisions, cases = [], 0, 0
    seconds = dict(original=0.0, frame=0.0)
    table = {}

    def candidates(labels, raw, anchor_t, settings):
        return table['value'].copy()

    for start in (0, 150, 300):
        window = stack[start:start + 120]
        for data in (window, window.astype(np.int32), np.ascontiguousarray(window.transpose(0, 2, 1))):
            ids = [int(v) for v in np.unique(data)[1:]]
            for trial in range(3):
                pre = [int(v) for v in rng.choice(ids, min(len(ids), 40), replace=False)]
                post = [int(v) for v in rng.choice(ids, min(len(ids), 40), replace=False)]
                rows = []
                for p in pre:
                    for q in rng.choice(post, 4, replace=False):
                        rows.append(dict(pre_identity=p, post_identity=int(q), allowed=bool(rng.random() < .8),
                                         cost=float(rng.random() * 3), gap_frames=int(rng.integers(1, 9))))
                table['value'] = pd.DataFrame(rows)
                params = dict(maximum_cost=2.5, minimum_assignment_margin=-1.0,
                              protected_identity_ids=[int(v) for v in rng.choice(ids, 10, replace=False)],
                              unprotected_canonical_policy=['longer_fragment', None][trial % 2])
                module.midpoint_bookend_candidates = candidates
                try:
                    t0 = time.perf_counter()
                    try:
                        expected = ('ok', original_reconcile(data, data, 60, params))
                    except Exception as error:
                        expected = ('error', type(error).__name__, str(error))
                    t1 = time.perf_counter()
                    module.midpoint_bookend_candidates = original_candidates
                    with adapter.install(module) as stats:
                        module.midpoint_bookend_candidates = candidates
                        try:
                            actual = ('ok', module.reconcile_midpoint_bookends(data, data, 60, params))
                        except Exception as error:
                            actual = ('error', type(error).__name__, str(error))
                        frame_calls = stats.get('frame_renames')
                    t2 = time.perf_counter()
                finally:
                    module.midpoint_bookend_candidates = original_candidates
                seconds['original'] += t1 - t0; seconds['frame'] += t2 - t1
                cases += 1
                try:
                    assert frame_calls == 1, frame_calls
                    assert expected[0] == actual[0], (expected[:3], actual[:3])
                    if expected[0] == 'ok':
                        (x, xd, xc), (y, yd, yc) = expected[1], actual[1]
                        assert type(x) is type(y) and x.dtype == y.dtype and np.array_equal(x, y)
                        pd.testing.assert_frame_equal(xd, yd, check_exact=True)
                        pd.testing.assert_frame_equal(xc, yc, check_exact=True)
                        decisions += len(xd)
                    else:
                        assert expected == actual
                except AssertionError as error:
                    failures.append(dict(start=start, dtype=str(data.dtype), trial=trial, error=str(error)[:300]))
    assert module.reconcile_midpoint_bookends is original_reconcile
    report = dict(status='pass' if not failures and decisions else 'fail', cases=cases, decisions=decisions,
                  seconds=seconds, failures=failures[:10])
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
