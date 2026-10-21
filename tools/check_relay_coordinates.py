"""Compare relay transition coordinates with the whole-frame pixel listing.

Real cell masks from the accepted stack plus empty and random masks; the
boxed listing must equal np.column_stack(np.nonzero(mask)) in dtype, shape
and values.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine/m22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    sys.path.insert(0, str(a.adapters.resolve()))
    import relay_pair_reuse as adapter
    stack = tifffile.imread(HISTORY / '11_issue017_rebase.tif')
    rng = np.random.default_rng(5)
    cases = [np.zeros((40, 50), bool), np.ones((40, 50), bool)] + [rng.random((40, 50)) < p for p in (.001, .05, .5)]
    for t in rng.choice(len(stack), 80, replace=False):
        for identity in rng.choice(np.unique(stack[t])[1:], 12):
            cases.append(stack[t] == identity)
    failures, seconds = [], dict(whole=0.0, boxed=0.0)
    for number, mask in enumerate(cases):
        t0 = time.perf_counter(); expected = np.column_stack(np.nonzero(mask))
        t1 = time.perf_counter(); actual = adapter._coordinates(mask)
        t2 = time.perf_counter()
        seconds['whole'] += t1 - t0; seconds['boxed'] += t2 - t1
        if not (expected.dtype == actual.dtype and expected.shape == actual.shape and np.array_equal(expected, actual)):
            failures.append(number)
    report = dict(status='pass' if not failures else 'fail', cases=len(cases), seconds=seconds, failures=failures[:20])
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
