"""Compare eligible-only matrix prepare with the previous whole-row prepare.

Runs the far-pair table check (complete assignments against the engine's own
_assign) while every Matrix.prepare call is also answered by the previous
whole-row version; both outputs must hold the same keys in the same order and
dictionaries equal in values, types and float bit patterns.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def exact(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return type(a) is type(b) and np.float64(a).tobytes() == np.float64(b).tobytes()
    return type(a) is type(b) and a == b


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a, rest = p.parse_known_args()
    sys.path.insert(0, str(a.adapter.resolve()))
    import tracking_matrix_batch as module
    current = module.Matrix.prepare
    report = dict(calls=0, nonempty=0, pairs=0, failures=[], seconds=dict(before=0.0, eligible=0.0))

    def prepare(self, batch, state, s):
        t0 = time.perf_counter(); expected = self._prepare_before(batch, state, s)
        t1 = time.perf_counter(); actual = current(self, batch, state, s)
        t2 = time.perf_counter()
        report['seconds']['before'] += t1 - t0; report['seconds']['eligible'] += t2 - t1
        report['calls'] += 1
        keys = list(actual.keys())
        ok = keys == list(expected)
        if ok and keys:
            report['nonempty'] += 1
            for key in keys:
                left, right = expected[key], actual.get(key)
                report['pairs'] += 1
                if list(left) != list(right) or not all(exact(left[k], right[k]) for k in left):
                    ok = False
                    break
        if not ok and len(report['failures']) < 20:
            report['failures'].append(dict(state=int(state.identity), s=int(s)))
        return actual

    module.Matrix.prepare = prepare
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    import check_sparse_pair_table
    sys.argv = [str(here / 'check_sparse_pair_table.py'), '--adapter', str(a.adapter), *rest]
    try:
        code = check_sparse_pair_table.main()
    except AssertionError as error:
        code = 1
        report['assignment_failure'] = repr(error)[:500]
    report['assignment_check'] = 'pass' if not code else 'fail'
    report['status'] = 'pass' if not code and not report['failures'] and report['calls'] else 'fail'
    a.report.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report, indent=1))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
