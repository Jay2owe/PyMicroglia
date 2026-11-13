"""Compare Issue 005's territory steps with and without the discarded audit.

Both territory functions run on real accepted label movies at several anchor
frames with the accepted territory settings, through the history's names
(audit skipped) and the territory module's originals. Labels and decision
tables must be identical; the skipped audit is an empty table, and the
territory module's own functions are untouched while installed.
"""
import argparse
import json
import sys
import time
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
    parser.add_argument('--frames', type=int, default=60)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import accepted_history as history
    import territory_reconciliation as territory
    import territory_audit_skip as adapter
    params = history._params('issue005', 'final')['territory']
    movies = [tifffile.imread(HISTORY / name)[:a.frames] for name in ('01_issue005.tif', '04_issue008.tif')]
    failures, seconds, cases = [], dict(original=0.0, replaced=0.0), 0
    originals = {name: getattr(territory, name) for name in adapter.NAMES}
    with adapter.install(history) as stats:
        assert stats['supported'], stats
        for name in adapter.NAMES:
            assert getattr(territory, name) is originals[name]
            assert getattr(history, name) is not originals[name]
        for movie in movies:
            for anchor in (0, a.frames // 2, a.frames - 1):
                for name in adapter.NAMES:
                    t0 = time.perf_counter()
                    expected = originals[name](movie, anchor, params)
                    t1 = time.perf_counter()
                    actual = getattr(history, name)(movie, anchor, params)
                    seconds['original'] += t1 - t0; seconds['replaced'] += time.perf_counter() - t1
                    cases += 1
                    try:
                        assert expected[0].dtype == actual[0].dtype and np.array_equal(expected[0], actual[0])
                        pd.testing.assert_frame_equal(expected[1], actual[1], check_exact=True)
                        assert isinstance(actual[2], pd.DataFrame) and actual[2].empty
                    except AssertionError as error:
                        failures.append(dict(name=name, anchor=anchor, error=str(error)[:200]))
        counters = dict(stats)
    for name in adapter.NAMES:
        assert getattr(history, name) is originals[name]
    assert counters['skipped'] == cases, counters
    report = dict(status='pass' if not failures else 'fail', frames=a.frames, cases=cases,
                  seconds=seconds, counters=counters, failures=failures[:10])
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
