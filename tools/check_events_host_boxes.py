"""Compare merge-event resolution with lazily formed, boxed host geometry.

The real stage input (initial identities saved by an earlier-stage run), the
pinned raw stack and the configured event settings. resolve_events without
the adapter is the reference; corrected labels, the event table and both
flag stacks must be equal, with plain observation masks and with the lazy
label masks the bounded runner uses.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
INPUTS = ROOT / 'development/cd68_full_red_green/rigid_batch/.j/A1/inputs'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True, help='composed snapshot (for the lazy reader)')
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve()), str(a.snapshot.resolve())]
    import events as module
    import events_host_boxes as adapter
    from pymicroglia.tracking.frozen_io import observation_reader
    config = json.loads((INPUTS / 'motion_config.json').read_text(encoding='utf-8'))
    params = config['events']
    labels = tifffile.imread(a.labels)
    raw = tifffile.imread(INPUTS / '95_full_A1.tif')
    plain_reader = module.frame_observations
    results, failures = [], []
    for reader in ('plain', 'lazy'):
        module.frame_observations = plain_reader if reader == 'plain' else observation_reader(plain_reader)
        t0 = time.perf_counter()
        expected = module.resolve_events(labels, raw, params)
        t1 = time.perf_counter()
        with adapter.install(module) as stats:
            assert stats['supported']
            actual = module.resolve_events(labels, raw, params)
        t2 = time.perf_counter()
        same = all(np.array_equal(x, y) and x.dtype == y.dtype for x, y in
                   ((expected[0], actual[0]), (expected[2], actual[2]), (expected[3], actual[3])))
        try:
            pd.testing.assert_frame_equal(expected[1], actual[1], check_exact=True)
        except AssertionError:
            same = False
        row = dict(reader=reader, equal=bool(same), events=len(expected[1]), original_s=round(t1 - t0, 1),
                   adapted_s=round(t2 - t1, 1), boxed=stats['boxed'], whole=stats['whole'])
        print(row, flush=True)
        results.append(row)
        if not same:
            failures.append(reader)
    module.frame_observations = plain_reader
    report = dict(status='pass' if not failures else 'fail', results=results, failures=failures)
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
