"""Compare on-demand midpoint bookend measurement with the original, exactly.

Real accepted-history stacks with the registered raw movie, at the middle
anchor and at random anchors of shorter windows, with plain masks and with the
frozen engine's lazy label masks. Candidate tables and reconciled labels,
tables and reunions must match.
"""
import argparse
import json
from pathlib import Path
import sys
import time


def outcome(module, labels, raw, anchor):
    import numpy as np
    table = module.midpoint_bookend_candidates(labels, raw, anchor, {})
    result, reunions, candidates = module.reconcile_midpoint_bookends(labels, raw, anchor, {})
    return table, candidates, reunions, np.asarray(result)


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'snapshot', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--windows', type=int, default=8)
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    import tifffile
    sys.path[:0] = [str(a.adapter.resolve()), str((a.engine / 'code').resolve())]
    import tracking
    import midpoint_reconciliation as module
    import midpoint_bookend_reuse as reuse
    sys.path.insert(0, str((a.snapshot / 'pymicroglia' / 'tracking').resolve()))
    from frozen_io import observation_reader
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    raw = tifffile.imread(folder / '07_trimmed_evidence/out/95_full_A1_registered_raw.tif')
    stacks = [tifffile.imread(folder / name) for name in ('01_issue005.tif', '10_blob_residency.tif')]
    rng = np.random.default_rng(31)
    cases = [(s, 0, len(stacks[s]), len(stacks[s]) // 2) for s in range(len(stacks))]
    for _ in range(a.windows):
        s = int(rng.integers(len(stacks)))
        start = int(rng.integers(0, len(raw) - 90))
        stop = start + int(rng.integers(40, 90))
        cases.append((s, start, stop, int(rng.integers(3, stop - start - 3))))
    plain = module.frame_observations
    lazy = observation_reader(tracking.frame_observations)
    report = dict(cases=0, candidate_rows=0, reunions=0, before_seconds=0.0, after_seconds=0.0)
    for number, (s, start, stop, anchor) in enumerate(cases):
        labels, movie = stacks[s][start:stop], raw[start:stop]
        module.frame_observations = lazy if number % 2 else plain
        try:
            tick = time.perf_counter(); expected = outcome(module, labels, movie, anchor)
            report['before_seconds'] += time.perf_counter() - tick
            with reuse.install(module) as stats:
                assert stats['supported']
                tick = time.perf_counter(); actual = outcome(module, labels, movie, anchor)
                report['after_seconds'] += time.perf_counter() - tick
        finally:
            module.frame_observations = plain
        for e, f in zip(expected[:3], actual[:3]):
            pd.testing.assert_frame_equal(e, f, check_exact=True)
        assert expected[3].dtype == actual[3].dtype and expected[3].tobytes() == actual[3].tobytes()
        report['cases'] += 1
        report['candidate_rows'] += len(expected[0])
        report['reunions'] += len(expected[2])
        report.setdefault('frames_measured', []).append((stop - start, stats['frames']))
    report['status'] = 'pass'
    print(report)
    a.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
