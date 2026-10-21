"""Compare retroactive-separation component lookup with the boxed copy.

Real accepted frames (plain and disk-backed) are queried for owners that are
present, absent and background, at points on, near and far from each owner,
with several radii. Masks and component numbers must be equal.
"""
import argparse, json, sys, time
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import retroactive_successor_separation as module
    import retroactive_separation_reuse as adapter
    import array_storage
    original = module._component_at
    rng = np.random.default_rng(20261020)
    failures, queries = [], 0
    times = dict(original=0.0, adapted=0.0)
    stack = tifffile.imread(HISTORY / '11_issue017_rebase.tif')
    for label in ('plain', 'working'):
        with ExitStack() as scope:
            labels = stack
            if label == 'working':
                pool = array_storage.Pool(a.scratch, minimum_bytes=1 << 20, temporary_ram_bytes=32 << 20)
                scope.enter_context(array_storage.install(pool))
                labels = np.array(stack)
            with adapter.install(module) as stats:
                assert stats['supported']
                boxed = module._component_at
            for t in rng.choice(len(labels), 40, replace=False):
                frame = labels[int(t)]
                present = np.unique(np.asarray(frame))
                owners = list(rng.choice(present[present > 0], 12)) + [0, int(present.max()) + 7]
                for owner in owners:
                    yy, xx = np.nonzero(np.asarray(frame) == owner)
                    for case in range(6):
                        if len(yy) and case < 3:
                            i = int(rng.integers(len(yy)))
                            y, x = yy[i] + rng.normal(0, 2 * case), xx[i] + rng.normal(0, 2 * case)
                        else:
                            y, x = rng.uniform(-5, frame.shape[0] + 5), rng.uniform(-5, frame.shape[1] + 5)
                        point = SimpleNamespace(x=float(x), y=float(y), radius_px=float(rng.choice([0.5, 3.0, 9.0, 40.0])))
                        t0 = time.perf_counter(); expected = original(frame, int(owner), point)
                        t1 = time.perf_counter(); actual = boxed(frame, int(owner), point)
                        t2 = time.perf_counter()
                        times['original'] += t1 - t0; times['adapted'] += t2 - t1
                        queries += 1
                        if expected[1] != actual[1] or not np.array_equal(expected[0], actual[0]) \
                                or actual[0].dtype != expected[0].dtype or type(actual[1]) is not type(expected[1]):
                            failures.append(f'{label}-{int(t)}-{int(owner)}-{case}')
    report = dict(status='pass' if not failures else 'fail', queries=queries, seconds=times,
                  failures=failures[:20])
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report, indent=1))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
