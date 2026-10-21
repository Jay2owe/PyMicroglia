"""Compare unbounded motion support with the union-box copy.

Pairs of real cell masks from consecutive accepted frames, synthetic masks
(both empty, one empty, single pixels at the borders, dense noise) and
disk-backed copies are scored with real-valued transition images containing
NaN. Result dictionaries must be equal in keys, values and types.
"""
import argparse, importlib, json, sys, time, types
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'


def same(a, b):
    return list(a) == list(b) and all(
        type(a[k]) is type(b[k]) and (a[k] == b[k] or (a[k] != a[k] and b[k] != b[k])) for k in a)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import tracking
    import motion_union_bounds as adapter
    package = types.ModuleType('candidate_adapters')
    package.__path__ = [str(a.adapters.resolve()), str(a.adapters.resolve().parent / 'speed_candidate')]
    sys.modules['candidate_adapters'] = package
    array_storage = importlib.import_module('candidate_adapters.array_storage')
    original = tracking.motion_support
    installed = ExitStack()
    stats = installed.enter_context(adapter.install(tracking, ENGINE / 'code'))
    replacement = tracking.motion_support
    rng = np.random.default_rng(20261020)
    stack = tifffile.imread(HISTORY / '11_issue017_rebase.tif')
    pairs = [(np.zeros((40, 50), bool), np.zeros((40, 50), bool))]
    for y, x in ((0, 0), (39, 49), (0, 49), (39, 0)):
        m = np.zeros((40, 50), bool); m[y, x] = True
        pairs += [(m, np.zeros((40, 50), bool)), (np.zeros((40, 50), bool), m), (m, m.T[:40, :50] if False else m)]
    for _ in range(200):
        pairs.append((rng.random((40, 50)) < rng.choice([.001, .05, .4]), rng.random((40, 50)) < rng.choice([.001, .05, .4])))
    for t in rng.choice(len(stack) - 1, 25, replace=False):
        for identity in rng.choice(np.unique(stack[int(t)])[1:], 15):
            pairs.append((stack[int(t)] == identity, stack[int(t) + 1] == identity))
    failures, cases = [], 0
    times = dict(original=0.0, adapted=0.0)
    for storage in ('plain', 'working'):
        with ExitStack() as scope:
            if storage == 'working':
                pool = array_storage.Pool(a.scratch, minimum_bytes=1 << 10, temporary_ram_bytes=0)
                scope.enter_context(array_storage.install(pool))
            for number, (earlier, later) in enumerate(pairs):
                lag = rng.normal(0, 2, earlier.shape).astype(np.float32)
                lag[rng.random(earlier.shape) < .03] = np.nan
                if storage == 'working':
                    earlier, later, lag = np.array(earlier), np.array(later), np.array(lag)
                t0 = time.perf_counter(); expected = original(earlier, later, lag)
                t1 = time.perf_counter(); actual = replacement(earlier, later, lag)
                t2 = time.perf_counter()
                times['original'] += t1 - t0; times['adapted'] += t2 - t1
                cases += 1
                if not same(expected, actual):
                    failures.append(f'{storage}-{number}')
    installed.close()
    report = dict(status='pass' if stats['supported'] and not failures else 'fail',
                  supported=stats['supported'], cases=cases, boxed=stats['boxed'], whole=stats['whole'],
                  seconds=times, failures=failures[:20])
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report, indent=1))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
