"""Compare substantial-soma separation with the boxed copy.

Frame ranges of real accepted stacks (plain and disk-backed), plus synthetic
frames with negative values, detached fragments and several bodies per
identity, are separated by the engine function and the adapted copy at two
core sizes. Object stacks and tables must be equal.
"""
import argparse, json, sys, time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'


def synthetic(rng):
    frames = np.zeros((6, 60, 70), np.int32)
    for t in range(len(frames)):
        for identity in range(1, 6):
            for body in range(int(rng.integers(1, 4))):
                y, x = rng.integers(0, 52), rng.integers(0, 62)
                size = int(rng.integers(1, 8))
                frames[t, y:y + size, x:x + size] = identity
        frames[t, 55:58, 2:5] = -3
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import global_soma_ledger as module
    import soma_separation_boxes as adapter
    import importlib, types
    package = types.ModuleType('candidate_adapters')
    package.__path__ = [str(a.adapters.resolve()), str(a.adapters.resolve().parent / 'speed_candidate')]
    sys.modules['candidate_adapters'] = package
    array_storage = importlib.import_module('candidate_adapters.array_storage')
    original = module.separate_substantial_soma_objects
    failures, rows = [], []
    times = dict(original=0.0, adapted=0.0)

    def compare(labels, core, label):
        t0 = time.perf_counter()
        try:
            expected = original(labels, core)
        except AssertionError as error:
            expected = error
        with adapter.install(module) as stats:
            assert stats['supported']
            t1 = time.perf_counter()
            try:
                actual = module.separate_substantial_soma_objects(labels, core)
            except AssertionError as error:
                actual = error
            t2 = time.perf_counter()
        assert module.separate_substantial_soma_objects is original
        times['original'] += t1 - t0; times['adapted'] += t2 - t1
        if isinstance(expected, AssertionError) or isinstance(actual, AssertionError):
            ok = type(expected) is type(actual) and str(expected) == str(actual)
        else:
            ok = (np.array_equal(expected[0], actual[0]) and expected[0].dtype == actual[0].dtype
                  and expected[1].equals(actual[1]) and list(expected[1].dtypes) == list(actual[1].dtypes))
        rows.append(dict(case=label, ok=bool(ok), boxed=stats['boxed'], whole=stats['whole']))
        if not ok:
            failures.append(label)

    rng = np.random.default_rng(20261020)
    for case in range(30):
        compare(synthetic(rng), int(rng.choice([1, 4, 12])), f'synthetic-{case}')
    for name, start in (('01_issue005.tif', 0), ('06_issue010.tif', 200), ('11_issue017_rebase.tif', 400)):
        stack = tifffile.imread(HISTORY / name)[start:start + 40]
        for core in (12, 30):
            compare(stack, core, f'{name}-{start}-{core}-plain')
        with ExitStack() as scope:
            pool = array_storage.Pool(a.scratch, minimum_bytes=1 << 20, temporary_ram_bytes=32 << 20)
            scope.enter_context(array_storage.install(pool))
            compare(np.array(stack), 12, f'{name}-{start}-12-working')
    report = dict(status='pass' if not failures else 'fail', seconds=times,
                  real=[r for r in rows if not r['case'].startswith('synthetic')],
                  synthetic=sum(r['case'].startswith('synthetic') for r in rows), failures=failures)
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report, indent=1))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
