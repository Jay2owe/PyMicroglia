"""Compare Issue 017 locked restoration and foreground accounting with continuity_repair.

Real stage pairs from the accepted history (Issue 012 baseline against the
seat-identity, blob-residency and Issue 017 results) are restored under empty,
small, large and absent locked sets, and accounted against the baseline's
unclaimed stack, plus candidates edited to break the accounting (a baseline
cell removed without an unclaimed value, a negative label), mixed integer
types, disk-backed movies and non-integer input (which keeps the original).
Returned arrays (values, dtype, shape) and raised errors must be equal.
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'


def summary(value):
    if isinstance(value, tuple):
        return tuple(summary(part) for part in value)
    return (np.asarray(value).dtype.str, np.shape(value), np.asarray(value).tobytes())


def outcome(function, *args):
    try:
        return ('ok', summary(function(*args)))
    except Exception as error:
        return ('error', type(error).__name__, str(error))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=120)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import accepted_history as history
    import continuity_repair_frames as adapter
    module = history._issue_module(history.CONTINUITY_OPS, 'continuity_repair')
    restore, account = module._restore_locked, module._account_foreground
    n = a.frames
    baseline = tifffile.imread(HISTORY / '08_issue012.tif')[:n]
    unclaimed = tifffile.imread(HISTORY / '08_issue012_unclaimed.tif')[:n]
    candidates = [tifffile.imread(HISTORY / name)[:n] for name in
                  ('09_seat_identity.tif', '10_blob_residency.tif', '11_issue017_rebase.tif')]
    rng = np.random.default_rng(17)
    ids = np.unique(baseline)[1:]
    folder = Path(tempfile.mkdtemp())

    def disk(value, name):
        held = np.memmap(folder / f'{name}.dat', value.dtype, 'w+', shape=value.shape)
        held[:] = value
        return held
    failures, seconds, cases = [], dict(original=0.0, replaced=0.0), 0
    with adapter.install(module) as stats:
        assert stats['supported']
        new_restore, new_account = module._restore_locked, module._account_foreground

        def compare(kind, old, new, *args):
            nonlocal cases
            t0 = time.perf_counter(); e = outcome(old, *args)
            t1 = time.perf_counter(); f = outcome(new, *args)
            seconds['original'] += t1 - t0; seconds['replaced'] += time.perf_counter() - t1
            cases += 1
            if e != f:
                failures.append(dict(kind=kind, case=cases, expected=str(e)[:160], actual=str(f)[:160]))
        locked_sets = [set(), {int(ids[0])}, set(map(int, rng.choice(ids, 5, replace=False))),
                       set(map(int, rng.choice(ids, len(ids) // 2, replace=False))), {10 ** 6}]
        for number, candidate in enumerate(candidates):
            for locked in locked_sets:
                compare('restore', restore, new_restore, candidate, baseline, locked)
            compare('account', account, new_account, candidate, baseline, unclaimed)
            broken = candidate.copy()
            cell = int(rng.choice(np.unique(broken[n // 2])[1:]))
            broken[n // 2][broken[n // 2] == cell] = 0
            compare('account-broken', account, new_account, broken, baseline, np.zeros_like(unclaimed))
            negative = candidate.astype(np.int32)
            negative[3][np.nonzero(baseline[3])[0][0], np.nonzero(baseline[3])[1][0]] = -1
            compare('account-negative', account, new_account, negative, baseline, unclaimed)
            mixed = candidate.astype(np.int32)
            compare('restore-mixed', restore, new_restore, mixed, baseline, locked_sets[2])
            compare('account-mixed', account, new_account, mixed, baseline.astype(np.int32), unclaimed)
            if number == 0:
                d_candidate, d_baseline, d_unclaimed = (disk(candidate, 'c'), disk(baseline, 'b'),
                                                        disk(unclaimed, 'u'))
                compare('restore-disk', restore, new_restore, d_candidate, d_baseline, locked_sets[3])
                compare('account-disk', account, new_account, d_candidate, d_baseline, d_unclaimed)
                compare('restore-float', restore, new_restore, candidate.astype(np.float32), baseline,
                        locked_sets[1])
                compare('account-float', account, new_account, candidate.astype(np.float32), baseline,
                        unclaimed)
        counters = dict(stats)
    assert module._restore_locked is restore and module._account_foreground is account
    assert counters['restored'] > 0 and counters['accounted'] > 0 and counters['fallback'] > 0, counters
    report = dict(status='pass' if not failures else 'fail', frames=n, cases=cases, seconds=seconds,
                  counters=counters, failures=failures[:10])
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
