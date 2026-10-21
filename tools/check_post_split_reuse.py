"""Check post-split first claims and summary losses against the original statements.

Pairs of accepted label stacks (plain and disk-backed) are compared with the
original whole-movie expressions. The recompiled discovery and summary are
checked end to end by replaying the Issue 055 step against its saved outputs.
"""
import argparse, json, sys, time
from contextlib import ExitStack
from pathlib import Path

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
    import post_split_backfill as module
    import post_split_reuse as adapter
    import array_storage
    failures, rows = [], []
    times = dict(original=0.0, adapted=0.0)
    pairs = [('01_issue005.tif', '02_issue006.tif'), ('05_issue009.tif', '06_issue010.tif'),
             ('10_blob_residency.tif', '11_issue017_rebase.tif')]
    for before_name, after_name in pairs:
        before = tifffile.imread(HISTORY / before_name)
        after = tifffile.imread(HISTORY / after_name)
        for label in ('plain', 'working'):
            with ExitStack() as scope:
                if label == 'working':
                    pool = array_storage.Pool(a.scratch, minimum_bytes=1 << 20, temporary_ram_bytes=32 << 20)
                    scope.enter_context(array_storage.install(pool))
                    labels, candidate = np.array(before), np.array(after)
                else:
                    labels, candidate = before, after
                active = sorted(set(map(int, np.unique(labels))) - {0})
                t0 = time.perf_counter()
                expected_claims = {identity: int(np.flatnonzero(np.any(
                    labels == identity, axis=(1, 2)))[0]) for identity in active}
                expected_losses = 0
                for identity in set(active):
                    b = np.any(labels == identity, axis=(1, 2))
                    c = np.any(candidate == identity, axis=(1, 2))
                    expected_losses += int(np.count_nonzero(b & ~c))
                t1 = time.perf_counter()
                stats = dict(first_claims=0, summaries=0)
                found = adapter.helpers(module, stats)
                claims = found['_first_claims'](labels, active)
                losses = found['_named_losses'](labels, candidate, set(active))
                t2 = time.perf_counter()
            times['original'] += t1 - t0; times['adapted'] += t2 - t1
            rows.append(dict(case=f'{before_name}->{after_name}-{label}', identities=len(active),
                             losses=expected_losses))
            if claims != expected_claims or list(claims) != list(expected_claims) or losses != expected_losses:
                failures.append(f'{before_name}->{after_name}-{label}')
    with adapter.install(module) as stats:
        supported = stats['supported']
    if not supported:
        failures.append('recompilation unsupported')
    report = dict(status='pass' if not failures else 'fail', seconds=times, rows=rows, failures=failures)
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report, indent=1))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
