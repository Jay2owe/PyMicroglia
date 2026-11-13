"""Compare the owner-presence rewrites with the original functions, exactly.

The terminal boundary partition (discover_and_apply) and the right-censored
reciprocal exchange (produce) run on their accepted A1 stage inputs, as plain
arrays and as memory-pool working arrays (how the benchmark holds the movie),
with and without owner_presence_frames; every returned array, table and summary
must be identical in type, dtype, shape and values. The single functions are
then compared directly: _presence for every owner of several real stacks plus
absent, zero, negative, very large, NumPy, float and text owners; the owner
table and _windows_from_labels on real stacks and synthetic movies (signed with
negatives, uint8, uint32 above the count width, uint64, single frames, empty,
two-dimensional and float stacks, which must run the original).
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'


def same(a, b):
    if type(a) is not type(b):
        return f'type {type(a).__name__} != {type(b).__name__}'
    if isinstance(a, np.ndarray):
        if a.dtype != b.dtype or a.shape != b.shape:
            return f'array {a.dtype}{a.shape} != {b.dtype}{b.shape}'
        return None if np.array_equal(np.asarray(a), np.asarray(b), equal_nan=a.dtype.kind in 'fc') else 'array values'
    if isinstance(a, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(a, b, check_exact=True)
        except AssertionError as error:
            return 'frame ' + str(error)[:200]
        return None if list(a.dtypes) == list(b.dtypes) else 'frame dtypes'
    if isinstance(a, dict):
        if list(a) != list(b):
            return 'dict keys'
        for key in a:
            problem = same(a[key], b[key])
            if problem:
                return f'{key}: {problem}'
        return None
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return 'length'
        for index, (x, y) in enumerate(zip(a, b)):
            problem = same(x, y)
            if problem:
                return f'[{index}] {problem}'
        return None
    if isinstance(a, set):
        return None if a == b and list(a) == list(b) else 'set'
    if isinstance(a, float):
        return None if np.float64(a).tobytes() == np.float64(b).tobytes() else 'float'
    return None if a == b else f'{a!r} != {b!r}'


def numpy_crash(stack, owner):
    """numpy 2.1 crashes comparing a strided integer array with an out-of-range int."""
    array = np.asarray(stack)
    if array.dtype.kind not in 'iu' or array.flags.c_contiguous or array.flags.f_contiguous:
        return False
    try:
        value = int(owner)
    except (TypeError, ValueError, OverflowError):
        return False
    limits = np.iinfo(array.dtype)
    return not limits.min <= value <= limits.max


def outcome(function, *args):
    try:
        return ('ok', function(*args))
    except Exception as error:
        return ('error', (type(error).__name__, str(error)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--part', choices=('all', 'stages', 'functions'), default='all',
                        help='run the stage comparisons and the single-function cases in separate processes')
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import tifffile
    import array_storage
    import terminal_boundary_vanished_seat as terminal
    import terminal_boundary_vanished_seat_integration as terminal_integration
    import right_censored_reciprocal_exchange as exchange
    import right_censored_reciprocal_exchange_integration as exchange_integration
    import continuity_confidence as confidence
    import retirement_aware_persistence
    import retirement_aware_persistence_integration
    import owner_presence_frames as adapter
    aliases = (retirement_aware_persistence, retirement_aware_persistence_integration)
    config = json.loads((ENGINE / 'config.json').read_text(encoding='utf-8'))
    a.scratch.mkdir(parents=True, exist_ok=True)
    pool = array_storage.Pool(a.scratch / 'pool')
    serial = [0]
    kept = []

    def pooled(stack):
        path = a.scratch / f'stack_{serial[0]}.dat'
        serial[0] += 1
        mapped = np.memmap(path, dtype=stack.dtype, mode='w+', shape=stack.shape)
        mapped[:] = stack
        mapped.flush()
        working = pool.adopt(mapped, path)
        assert type(working).__name__ == 'WorkingArray' and working._pool is pool
        # Held to the end: an adopted file is closed as soon as its array goes.
        kept.append(working)
        return working

    report = dict(failures=[], stages={}, seconds={}, cases=dict(presence=0, tables=0, windows=0))
    stats_seen = {}

    def fail(**item):
        report['failures'].append(item)

    def installed():
        return adapter.install(terminal, exchange, confidence, aliases)

    # Whole stages on their accepted inputs.
    raw = tifffile.imread(next((HISTORY / '07_trimmed_evidence/out').glob('*_registered_raw.tif')))
    points = pd.read_csv(HISTORY / '26_latent_body_tracks/out/latent_track_points.csv')
    thresholds = pd.read_csv(HISTORY / '25_raw_physical_hypotheses/out/frame_evidence_thresholds.csv')
    _, terminal_params = terminal_integration._configured(config)
    _, exchange_params = exchange_integration._configured(config)
    stages = (
        ('terminal_boundary_vanished_seat', '105_well_b3_issue016_body_scale_ownerless_allocation',
         lambda labels, unclaimed: terminal.discover_and_apply(labels, unclaimed, raw, points.copy(), thresholds.copy(), terminal_params)),
        ('right_censored_reciprocal_exchange', '93_well_b3_issue006_right_censored_seat_partition',
         lambda labels, unclaimed: exchange.produce(labels, unclaimed, points.copy(), exchange_params)),
    )
    if a.part == 'functions':
        stages = ()
    for name, stem, call in stages:
        labels = tifffile.imread(HISTORY / (stem + '.tif'))
        unclaimed = tifffile.imread(HISTORY / (stem + '_unclaimed.tif'))
        for form in ('plain', 'pool'):
            def inputs():
                if form == 'plain':
                    return labels.copy(), unclaimed.copy()
                return pooled(labels), pooled(unclaimed)
            t0 = time.perf_counter()
            expected = outcome(call, *inputs())
            t1 = time.perf_counter()
            with installed() as stats:
                assert name in stats['supported'], stats
                actual = outcome(call, *inputs())
            t2 = time.perf_counter()
            stats_seen[f'{name}:{form}'] = stats
            report['stages'][f'{name}:{form}'] = dict(status=expected[0], original_seconds=round(t1 - t0, 2),
                                                      adapted_seconds=round(t2 - t1, 2))
            problem = same(expected, actual)
            if problem:
                fail(stage=name, form=form, problem=problem)
        del labels, unclaimed

    if a.part == 'stages':
        report['status'] = 'pass' if not report['failures'] and len(report['stages']) == 4 else 'fail'
        report['part'] = a.part
        a.output.write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
        print(json.dumps(report, indent=1, default=str))
        return 0 if report['status'] == 'pass' else 1
    # Single functions on real stacks and synthetic movies.
    rng = np.random.default_rng(1024)
    real = {}
    for stem in ('105_well_b3_issue016_body_scale_ownerless_allocation', '93_well_b3_issue006_right_censored_seat_partition',
                 '01_issue005', '08_issue012'):
        real[stem] = tifffile.imread(HISTORY / (stem + '.tif'))
    signed = rng.integers(-3, 9, (12, 20, 30)).astype(np.int16)
    synthetic = {
        'signed_negatives': signed,
        'int8': rng.integers(-2, 5, (5, 7, 9)).astype(np.int8),
        'uint8': rng.integers(0, 6, (9, 16, 16)).astype(np.uint8),
        'uint32_wide': (rng.integers(0, 4, (6, 10, 12)) * 70001).astype(np.uint32),
        'uint64': rng.integers(0, 5, (6, 10, 12)).astype(np.uint64),
        'single_frame': rng.integers(0, 4, (1, 15, 15)).astype(np.uint16),
        'all_zero': np.zeros((4, 8, 8), np.uint16),
        'empty': np.zeros((0, 8, 8), np.uint16),
        'two_dimensional': rng.integers(0, 4, (10, 12)).astype(np.uint16),
        'float': rng.integers(0, 4, (5, 6, 7)).astype(np.float32),
        'noncontiguous': rng.integers(0, 6, (8, 30, 40)).astype(np.uint16)[:, ::2, 1::3],
    }
    stacks = [(name, stack) for name, stack in real.items()] + list(synthetic.items())
    stacks += [('pool:' + name, pooled(stack)) for name, stack in list(real.items())[:2]]
    stacks += [('pool:signed_negatives', pooled(signed))]
    original_presence = terminal._presence
    original_windows = confidence._windows_from_labels
    exchange_globals = vars(exchange)

    def original_table(labels):
        return {
            int(owner): set(map(int, frames))
            for owner, frames in ((owner, np.flatnonzero(np.any(
                labels == int(owner), axis=(1, 2))))
                for owner in np.unique(labels) if int(owner) > 0)}

    seconds = dict(presence_original=0.0, presence_adapted=0.0, table_original=0.0, table_adapted=0.0,
                   windows_original=0.0, windows_adapted=0.0)
    with installed() as stats:
        table = exchange_globals['_owner_presence_frames']
        shared = {}
        for name, stack in stacks:
            values = np.unique(np.asarray(stack)).tolist() if np.asarray(stack).size else []
            owners = [v for v in values if v][:60] + [0, -1, 10 ** 6, 2 ** 70, np.int64(3), np.uint16(2), 3.0, '2', 'x', True]
            if np.asarray(stack).ndim == 3 and np.asarray(stack).dtype.kind in 'iu' and np.asarray(stack).size:
                owners.append(int(np.asarray(stack).max()) + 1)
            for owner in owners:
                if numpy_crash(stack, owner):
                    report.setdefault('skipped_numpy_crash', []).append((name, repr(owner)))
                    continue
                t0 = time.perf_counter()
                expected = outcome(original_presence, stack, owner)
                t1 = time.perf_counter()
                actual = outcome(terminal._presence, stack, owner)
                seconds['presence_original'] += t1 - t0
                seconds['presence_adapted'] += time.perf_counter() - t1
                report['cases']['presence'] += 1
                problem = same(expected, actual)
                if problem:
                    fail(function='_presence', stack=name, owner=repr(owner), problem=problem)
                # Inside a discover_and_apply call on this very movie the per-call table answers.
                scope = terminal.discover_and_apply.presence_scope
                scope.append(shared.setdefault(name, [stack, None]))
                try:
                    scoped = outcome(terminal._presence, stack, owner)
                finally:
                    scope.pop()
                report['cases']['presence'] += 1
                problem = same(expected, scoped)
                if problem:
                    fail(function='_presence scoped', stack=name, owner=repr(owner), problem=problem)
            t0 = time.perf_counter()
            expected = outcome(original_table, stack)
            t1 = time.perf_counter()
            actual = outcome(table, stack)
            seconds['table_original'] += t1 - t0
            seconds['table_adapted'] += time.perf_counter() - t1
            report['cases']['tables'] += 1
            problem = same(expected, actual)
            if problem:
                fail(function='owner_presence', stack=name, problem=problem)
            t0 = time.perf_counter()
            expected = outcome(original_windows, stack)
            t1 = time.perf_counter()
            actual = outcome(confidence._windows_from_labels, stack)
            seconds['windows_original'] += t1 - t0
            seconds['windows_adapted'] += time.perf_counter() - t1
            report['cases']['windows'] += 1
            problem = same(expected, actual)
            if problem:
                fail(function='_windows_from_labels', stack=name, problem=problem)
        aliased = all(module._windows_from_labels is confidence._windows_from_labels for module in aliases)
        stats_seen['functions'] = dict(stats)
    restored = (terminal._presence is original_presence and not hasattr(terminal.discover_and_apply, '__wrapped__') and confidence._windows_from_labels is original_windows
                and '_owner_presence_frames' not in exchange_globals
                and all(module._windows_from_labels is original_windows for module in aliases))
    report['seconds'] = {key: round(value, 2) for key, value in seconds.items()}
    report['stats'] = stats_seen
    report['aliases_patched'] = aliased
    report['restored'] = restored
    used = stats_seen['functions']
    report['status'] = 'pass' if (not report['failures'] and aliased and restored
                                  and used['presence_chunked'] and used['presence_tabled'] and used['tables'] and used['windows']
                                  and used['presence_fallback'] and used['windows_fallback']) else 'fail'
    report['failures'] = report['failures'][:30]
    report['part'] = a.part
    a.output.write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
    print(json.dumps(report, indent=1, default=str))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
