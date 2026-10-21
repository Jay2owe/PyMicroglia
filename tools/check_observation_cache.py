"""Compare cached frame observations with the frozen observation reader.

Real label and raw frames (plain and disk-backed) are measured by the reader
and by the cache, repeatedly, with and without shapes and per-cell lookups,
after edits to a label frame, and after callers edit returned rows. Every row
must hold the same keys in the same order and equal values of the same types.
"""
import argparse, importlib.util, json, sys, time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'
RAW = HISTORY / '07_trimmed_evidence/out/95_full_A1_registered_raw.tif'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def same(expected, actual, label_mask):
    if list(expected) != list(actual):
        return 'local ids'
    for local_id, left in expected.items():
        right = actual[local_id]
        if list(left) != list(right):
            return f'{local_id} keys'
        for field, a in left.items():
            b = right[field]
            if field == 'mask':
                if not isinstance(b, label_mask) or b.identity != a.identity \
                        or not np.array_equal(np.asarray(a), np.asarray(b)):
                    return f'{local_id} mask'
            elif isinstance(a, np.ndarray):
                if type(a) is not type(b) or a.dtype != b.dtype or not np.array_equal(a, b):
                    return f'{local_id} {field}'
            elif type(a) is not type(b) or a != b:
                return f'{local_id} {field}'
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import tracking
    import array_storage
    frozen = load(a.snapshot / 'pymicroglia/tracking/frozen_io.py', 'frozen_io_check')
    cache_module = load(a.adapters / 'observation_cache.py', 'observation_cache_check')
    labels_stack = tifffile.imread(HISTORY / '11_issue017_rebase.tif')
    raw_stack = tifffile.imread(RAW)
    rng = np.random.default_rng(20261020)
    frames = sorted(rng.choice(len(labels_stack), 40, replace=False).tolist())
    failures, calls = [], 0
    times = dict(reader=0.0, cache=0.0)
    for storage in ('plain', 'working'):
        with ExitStack() as scope:
            labels, raw, copy = labels_stack, raw_stack, None
            if storage == 'working':
                pool = array_storage.Pool(a.scratch, minimum_bytes=1 << 20, temporary_ram_bytes=32 << 20)
                scope.enter_context(array_storage.install(pool))
                labels, raw, copy = np.array(labels_stack), np.array(raw_stack), pool.snapshot
            reader = frozen.observation_reader(tracking.frame_observations, copy)
            cache = cache_module.ObservationCache(reader, frozen.LabelMask)

            def check(frame_labels, frame_raw, args, case):
                nonlocal calls
                t0 = time.perf_counter(); expected = reader(frame_labels, frame_raw, *args)
                t1 = time.perf_counter(); actual = cache(frame_labels, frame_raw, *args)
                t2 = time.perf_counter()
                times['reader'] += t1 - t0; times['cache'] += t2 - t1
                calls += 1
                problem = same(expected, actual, frozen.LabelMask)
                if problem:
                    failures.append(f'{storage}-{case}: {problem}')
                return actual

            for repeat in range(3):
                for t in frames:
                    ids = np.unique(np.asarray(labels_stack[t]))[1:]
                    lookups = (
                        {int(i): {int(rng.integers(9))} for i in ids[::3]},
                        {int(i): int(rng.integers(1, 50)) for i in ids[::4]},
                        {int(i): bool(rng.integers(2)) for i in ids[::5]})
                    args = [(), lookups, (None, None, None, False)][repeat]
                    rows = check(labels[t], raw[t], args, f'{repeat}-{t}')
                    for row in rows.values():
                        row['position'][0] += 5.0
                        row['mask'] = None
                        if row['shape'] is not None:
                            row['shape'][:] = -1
                        row['green_track_ids'].add(-1)
                    check(labels[t], raw[t], args, f'{repeat}-{t}-after-edit')
            for t in frames[:10]:
                edited = np.array(labels_stack[t])
                ids = np.unique(edited)[1:]
                edited[edited == ids[0]] = ids[-1]
                check(edited, raw[t], (), f'edited-{t}')
                edited[:5, :5] = 0
                check(edited, raw[t], (), f'edited-corner-{t}')
                check(labels[t], raw[t] + 1 if storage == 'plain' else np.asarray(raw[t]) + 1, (), f'raw-{t}')
            check(np.zeros_like(labels_stack[0]), raw[0], (), 'empty')
            check(np.zeros_like(labels_stack[0]), raw[0], (), 'empty-again')
            failures.extend(f'{storage}-stats {cache.stats}' for _ in range(0 if cache.stats['hits'] else 1))
    report = dict(status='pass' if not failures else 'fail', calls=calls, seconds=times,
                  failures=failures[:20])
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report, indent=1))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
