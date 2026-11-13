"""Check integer_unique's background-remainder counting against np.unique, exactly.

Mostly-zero, dense and mixed label arrays of each integer dtype the bounded
histogram accepts, small and above the parallel threshold, with and without
counts; the real accepted movie; and the same movie as a disk-backed working
array under a storage pool. Values, counts and their dtypes must be equal.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
MOVIE = (ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine/m22_accepted_history'
         / 'pymicroglia_95_full_A1/mid/accepted_history/11_issue017_rebase.tif')


def same(expected, actual):
    if isinstance(expected, tuple):
        return all(same(x, y) for x, y in zip(expected, actual)) and len(expected) == len(actual)
    actual = np.asarray(actual).view(np.ndarray)
    return expected.dtype == actual.dtype and np.array_equal(expected, actual)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    a = parser.parse_args()
    sys.path.insert(0, str(a.adapters.resolve()))
    import array_storage as storage
    rng = np.random.default_rng(20261022)
    failures, cases = [], 0

    def check(label, data):
        nonlocal cases
        for counts in (False, True):
            expected = np.unique(np.asarray(data).view(np.ndarray), return_counts=counts)
            actual = storage.integer_unique(data, np.unique, return_counts=counts)
            cases += 1
            if not same(expected, actual):
                failures.append(dict(case=label, counts=counts))

    for dtype in (np.uint8, np.uint16, np.int16, np.int32, np.uint32, np.int64):
        for size in (1, 5, 1000, (1 << 20) + 7, storage.PARALLEL_ELEMENTS + 3):
            high = min(np.iinfo(dtype).max, 3000)
            for fraction in (0.0, 0.02, 0.49, 0.51, 0.97, 1.0):
                data = np.zeros(size, dtype)
                occupied = rng.random(size) < fraction
                data[occupied] = rng.integers(1, high + 1, int(occupied.sum()))
                check(f'{np.dtype(dtype).name} size={size} occupied={fraction}', data)
            # Halves of different density so chunks take both branches.
            data = np.zeros(size, dtype)
            data[size // 2:] = rng.integers(0, high + 1, size - size // 2)
            check(f'{np.dtype(dtype).name} size={size} half dense', data)
    movie = tifffile.imread(MOVIE)
    t0 = time.perf_counter()
    check('accepted movie', movie)
    seconds = time.perf_counter() - t0
    for t in range(0, len(movie), 59):
        check(f'accepted frame {t}', movie[t])
    pool = storage.Pool(a.scratch, minimum_bytes=1 << 10, temporary_ram_bytes=1 << 20)
    with storage.install(pool):
        working = np.array(movie)
        check('accepted movie as working array', working)
        check('working frame', working[200])
    report = dict(status='pass' if not failures else 'fail', cases=cases,
                  movie_seconds_both_calls=round(seconds, 3), failures=failures[:20])
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
