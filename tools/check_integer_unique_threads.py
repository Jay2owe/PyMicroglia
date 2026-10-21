"""Compare the threaded whole-array integer histogram with np.unique, exactly.

Arrays below and above the threading threshold, five integer dtypes, negative
and wide ranges, with and without counts; values and dtypes must match.
"""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    sys.path.insert(0, str(a.adapter.resolve()))
    import array_storage
    rng = np.random.default_rng(1)
    cases = 0
    for dtype in (np.uint8, np.uint16, np.int32, np.int64, np.uint32):
        for size in (1000, array_storage.PARALLEL_ELEMENTS + 13, 40 * 1024**2 + 7):
            for low, high in ((0, 50), (0, 3000), (-5, 200), (0, (1 << 20) + 5)):
                if np.iinfo(dtype).min > low or np.iinfo(dtype).max < high:
                    continue
                values = rng.integers(low, high, size=size).astype(dtype)
                for counts in (False, True):
                    expected = np.unique(values, return_counts=counts)
                    actual = array_storage.integer_unique(values, np.unique, return_counts=counts)
                    pairs = zip(expected, actual) if counts else [(expected, actual)]
                    assert all(np.array_equal(e, f) and e.dtype == f.dtype for e, f in pairs), (dtype, size, low, high)
                    cases += 1
    movie = rng.integers(0, 3000, size=(1501, 512, 512)).astype(np.int32)
    tick = time.perf_counter()
    expected = np.unique(movie)
    numpy_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    actual = array_storage.integer_unique(movie, np.unique)
    threaded_seconds = time.perf_counter() - tick
    assert np.array_equal(expected, actual) and expected.dtype == actual.dtype
    report = dict(status='pass', cases=cases + 1, movie_shape=list(movie.shape),
                  numpy_unique_seconds=numpy_seconds, threaded_seconds=threaded_seconds)
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
