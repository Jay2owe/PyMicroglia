"""Compare the packed-key unique of two-column integer arrays with np.unique(axis=0), exactly.

Eight integer dtypes; random, clustered, negative and extreme values (spans
that do and do not fit the packed key); real successive-frame node pairs from
an accepted label movie; with and without counts. Rows, counts, dtypes, shapes
and contiguity must match. The installed pool must route axis-0 two-column
calls (and only those) to the packed form.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    import tifffile
    from skimage.measure import label
    sys.path.insert(0, str(a.adapter.resolve()))
    import array_storage
    original = np.unique
    rng = np.random.default_rng(88)
    report = dict(status='pass', cases=0, routed=0, seconds=dict(numpy=0.0, packed=0.0))

    def compare(values, counts):
        t0 = time.perf_counter()
        expected = original(values, axis=0, return_counts=counts)
        t1 = time.perf_counter()
        actual = array_storage.pair_unique(values, original, return_counts=counts)
        t2 = time.perf_counter()
        report['seconds']['numpy'] += t1 - t0
        report['seconds']['packed'] += t2 - t1
        pairs = zip(expected, actual) if counts else [(expected, actual)]
        for e, f in pairs:
            assert e.dtype == f.dtype and e.shape == f.shape and np.array_equal(e, f), (values.dtype, values.shape)
            assert f.flags.c_contiguous == e.flags.c_contiguous
        report['cases'] += 1

    for dtype in (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64):
        info = np.iinfo(dtype)
        ranges = [(0, 3), (0, 50), (max(info.min, -40), min(info.max, 40)), (info.min, info.max)]
        if info.bits >= 32:
            ranges += [(0, 1 << 20), (max(info.min, -(1 << 31)), min(info.max, 1 << 31)), (info.max - 1000, info.max)]
        for low, high in ranges:
            for rows in (1, 2, 255, 256, 257, 5000, 200000):
                values = rng.integers(low, high, size=(rows, 2), endpoint=True, dtype=dtype)
                for counts in (False, True):
                    compare(values, counts)
            # Strided input (a column view of a wider array).
            wide = rng.integers(low, high, size=(3000, 3), endpoint=True, dtype=dtype)
            compare(wide[:, 1:], True)
            compare(np.asfortranarray(wide[:, :2]), False)
    # Real node pairs between successive frames, as the path linkers build them.
    movie = tifffile.imread(a.labels, key=range(0, 60))
    previous = None
    offset = 0
    for frame in movie:
        nodes = label(frame, background=0, connectivity=2).astype(np.int32)
        nodes[nodes > 0] += offset
        offset = int(nodes.max())
        if previous is not None:
            shared = (previous > 0) & (nodes > 0)
            values = np.column_stack((previous[shared], nodes[shared]))
            for counts in (False, True):
                compare(values, counts)
        previous = nodes
    # Routing through the installed pool.
    with tempfile.TemporaryDirectory() as folder:
        pool = array_storage.Pool(folder)
        calls = []
        packed = array_storage.pair_unique

        def counted(*args, **kwargs):
            calls.append(1)
            return packed(*args, **kwargs)
        array_storage.pair_unique = counted
        try:
            with array_storage.install(pool):
                values = rng.integers(0, 900, size=(4000, 2)).astype(np.int32)
                assert np.array_equal(np.unique(values, axis=0), original(values, axis=0))
                routed = len(calls)
                np.unique(values)  # flattened: not a pair call
                np.unique(values[:100], axis=0)  # below the row threshold
                np.unique(values.astype(float), axis=0)
                np.unique(rng.integers(0, 9, size=(4000, 3)), axis=0)
                np.unique(values, axis=1)
                np.unique(values, axis=0, return_index=True)
                assert len(calls) == routed == 1, calls
        finally:
            array_storage.pair_unique = packed
        report['routed'] = routed
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
