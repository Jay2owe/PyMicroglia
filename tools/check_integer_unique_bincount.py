"""Compare the storage adapter's integer unique (with its bincount route) with np.unique.

Random integer arrays of every integer dtype, with and without negative values,
small and large largest values (either side of the bincount bound), several
sizes (single chunk, many chunks, chunk boundaries), memory-mapped inputs, and
real label frames and a whole label movie: sorted values, their dtype and the
counts must equal NumPy's.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / ('development/cd68_full_red_green/rigid_analysis/A1/tracking/engine/'
                  'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--storage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    spec = importlib.util.spec_from_file_location('storage_under_check', a.storage)
    storage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(storage)
    original = np.unique
    rng = np.random.default_rng(20261020)
    cases = []
    for dtype in (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64):
        info = np.iinfo(dtype)
        for low, high in ((0, 5), (0, 417), (3, 9000), (0, (1 << 20) - 1), (0, 1 << 20), (-50, 50),
                          (0, 5_000_000), (info.max - 20, info.max), (info.min, info.min + 10)):
            low, high = max(low, info.min), min(high, info.max)
            if low > high:
                continue
            for size in (1, 7, 65_536, 1024 ** 2, 1024 ** 2 + 3, 2_500_000):
                values = rng.integers(low, high, size, dtype=dtype, endpoint=True)
                cases.append((f'{np.dtype(dtype).name}-{low}-{high}-{size}', values))
    stack = tifffile.imread(HISTORY / '10_blob_residency.tif')
    for t in range(0, len(stack), 25):
        cases.append((f'frame-{t}', stack[t]))
        cases.append((f'frame-int32-{t}', stack[t].astype(np.int32)))
    cases.append(('movie', stack))
    failures, routes = [], dict(bincount=0, other=0)
    seconds = dict(numpy=0.0, adapter=0.0)
    with tempfile.TemporaryDirectory() as scratch:
        mapped = np.memmap(Path(scratch) / 'labels.dat', dtype=stack.dtype, mode='w+', shape=stack[:40].shape)
        mapped[:] = stack[:40]
        cases.append(('memmap', mapped))
        for name, values in cases:
            for counts in (False, True):
                tick = time.perf_counter()
                expected = original(values, return_counts=counts)
                middle = time.perf_counter()
                actual = storage.integer_unique(values, original, return_counts=counts)
                seconds['numpy'] += middle - tick
                seconds['adapter'] += time.perf_counter() - middle
                pairs = zip(expected, actual) if counts else [(expected, actual)]
                if not all(e.dtype == f.dtype and e.shape == f.shape and np.array_equal(e, f) for e, f in pairs):
                    failures.append((name, counts))
            data = np.asarray(values)
            uses = (data.size and data.min() >= 0 and data.max() < storage.BINCOUNT_SPAN
                    and np.can_cast(data.dtype, np.intp, 'safe'))
            routes['bincount' if uses else 'other'] += 1
        total = len(cases)
        cases.clear()
        del values, data
        mapped._mmap.close()
        del mapped
    report = dict(status='pass' if not failures else 'fail', cases=total, routes=routes,
                  seconds=seconds, failures=failures[:20])
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
