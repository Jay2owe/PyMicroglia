"""Check that reusing released working-array files changes no value, only the file lifecycle.

The same random sequence of storage-hooked operations (copies, casts, ufuncs,
zeros/ones/full/empty-then-fill, where, concatenate, stacking, explicit deletes
and kept aliases) runs under a pool that deletes released files and a pool
that keeps them for reuse; every result must be equal in type, dtype, shape,
layout and values. Separately: a reused buffer reads as zeros (zeros, a
zero-skipping fill and a fresh ufunc output), a live view blocks reuse, only an
exact byte size is reused, the retained total never exceeds the limit, and
close() and collection remove every file.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import tempfile
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--storage', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=4000)
    a = p.parse_args()
    import numpy as np
    from benchmark_working_storage import load
    array_storage = load(a.storage.resolve() / 'array_storage.py')
    Pool, install, WorkingArray = array_storage.Pool, array_storage.install, array_storage.WorkingArray

    def sequence(pool, seed):
        rng = np.random.default_rng(seed)
        results, kept = [], []
        with install(pool):
            for step in range(a.steps):
                shape = [(6, 20, 30), (6, 20, 30), (3, 40, 30), (12, 10, 10)][int(rng.integers(0, 4))]
                dtype = [np.uint16, np.int32, np.float32, bool][int(rng.integers(0, 4))]
                kind = int(rng.integers(0, 9))
                base = (rng.integers(0, 50, shape) if dtype is not bool else rng.random(shape) < .4).astype(dtype)
                if kind == 0:
                    value = np.array(base)
                elif kind == 1:
                    value = np.array(base).copy()
                elif kind == 2:
                    value = np.array(base).astype(np.float64) / 3
                elif kind == 3:
                    value = np.zeros(shape, dtype)
                elif kind == 4:
                    value = np.empty(shape, dtype); value[...] = base
                elif kind == 5:
                    value = np.full(shape, 7, dtype)
                elif kind == 6:
                    value = np.where(np.array(base) > 0, np.array(base), 0)
                elif kind == 7:
                    value = np.add(np.array(base), np.ones(shape, dtype))
                else:
                    value = np.concatenate([np.array(base)[:1], np.zeros_like(base)[1:]], axis=0)
                results.append((type(value).__name__, value.dtype.str, value.shape,
                                bool(value.flags.c_contiguous), np.asarray(value).tobytes()))
                if rng.random() < .1:
                    kept.append(np.asarray(value)[:1])  # a live alias keeps the file
                if len(kept) > 8:
                    kept.pop(0)
                del value
                if step % 50 == 0:
                    gc.collect()
        del kept
        gc.collect()
        return results

    report = dict(status='pass')
    with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
        plain = Pool(left, minimum_bytes=1024)
        reusing = Pool(right, minimum_bytes=1024, released_bytes=4 * 6 * 20 * 30 * 8)
        tick = time.perf_counter()
        expected = sequence(plain, 9)
        middle = time.perf_counter()
        actual = sequence(reusing, 9)
        end = time.perf_counter()
        assert expected == actual
        assert reusing.reused_allocations > 0 and plain.reused_allocations == 0
        assert reusing.released_bytes <= reusing.released_limit
        report.update(steps=a.steps, reused=reusing.reused_allocations, allocations=reusing.allocations,
                      seconds=dict(deleting=middle - tick, reusing=end - middle))
        plain.close(); reusing.close(); gc.collect()
        assert not list(Path(left).iterdir()) and not list(Path(right).iterdir())

    with tempfile.TemporaryDirectory() as folder:
        movie = (8, 64, 64)
        pool = Pool(folder, minimum_bytes=1024, released_bytes=2 * 8 * 64 * 64 * 4)
        with install(pool):
            first = np.full(movie, 5, np.int32)
            path = first.filename
            del first; gc.collect()
            assert len(pool.released) == 1 and Path(path).exists()
            again = np.zeros(movie, np.int32)                 # zero fill skipped for files
            assert pool.reused_allocations == 1 and again.filename == path
            assert isinstance(again, WorkingArray) and not np.asarray(again).any()
            again[...] = 9; del again; gc.collect()
            empty = np.empty(movie, np.int32)
            assert not np.asarray(empty).any(); empty[...] = 3
            view = np.asarray(empty)[2:]; del empty; gc.collect()
            assert not pool.released                        # the live view keeps it
            other = np.zeros((8, 64, 64), np.int16)         # a different byte size
            assert other.filename != path
            del view; gc.collect()
            assert len(pool.released) == 1
            same_size = np.ones((4, 128, 64), np.int32)   # same bytes, other shape
            assert same_size.filename == path and np.asarray(same_size).all()
            del same_size, other; gc.collect()
            out = np.add(np.array(np.ones(movie, np.int32)), 0)
            assert np.asarray(out).sum() == np.prod(movie)
            del out; gc.collect()
            assert pool.released_bytes <= pool.released_limit
        pool.close(); gc.collect()
        assert not list(Path(folder).iterdir())
    report['lifecycle'] = 'pass'
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
