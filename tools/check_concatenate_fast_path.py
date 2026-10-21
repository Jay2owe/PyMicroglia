"""Compare np.concatenate under the old and new working-array storage, exactly.

Random inputs of every numeric kind, mixed promotions, sizes either side of
the disk threshold, generators, scalars, bad axes and explicit dtypes must give
the same array type, dtype, values or exception under both storages.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time


def cases(np, rng):
    kinds = [np.bool_, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64,
             np.uint64, np.float16, np.float32, np.float64, np.complex64, np.complex128,
             np.clongdouble, np.dtype('U3'), object]
    for _ in range(3000):
        count = int(rng.integers(0, 4)) if rng.random() < 0.1 else int(rng.integers(1, 4))
        width = int(rng.choice([1, 3, 50, 4000, 70000, 140000, 300000]))
        ndim = int(rng.integers(1, 3))
        axis = rng.choice([0, 1, -1, -2, 2, None]) if rng.random() < 0.15 else (0 if ndim == 1 else int(rng.choice([0, 1])))
        rows = int(rng.integers(1, 4))
        family = kinds[:15] if rng.random() < 0.9 else kinds
        values = []
        for _ in range(count):
            kind = family[int(rng.integers(len(family)))] if rng.random() < 0.5 else family[11]
            if ndim == 1:
                shape = (int(rng.integers(1, width + 1)),)
            elif axis in (1, -1):
                shape = (rows, int(rng.integers(1, width + 1)))
            else:
                shape = (int(rng.integers(1, 4)), width)
            values.append((rng.random(shape) * 200).astype(kind))
        if rng.random() < 0.03:
            values.append(3)
        container = [list, tuple, iter][int(rng.choice(3, p=[0.6, 0.35, 0.05]))]
        dtype = None if rng.random() < 0.9 else np.dtype(kinds[int(rng.integers(len(kinds) - 2))])
        yield container, values, axis, dtype


def run(st, scratch, np, seed):
    rng = np.random.default_rng(seed)
    out = []
    pool = st.Pool(Path(scratch), minimum_bytes=1024**2, temporary_ram_bytes=32 * 1024**2)
    with st.install(pool):
        for container, values, axis, dtype in cases(np, rng):
            try:
                kwargs = {} if dtype is None else dict(dtype=dtype, casting='unsafe')
                result = np.concatenate(container(values), axis=axis, **kwargs)
                # Digest inside the pool: its mapped files close with it.
                data = result.tolist() if result.dtype.kind == 'O' else hashlib.sha256(result.tobytes()).hexdigest()
                out.append((type(result).__name__, result.dtype.str, result.shape, data))
                del result
            except Exception as error:
                out.append(('error', type(error).__name__, str(error)))
    pool.close()
    return out


def main():
    p = argparse.ArgumentParser()
    for name in ('old', 'new', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    sys.path.insert(0, str(Path(__file__).parent))
    from benchmark_working_storage import load
    with tempfile.TemporaryDirectory(prefix='concat-old-') as one, tempfile.TemporaryDirectory(prefix='concat-new-') as two:
        expected = run(load(a.old.resolve()), one, np, 9)
        actual = run(load(a.new.resolve()), two, np, 9)
        assert len(expected) == len(actual)
        for e, f in zip(expected, actual):
            assert e == f, (e[:3], f[:3])
        timing = {}
        for label, path, folder in (('old', a.old, one), ('new', a.new, two)):
            st = load(path.resolve())
            pool = st.Pool(Path(folder), minimum_bytes=1024**2)
            small = [np.arange(5), np.arange(7)]
            with st.install(pool):
                tick = time.perf_counter()
                for _ in range(100000):
                    np.concatenate(small)
                timing[label + '_microseconds'] = (time.perf_counter() - tick) * 10
            pool.close()
    report = dict(status='pass', cases=len(expected),
                  errors=sum(e[0] == 'error' for e in expected),
                  types=dict(collections.Counter(e[0] for e in expected)),
                  large_results=int(sum(e[0] != 'error' and np.prod(e[2]) * np.dtype(e[1]).itemsize >= 1024**2
                                    for e in expected)), **timing)
    print(report)
    a.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
