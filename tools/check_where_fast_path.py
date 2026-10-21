"""Compare three-argument np.where under the old and new storage, exactly.

Conditions and values as plain arrays, disk-backed working arrays and their
frame views, Python and NumPy scalars, lists, broadcastable and incompatible
shapes, booleans, integers, floats, complex and strings, either side of the
disk threshold: results must give the same array type, dtype, shape and bytes,
or the same exception.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time


def run(st, scratch, np, seed):
    rng = np.random.default_rng(seed)
    out = []
    pool = st.Pool(Path(scratch), minimum_bytes=1024**2, temporary_ram_bytes=32 * 1024**2)
    with st.install(pool):
        raw = rng.integers(0, 50, (24, 160, 170)).astype(np.uint16)
        floats = rng.random((24, 160, 170))
        stacks = [pool.array(raw, copy=True), pool.array(floats, copy=True)]
        assert all(isinstance(s, st.WorkingArray) for s in stacks)

        def operand(shape):
            choice = int(rng.integers(12))
            if choice == 0:
                return float(rng.random())
            if choice == 1:
                return int(rng.integers(-5, 50))
            if choice == 2:
                return np.float32(rng.random())
            if choice == 3:
                return rng.random(shape)
            if choice == 4:
                return rng.integers(0, 9, shape).astype(rng.choice([np.int8, np.uint16, np.int64]))
            if choice == 5:
                return rng.random(shape[-1:])
            if choice == 6:
                return rng.random((3, 5))
            if choice == 7:
                return (rng.random(shape) + 1j * rng.random(shape))
            if choice == 8:
                return np.full(shape, 'ab')
            if choice == 9:
                return list(range(shape[-1])) if shape else 3
            if choice == 10:
                return True
            stack = stacks[int(rng.integers(2))]
            if shape == stack.shape:
                return stack
            return stack[int(rng.integers(len(stack)))][:shape[0], :shape[1]] if len(shape) == 2 else rng.random(shape)

        for number in range(6000):
            size = int(rng.integers(4))
            shape = [(), (int(rng.integers(1, 60)),), (int(rng.integers(1, 160)), int(rng.integers(1, 170))),
                     (24, 160, 170)][size]
            if size == 3:
                condition = stacks[int(rng.integers(2))] > float(rng.random() * 25)
            elif size == 2 and rng.random() < 0.3:
                stack = stacks[int(rng.integers(2))]
                condition = stack[int(rng.integers(len(stack)))][:shape[0], :shape[1]] > 10
            elif rng.random() < 0.05:
                condition = bool(rng.integers(2))
            else:
                condition = rng.random(shape) < 0.5
            values = (operand(shape), operand(shape))
            try:
                result = np.where(condition, *values)
                data = np.asarray(result)
                out.append((type(result).__name__, data.dtype.str, data.shape,
                            hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()))
                del result, data
            except Exception as error:
                out.append(('error', type(error).__name__, str(error)))
        for _ in range(200):
            condition = rng.random((int(rng.integers(1, 40)), 7)) < 0.5
            try:
                result = np.where(condition)
                out.append(tuple(hashlib.sha256(np.asarray(r).tobytes()).hexdigest() for r in result))
            except Exception as error:
                out.append(('error', type(error).__name__, str(error)))
        del stacks
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
    with tempfile.TemporaryDirectory(prefix='where-old-') as one, tempfile.TemporaryDirectory(prefix='where-new-') as two:
        tick = time.perf_counter()
        expected = run(load(a.old.resolve()), one, np, 33)
        before = time.perf_counter() - tick
        tick = time.perf_counter()
        actual = run(load(a.new.resolve()), two, np, 33)
        after = time.perf_counter() - tick
    assert len(expected) == len(actual)
    for number, (e, f) in enumerate(zip(expected, actual)):
        assert e == f, (number, e, f)
    report = dict(status='pass', cases=len(expected), types=dict(collections.Counter(e[0] for e in expected)),
                  old_seconds=before, new_seconds=after)
    print(report)
    a.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
