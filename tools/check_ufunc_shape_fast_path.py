"""Compare ufuncs on working arrays under the old and new storage, exactly.

Whole stacks and frame views of disk-backed working arrays meet arrays of the
same shape, broadcastable and incompatible shapes, Python and NumPy scalars,
out= and dtype= arguments; results either side of the disk threshold must give
the same array type, dtype, shape and bytes, or the same exception.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

UFUNCS = ('add', 'subtract', 'multiply', 'true_divide', 'equal', 'less_equal', 'maximum',
          'logical_and', 'sqrt', 'abs', 'log2', 'hypot')


def run(st, scratch, np, seed):
    rng = np.random.default_rng(seed)
    out = []
    pool = st.Pool(Path(scratch), minimum_bytes=1024**2, temporary_ram_bytes=32 * 1024**2)
    with st.install(pool):
        raw = rng.integers(0, 50, (24, 160, 170)).astype(np.uint16)
        floats = rng.random((24, 160, 170))
        stacks = [pool.array(raw, copy=True), pool.array(floats, copy=True)]
        assert all(isinstance(s, st.WorkingArray) for s in stacks)
        for _ in range(4000):
            stack = stacks[int(rng.integers(2))]
            view = stack if rng.random() < 0.15 else stack[int(rng.integers(len(stack)))]
            if rng.random() < 0.1:
                view = view[..., 10:90]
            name = UFUNCS[int(rng.integers(len(UFUNCS)))]
            ufunc = getattr(np, name)
            choice = int(rng.integers(7))
            if choice == 0:
                other = rng.random(view.shape)
            elif choice == 1:
                other = float(rng.random())
            elif choice == 2:
                other = np.uint16(int(rng.integers(0, 50)))
            elif choice == 3:
                other = rng.random(view.shape[-1:])
            elif choice == 4:
                other = rng.random((3, 5))
            elif choice == 5:
                other = view
            else:
                other = int(rng.integers(0, 50))
            kwargs = {}
            if rng.random() < 0.05:
                kwargs['dtype'] = np.float64
            try:
                if ufunc.nin == 1:
                    result = ufunc(view, **kwargs)
                elif rng.random() < 0.05 and np.shape(other) in ((), view.shape):
                    target = np.empty(view.shape, np.result_type(view, other))
                    result = ufunc(view, other, out=target, **kwargs)
                else:
                    result = ufunc(view, other, **kwargs) if rng.random() < 0.8 else ufunc(other, view, **kwargs)
                data = np.asarray(result)
                out.append((type(result).__name__, data.dtype.str, data.shape,
                            hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()))
                del result, data
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
    with tempfile.TemporaryDirectory(prefix='ufunc-old-') as one, tempfile.TemporaryDirectory(prefix='ufunc-new-') as two:
        tick = time.perf_counter()
        expected = run(load(a.old.resolve()), one, np, 21)
        before = time.perf_counter() - tick
        tick = time.perf_counter()
        actual = run(load(a.new.resolve()), two, np, 21)
        after = time.perf_counter() - tick
    assert len(expected) == len(actual)
    for e, f in zip(expected, actual):
        assert e == f, (e, f)
    report = dict(status='pass', cases=len(expected), types=dict(collections.Counter(e[0] for e in expected)),
                  old_seconds=before, new_seconds=after)
    print(report)
    a.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
