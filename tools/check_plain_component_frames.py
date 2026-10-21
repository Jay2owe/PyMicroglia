"""Compare component tables of disk-backed frames through plain views.

Frames of two accepted stacks are read from disk-backed working arrays, and
synthetic integer frames from mapped files, with the engine's own function and
through the plain-view wrapper. Component maps, owners and areas must match in
values, dtypes and array types.
"""
import argparse, json, sys, tempfile, time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import component_accounting as accounting
    import plain_component_frames as adapter
    import array_storage
    original = accounting.label_value_components
    failures, counts = [], dict(frames=0, plain_views=0)
    times = dict(original=0.0, adapted=0.0)

    def compare(frame, label):
        t0 = time.perf_counter(); expected = original(frame)
        with adapter.install(accounting) as stats:
            t1 = time.perf_counter(); actual = accounting.label_value_components(frame)
            t2 = time.perf_counter()
            counts['plain_views'] += stats['plain_views']
        assert accounting.label_value_components is original
        times['original'] += t1 - t0; times['adapted'] += t2 - t1
        counts['frames'] += 1
        for e, c in zip(expected, actual):
            if not (type(e) is type(c) and e.dtype == c.dtype and e.shape == c.shape
                    and np.array_equal(e, c)):
                failures.append(label)
                break

    with ExitStack() as scope:
        pool = array_storage.Pool(a.scratch, minimum_bytes=1 << 20, temporary_ram_bytes=32 << 20)
        scope.enter_context(array_storage.install(pool))
        for name in ('01_issue005.tif', '11_issue017_rebase.tif'):
            working = np.array(tifffile.imread(HISTORY / name))
            assert isinstance(working, array_storage.WorkingArray)
            for t in range(len(working)):
                compare(working[t], f'{name}-{t}')
    rng = np.random.default_rng(20261020)
    folder = Path(tempfile.mkdtemp(dir=a.scratch))
    for case in range(300):
        shape = (int(rng.integers(1, 40)), int(rng.integers(1, 40)))
        dtype = rng.choice([np.uint8, np.uint16, np.int32, np.int64])
        values = rng.integers(0, 6, shape) * (rng.random(shape) < float(rng.choice([0, .3, .8])))
        mapped = np.memmap(folder / f'{case}.dat', dtype=dtype, mode='w+', shape=shape)
        mapped[:] = values
        compare(mapped, f'synthetic-{case}')
        compare(np.ascontiguousarray(values.astype(dtype)), f'synthetic-plain-{case}')
        del mapped
    report = dict(status='pass' if not failures else 'fail', counts=counts, seconds=times,
                  failures=failures[:20])
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report, indent=1))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
