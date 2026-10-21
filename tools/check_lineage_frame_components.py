"""Compare region-object-free local-lineage frame components with the original.

Every frame of two accepted label stacks, read plain and from disk-backed
working arrays, plus synthetic frames with negative values, touching
identities and empty frames. Component numbers maps and component lists must
be identical.
"""
import argparse, json, sys, time
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
    import local_lineage_reservations as module
    import lineage_frame_components as adapter
    import array_storage
    original = module._frame_components
    failures, counts = [], dict(frames=0, components=0)
    times = dict(original=0.0, adapted=0.0)

    def compare(frame, label):
        t0 = time.perf_counter(); expected = original(frame)
        with adapter.install(module) as stats:
            assert stats['supported']
            t1 = time.perf_counter(); actual = module._frame_components(frame)
            t2 = time.perf_counter()
            assert stats['fallback'] == 0
        times['original'] += t1 - t0; times['adapted'] += t2 - t1
        counts['frames'] += 1; counts['components'] += len(expected.components)
        if not (np.array_equal(expected.numbers, actual.numbers)
                and expected.numbers.dtype == actual.numbers.dtype
                and expected.components == actual.components
                and all(type(getattr(e, f)) is type(getattr(c, f))
                        for e, c in zip(expected.components, actual.components)
                        for f in ('number', 'identity', 'area', 'centre_y', 'centre_x'))):
            failures.append(label)

    for name in ('01_issue005.tif', '11_issue017_rebase.tif'):
        stack = tifffile.imread(HISTORY / name)
        for t in range(len(stack)):
            compare(stack[t], f'{name}-{t}')
        with ExitStack() as scope:
            pool = array_storage.Pool(a.scratch, minimum_bytes=1 << 20, temporary_ram_bytes=32 << 20)
            scope.enter_context(array_storage.install(pool))
            working = np.array(stack)
            assert isinstance(working, array_storage.WorkingArray)
            for t in range(0, len(working), 5):
                compare(working[t], f'{name}-working-{t}')
    rng = np.random.default_rng(20261020)
    for case in range(500):
        shape = (int(rng.integers(1, 40)), int(rng.integers(1, 40)))
        frame = rng.integers(-2, 6, shape) * (rng.random(shape) < float(rng.choice([0, .2, .6, 1])))
        compare(frame.astype(rng.choice([np.int32, np.int64, np.uint16]) if frame.min() >= 0 else np.int32),
                f'synthetic-{case}')
    report = dict(status='pass' if not failures else 'fail', counts=counts, seconds=times,
                  failures=failures[:20])
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report, indent=1))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
