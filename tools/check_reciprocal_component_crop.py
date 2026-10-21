"""Compare cropped pair components with reciprocal_lineages.frame_components, exactly."""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--stack', default='11_issue017_rebase.tif')
    p.add_argument('--pairs', type=int, default=4000)
    p.add_argument('--working-arrays', type=Path, default=None,
                   help='scratch folder: read frames from disk-backed working arrays')
    a = p.parse_args()
    import numpy as np
    import tifffile
    sys.path[:0] = [str(a.adapter.resolve()), str((a.engine / 'code').resolve())]
    import reciprocal_lineages as module
    from array_bounds import occupied_bounds
    import reciprocal_component_crop as crop
    stack = tifffile.imread(a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history' / a.stack)
    rng = np.random.default_rng(5)
    original = module.frame_components
    cases = []
    for _ in range(a.pairs):
        t = int(rng.integers(0, len(stack)))
        present = np.unique(stack[t]); present = present[present > 0]
        if rng.random() < 0.1 or len(present) < 2:
            pair = (int(rng.integers(1, 500)), int(rng.integers(1, 500)))  # absent or partly absent
        else:
            pair = tuple(int(v) for v in rng.choice(present, 2, replace=False))
        cases.append((t, pair, int(rng.choice([1, 5, 20, 60]))))
    tick = time.perf_counter()
    expected = [original(stack[t], pair, t + 1, m) for t, pair, m in cases]
    before = time.perf_counter() - tick
    from contextlib import ExitStack
    with ExitStack() as scope:
        if a.working_arrays is not None:
            import array_storage
            pool = array_storage.Pool(a.working_arrays, minimum_bytes=1 << 20, temporary_ram_bytes=32 << 20)
            scope.enter_context(array_storage.install(pool))
            stack = np.array(stack)
            assert isinstance(stack, array_storage.WorkingArray)
        stats = scope.enter_context(crop.install(module, occupied_bounds))
        assert stats['supported']
        tick = time.perf_counter()
        actual = [module.frame_components(stack[t], pair, t + 1, m) for t, pair, m in cases]
        after = time.perf_counter() - tick
    assert stats['fallback_calls'] == 0
    assert expected == actual
    assert all(type(e[k]) is type(f[k]) for E, A in zip(expected, actual) for e, f in zip(E, A) for k in e)
    report = dict(status='pass', cases=len(cases), components=sum(map(len, expected)),
                  before_seconds=before, after_seconds=after, counters=stats)
    print(report)
    a.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
