"""Compare joint-box census mask distances with the full-frame transform, exactly."""
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
    p.add_argument('--stack', default='10_blob_residency.tif')
    p.add_argument('--pairs', type=int, default=1500)
    a = p.parse_args()
    import numpy as np
    import tifffile
    sys.path[:0] = [str(a.adapter.resolve()), str((a.engine / 'code').resolve())]
    import accepted_history as history
    from array_bounds import occupied_bounds
    import census_distance_crop as crop
    module = history._issue_module(history.I017, 'm1_census')
    stack = tifffile.imread(a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history' / a.stack)
    rng = np.random.default_rng(17)
    cases = []
    while len(cases) < a.pairs:
        t, u = (int(v) for v in rng.integers(0, len(stack), 2))
        ids = np.unique(stack[t]); ids = ids[ids > 0]
        if not len(ids):
            continue
        identity = int(rng.choice(ids))
        other = identity if rng.random() < 0.7 else int(rng.integers(0, 450))
        cases.append((stack[t] == identity, stack[u] == other))
    edge = np.zeros((40, 50), bool); edge[0, 0] = True
    far = np.zeros((40, 50), bool); far[-1, -1] = True
    cases += [(edge, far), (far, edge), (edge, edge), (edge, np.zeros_like(edge)), (np.zeros_like(edge), far)]
    original = module.mask_distance
    tick = time.perf_counter(); expected = [original(x, y) for x, y in cases]; before = time.perf_counter() - tick
    with crop.install(module, occupied_bounds) as stats:
        assert stats['supported']
        tick = time.perf_counter(); actual = [module.mask_distance(x, y) for x, y in cases]; after = time.perf_counter() - tick
    same = all((e == f) or (e != e and f != f) for e, f in zip(expected, actual))
    assert same
    report = dict(status='pass', cases=len(cases), before_seconds=before, after_seconds=after, counters=stats)
    print(report)
    a.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
