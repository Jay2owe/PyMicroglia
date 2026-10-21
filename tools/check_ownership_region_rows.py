"""Compare find_objects ownership observations with the regionprops original, exactly.

Real frames of three accepted-history stacks with and without masks, plus
synthetic frames with touching identities, diagonal links and empty frames:
every observation's identity, area, centroid bits, pixel coordinates and mask
must match, through the history loader as well.
"""
import argparse
import json
from pathlib import Path
import sys
import time


def flatten(rows):
    import numpy as np
    out = []
    for row in rows:
        out.append((row.identity, row.area, np.asarray(row.centroid).dtype.str, np.asarray(row.centroid).tobytes(),
                    row.pixels.dtype.str, row.pixels.shape, row.pixels.tobytes(),
                    None if row.mask is None else (row.mask.dtype.str, row.mask.shape, row.mask.tobytes())))
    return out


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--frames', type=int, default=500)
    a = p.parse_args()
    import numpy as np
    import tifffile
    sys.path[:0] = [str(a.adapter.resolve()), str((a.engine / 'code').resolve())]
    import accepted_history as history
    import ownership_region_rows as rows
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    stacks = [tifffile.imread(folder / name) for name in ('07_issue011.tif', '08_issue012.tif', '10_blob_residency.tif')]
    rng = np.random.default_rng(12)
    cases = []
    for _ in range(a.frames):
        s = int(rng.integers(len(stacks)))
        cases.append((stacks[s][int(rng.integers(len(stacks[s])))], bool(rng.integers(2))))
    for number in range(200):
        frame = (rng.integers(0, 5, (int(rng.integers(3, 30)), int(rng.integers(3, 30))))
                 * (rng.random((1,)) < 2)).astype([np.uint16, np.int32][number % 2])
        frame[rng.random(frame.shape) < 0.5] = 0
        cases.append((frame, bool(number % 3)))
    cases.append((np.zeros((10, 10), np.uint16), True))
    with rows.install_loader(history) as loaded:
        module = history._issue_module(history.I012, 'm1_reconcile_domain_ownership')
        assert loaded and loaded[0]['supported'], loaded
        replaced = module.observations
        original = module._ownership_region_original
        tick = time.perf_counter(); expected = [flatten(original(f, m)) for f, m in cases]; before = time.perf_counter() - tick
        tick = time.perf_counter(); actual = [flatten(replaced(f, m)) for f, m in cases]; after = time.perf_counter() - tick
        # Source metrics as dormancy detection asks them (dilation 1), and the
        # other dilations that keep the original: real consecutive frames, every
        # observation against every identity under it plus absent identities,
        # with a lag image that gives both red and blue pixels.
        replaced_metrics, original_metrics = module.source_metrics, None
        metric_cases = 0
        metric_seconds = dict(original=0.0, replaced=0.0)
    original_metrics = module.source_metrics
    assert replaced_metrics is not original_metrics
    for _ in range(max(a.frames // 5, 20)):
        stack = stacks[int(rng.integers(len(stacks)))]
        t = int(rng.integers(1, len(stack)))
        previous, current = stack[t - 1], stack[t]
        lookup = module.identity_points(previous)
        lag = rng.normal(0, 2, previous.shape).astype(np.float32)
        for item in original(current, False):
            under = sorted(set(map(int, np.unique(previous[tuple(item.pixels.T)]))) - {0})
            for identity in under + [int(rng.integers(1, 3000)), 0]:
                for dilation in (1, 1, 0, 2):
                    tick = time.perf_counter()
                    e = original_metrics(previous, item, identity, lag, dilation, lookup)
                    middle = time.perf_counter()
                    f = replaced_metrics(previous, item, identity, lag, dilation, lookup)
                    metric_seconds['original'] += middle - tick
                    metric_seconds['replaced'] += time.perf_counter() - middle
                    assert e == f and (e is None or all(type(e[k]) is type(f[k]) for k in e)), (t, identity, dilation)
                    metric_cases += 1
    assert module.observations is original
    for number, (e, f) in enumerate(zip(expected, actual)):
        assert e == f, number
    report = dict(status='pass', frames=len(cases), observations=sum(len(e) for e in expected),
                  before_seconds=before, after_seconds=after, counters=loaded[0],
                  source_metric_cases=metric_cases, source_metric_seconds=metric_seconds)
    print(report)
    a.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
