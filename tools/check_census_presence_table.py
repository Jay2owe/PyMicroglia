"""Compare the count-table census with the original measure, exactly.

Real accepted-history stacks (full length) and synthetic movies with gaps,
single-frame segments, border cells, compatibility ids, empty movies and
several dtypes must give identical census, segment, gap, termination and
presence rows and summaries (float bits included).
"""
import argparse
import json
from pathlib import Path
import sys
import time


def same(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return list(a) == list(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return a == b or (a != a and b != b)
    return a == b


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--synthetic', type=int, default=150)
    a = p.parse_args()
    import numpy as np
    import tifffile
    sys.path[:0] = [str(a.adapter.resolve()), str((a.engine / 'code').resolve())]
    import accepted_history as history
    import census_presence_table as table
    module = history._issue_module(history.I017, 'm1_census')
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    rng = np.random.default_rng(17)
    cases = []
    for name in ('10_blob_residency.tif', '04_issue008.tif'):
        stack = tifffile.imread(folder / name)
        ids = np.unique(stack); ids = ids[ids > 0]
        cases.append((stack, 7, 30.0, set(int(v) for v in rng.choice(ids, 20))))
    for number in range(a.synthetic):
        shape = (int(rng.integers(1, 12)), int(rng.integers(3, 25)), int(rng.integers(3, 25)))
        movie = rng.integers(0, int(rng.integers(1, 9)), shape) * (rng.random(shape) < 0.3)
        movie[rng.random(shape[0]) < 0.3] = 0
        movie = movie.astype([np.uint16, np.int32, np.uint8, np.int64][number % 4])
        cases.append((movie, int(rng.integers(0, 5)), float(rng.choice([30.0, 12.5])),
                      set(int(v) for v in rng.integers(1, 8, 2)) if rng.random() < 0.5 else None))
    cases.append((np.zeros((4, 5, 5), np.uint16), 0, 30.0, None))
    original = module.measure
    tick = time.perf_counter(); expected = [original(*c) for c in cases]; before = time.perf_counter() - tick
    with table.install(module) as stats:
        assert stats['supported']
        tick = time.perf_counter(); actual = [module.measure(*c) for c in cases]; after = time.perf_counter() - tick
    assert module.measure is original
    for number, (e, f) in enumerate(zip(expected, actual)):
        assert same(e, f), number
    report = dict(status='pass', cases=len(cases), real_identities=[len(e['census']) for e in expected[:2]],
                  gaps=sum(len(e['gaps']) for e in expected), before_seconds=before, after_seconds=after,
                  counters=stats)
    print(report)
    a.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
