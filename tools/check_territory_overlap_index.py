"""Compare box-indexed territory overlap tables with the all-pairs original, exactly.

Consecutive and random frame pairs of real accepted-history stacks, plus
shuffled, empty and single-cell observation lists, must give the same four
tables (dtype, shape, values) and the same strong mutual links.
"""
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
    p.add_argument('--pairs', type=int, default=1200)
    a = p.parse_args()
    import numpy as np
    import tifffile
    sys.path[:0] = [str(a.adapter.resolve()), str((a.engine / 'code').resolve())]
    import territory_reconciliation as module
    import territory_overlap_index as index
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    stacks = [tifffile.imread(folder / name) for name in ('01_issue005.tif', '10_blob_residency.tif')]
    rng = np.random.default_rng(23)
    cache = {}

    def observations(s, t):
        if (s, t) not in cache:
            cache[s, t] = module._observations(stacks[s][t])
        return cache[s, t]
    cases = []
    while len(cases) < a.pairs:
        s = int(rng.integers(len(stacks)))
        t = int(rng.integers(1, len(stacks[s])))
        u = t - 1 if rng.random() < 0.7 else int(rng.integers(len(stacks[s])))
        previous, current = list(observations(s, u)), list(observations(s, t))
        if rng.random() < 0.1:
            rng.shuffle(previous)
        if rng.random() < 0.05:
            current = current[:1]
        cases.append((previous, current))
    cases += [([], []), (observations(0, 3), []), ([], observations(0, 3)), (observations(0, 3), observations(0, 3))]
    params = dict(minimum_intersection_px=12)
    original = module._overlap_tables
    tick = time.perf_counter()
    expected = [(original(x, y), module._strong_mutual_links(x, y, params)) for x, y in cases]
    before = time.perf_counter() - tick
    with index.install(module) as stats:
        assert stats['supported']
        tick = time.perf_counter()
        actual = [(module._overlap_tables(x, y), module._strong_mutual_links(x, y, params)) for x, y in cases]
        after = time.perf_counter() - tick
    assert module._overlap_tables is original
    links = 0
    for (e, el), (f, fl) in zip(expected, actual):
        assert e.keys() == f.keys()
        for key in e:
            assert e[key].dtype == f[key].dtype and e[key].shape == f[key].shape
            assert e[key].tobytes() == f[key].tobytes(), key
        assert len(el) == len(fl)
        for x, y in zip(el, fl):
            assert x.keys() == y.keys() and all(type(x[k]) is type(y[k]) and (x[k] == y[k]) for k in x)
        links += len(el)
    report = dict(status='pass', cases=len(cases), links=links,
                  overlapping_pairs=int(sum(int(np.count_nonzero(e['intersection'])) for e, _ in expected)),
                  all_pairs=int(sum(e['intersection'].size for e, _ in expected)),
                  before_seconds=before, after_seconds=after, counters=stats)
    print(report)
    a.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
