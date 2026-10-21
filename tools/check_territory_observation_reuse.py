"""Check frame-content cache invalidation and complete territory decisions."""
import argparse
import json
from pathlib import Path
import platform
import sys
from concurrent.futures import ThreadPoolExecutor

platform.platform = lambda *a, **k: 'Windows-territory-observation-check'
import numpy as np
import pandas as pd


def same_rows(left, right):
    assert len(left) == len(right)
    for old, new in zip(left, right):
        assert old.keys() == new.keys()
        for key in old:
            if isinstance(old[key], np.ndarray):assert np.array_equal(old[key], new[key])
            else:assert old[key] == new[key]


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'report'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    sys.path[:0] = [str(a.engine.resolve()/'code'), str(a.adapter.resolve())]
    import territory_reconciliation as module
    import territory_observation_reuse as reuse
    original = module._observations
    rng = np.random.default_rng(117)
    cases = []
    for case in range(8):
        labels = np.zeros((12, 31, 37), np.uint16)
        labels[:, 2:9, 3:11] = 1
        labels[:, 13:22, 21:30] = 2
        for t in range(12):
            labels[t, 4+t:7+t, 16:20] = 3 if t < 5 else 4
            for _ in range(4):
                labels[t, rng.integers(0,31), rng.integers(0,37)] = rng.integers(5, 11)
        if case%2:labels=labels.transpose(0,2,1)
        if case==4:labels[labels==1]=65000
        if case==5:labels[:]=0
        if case==6:labels=labels.astype(np.int32)
        expected = [fn(labels, 5, {}) for fn in
                    (module.optimise_territory_ownership_graph, module.reconcile_same_territory_replacements)]
        with reuse.install(module) as counters:
            assert counters['supported']
            frame = labels[0].copy()
            same_rows(original(frame), module._observations(frame))
            cached = module._observations(frame)
            if cached:
                cached[0]['position'][:] = -100
                cached[0]['local_mask'][:] = False
            same_rows(original(frame), module._observations(frame))
            frame[0,0] = 12
            same_rows(original(frame), module._observations(frame))
            with ThreadPoolExecutor(max_workers=2) as pool:
                rows = list(pool.map(module._observations, [frame, frame.copy()]*4))
            for row in rows:same_rows(original(frame), row)
            actual = [fn(labels, 5, {}) for fn in
                      (module.optimise_territory_ownership_graph, module.reconcile_same_territory_replacements)]
        assert module._observations is original
        assert counters['hits'] > 0
        for old, new in zip(expected, actual):
            assert np.array_equal(old[0], new[0])
            for before, after in zip(old[1:], new[1:]):
                pd.testing.assert_frame_equal(before, after, check_exact=True)
                assert before.to_csv(index=False) == after.to_csv(index=False)
        cases.append(dict(case=case, counters=counters))
    report = dict(status='pass', cases=cases,
                  scope='Exact complete masks, ordered decisions and audits; edits, returned-value mutation, transposes, sparse names, concurrent reads')
    a.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status='pass', cases=len(cases))))


if __name__ == '__main__':
    main()
