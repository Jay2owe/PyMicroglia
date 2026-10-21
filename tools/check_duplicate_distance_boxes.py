"""Compare Issue 012's substantial_duplicate with its boxed form, exactly.

Real frames from the Issue 011 output (the Issue 012 input) and the Issue 012
output: every identity, plus synthetic frames whose identity is split into
parts at distances on both sides of both thresholds, with neighbours touching,
near and far, and thresholds of 0, fractional and large. The boolean answer
must be the same for every call.
"""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--engine', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frames', type=int, default=12)
    a = p.parse_args()
    sys.path[:0] = [str(a.engine.resolve() / 'code'), str(a.adapter.resolve())]
    import numpy as np
    import tifffile
    import accepted_history as ah
    import ownership_observation_reuse as adapter
    module = ah._issue_module(ah.I012, 'm1_reconcile_domain_ownership')
    cfg = ah._params('issue012', 'domain_ownership')['ownership']
    original = module.substantial_duplicate
    stats = dict(duplicate_boxed=0, duplicate_fallback=0)
    boxed = adapter.boxed_duplicate(original, stats)
    settings = [(int(cfg.get('duplicate_minimum_area_px', 48)), float(cfg.get('duplicate_minimum_distance_px', 12.0)),
                 float(cfg.get('same_owner_maximum_fragment_gap_px', 2.0)))]
    settings += [(1, 0.0, 0.0), (4, 2.5, 1.5), (8, 30.0, 9.0), (20, 6.0, 0.4)]
    times = dict(original=0.0, adapted=0.0)
    calls = 0
    trues = 0

    def compare(frame, identity, setting):
        nonlocal calls, trues
        t0 = time.perf_counter()
        expected = original(frame, identity, *setting)
        t1 = time.perf_counter()
        actual = boxed(frame, identity, *setting)
        t2 = time.perf_counter()
        times['original'] += t1 - t0
        times['adapted'] += t2 - t1
        assert type(expected) is type(actual) and expected == actual, (identity, setting, expected, actual)
        calls += 1
        trues += bool(expected)

    folder = a.engine / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'
    rng = np.random.default_rng(612)
    for name in ('07_issue011.tif', '08_issue012.tif'):
        stack = tifffile.imread(folder / name)
        for t in rng.choice(len(stack), a.frames, replace=False):
            frame = stack[int(t)]
            for identity in np.unique(frame)[1:]:
                compare(frame, int(identity), settings[0])
            for identity in rng.choice(np.unique(frame)[1:], 15):
                for setting in settings[1:]:
                    compare(frame, int(identity), setting)
    for case in range(1500):
        frame = np.zeros((rng.integers(20, 90), rng.integers(20, 90)), np.int32)
        h, w = frame.shape
        for _ in range(rng.integers(2, 6)):
            y, x = rng.integers(0, h), rng.integers(0, w)
            frame[y:y + rng.integers(1, 12), x:x + rng.integers(1, 12)] = 1
        for other in range(2, rng.integers(2, 7)):
            y, x = rng.integers(0, h), rng.integers(0, w)
            frame[y:y + rng.integers(1, 6), x:x + rng.integers(1, 6)] = other
        for setting in settings[1:]:
            compare(frame, 1, setting)
    assert stats['duplicate_fallback'] == 0
    report = dict(status='pass', calls=calls, true_answers=trues, seconds=times, stats=stats)
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
