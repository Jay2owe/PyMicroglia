"""Compare stationary gap recovery's component masks and predecessor vote.

Real accepted label frames (plain and memory-mapped) are split into one
identity's components by the frozen function and by the boxed adapter, for
every identity present plus an absent one and several area floors; each
component then asks both predecessor votes with several forbidden sets.
Masks (count, order, shape, dtype and pixels) and chosen predecessors must be
identical. Synthetic frames add components touching every edge and corner.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'


def same_masks(expected, actual):
    if len(expected) != len(actual):
        return f'count {len(expected)} != {len(actual)}'
    for index, (a, b) in enumerate(zip(expected, actual)):
        if type(a) is not type(b) or a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
            return f'mask {index}'
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=40)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import stationary_gap_recovery as module
    import stationary_gap_masks as adapter
    original_masks, original_predecessor = module._component_masks, module._safe_predecessor
    stacks = []
    for name in ('11_issue017_rebase.tif', '08_issue012.tif'):
        movie = tifffile.imread(HISTORY / name)
        picks = np.linspace(1, len(movie) - 1, a.frames).astype(int)
        stacks.append((name, movie, picks))
    a.scratch.mkdir(parents=True, exist_ok=True)
    name, movie, picks = stacks[0]
    mapped = np.memmap(a.scratch / 'labels.dat', dtype=movie.dtype, mode='w+', shape=movie.shape)
    mapped[:] = movie
    mapped.flush()
    stacks.append(('memmap ' + name, mapped, picks[::4]))
    rng = np.random.default_rng(23)
    synthetic = np.zeros((3, 40, 50), np.int32)
    synthetic[1, :3, :3] = 7; synthetic[1, -2:, -4:] = 7; synthetic[1, 10:14, 0] = 7
    synthetic[1, 0, 20:30] = 7; synthetic[1, 20, -1] = 7; synthetic[1, 30:, 25:27] = 9
    synthetic[0, :5, :] = 4; synthetic[0, -6:, -6:] = 9; synthetic[0, 15:20, 20:25] = 11
    synthetic[2] = rng.integers(0, 4, synthetic.shape[1:])
    stacks.append(('synthetic', synthetic, np.array([1, 2])))
    failures, cases, votes = [], 0, 0
    seconds = dict(original=0.0, adapter=0.0)
    with adapter.install(module) as stats:
        assert stats['supported'], stats
        for name, movie, picks in stacks:
            for t in picks:
                frame = movie[t]
                present = [int(v) for v in np.unique(np.asarray(frame)) if v > 0]
                for identity in present + [int(max(present, default=0)) + 1]:
                    for minimum in (1, 0, 6, 40):
                        t0 = time.perf_counter()
                        expected = original_masks(frame, identity, minimum)
                        t1 = time.perf_counter()
                        actual = module._component_masks(frame, identity, minimum)
                        seconds['original'] += t1 - t0; seconds['adapter'] += time.perf_counter() - t1
                        cases += 1
                        problem = same_masks(expected, actual)
                        if problem:
                            failures.append(dict(stack=name, t=int(t), identity=identity, minimum=minimum, error=problem))
                            continue
                        if minimum != 1:
                            continue
                        for mask in expected:
                            for forbidden in ({identity}, set(), set(map(int, rng.choice(present, min(3, len(present)), replace=False)))):
                                t0 = time.perf_counter()
                                left = original_predecessor(movie, int(t), mask, forbidden)
                                t1 = time.perf_counter()
                                right = module._safe_predecessor(movie, int(t), mask, forbidden)
                                seconds['original'] += t1 - t0; seconds['adapter'] += time.perf_counter() - t1
                                votes += 1
                                if type(left) is not type(right) or left != right:
                                    failures.append(dict(stack=name, t=int(t), identity=identity, vote=[left, right]))
            # Frame 0 and an empty mask keep the original answers.
            empty = np.zeros(movie.shape[1:], bool)
            for frame_number in (0, int(picks[0])):
                if module._safe_predecessor(movie, frame_number, empty, set()) != original_predecessor(movie, frame_number, empty, set()):
                    failures.append(dict(stack=name, t=frame_number, error='empty or first-frame vote'))
        counters = dict(stats)
    assert module._component_masks is original_masks and module._safe_predecessor is original_predecessor
    # The memory-mapped copy is removed by the caller once this process exits.
    report = dict(status='pass' if not failures else 'fail', cases=cases, votes=votes,
                  seconds=seconds, counters=counters, failures=failures[:10])
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
