"""Compare the measured-core gap test with and without the scalar overlap, exactly.

Both pinned frozen versions of measured_soma_gap_memory run
core_scaled_gap_supported on synthetic moving cores (integer and half-pixel
centres and radii, so disks often touch exactly), with source and target masks
around chosen cells at gaps of 1 to 12 frames in both directions. Whole-frame
masks over more than 64 core pairs send some calls through the array path. Each answer from the installed adapter must equal the frozen
method's answer on a fresh memory; boundary ties, True and False answers and
both overlap paths must all occur.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--frozen', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', type=int, default=6000)
    a = p.parse_args()
    import numpy as np
    sys.path[:0] = [str(a.adapter.resolve())]
    import measured_gap_path_reuse as adapter
    rng = np.random.default_rng(37)
    size, frames, cells = 96, 40, 14
    # Cells drift by whole or half pixels; radii 2-5 in half steps.
    start = rng.integers(10, size - 10, (cells, 2)).astype(float)
    drift = rng.choice([-1., -.5, 0., .5, 1.], (frames, cells, 2))
    positions = start + np.cumsum(drift, axis=0)
    radii = rng.choice([2., 2.5, 3., 3.5, 4., 5.], cells)
    points = {}
    for t in range(frames):
        keep = rng.random(cells) > .15
        table = np.column_stack((positions[t, keep], radii[keep]))
        if t % 9 == 4:
            # A crowded frame (more than 64 pairs with a crowded partner).
            extra = np.column_stack((rng.uniform(5, size - 5, (70, 2)), rng.uniform(1, 4, 70)))
            table = np.vstack((table, extra))
        if t % 13 == 6:
            table = table + rng.normal(0, 1e-3, table.shape)
        points[t] = table
    yy, xx = np.mgrid[:size, :size]

    def mask_around(t, cell, grow):
        y, x = positions[t, cell]
        return (yy - y) ** 2 + (xx - x) ** 2 <= (radii[cell] + grow) ** 2

    class State:
        def __init__(self, mask, t):
            self.mask, self.t = mask, t
    cases = []
    for _ in range(a.cases):
        s = int(rng.integers(0, frames))
        gap = int(rng.integers(1, 13)) * int(rng.choice([-1, 1]))
        t = min(max(s + gap, 0), frames - 1)
        if t == s:
            continue
        left = int(rng.integers(0, cells))
        right = left if rng.random() < .6 else int(rng.integers(0, cells))
        cases.append((State(mask_around(s, left, rng.choice([0, 2, 6])), s),
                      dict(mask=mask_around(t, right, rng.choice([0, 2, 6]))), t))
        if rng.random() < .05:
            # Whole-frame masks: every core on both sides (more than 64 pairs).
            cases.append((State(np.ones((size, size), bool), s), dict(mask=np.ones((size, size), bool)), t))
    # Exact boundary ties of the disk test, counted on the original arithmetic.
    ties = 0
    report = dict(status='pass', calls=0, true=0, false=0, versions={})
    timings = dict(original=0.0, adapted=0.0)
    for version in ('r213_a010', 'r215_a003'):
        path = a.frozen / version / 'measured_soma_gap_memory.py'
        spec = importlib.util.spec_from_file_location(f'gap_memory_{version}', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cls = module.MeasuredSomaGapMemory
        original = cls.core_scaled_gap_supported
        memory = cls(points, 1, [])
        tick = time.perf_counter()
        expected = [original(memory, source, observation, t) for source, observation, t in cases]
        timings['original'] += time.perf_counter() - tick
        with adapter.install(module) as stats:
            assert stats['supported'] and cls.core_scaled_gap_supported is not original
            memory = cls(points, 1, [])
            tick = time.perf_counter()
            actual = [cls.core_scaled_gap_supported(memory, source, observation, t)
                      for source, observation, t in cases]
            timings['adapted'] += time.perf_counter() - tick
        assert cls.core_scaled_gap_supported is original
        mismatches = [i for i, (e, g) in enumerate(zip(expected, actual)) if e is not g and e != g]
        assert not mismatches, (version, mismatches[:10])
        assert all(type(g) is type(e) for e, g in zip(expected, actual))
        report['calls'] += len(cases)
        report['true'] += sum(map(bool, expected))
        report['false'] += len(expected) - sum(map(bool, expected))
        report['versions'][version] = dict(stats)
        assert stats['overlap_scalar'] > 0 and stats['overlap_arrays'] > 0, stats
    for source, observation, t in cases:
        old = memory.contained_cores(source.mask, source.t)
        new = memory.contained_cores(observation['mask'], t)
        if len(old) and len(new):
            d = ((old[:, None, :2] - new[None, :, :2]) ** 2).sum(axis=2)
            u = (old[:, None, 2] + new[None, :, 2]) ** 2 * abs(int(t) - int(source.t))
            ties += int(np.any(d == u))
    report['boundary_ties'] = ties
    assert ties > 0 and report['true'] > 0 and report['false'] > 0
    report['seconds'] = timings
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
