"""Measured-core walks on real tracker geometry: remembered walk states give the original paths, exactly.

Walks start from every measured core of many target frames and run back (and
forward) to source frames up to --lag frames away, in shuffled order, so later
walks reach the states of earlier ones. Every returned path (frames, point
dtypes and bytes) must equal the frozen trace_measured_core_path.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--module', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--points', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--targets', type=int, default=60)
    p.add_argument('--lag', type=int, default=40)
    p.add_argument('--labels', type=Path)
    p.add_argument('--frozen-io', type=Path, help='frozen_io.py giving LabelMask, loaded before the adapter')
    a = p.parse_args()
    sys.path.insert(0, str(a.adapter.resolve()))
    import numpy as np
    import pandas as pd
    if a.frozen_io is not None:
        io_spec = importlib.util.spec_from_file_location('frozen_io', a.frozen_io)
        frozen_io = importlib.util.module_from_spec(io_spec)
        sys.modules['frozen_io'] = frozen_io
        io_spec.loader.exec_module(frozen_io)
    import measured_gap_path_reuse as reuse
    spec = importlib.util.spec_from_file_location('measured_gap_original', a.module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.trace_measured_core_path
    points = pd.read_csv(a.points, low_memory=False)
    geometry = module.observed_core_geometry(points)
    frames = sorted(geometry)
    rng = np.random.default_rng(213)
    targets = sorted(rng.choice(frames[a.lag:-a.lag], size=min(a.targets, len(frames) - 2 * a.lag), replace=False))
    requests = []
    for target in targets:
        for core in geometry[target]:
            for lag in rng.choice(np.arange(2, a.lag + 1), size=6, replace=False):
                requests.append((core, int(target), int(target) - int(lag)))
                requests.append((core, int(target), int(target) + int(lag)))
    order = rng.permutation(len(requests))
    # Group by target as the tracker does, with targets visited in time order
    # in one pass and shuffled in another.
    passes = {'time_order': sorted(range(len(requests)), key=lambda i: (requests[i][1], i)), 'shuffled': list(order)}
    report = dict(status='pass', module=str(a.module), requests=len(requests), passes={})
    for name, sequence in passes.items():
        t0 = time.perf_counter()
        expected = [original(geometry, requests[i][0], requests[i][1], requests[i][2]) for i in sequence]
        t1 = time.perf_counter()
        with reuse.install(module) as stats:
            actual = [module.trace_measured_core_path(geometry, requests[i][0], requests[i][1], requests[i][2]) for i in sequence]
        t2 = time.perf_counter()
        steps = 0
        for left, right in zip(expected, actual):
            assert len(left) == len(right)
            for (lf, lp), (rf, rp) in zip(left, right):
                assert lf == rf and type(lf) is type(rf) and lp.dtype == rp.dtype and lp.tobytes() == rp.tobytes()
            steps += len(left)
        report['passes'][name] = dict(original_seconds=t1 - t0, adapted_seconds=t2 - t1, path_points=steps, stats=dict(stats))
    assert module.trace_measured_core_path is original

    # contained_cores and core_area on real label frames: lazy label masks
    # (writeable and read-only labels), plain boolean masks, cropped masks of
    # another shape and frames without measured cores.
    if a.labels is not None:
        import tifffile
        from frozen_io import LabelMask
        movie = tifffile.imread(a.labels)
        checked = 0
        times = dict(original=0.0, adapted=0.0)
        memory = module.MeasuredSomaGapMemory(geometry, 1, [])
        frame_list = [int(t) for t in rng.choice(len(movie), size=min(80, len(movie)), replace=False)]
        cases = []
        for t in frame_list:
            writable = movie[t].copy()
            frozen = movie[t].copy(); frozen.flags.writeable = False
            for identity in [int(v) for v in np.unique(movie[t])[1:12]] + [9999]:
                cases.append((LabelMask(writable, identity), t))
                cases.append((LabelMask(frozen, identity), t))
                cases.append((writable == identity, t))
                cases.append(((writable == identity)[:300, 50:], t))
                cases.append((writable == identity, len(movie) + 50))
        for repeat in range(2):
            t0 = time.perf_counter()
            expected = [(module.MeasuredSomaGapMemory.contained_cores(memory, mask, t),
                         module.MeasuredSomaGapMemory.core_area(memory, mask, t)) for mask, t in cases]
            t1 = time.perf_counter()
            with reuse.install(module) as core_stats:
                adapted = module.MeasuredSomaGapMemory(geometry, 1, [])
                actual = [(adapted.contained_cores(mask, t), adapted.core_area(mask, t)) for mask, t in cases]
                actual += [(adapted.contained_cores(mask, t), adapted.core_area(mask, t)) for mask, t in cases]
            t2 = time.perf_counter()
            times['original'] += t1 - t0; times['adapted'] += (t2 - t1) / 2
            for (left_cores, left_area), (right_cores, right_area) in zip(expected + expected, actual):
                assert left_cores.dtype == right_cores.dtype and left_cores.shape == right_cores.shape
                assert left_cores.tobytes() == right_cores.tobytes()
                assert left_area == right_area and type(left_area) is type(right_area)
                checked += 1
        report['cores'] = dict(calls=checked, seconds=times, stats=dict(core_stats))
        assert core_stats['core_direct'] > 0 and core_stats['core_reuses'] > 0, core_stats
    report['sources'] = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in (Path(__file__), a.module, Path(reuse.__file__))}
    a.output.write_text(json.dumps(report, indent=1, default=str) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'sources'}, default=str))


if __name__ == '__main__':
    main()
