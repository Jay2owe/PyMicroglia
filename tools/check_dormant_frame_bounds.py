"""Compare dormant-path frame bounds from find_objects with the np.unique route.

Every frame of two accepted label movies (plain and as memory-mapped views),
plus synthetic frames: no background, all background, signed with and without
negative values, sparse large labels, single pixels at the edges. Each bounds
dictionary must have the same keys and the same box slices as the compact
inverse-label route it replaces.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'


def unique_route(frame):
    values, inverse = np.unique(frame, return_inverse=True)
    numbered = inverse.reshape(frame.shape) + 1
    return dict(zip(map(int, values), ndimage.find_objects(numbered)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import dormant_path_reuse
    holder = dormant_path_reuse.__dict__
    cls = next(v for v in holder.values() if isinstance(v, type) and 'frame_bounds' in vars(v))
    probe = cls.__new__(cls)
    from collections import OrderedDict
    probe.bounds, probe.ndi, probe.stats = OrderedDict(), ndimage, dict(frame_indexes=0)
    frames = []
    for name in ('07_issue011.tif', '08_issue012.tif'):
        movie = tifffile.imread(HISTORY / name)
        frames += [(f'{name}-{t}', movie[t]) for t in range(len(movie))]
    a.scratch.mkdir(parents=True, exist_ok=True)
    mapped = np.memmap(a.scratch / 'frames.dat', dtype=movie.dtype, mode='w+', shape=movie[:20].shape)
    mapped[:] = movie[:20]
    frames += [(f'memmap-{t}', mapped[t]) for t in range(20)]
    rng = np.random.default_rng(12)
    full = rng.integers(1, 9, (30, 40)).astype(np.uint16)
    signed = rng.integers(0, 6, (30, 40)).astype(np.int32)
    negative = signed.copy(); negative[3, 4] = -2
    sparse = np.zeros((50, 50), np.int64); sparse[0, 0] = 2000; sparse[-1, -1] = 3; sparse[10:12, 20:30] = 2499
    edge = np.zeros((9, 9), np.uint8); edge[0, 8] = 1; edge[8, 0] = 4
    frames += [('full', full), ('empty', np.zeros((5, 7), np.uint16)), ('signed', signed),
               ('negative', negative), ('sparse', sparse), ('edge', edge), ('single', np.array([[3]], np.int32))]
    failures, seconds = [], dict(unique=0.0, objects=0.0)
    for label, frame in frames:
        t0 = time.perf_counter()
        expected = unique_route(np.asarray(frame))
        t1 = time.perf_counter()
        actual = probe.frame_bounds(frame, label)
        seconds['unique'] += t1 - t0; seconds['objects'] += time.perf_counter() - t1
        if set(expected) != set(actual) or any(expected[k] != actual[k] for k in expected):
            failures.append(label)
    report = dict(status='pass' if not failures else 'fail', frames=len(frames), seconds=seconds, failures=failures[:10])
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
