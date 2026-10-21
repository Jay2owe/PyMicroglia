"""Local fragment reunion discovery: the foreground pair scan equals the original, exactly.

Runs the frozen local_fragment_reunion.discover_frame with and without
fragment_pair_scan over every frame of several accepted label movies (with
their latent track points) and over edge cases (empty, single component,
components touching every border, adjacent equal labels, a wide label range,
non-contiguous frames). Components, the nearest-gap dictionary (keys, order,
values) and the audit table (columns, dtypes, values, order) must be equal.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--module', type=Path, required=True, help='folder holding local_fragment_reunion.py')
    p.add_argument('--adapter', type=Path, required=True, help='folder holding fragment_pair_scan.py')
    p.add_argument('--history', type=Path, required=True, help='accepted_history folder')
    p.add_argument('--movies', nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import pandas as pd
    import tifffile
    sys.path[:0] = [str(a.module.resolve()), str(a.adapter.resolve())]
    import local_fragment_reunion as module
    import fragment_pair_scan as adapter
    points = pd.read_csv(a.history / '26_latent_body_tracks/out/latent_track_points.csv')
    groups = {int(frame): group for frame, group in points.groupby('frame')}
    empty = points.iloc[:0]
    params = module.Params()

    def run(frame, group):
        components = module._components(frame)
        pairs = module._nearby_pairs(components[0], params.maximum_gap_px)
        table, labelled = module.discover_frame(frame, group, 0, params)
        return components, list(pairs.items()), table, labelled

    def same(left, right, where):
        (c0, p0, t0, l0), (c1, p1, t1, l1) = left, right
        for x, y in zip(c0, c1):
            assert type(x) is type(y) and x.dtype == y.dtype and np.array_equal(x, y), where
            assert x.tobytes() == y.tobytes(), where
        assert p0 == p1 and all(type(u[1]) is type(v[1]) for u, v in zip(p0, p1)), where
        assert list(t0.columns) == list(t1.columns) and list(t0.dtypes) == list(t1.dtypes), where
        pd.testing.assert_frame_equal(t0, t1, check_exact=True)
        assert l0.dtype == l1.dtype and np.array_equal(l0, l1), where

    cases = []
    for name in a.movies:
        movie = tifffile.imread(a.history / name)
        cases += [(f'{name}:{t}', np.ascontiguousarray(movie[t]), groups.get(t, empty)) for t in range(len(movie))]
    rng = np.random.default_rng(7)
    edge = [np.zeros((40, 50), np.uint16), np.ones((40, 50), np.uint16)]
    frame = np.zeros((40, 50), np.int32); frame[:, :3] = 5; frame[:, -3:] = 9; frame[0, 10:20] = 7; frame[-1, 10:20] = 7
    frame[20, 20] = 3; frame[20, 22] = 4; frame[21, 21] = 3; frame[10:13, 30] = 2**30
    edge.append(frame)
    blobs = np.zeros((60, 60), np.uint16)
    for k in range(1, 40):
        y, x = rng.integers(0, 58, 2); blobs[y:y + 2, x:x + 3] = rng.integers(1, 6)
    edge += [blobs, blobs.astype(np.int64), blobs[::2, ::-1].copy(), blobs > 0]
    cases += [(f'edge:{i}', f, empty) for i, f in enumerate(edge)]
    seconds = dict(original=0.0, adapted=0.0)
    for where, frame, group in cases:
        t0 = time.perf_counter(); expected = run(frame, group)
        t1 = time.perf_counter()
        with adapter.install(module) as installed:
            assert installed
            actual = run(frame, group)
        t2 = time.perf_counter()
        seconds['original'] += t1 - t0; seconds['adapted'] += t2 - t1
        same(expected, actual, where)
    report = dict(status='pass', frames=len(cases), seconds=seconds,
                  sources={str(f): hashlib.sha256(Path(f).read_bytes()).hexdigest()
                           for f in (Path(__file__), Path(module.__file__), Path(adapter.__file__))})
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'sources'}))


if __name__ == '__main__':
    main()
