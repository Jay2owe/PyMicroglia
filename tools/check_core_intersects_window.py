"""Compare the frozen _core_intersects with the empty-window shortcut, exactly.

The frozen function is imported from the completed-feedback code; the adapter
must engage on it (source digest), replace every alias and restore them. Real
accepted outlines (every name of real frames) are tested against every tracker
point of the frame; random bool, integer (odd and even values), memmap and masked-array masks
are tested against random points including edge, outside, zero and negative
radius, and non-finite points, whose errors must match.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time


def outcome(function, mask, point):
    try:
        return ('value', function(mask, point))
    except Exception as error:
        return ('raised', type(error).__name__, str(error))


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'code', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--frames', type=int, default=12)
    p.add_argument('--start', type=int, default=200)
    p.add_argument('--random', type=int, default=60000)
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    import tifffile
    sys.path[:0] = [str(a.code.resolve()), str(a.adapter.resolve())]
    import shared_parent_soma_ownership_a002 as module
    import core_intersects_window as adapter
    original = module._core_intersects
    assert adapter._verified(original)
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    labels = tifffile.imread(folder / '99_well_b3_issue013_anchored_projection_owner_relay.tif', key=range(a.start, a.start + a.frames))
    points = pd.read_csv(folder / '26_latent_body_tracks/out/latent_track_points.csv', low_memory=False)
    cases = []
    for t, frame in enumerate(labels):
        here = list(points[points.frame == a.start + t].itertuples(index=False))
        for name in np.unique(frame)[1:]:
            mask = frame == name
            cases += [(mask, q) for q in here]
    real = len(cases)
    rng = np.random.default_rng(3)
    scratch = Path(tempfile.mkdtemp(prefix='core-window-'))
    stored = np.zeros((64, 64), bool)
    stored[20:30, 5:9] = True
    stored.tofile(scratch / 'mask.dat')
    mapped = np.memmap(scratch / 'mask.dat', dtype=bool, mode='r', shape=(64, 64))
    Point = type('Point', (), {})
    for trial in range(a.random):
        h, w = int(rng.integers(1, 50)), int(rng.integers(1, 50))
        kind = trial % 5
        if kind == 0:
            mask = rng.random((h, w)) < rng.uniform(0, .2)
        elif kind == 1:
            mask = (rng.integers(0, 4, (h, w)) * (rng.random((h, w)) < .1)).astype(np.int32)
        elif kind == 2:
            mask = np.zeros((h, w), bool)
        elif kind == 3:
            mask = mapped
            h, w = mask.shape
        else:
            # A masked array's .any() skips masked pixels.
            mask = np.ma.array(rng.random((h, w)) < .1, mask=rng.random((h, w)) < .5)
        q = Point()
        q.x = float(rng.uniform(-8, w + 8)); q.y = float(rng.uniform(-8, h + 8))
        q.radius_px = float(rng.choice([0.0, -2.0, rng.uniform(0, 9)]))
        if trial % 997 == 0:
            q.x = float('nan')
        cases.append((mask, q))
    tick = time.perf_counter()
    expected = [outcome(original, m, q) for m, q in cases]
    middle = time.perf_counter()
    stats = {}
    with adapter.install(stats):
        assert module._core_intersects is adapter._core_intersects and stats['versions'] == 1
        actual = [outcome(module._core_intersects, m, q) for m, q in cases]
    end = time.perf_counter()
    assert module._core_intersects is original
    mismatches = [i for i, (x, y) in enumerate(zip(expected, actual)) if x != y]
    assert not mismatches, [(i, expected[i], actual[i], vars(cases[i][1]) if hasattr(cases[i][1], "__dict__") else cases[i][1]) for i in mismatches[:3]]
    assert all(type(x[1]) is bool for x in actual if x[0] == 'value')
    mapped._mmap.close()
    del mapped
    import shutil
    shutil.rmtree(scratch, ignore_errors=True)
    report = dict(status='pass', real_cases=real, random_cases=len(cases) - real,
                  true=sum(x == ('value', True) for x in expected),
                  raised=sum(x[0] == 'raised' for x in expected),
                  seconds=dict(original=middle - tick, adapted=end - middle))
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
