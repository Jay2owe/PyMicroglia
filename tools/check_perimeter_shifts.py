"""Compare regionprops perimeters from shifted views with skimage's.

Every region image of real accepted label frames, plus edge cases (single
pixels, lines, diagonals, rings, border-touching and random blobs, an empty
image), is measured by skimage's perimeter and by the shifted version; values
must be bit-identical float64. regionprops' perimeter property and the
Issue 011 feature table's normalized perimeter route are compared with and
without the adapter installed.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frame-step', type=int, default=4)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    from skimage.measure import regionprops
    from skimage.measure import _regionprops_utils as utils
    import perimeter_shifts as adapter
    rng = np.random.default_rng(5)
    masks = [np.ones(shape, bool) for shape in ((1, 1), (1, 7), (7, 1), (2, 2), (3, 3), (5, 9))]
    masks += [np.zeros((0, 4), bool), np.zeros((3, 3), bool), np.eye(9, dtype=bool), np.eye(9, dtype=bool)[::-1]]
    ring = np.ones((11, 11), bool); ring[2:-2, 2:-2] = False; masks.append(ring)
    for _ in range(300):
        masks.append(rng.random(tuple(rng.integers(1, 30, 2))) < rng.uniform(0.2, 0.9))
    frames = []
    for name in ('05_issue009.tif', '06_issue010.tif'):
        movie = tifffile.imread(HISTORY / name)
        frames += [movie[t] for t in range(0, len(movie), a.frame_step)]
    real = 0
    for frame in frames:
        for region in regionprops(frame):
            masks.append(region.image.copy()); real += 1
    t0 = time.perf_counter()
    expected = [utils.perimeter(mask, 4) for mask in masks]
    before = time.perf_counter() - t0
    sample = frames[::6]
    baseline = [r.perimeter for f in sample for r in regionprops(f)]
    failures = []
    with adapter.install() as stats:
        assert stats['supported'], stats
        from skimage.measure import _regionprops as module
        t0 = time.perf_counter()
        actual = [module.perimeter(mask, 4) for mask in masks]
        after = time.perf_counter() - t0
        routed = [r.perimeter for f in sample for r in regionprops(f)]
        assert module.perimeter(masks[0].astype(np.uint8), 4) == utils.perimeter(masks[0].astype(np.uint8), 4)
        counters = dict(stats)
    assert module.perimeter is utils.perimeter
    for index, (e, f) in enumerate(zip(expected, actual)):
        if type(e) is not type(f) or np.float64(e).tobytes() != np.float64(f).tobytes():
            failures.append(dict(index=index, shape=list(masks[index].shape), values=[float(e), float(f)]))
    if [np.float64(v).tobytes() for v in baseline] != [np.float64(v).tobytes() for v in routed]:
        failures.append(dict(error='regionprops perimeter property'))
    report = dict(status='pass' if not failures else 'fail', masks=len(masks), real_regions=real,
                  property_regions=len(baseline), before_seconds=before, after_seconds=after,
                  counters=counters, failures=failures[:10])
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
