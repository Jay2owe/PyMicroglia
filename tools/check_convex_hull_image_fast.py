"""Compare the qhull-free convex_hull_image with skimage's on real and edge-case masks."""
import argparse
import json
from pathlib import Path
import sys
import time
import warnings


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--stacks', nargs='+', default=['06_issue010.tif', '12_issue023_generalized.tif'])
    p.add_argument('--frame-step', type=int, default=3)
    a = p.parse_args()
    import numpy as np
    import tifffile
    from scipy import ndimage as ndi
    from skimage.measure import regionprops
    from skimage.morphology import convex_hull as module
    sys.path.insert(0, str(a.adapter.resolve()))
    import convex_hull_image_fast as fast
    original = module.convex_hull_image
    masks = []
    rng = np.random.default_rng(11)
    # Edge cases: single pixels, lines, diagonals, rings, borders, random blobs.
    for shape in ((1, 1), (1, 7), (7, 1), (3, 3), (5, 9)):
        masks.append(np.ones(shape, bool))
    line = np.zeros((9, 9), bool); line[4, :] = True; masks.append(line)
    masks.append(np.eye(9, dtype=bool)); masks.append(np.eye(9, dtype=bool)[::-1])
    ring = np.ones((11, 11), bool); ring[2:-2, 2:-2] = False; masks.append(ring)
    for _ in range(4000):
        h, w = rng.integers(1, 40, 2)
        mask = rng.random((h, w)) < rng.uniform(0.02, 0.9)
        masks.append(mask)
    reference = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    real = 0
    for name in a.stacks:
        stack = tifffile.imread(reference / name)
        for frame in stack[::a.frame_step]:
            for region in regionprops(frame):
                masks.append(region.image.copy()); real += 1
    expected = []
    tick = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for mask in masks:
            expected.append(original(mask))
        before = time.perf_counter() - tick
        with fast.install() as stats:
            assert stats['supported']
            tick = time.perf_counter()
            actual = [module.convex_hull_image(mask) for mask in masks]
            after = time.perf_counter() - tick
            # The regionprops route that solidity uses.
            sample = masks[-min(real, 2000):]
            props = [(r.solidity, r.area_convex) for m in sample for r in regionprops(m.astype(np.uint8))]
        baseline = [(r.solidity, r.area_convex) for m in sample for r in regionprops(m.astype(np.uint8))]
    mismatches = sum(not (e.dtype == f.dtype and e.shape == f.shape and np.array_equal(e, f))
                     for e, f in zip(expected, actual))
    assert mismatches == 0, mismatches
    assert props == baseline
    report = dict(status='pass', masks=len(masks), real_components=real, before_seconds=before,
                  after_seconds=after, counters=stats)
    print(report)
    a.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
