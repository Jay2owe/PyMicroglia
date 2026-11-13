"""Compare counted mask centres with scipy's center_of_mass and the lineage _centre.

Masks: every identity of real accepted label frames, unions of neighbours,
random masks, single pixels, full frames, frame-edge strips, an empty mask
(original path), uint8 and 3-D masks and labelled calls (original path).
Results must have the same type and bits; the installed names must route to
the counted version and be restored afterwards.
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


def bits(value):
    if isinstance(value, tuple):
        return tuple((type(v).__name__, np.float64(v).tobytes()) for v in value)
    return type(value).__name__, repr(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frame-step', type=int, default=24)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import warnings
    warnings.simplefilter('ignore')
    import scipy.ndimage as ndi
    import physical_lineage_aliases as lineage
    import mask_centre_counts as adapter
    original_centre, original_lineage = ndi.center_of_mass, lineage._centre
    rng = np.random.default_rng(17)
    masks = []
    for name in ('09_seat_identity.tif', '11_issue017_rebase.tif'):
        movie = tifffile.imread(HISTORY / name)
        for t in range(0, len(movie), a.frame_step):
            frame = movie[t]
            ids = [int(v) for v in np.unique(frame) if v]
            masks += [frame == i for i in ids]
            masks += [(frame == i) | (frame == j) for i, j in zip(ids[::7], ids[1::7])]
            masks.append(frame > 0)
    shape = masks[0].shape
    for _ in range(200):
        masks.append(rng.random(shape) < rng.uniform(0.0005, 0.6))
    one = np.zeros(shape, bool); one[0, 0] = True; masks.append(one)
    one = np.zeros(shape, bool); one[-1, -1] = True; masks.append(one)
    masks += [np.ones(shape, bool), np.ones((1, 1), bool), np.ones((1, 9), bool)]
    strip = np.zeros(shape, bool); strip[:, -1] = True; masks.append(strip)
    others = [np.zeros(shape, bool), masks[0].astype(np.uint8), np.stack([masks[0], masks[1]]),
              np.asfortranarray(masks[2])]
    failures, seconds = [], dict(original=0.0, counted=0.0)
    t0 = time.perf_counter()
    expected = [original_centre(m) for m in masks]
    expected_lineage = [original_lineage(m) for m in masks]
    seconds['original'] = time.perf_counter() - t0
    expected_other = [bits(original_centre(m)) for m in others]
    labelled = bits(original_centre(masks[0], masks[0].astype(int), 1))
    with adapter.install() as stats, adapter.install_lineage(lineage) as lineage_stats:
        assert stats['supported'] and lineage_stats['supported'] and ndi.center_of_mass is not original_centre
        t0 = time.perf_counter()
        actual = [ndi.center_of_mass(m) for m in masks]
        actual_lineage = [lineage._centre(m) for m in masks]
        seconds['counted'] = time.perf_counter() - t0
        actual_other = [bits(ndi.center_of_mass(m)) for m in others]
        actual_labelled = bits(ndi.center_of_mass(masks[0], masks[0].astype(int), 1))
        counters = dict(stats), dict(lineage_stats)
    assert ndi.center_of_mass is original_centre and lineage._centre is original_lineage
    for index, (e, f, g, h) in enumerate(zip(expected, actual, expected_lineage, actual_lineage)):
        if bits(e) != bits(f) or bits(g) != bits(h):
            failures.append(dict(index=index, scipy=[repr(e), repr(f)], lineage=[repr(g), repr(h)]))
    if expected_other != actual_other or labelled != actual_labelled:
        failures.append(dict(error='original-path cases differ', expected=str(expected_other), actual=str(actual_other)))
    report = dict(status='pass' if not failures else 'fail', masks=len(masks), other_cases=len(others) + 1,
                  seconds=seconds, counters=counters, failures=failures[:10])
    a.output.write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
    print(json.dumps(report, default=str))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
