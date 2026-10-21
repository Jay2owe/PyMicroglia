"""Compare boxed physical-lineage core measurements with the whole-frame originals.

Real cell masks, unions of touching cells and dilated/thinned versions from
accepted-label frames (including cells on the frame edge), plus empty, full,
single-pixel, one-pixel-wide and random masks; core sizes, the returned core
frames and substantial-core counts for several minimum sizes must be equal.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'


def same(a, b):
    if isinstance(a, tuple):
        return (len(a) == len(b) and type(a[0]) is type(b[0]) and a[0] == b[0]
                and ((a[1] is None and b[1] is None) or (
                    a[1] is not None and b[1] is not None and type(a[1]) is type(b[1])
                    and a[1].dtype == b[1].dtype and np.array_equal(a[1], b[1]))))
    return type(a) is type(b) and a == b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import physical_lineage_aliases as module
    import physical_core_boxes as adapter
    stack = tifffile.imread(HISTORY / '11_issue017_rebase.tif')
    rng = np.random.default_rng(20261021)
    masks = []
    for t in rng.choice(len(stack), 50, replace=False):
        frame = stack[t]
        ids = np.unique(frame)[1:]
        edge = set(np.unique(np.concatenate([frame[0], frame[-1], frame[:, 0], frame[:, -1]]))) - {0}
        chosen = list(rng.choice(ids, 20, replace=False)) + list(edge)[:4]
        for identity in chosen:
            cell = frame == identity
            masks.append(cell)
            ring = ndi.binary_dilation(cell, iterations=2) & ~cell
            others = np.unique(frame[ring]); others = others[others != 0]
            if len(others):
                masks.append(cell | (frame == int(rng.choice(others))))
            masks.append(ndi.binary_dilation(cell, iterations=int(rng.integers(1, 4))))
            masks.append(cell & (rng.random(cell.shape) < .8))
    shape = stack.shape[1:]
    extra = [np.zeros(shape, bool), np.ones(shape, bool)]
    one = np.zeros(shape, bool); one[0, 0] = True; extra.append(one)
    line = np.zeros(shape, bool); line[10, 5:60] = True; extra.append(line)
    corner = np.zeros(shape, bool); corner[-6:, -9:] = True; extra.append(corner)
    extra += [rng.random(shape) < p for p in (.01, .5, .9)]
    extra.append(np.ones((4, 4), np.uint8))
    masks += extra
    originals = (module.largest_core, module.substantial_core_count)
    failures, seconds = [], dict(original=0.0, boxed=0.0)
    with adapter.install(module) as stats:
        boxed = (module.largest_core, module.substantial_core_count)
        for number, mask in enumerate(masks):
            t0 = time.perf_counter()
            expected = [originals[0](mask)] + [originals[1](mask, m) for m in (0, 1, 5, 20)]
            t1 = time.perf_counter()
            actual = [boxed[0](mask)] + [boxed[1](mask, m) for m in (0, 1, 5, 20)]
            t2 = time.perf_counter()
            seconds['original'] += t1 - t0; seconds['boxed'] += t2 - t1
            if not all(same(x, y) for x, y in zip(expected, actual)):
                failures.append(number)
        counters = dict(stats)
    assert (module.largest_core, module.substantial_core_count) == originals
    report = dict(status='pass' if not failures else 'fail', masks=len(masks), seconds=seconds,
                  counters=counters, failures=failures[:20])
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
