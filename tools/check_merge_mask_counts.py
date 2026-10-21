"""Compare persistent-merge partition scoring with Boolean counts.

Hosts are unions of two real neighbouring cells; previous masks come from the
preceding frame; seeds are the cells' rounded centres and random host pixels;
raw images and transition frames are real. Original and adapted scores (every
value, including both partition masks) and identity shapes must be equal.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'

PARAMS = dict(minimum_seed_separation_px=3, peak_sigma_px=1.5, minimum_partition_px=8,
              position_weight=1.0, position_scale_px=10.0, area_weight=1.0, identity_shape_weight=0.5,
              identity_overlap_weight=0.5, motion_weight=1.0, body_shape_weight=0.5,
              motion_destination_weight=0.5)


def same(a, b):
    if a is None or b is None:
        return a is b
    if list(a) != list(b):
        return False
    for key in a:
        x, y = a[key], b[key]
        if isinstance(x, np.ndarray):
            if type(x) is not type(y) or x.dtype != y.dtype or not np.array_equal(x, y):
                return False
        elif type(x) is not type(y) or not (x == y or (x != x and y != y)):
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import persistent_merges as module
    import merge_mask_counts as adapter
    original_score, original_shape = module._partition_score, module._identity_shape
    labels = tifffile.imread(HISTORY / '11_issue017_rebase.tif')
    raw = tifffile.imread(HISTORY / '07_trimmed_evidence/out/95_full_A1_registered_raw.tif')
    rng = np.random.default_rng(20261020)
    failures, cases, scored = [], 0, 0
    times = dict(original=0.0, adapted=0.0)
    with adapter.install(module) as stats:
        assert stats['supported']
        for t in rng.choice(np.arange(1, len(labels)), 40, replace=False):
            frame, before = labels[int(t)], labels[int(t) - 1]
            lag = rng.normal(0, 2, frame.shape).astype(np.float32)
            smooth = ndi.gaussian_filter(raw[int(t)].astype(np.float32), sigma=1.5)
            ids = np.unique(frame)[1:]
            for identity in rng.choice(ids, 8):
                own = frame == identity
                ring = ndi.binary_dilation(own, iterations=3) & ~own
                neighbours = np.unique(frame[ring]); neighbours = neighbours[neighbours > 0]
                if not len(neighbours):
                    continue
                other = int(rng.choice(neighbours))
                host = own | (frame == other)
                pa, pb = dict(mask=before == identity), dict(mask=before == other)
                centres = [tuple(np.round(ndi.center_of_mass(m)).astype(int)) for m in (own, frame == other)]
                yy, xx = np.nonzero(host)
                seeds = [tuple(centres[0]), tuple(centres[1])] + [
                    (int(yy[i]), int(xx[i])) for i in rng.integers(0, len(yy), 2)]
                shapes = (original_shape(own), original_shape(frame == other))
                destination = (frame == identity) if rng.random() < .5 else None
                for sa in seeds:
                    for sb in seeds:
                        args = (host, raw[int(t)], pa, pb, np.array(centres[0], float), np.array(centres[1], float),
                                float(own.sum()), float((frame == other).sum()), shapes[0], shapes[1], lag,
                                sa, sb, PARAMS, destination, smooth)
                        t0 = time.perf_counter(); expected = original_score(*args)
                        t1 = time.perf_counter(); actual = module._partition_score(*args)
                        t2 = time.perf_counter()
                        times['original'] += t1 - t0; times['adapted'] += t2 - t1
                        cases += 1; scored += expected is not None
                        if not same(expected, actual):
                            failures.append(f'{int(t)}-{int(identity)}-{other}-{sa}-{sb}')
                for mask in (own, host, pa['mask']):
                    if not np.array_equal(original_shape(mask), module._identity_shape(mask)):
                        failures.append(f'shape-{int(t)}-{int(identity)}')
                    if original_shape(mask).dtype != module._identity_shape(mask).dtype:
                        failures.append(f'shape-dtype-{int(t)}')
        counted = stats['counted']
    report = dict(status='pass' if not failures else 'fail', cases=cases, scored=scored, counted=counted,
                  seconds=times, failures=failures[:20])
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report, indent=1))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
