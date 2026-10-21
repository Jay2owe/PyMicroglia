"""Compare persistent-merge partition scoring with the boxed scoring.

Hosts are unions of two real neighbouring cells (cells touching the frame
edge are sampled deliberately); previous masks come from the preceding frame,
as plain Boolean images or lazy label masks; seeds are the cells' rounded
centres, random host pixels and random pixels outside the host; destinations
are absent, plain or lazy; smoothing is supplied or left to the function. The
engine's own function is the reference: every returned value (both partition
images included) must be equal in type and value.
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
        elif isinstance(x, float) and np.float64(x).tobytes() != np.float64(y).tobytes():
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True, help='composed snapshot (for LabelMask)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=40)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve()), str(a.snapshot.resolve())]
    import persistent_merges as module
    import merge_partition_boxes as adapter
    from pymicroglia.tracking.frozen_io import LabelMask
    original = module._partition_score
    labels = tifffile.imread(HISTORY / '11_issue017_rebase.tif')
    raw = tifffile.imread(HISTORY / '07_trimmed_evidence/out/95_full_A1_registered_raw.tif')
    rng = np.random.default_rng(20261021)
    failures, cases, scored = [], 0, 0
    times = dict(original=0.0, boxed=0.0)
    with adapter.install(module) as stats:
        assert stats['supported']
        for t in rng.choice(np.arange(1, len(labels)), a.frames, replace=False):
            t = int(t)
            frame, before = labels[t], labels[t - 1]
            lag = rng.normal(0, 2, frame.shape).astype(np.float32)
            lag[rng.random(frame.shape) < .02] = np.nan
            smooth = ndi.gaussian_filter(raw[t].astype(np.float32), sigma=1.5)
            ids = np.unique(frame)[1:]
            edge_ids = np.unique(np.concatenate([frame[0], frame[-1], frame[:, 0], frame[:, -1]]))
            edge_ids = edge_ids[edge_ids > 0]
            chosen = list(rng.choice(ids, 6)) + (list(rng.choice(edge_ids, 2)) if len(edge_ids) else [])
            for identity in chosen:
                own = frame == identity
                ring = ndi.binary_dilation(own, iterations=3) & ~own
                neighbours = np.unique(frame[ring]); neighbours = neighbours[neighbours > 0]
                if not len(neighbours):
                    continue
                other = int(rng.choice(neighbours))
                host = own | (frame == other)
                lazy = rng.random() < .5
                if lazy:
                    snapshot = before.copy()
                    pa, pb = dict(mask=LabelMask(snapshot, int(identity))), dict(mask=LabelMask(snapshot, other))
                else:
                    pa, pb = dict(mask=before == identity), dict(mask=before == other)
                centres = [tuple(int(v) for v in np.round(ndi.center_of_mass(m))) for m in (own, frame == other)]
                yy, xx = np.nonzero(host)
                seeds = centres + [(int(yy[i]), int(xx[i])) for i in rng.integers(0, len(yy), 2)]
                seeds.append((int(rng.integers(0, frame.shape[0])), int(rng.integers(0, frame.shape[1]))))
                shapes = (module._identity_shape(own), module._identity_shape(frame == other))
                choice = rng.random()
                destination = (None if choice < .4 else (frame == identity) if choice < .7
                               else LabelMask(frame.copy(), int(identity)))
                given = smooth if rng.random() < .85 else None
                for sa in seeds:
                    for sb in seeds:
                        args = (host, raw[t], pa, pb, np.array(centres[0], float), np.array(centres[1], float),
                                float(own.sum()), float((frame == other).sum()), shapes[0], shapes[1], lag,
                                sa, sb, PARAMS, destination, given)
                        t0 = time.perf_counter(); expected = original(*args)
                        t1 = time.perf_counter(); actual = module._partition_score(*args)
                        t2 = time.perf_counter()
                        times['original'] += t1 - t0; times['boxed'] += t2 - t1
                        cases += 1; scored += expected is not None
                        if not same(expected, actual):
                            failures.append(f'{t}-{int(identity)}-{other}-{sa}-{sb}-{lazy}')
        counts = dict(boxed=stats['boxed'], fallback=stats['fallback'])
    report = dict(status='pass' if not failures and counts['boxed'] else 'fail', cases=cases, scored=scored,
                  **counts, seconds=times, failures=failures[:20])
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report, indent=1))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
