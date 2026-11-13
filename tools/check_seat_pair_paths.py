"""Compare seat-identity saddle depths, seat claims and freed bodies with the originals.

The Issue 013 module is loaded from the engine and its own soma detector finds
cores on accepted labels and the registered raw. Every pair of cores sharing a
component is measured on the raw, on raw coarsened into plateaus and on a flat
image, with cores held from the previous frame (partly outside the body) as
well. Seat tracks over a run of frames give the claims, also on labels edited
into two-name ties under each core and with a label too large for the count
table. Freed bodies use seats painted on real cells, second bodies painted
over neighbours, and evicted names that are absent, present or seated, at two
stranded-piece sizes. Raw frames, label movies and frames being freed are
also given as memory-mapped arrays, as the pooled run passes them. Every
result must equal the original's, including dtype and order.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'
RAW = HISTORY / '07_trimmed_evidence/out/95_full_A1_registered_raw.tif'


def same(a, b):
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=10)
    parser.add_argument('--track-frames', type=int, default=60)
    parser.add_argument('--scratch', type=Path, required=True)
    a = parser.parse_args()
    a.scratch.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(ENGINE / 'code/corrections/seat_identity'), str(ENGINE / 'code'), str(a.adapters.resolve())]
    path = ENGINE / 'code/corrections/seat_identity/seat_identity.py'
    spec = importlib.util.spec_from_file_location('seat_identity', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    import seat_pair_paths
    stats = {}
    params = json.loads((ENGINE / 'code/accepted_history_parameters.json').read_text(encoding='utf-8'))
    cfg = params['issue013']['seat_identity']['seats']
    fast_ratio = seat_pair_paths.valley_ratio(module.valley_ratio, stats.setdefault('ratio', dict(boxed_ratios=0, original_ratios=0)))
    claim_stats = dict(counted_claims=0, original_claims=0)
    fast_claims = seat_pair_paths.seat_claims(module, claim_stats)
    free_stats = dict(counted_frees=0, original_frees=0)
    fast_free = seat_pair_paths.free_second_bodies(module, free_stats)
    labels = tifffile.imread(HISTORY / '08_issue012.tif')
    picks = sorted(set(np.linspace(0, len(labels) - 1, a.frames).round().astype(int).tolist())
                   | set(range(a.track_frames)))
    raw = {t: tifffile.imread(RAW, key=t) for t in picks}
    cores = {t: module.soma_cores(raw[t], labels[t] > 0, float(cfg['percentile']),
                                  int(cfg['minimum_core_area_px'])) for t in picks}
    failures, calls = [], dict(ratio=0, claims=0, free=0)
    seconds = {name: dict(original=0.0, fast=0.0) for name in calls}

    def compare(name, original, fast, *args):
        t0 = time.perf_counter(); expected = original(*args); t1 = time.perf_counter()
        actual = fast(*args); seconds[name]['original'] += t1 - t0
        seconds[name]['fast'] += time.perf_counter() - t1
        calls[name] += 1
        if not same(expected, actual):
            failures.append(name)

    ratio_frames = np.linspace(0, len(labels) - 1, a.frames).round().astype(int).tolist()
    for t in ratio_frames:
        numbered, components = module.ndi.label(labels[t] > 0, structure=module.STRUCTURE)
        previous = cores.get(t - 1, [])
        mapped = np.memmap(a.scratch / f'raw_{t}.dat', dtype=raw[t].dtype, mode='w+', shape=raw[t].shape)
        mapped[:] = raw[t]
        images = dict(raw=raw[t], plateau=(raw[t] // 64).astype(raw[t].dtype),
                      flat=np.zeros_like(raw[t]), mapped=mapped)
        for number in range(1, components + 1):
            component = numbered == number
            here = [core['mask'] for core in cores[t] + previous if (core['mask'] & component).any()]
            for i in range(len(here)):
                for j in range(len(here)):
                    if i == j:
                        continue
                    for image in images.values():
                        compare('ratio', module.valley_ratio, fast_ratio, image, component, here[i], here[j])
    tracks = module.track_seats([cores[t] for t in range(a.track_frames)], float(cfg['maximum_step_px']),
                                int(cfg['maximum_gap_frames']))
    window = np.ascontiguousarray(labels[:a.track_frames])
    tied = window.copy(); tied[:, :, 1::2] = np.where(tied[:, :, 1::2] > 0, tied[:, :, 1::2] + 1, 0)
    large = window.copy(); large[large == large.max()] = 9000
    mapped = np.memmap(a.scratch / 'labels.dat', dtype=window.dtype, mode='w+', shape=window.shape)
    mapped[:] = window
    for movie in (window, tied, large, mapped):
        for minimum in ((5, 8), (1, 1), (30, 40)):
            compare('claims', module.seat_claims, fast_claims, movie, tracks, *minimum)
    rng = np.random.default_rng(31)
    for t in ratio_frames:
        current = labels[t].copy()
        names = [int(v) for v in np.unique(current) if v]
        if len(names) < 12:
            continue
        chosen = rng.choice(names, 12, replace=False).tolist()
        seated, painted = chosen[:6], chosen[6:]
        seat_regions = np.zeros(current.shape, np.uint16)
        evicted = {}
        for k, identity in enumerate(seated):
            seat_regions[current == identity] = identity
            current[labels[t] == painted[k]] = identity            # a second body on a neighbour
            evicted[identity] = [painted[k], 60000 + k, seated[(k + 1) % 6], names[k]]
        held = np.memmap(a.scratch / f'current_{t}.dat', dtype=current.dtype, mode='w+', shape=current.shape)
        held[:] = current
        for frame in (current, held):
            for stranded in (10, 400):
                compare('free', module.free_second_bodies, fast_free, frame, seat_regions, evicted,
                        int(cfg['minimum_body_px']), stranded)
    report = dict(status='pass' if not failures and all(calls.values()) else 'fail', calls=calls,
                  stats=dict(ratio=stats['ratio'], claims=claim_stats, free=free_stats),
                  seconds=seconds, failures=failures[:10])
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
