"""Compare the production physical detector with and without batched_detection_cores, exactly.

The detector is composed as optimized_pipeline composes it (bucketed peak
search, relative visibility with the reused core structure, fractional raw
peaks), with and without the bounded ndi.label adapter. Every frame of the A1
registered raw movie runs with the accepted raw-detection settings and with the
defaults; a subset runs with altered window radius, radius bounds, quantiles,
peak distance and smoothing, and synthetic frames cover empty evidence, frames
smaller than a window, other integer types, negative values, float frames and
invalid settings. Rows and thresholds must be equal in values, types and float
bits, and errors equal in type and message.
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'


def same(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return np.float64(a).tobytes() == np.float64(b).tobytes()
    if isinstance(a, dict):
        return list(a) == list(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def outcome(function, *args):
    try:
        return ('ok', function(*args))
    except Exception as error:
        return ('error', (type(error).__name__, str(error)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--subset', type=int, default=40)
    parser.add_argument('--scratch', type=Path, required=True)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import tifffile
    import raw_physical_hypotheses as module
    from peak_search_speed import bucketed_peak_search
    from relative_detection_visibility import relative_detection_visibility
    from fractional_raw_peaks import fractional_raw_peaks
    import batched_detection_cores
    import bounded_label
    import array_storage
    config = json.loads((ENGINE / 'config.json').read_text(encoding='utf-8'))
    accepted = config['accepted_postprocessing']['field_wide_owner_consensus'].get('raw_detection', {})
    raw = tifffile.imread(next((HISTORY / '07_trimmed_evidence/out').glob('*_registered_raw.tif')))
    # The benchmark adopts pinned inputs into the memory pool, so production
    # frames are the pool's WorkingArray views; plain memory maps are covered too.
    a.scratch.mkdir(parents=True, exist_ok=True)
    pool = array_storage.Pool(a.scratch / 'pool')
    mapped = np.memmap(a.scratch / 'raw.dat', dtype=raw.dtype, mode='w+', shape=raw.shape)
    mapped[:] = raw
    mapped.flush()
    working = pool.adopt(mapped, a.scratch / 'raw.dat')
    working.flags.writeable = False
    assert type(working[0]).__name__ == 'WorkingArray' and working[0]._pool is pool
    plain_map = np.memmap(a.scratch / 'raw.dat', dtype=raw.dtype, mode='r', shape=raw.shape)
    rng = np.random.default_rng(20261024)
    subset = raw[np.linspace(0, len(raw) - 1, a.subset).astype(int)]
    variants = [
        ('window_0', dict(accepted, core_window_radius_px=0)),
        ('window_1', dict(accepted, core_window_radius_px=1)),
        ('window_3', dict(accepted, core_window_radius_px=3)),
        ('window_20', dict(accepted, core_window_radius_px=20)),
        ('window_64', dict(accepted, core_window_radius_px=64)),
        ('window_65_fallback', dict(accepted, core_window_radius_px=65)),
        ('window_negative_fallback', dict(accepted, core_window_radius_px=-2)),
        ('radius_bounds_swapped', dict(accepted, minimum_core_radius_px=5.0, maximum_core_radius_px=2.0)),
        ('radius_bounds_text', dict(accepted, minimum_core_radius_px='2.5', maximum_core_radius_px=6)),
        ('radius_bound_invalid', dict(accepted, minimum_core_radius_px='wide')),
        ('quantiles', dict(accepted, weak_peak_quantile=0.2, strong_peak_quantile=0.9)),
        ('quantiles_invalid', dict(accepted, weak_peak_quantile=0.8, strong_peak_quantile=0.7)),
        ('distance_0', dict(accepted, minimum_peak_distance_px=0.0)),
        ('distance_2_5', dict(accepted, minimum_peak_distance_px=2.5)),
        ('sigma', dict(accepted, core_sigma_px=0.7, background_sigma_px=5.0)),
    ]
    synthetic = [
        ('zeros', np.zeros((64, 64), np.uint16), 0),
        ('small_frame', raw[100][200:205, 200:206].copy(), 1),
        ('uint8', (raw[50] // 64).astype(np.uint8), 2),
        ('int16_negative', (raw[60].astype(np.int32) - 300).astype(np.int16), 3),
        ('int32_large', raw[70].astype(np.int32) * 30000, np.int64(4)),
        ('float32_fallback', raw[80].astype(np.float32) + 0.25, 5),
        ('frame_index_float', raw[90], 6.0),
        ('noncontiguous', raw[110][::2, ::2], 7),
        ('three_dimensional', raw[0:2], 8),
        ('single_spot', np.pad(np.full((3, 3), 900, np.uint16), 30), 9),
        ('random', rng.integers(0, 3000, (120, 90)).astype(np.uint16), 10),
    ]
    cases = [('accepted', accepted, frame, index) for index, frame in enumerate(raw)]
    cases += [('defaults', {}, frame, index) for index, frame in enumerate(raw)]
    cases += [(name, params, frame, index) for name, params in variants for index, frame in enumerate(subset)]
    cases += [(name, accepted, frame, index) for name, frame, index in synthetic]
    cases += [(name + ':defaults', {}, frame, index) for name, frame, index in synthetic]
    cases += [('pool', accepted, working[index], index) for index in range(len(raw))]
    cases += [('memmap', accepted, plain_map[index], index) for index in np.linspace(0, len(raw) - 1, a.subset).astype(int)]
    cases += [('pool_small', accepted, working[100][200:205, 200:206], 1), ('pool_strided', accepted, working[110][::2, ::2], 7)]
    report = dict(cases=len(cases) * 2, failures=[], seconds=dict(original=0.0, batched=0.0), stats={}, rows=0, errors=0)
    for bounded in (False, True):
        with ExitStack() as stack:
            if bounded:
                stack.enter_context(bounded_label.install())
            stack.enter_context(bucketed_peak_search(module))
            stack.enter_context(relative_detection_visibility(module, reuse_core_structure=True))
            stack.enter_context(fractional_raw_peaks(module))
            production = module.detect_frame
            expected = []
            t0 = time.perf_counter()
            for name, params, frame, index in cases:
                expected.append(outcome(production, frame, index, params))
            t1 = time.perf_counter()
            with batched_detection_cores.install(module) as stats:
                assert stats['supported'], 'adapter did not recognise the production detector'
                actual = [outcome(module.detect_frame, frame, index, params) for name, params, frame, index in cases]
            t2 = time.perf_counter()
            assert module.detect_frame is production
        report['seconds']['original'] += t1 - t0
        report['seconds']['batched'] += t2 - t1
        report['stats']['bounded' if bounded else 'plain'] = stats
        for (name, params, frame, index), left, right in zip(cases, expected, actual):
            if left[0] == 'ok':
                report['rows'] += len(left[1][0])
            else:
                report['errors'] += 1
            if not same(left, right):
                report['failures'].append(dict(case=name, index=repr(index), bounded=bounded,
                                               expected=repr(left)[:300], actual=repr(right)[:300]))
    # The owner-consensus frame pool: two threads with the production detector,
    # against the adapter's pool size with the batched detector.
    import parallel_detection_context as detection_pool
    with ExitStack() as stack:
        stack.enter_context(bounded_label.install())
        stack.enter_context(bucketed_peak_search(module))
        stack.enter_context(relative_detection_visibility(module, reuse_core_structure=True))
        stack.enter_context(fractional_raw_peaks(module))
        t0 = time.perf_counter()
        expected = list(detection_pool.detect_frames(working, accepted, module.detect_frame))
        t1 = time.perf_counter()
        with batched_detection_cores.install(module) as cores, batched_detection_cores.detection_workers(detection_pool) as workers:
            assert workers['supported']
            actual = list(detection_pool.detect_frames(working, accepted, module.detect_frame))
        t2 = time.perf_counter()
    report['pool'] = dict(frames=len(raw), equal=same(expected, actual), workers=workers, cores=cores,
                          original_seconds=round(t1 - t0, 2), adapted_seconds=round(t2 - t1, 2))
    report['status'] = 'pass' if not report['failures'] and report['rows'] and report['pool']['equal'] and cores['frames'] == len(raw) and all(
        s['peaks'] for s in report['stats'].values()) and not any(
        s['fallback_types'].get('WorkingArray') for s in report['stats'].values()) else 'fail'
    report['failures'] = report['failures'][:20]
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report, indent=1))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
