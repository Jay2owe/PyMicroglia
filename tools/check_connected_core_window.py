"""Compare windowed connected cores in ownerless body recovery with the original, exactly.

python check_connected_core_window.py --engine <engine> --adapter <identity_candidate dir> --output <report.json>

Synthetic frames (float32, float64, integer and boolean; some smaller than the
background kernel) with random sigmas, window radii (including 0 and negative),
weak thresholds and points inside, on and beyond every edge, plus NaN points;
then real A1 raw frames at every tracker point of a sample of frames with the
default parameters. The core (dtype, shape, values), region and details must
equal the original's exactly, and errors must match.
"""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    import tifffile
    sys.path[:0] = [str(a.adapter.resolve()), str((a.engine / 'code').resolve())]
    import ownerless_body_recovery as module
    import connected_core_window as adapter
    original = module._connected_core
    timing = [0.0, 0.0]
    compared = errors = 0

    class Point:
        def __init__(self, x, y):
            self.x, self.y = x, y

    def outcome(function, *args):
        try:
            tick = time.perf_counter()
            value = function(*args)
            return 'ok', value, time.perf_counter() - tick
        except Exception as error:
            return 'error', (type(error).__name__, str(error)), 0.0

    def compare(frame, point, weak, params):
        nonlocal compared, errors
        old = outcome(original, frame, point, weak, params)
        new = outcome(module._connected_core, frame, point, weak, params)
        timing[0] += old[2]; timing[1] += new[2]
        compared += 1
        assert old[0] == new[0], (old[:2], new[:2])
        if old[0] == 'error':
            errors += 1
            assert old[1] == new[1], (old[1], new[1])
            return
        (core, region, details), (core2, region2, details2) = old[1], new[1]
        assert type(core) is type(core2) and core.dtype == core2.dtype and core.shape == core2.shape
        assert np.array_equal(core, core2) and region == region2
        assert list(details) == list(details2)
        for key in details:
            assert type(details[key]) is type(details2[key]) and repr(details[key]) == repr(details2[key]), key

    rng = np.random.default_rng(8)
    with adapter.install(module) as stats:
        assert stats['supported'], 'pinned sources differ'
        for case in range(300):
            height, width = int(rng.integers(3, 120)), int(rng.integers(3, 120))
            kind = case % 4
            frame = (rng.random((height, width)) * 50).astype([np.float32, np.float64, np.uint16, np.float32][kind])
            if kind == 3:
                frame = frame > 25
            params = dict(core_sigma_px=float(rng.choice([0.5, 1.2, 2.0, 3.3])),
                          background_sigma_px=float(rng.choice([4.0, 8.0, 12.5, 30.0])),
                          core_window_radius_px=int(rng.choice([7, 0, 3, 15, -1])),
                          core_peak_fraction=float(rng.uniform(0.1, 0.9)),
                          weak_threshold_fraction=float(rng.uniform(0.2, 0.8)))
            for _ in range(6):
                x = float(rng.uniform(-10, width + 10)); y = float(rng.uniform(-10, height + 10))
                if rng.random() < .2:
                    x, y = round(x) + .5, round(y) - .5
                compare(frame, Point(x, y), float(rng.uniform(0, 20)), params)
            compare(frame, Point(np.nan, 3.0), 1.0, params)
        compare(np.zeros((9, 9), np.float32), Point(4, 4), 1.0, dict(core_sigma_px=0.0))
        compare(np.zeros((9, 9), np.complex64), Point(4, 4), 1.0, {})
        folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
        raw = tifffile.imread(folder / '07_trimmed_evidence/out/95_full_A1_registered_raw.tif')
        points = pd.read_csv(folder / '26_latent_body_tracks/out/latent_track_points.csv')
        observed = points[points.state == 'observed']
        real = 0
        for frame_index in range(0, len(raw), 9):
            for point in observed[observed.frame.astype(int) == frame_index].itertuples(index=False):
                compare(raw[frame_index], point, 3.0, {})
                real += 1
    assert module._connected_core is original
    report = dict(status='pass', compared=compared, original_errors=errors, real_points=real,
                  original_seconds=round(timing[0], 2), adapter_seconds=round(timing[1], 2), counters=stats)
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
