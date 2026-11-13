"""Compare the scalar motion-support sampler with the engine's cached sampler.

Both wrappers are installed on a stand-in linker module holding the frozen
latent-body support function, and sample the same planes along the same
links: real registered raw frames, signed and float planes with signed zeros,
negative values and border maxima, boolean and non-finite planes, and links
given as Python floats and integers, NumPy float32/float64/int64 values,
exact half-pixel points, far outside points, infinities and NaN. Every result
must be the same float with the same bits, sign of zero included.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
RAW = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history/07_trimmed_evidence/out/95_full_A1_registered_raw.tif'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--links', type=int, default=20000)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import motion_sampling_speed
    import latent_body_tracks
    import motion_support_scalars as adapter
    rng = np.random.default_rng(32)
    raw = tifffile.imread(RAW, key=range(0, 40, 10))
    planes = [raw[i] for i in range(len(raw))]
    signed = rng.integers(-500, 500, (60, 80)).astype(np.int32)
    zeros = np.where(rng.random((60, 80)) < 0.5, -0.0, 0.0).astype(np.float32)
    mixed = (rng.normal(0, 1, (60, 80)) * (rng.random((60, 80)) < 0.3)).astype(np.float64)
    mixed[mixed == 0] = -0.0
    negative = -np.abs(rng.normal(5, 2, (50, 50))).astype(np.float32)
    nonfinite = rng.random((40, 40)).astype(np.float32); nonfinite[3, 3] = np.nan
    planes += [signed, zeros, mixed, negative, rng.random((30, 30)) < 0.2, nonfinite,
               (rng.random((30, 30)) * 70000).astype(np.float64), rng.random((20, 20)).astype(np.float16)]

    def coordinate(limit):
        kind = rng.integers(0, 9)
        value = rng.uniform(-6, limit + 6)
        if kind == 0: return float(np.floor(value)) + 0.5
        if kind == 1: return int(value)
        if kind == 2: return np.float32(value)
        if kind == 3: return np.int64(int(value))
        if kind == 4: return float(rng.choice([np.inf, -np.inf, np.nan, 1e300, -1e300]))
        if kind == 5: return np.float64(value)
        if kind == 6: return bool(rng.integers(0, 2))
        return float(value)

    failures, calls = [], 0
    seconds = dict(original=0.0, scalar=0.0)
    for plane in planes:
        links = [(coordinate(plane.shape[1]), coordinate(plane.shape[0]),
                  coordinate(plane.shape[1]), coordinate(plane.shape[0])) for _ in range(a.links // len(planes))]
        results = []
        for wrapper in (motion_sampling_speed.cached_motion_samples, adapter.make(motion_sampling_speed.cached_motion_samples)):
            linker = SimpleNamespace(_motion_support_plane=latent_body_tracks._motion_support_plane)
            with wrapper(linker):
                t0 = time.perf_counter()
                values = []
                for link in links:
                    try:
                        values.append(linker._motion_support_plane(plane, *link))
                    except Exception as error:
                        values.append(('error', type(error).__name__))
                elapsed = time.perf_counter() - t0
            seconds['original' if not results else 'scalar'] += elapsed
            results.append(values)
        for link, left, right in zip(links, *results):
            calls += 1
            same = (type(left) is type(right) and
                    (left == right if isinstance(left, tuple) else np.float64(left).tobytes() == np.float64(right).tobytes()))
            if not same:
                failures.append(dict(dtype=str(plane.dtype), link=[repr(v) for v in link], values=[repr(left), repr(right)]))
    stats = dict(adapter.make(None).stats)
    with adapter.install(motion_sampling_speed) as installed:
        assert installed['supported'] and motion_sampling_speed.cached_motion_samples.__wrapped__ is not None
    report = dict(status='pass' if not failures else 'fail', calls=calls, seconds=seconds, failures=failures[:10])
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
