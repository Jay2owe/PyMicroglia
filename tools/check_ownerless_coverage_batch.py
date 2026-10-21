"""Compare batched ownerless coverage with the frozen per-point sampler, exactly."""
import argparse
import json
from pathlib import Path
import platform
import sys
import time


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'snapshot', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--stacks', nargs='+', default=[
        '22_issue038_ownerless_body_recovery', '29_issue048_late_owner_backfill',
        '61_well_b2_issue005_bracketed_mixed_owner_flash'])
    a = p.parse_args()
    platform.platform = lambda *x, **k: 'Windows-ownerless-coverage-check'
    sys.path[:0] = [str(a.snapshot.resolve()), str((a.engine / 'code').resolve())]
    import numpy as np
    import pandas as pd
    import tifffile
    import ownerless_body_recovery as module
    from pymicroglia.tracking import ownerless_coverage_batch as batch
    reference = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    points = pd.read_csv(reference / '26_latent_body_tracks/out/latent_track_points.csv')
    rng = np.random.default_rng(20261020)
    cases = []
    for stem in a.stacks:
        cases.append((stem, points, tifffile.imread(reference / f'{stem}.tif'),
                      tifffile.imread(reference / f'{stem}_unclaimed.tif')))
    # Edge cases: borders, outside the image, tiny and huge radii, ties.
    labels = np.zeros((3, 23, 29), np.uint16)
    labels[:, 3:9, 4:10] = 5; labels[:, 3:9, 10:16] = 2; labels[1, 15:20, 20:29] = 7
    unclaimed = np.zeros_like(labels); unclaimed[2, 10:14, 0:5] = 3
    n = 4000
    synthetic = pd.DataFrame(dict(
        track_id=np.arange(n), frame=rng.integers(0, 3, n),
        x=rng.uniform(-8, 37, n), y=rng.uniform(-8, 31, n),
        radius_px=rng.choice([0.0, 0.5, 2.6666, 3.0, 7.9, 8.0, 40.0], n) + rng.uniform(0, 1e-9, n),
        state=rng.choice(['observed', 'latent_visible', 'latent_hidden'], n)))
    synthetic.loc[:9, 'x'] = 9.5; synthetic.loc[:9, 'y'] = 5.5  # exact two-way ties
    cases.append(('synthetic_edges', synthetic, labels, unclaimed))
    results = {}
    with batch.install(module) as stats:
        assert stats['supported']
        fast = module.attach_coverage
    for name, frame_points, stack, ledger in cases:
        tick = time.perf_counter(); expected = batch_original(module, fast)(frame_points, stack, ledger); before = time.perf_counter() - tick
        tick = time.perf_counter(); actual = fast(frame_points, stack, ledger); after = time.perf_counter() - tick
        pd.testing.assert_frame_equal(expected, actual, check_exact=True)
        assert all(expected[c].dtype == actual[c].dtype for c in expected)
        results[name] = dict(points=len(frame_points), before_seconds=before, after_seconds=after, exact=True)
        print(name, results[name], flush=True)
    assert stats['fallback_calls'] == 0, stats
    report = dict(status='pass', cases=results, counters=stats)
    a.output.write_text(json.dumps(report, indent=2) + '\n')


def batch_original(module, fast):
    # The adapter restored the original on exit; keep the frozen function.
    original = module.attach_coverage
    assert original is not fast
    return original


if __name__ == '__main__':
    main()
