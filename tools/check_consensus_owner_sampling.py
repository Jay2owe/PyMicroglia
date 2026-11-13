"""Compare owner_consensus.attach_owners with and without the batched exact votes.

The frozen module is loaded from the A1 engine. The real latent-body track
points are attached on accepted labels from three points in the history, with
the default radius bounds and with other bounds (narrow, wide, fractional,
zero and equal), on the recorded table, a shuffled one, one with points on the
image edge, one with a missing radius and an empty one; every table must be
identical to the original's, and an unordered bound must raise the same error.
Then the whole dominant-seat projection diversion discovery (which samples
through the module in its bracketed mixed-owner flash base) is compared on a
window of frames with the run's configured parameters.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


def outcome(function, *args):
    try:
        return ('value', function(*args))
    except Exception as error:
        return ('error', type(error).__name__, str(error))


def same(x, y):
    if type(x) is not type(y):
        return False
    if isinstance(x, (tuple, list)):
        return len(x) == len(y) and all(same(a, b) for a, b in zip(x, y))
    if isinstance(x, dict):
        return list(x) == list(y) and all(same(x[k], y[k]) for k in x)
    if isinstance(x, np.ndarray):
        return x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y)
    if isinstance(x, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(x, y, check_exact=True)
            return x.to_csv(index=False) == y.to_csv(index=False)
        except AssertionError:
            return False
    return x == y


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--engine', type=Path, required=True)
    p.add_argument('--adapters', type=Path, required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--window', default='0:160')
    p.add_argument('--config', type=Path, required=True)
    a = p.parse_args()
    sys.path[:0] = [str(a.adapters.resolve()), str((a.engine / 'code').resolve())]
    import owner_consensus
    import bracketed_mixed_owner_flash as flash
    import dominant_seat_projection_diversion as diversion
    import dominant_seat_projection_diversion_integration as integration
    import common
    import correction_owner_sampling as sampling
    history = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    points = pd.read_csv(history / '26_latent_body_tracks/out/latent_track_points.csv', low_memory=False)
    original = owner_consensus.attach_owners
    failures, cases = [], {}
    bounds = [None, {}, dict(minimum_owner_sample_radius_px=1.0, maximum_owner_sample_radius_px=4.0),
              dict(minimum_owner_sample_radius_px=3.0, maximum_owner_sample_radius_px=9.5),
              dict(minimum_owner_sample_radius_px=0.0, maximum_owner_sample_radius_px=0.0),
              dict(minimum_owner_sample_radius_px=2.5, maximum_owner_sample_radius_px=2.5),
              dict(minimum_owner_sample_radius_px=5.0, maximum_owner_sample_radius_px=3.0)]
    rng = np.random.default_rng(17)
    for name in ('61_well_b2_issue005_bracketed_mixed_owner_flash.tif', '05_issue009.tif', '08_issue012.tif'):
        labels = tifffile.imread(history / name)
        edge = points.sample(400, random_state=1).copy()
        edge.loc[edge.index[:100], 'x'] = 0.2
        edge.loc[edge.index[100:200], 'y'] = labels.shape[1] - 1.4
        edge.loc[edge.index[200:300], 'x'] = labels.shape[2] - 0.6
        missing = points.sample(300, random_state=2).copy()
        missing.loc[missing.index[:5], 'radius_px'] = np.nan
        tables = dict(recorded=points, shuffled=points.sample(frac=1, random_state=3), edge=edge,
                      missing=missing, empty=points.iloc[0:0])
        for table_name, table in tables.items():
            for index, params in enumerate(bounds):
                if table_name not in ('recorded', 'shuffled') and index > 2:
                    continue
                args = (table, labels) if params is None else (table, labels, params)
                label = f'{name}:{table_name}:bounds{index}'
                t0 = time.perf_counter()
                expected = outcome(original, *args)
                t1 = time.perf_counter()
                with sampling.install(owner_consensus) as stats:
                    assert stats['supported']
                    actual = outcome(owner_consensus.attach_owners, *args)
                    # A second call reads the held votes for unchanged frames.
                    again = outcome(owner_consensus.attach_owners, *args)
                t2 = time.perf_counter()
                cases[label] = dict(kind=expected[0], original_seconds=round(t1 - t0, 3),
                                    adapted_seconds_two_calls=round(t2 - t1, 3),
                                    measured=stats['measured'], reused=stats['reused'],
                                    fallback_calls=stats['fallback_calls'])
                if not same(expected, actual) or not same(expected, again):
                    failures.append(label)
        del labels
    start, stop = map(int, a.window.split(':'))
    labels = tifffile.imread(history / '61_well_b2_issue005_bracketed_mixed_owner_flash.tif', key=range(start, stop))
    raw = tifffile.imread(history / '07_trimmed_evidence/out/95_full_A1_registered_raw.tif', key=range(start, stop))
    window = points[(points.frame >= start) & (points.frame < stop)].copy()
    window['frame'] -= start
    t0 = time.perf_counter()
    _, params = integration._configured(common.Config.load(a.config.resolve()).values)
    expected = outcome(diversion.discover, labels, raw, window, params)
    t1 = time.perf_counter()
    with sampling.install(owner_consensus) as stats:
        actual = outcome(diversion.discover, labels, raw, window, params)
    t2 = time.perf_counter()
    cases['diversion.discover'] = dict(kind=expected[0], original_seconds=round(t1 - t0, 2),
                                   adapted_seconds=round(t2 - t1, 2), measured=stats['measured'])
    if not same(expected, actual) or stats['measured'] == 0:
        failures.append('diversion.discover')
    if owner_consensus.attach_owners is not original or flash.owner_consensus.attach_owners is not original:
        failures.append('restore')
    report = dict(status='pass' if not failures else 'fail', cases=cases, failures=failures,
                  sources={str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                           for f in [Path(__file__), Path(sampling.__file__), Path(owner_consensus.__file__)]})
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], failures=failures[:20], n=len(cases),
                          flash=cases['diversion.discover'],
                          recorded={k: v for k, v in cases.items() if k.endswith('recorded:bounds0')})))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
