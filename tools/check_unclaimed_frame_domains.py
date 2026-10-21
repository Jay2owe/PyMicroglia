"""Compare per-frame unclaimed-body discovery with the engine's original.

Real (labels, unclaimed) checkpoint pairs with the registered raw movie, at
the accepted parameters and at settings that reach every decision, plus
synthetic movies with gaps, ties and refusals. Events (including their
movie-sized domains) and both audit tables must be identical.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'
ACCEPTED = dict(minimum_duration_fraction=0.5, minimum_area_px=8, minimum_largest_fraction=0.1,
                minimum_eroded_core_px=15, minimum_median_raw=0)


def same(expected, actual):
    e_events, e_source, e_frames = expected
    a_events, a_source, a_frames = actual
    if len(e_events) != len(a_events):
        return 'event count'
    for e, a in zip(e_events, a_events):
        if set(e) != set(a):
            return 'event keys'
        for key in e:
            if key == 'domain':
                if (e[key].shape != a[key].shape or e[key].dtype != a[key].dtype
                        or not np.array_equal(e[key], a[key])):
                    return 'event domain'
            elif type(e[key]) is not type(a[key]) or e[key] != a[key]:
                return f'event {key}'
    for e, a, name in ((e_source, a_source, 'source'), (e_frames, a_frames, 'frames')):
        try:
            pd.testing.assert_frame_equal(e, a, check_exact=True)
        except AssertionError as error:
            return f'{name} table: {error}'
        for column in e.columns:
            if [type(v) for v in e[column]] != [type(v) for v in a[column]]:
                return f'{name} value types in {column}'
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--synthetic', type=int, default=300)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import unclaimed_body_identity as module
    import unclaimed_frame_domains as adapter
    original = module.discover_unclaimed_bodies
    failures, rows = [], []
    times = dict(original=0.0, framed=0.0)

    def compare(labels, unclaimed, raw, params, label):
        t0 = time.perf_counter(); expected = original(labels, unclaimed, raw, params)
        with adapter.install(module) as stats:
            assert stats['supported']
            t1 = time.perf_counter(); actual = module.discover_unclaimed_bodies(labels, unclaimed, raw, params)
            t2 = time.perf_counter()
        assert module.discover_unclaimed_bodies is original
        times['original'] += t1 - t0; times['framed'] += t2 - t1
        problem = same(expected, actual)
        decisions = expected[1]['decision'].value_counts().to_dict() if len(expected[1]) else {}
        rows.append(dict(case=label, events=len(expected[0]), sources=len(expected[1]), decisions=decisions))
        if problem:
            failures.append(dict(case=label, problem=problem[:400]))

    raw = tifffile.imread(HISTORY / '07_trimmed_evidence/out/95_full_A1_registered_raw.tif')
    settings = [ACCEPTED,
                dict(ACCEPTED, minimum_duration_fraction=0.0, minimum_eroded_core_px=0),
                dict(ACCEPTED, minimum_area_px=1, minimum_largest_fraction=0.9, minimum_median_raw=150)]
    for stem in ('06_issue010', '08_issue012', '09_seat_identity', '11_issue017_rebase'):
        labels = tifffile.imread(HISTORY / f'{stem}.tif')
        unclaimed = tifffile.imread(HISTORY / f'{stem}_unclaimed.tif')
        for number, params in enumerate(settings):
            compare(labels, unclaimed, raw, params, f'{stem}-settings{number}')
        # Free the claimed overlap so accepted bodies are reachable.
        freed = labels.copy(); freed[unclaimed > 0] = 0
        compare(freed, unclaimed, raw, settings[1], f'{stem}-freed')
    rng = np.random.default_rng(20261020)
    for case in range(a.synthetic):
        shape = (int(rng.integers(1, 9)), int(rng.integers(5, 24)), int(rng.integers(5, 24)))
        unclaimed = (rng.random(shape) < float(rng.choice([.05, .2, .5]))) * rng.integers(1, 4, shape)
        unclaimed = unclaimed.astype(rng.choice([np.uint16, np.int32]))
        labels = ((rng.random(shape) < .3) & (unclaimed == 0)).astype(np.uint16) * 7
        if rng.random() < .5:
            labels[:] = 0
        raw_small = rng.integers(0, 400, shape).astype(np.uint16)
        params = dict(minimum_duration_fraction=float(rng.choice([0, .3, .9])),
                      minimum_area_px=int(rng.integers(0, 6)),
                      minimum_largest_fraction=float(rng.choice([0, .5, 1])),
                      minimum_eroded_core_px=int(rng.integers(0, 3)),
                      minimum_median_raw=float(rng.choice([0, 100, 300])))
        compare(labels, unclaimed, raw_small, params, f'synthetic-{case}')
    report = dict(status='pass' if not failures else 'fail', seconds=times,
                  real=[r for r in rows if not r['case'].startswith('synthetic')],
                  synthetic=a.synthetic, failures=failures[:20])
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'real'}, indent=1))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
