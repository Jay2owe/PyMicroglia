"""Compare whole retroactive successor discovery with and without its adapter.

The accepted Issue 006 (A5) stage input, the current-run latent track points
and the registered raw movie, with the stage's configured parameters. Cases:
the real points; every other track removed; the strong flags written as text
(so the mapped flags come from text); and a later stage's labels. The
candidate movie, the audit table and the frame table must be equal.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', default='real,half_tracks,text_strong,later_labels')
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import retroactive_successor_separation as module
    import retroactive_successor_integration as integration
    import retroactive_separation_reuse as reuse
    config = json.loads((ENGINE / 'config.json').read_text(encoding='utf-8'))
    _, params = integration._configured(config)
    raw = tifffile.imread(next((HISTORY / '07_trimmed_evidence/out').glob('*_registered_raw.tif')))
    points = pd.read_csv(HISTORY / '26_latent_body_tracks/out/latent_track_points.csv')
    tracks = np.sort(points.track_id.unique())
    text_strong = points.copy()
    text_strong['strong'] = text_strong['strong'].map({True: 'True', False: 'false'})
    cases = [('real', '53_well_a5_issue005_projection_tolerant_seat_preservation.tif', points),
             ('half_tracks', '53_well_a5_issue005_projection_tolerant_seat_preservation.tif',
              points[points.track_id.isin(tracks[::2])].reset_index(drop=True)),
             ('text_strong', '53_well_a5_issue005_projection_tolerant_seat_preservation.tif', text_strong),
             ('later_labels', '55_well_a5_issue004_terminal_two_core_split_inheritance.tif', points)]
    failures, results = [], []
    totals = {}
    chosen = a.cases.split(',')
    for name, labels_name, table in [case for case in cases if case[0] in chosen]:
        labels = tifffile.imread(HISTORY / labels_name)
        t0 = time.perf_counter()
        expected = module.discover_and_apply(labels, table, raw, params)
        t1 = time.perf_counter()
        with reuse.install(module) as stats:
            assert stats['supported']
            actual = module.discover_and_apply(labels, table, raw, params)
        t2 = time.perf_counter()
        for key, value in stats.items():
            if type(value) is int:
                totals[key] = totals.get(key, 0) + value
        try:
            assert expected[0].dtype == actual[0].dtype and np.array_equal(expected[0], actual[0])
            pd.testing.assert_frame_equal(expected[1], actual[1], check_exact=True)
            pd.testing.assert_frame_equal(expected[2], actual[2], check_exact=True)
            equal = True
        except AssertionError as error:
            equal = False
            failures.append(dict(case=name, error=str(error)[:400]))
        row = dict(case=name, equal=equal, audit_rows=len(expected[1]), frame_rows=len(expected[2]),
                   eligible=int(expected[1]['eligible'].astype(bool).sum()) if 'eligible' in expected[1] else None,
                   changed_pixels=int(np.count_nonzero(expected[0] != labels)),
                   counters={k: v for k, v in stats.items() if type(v) is int},
                   original_seconds=round(t1 - t0, 1), adapted_seconds=round(t2 - t1, 1))
        results.append(row)
        print(json.dumps(row), file=sys.stderr, flush=True)
    report = dict(status='pass' if not failures and totals.get('profiles') and totals.get('counted') else 'fail',
                  totals=totals, failures=failures, results=results)
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
