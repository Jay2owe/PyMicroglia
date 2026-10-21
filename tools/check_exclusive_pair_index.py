"""Compare exclusive-pair discovery with its indexed adapter, exactly.

The real Issue 001 input (labels, track points and encounter frames) runs the
original discovery and the adapter; the audit and scored tables must be
identical, and the audit must equal the reference run's bytes. Random subsets of
the encounter rows, with some tracks marked in-encounter throughout (so pairs reach
tracks with no owner support) and some strong flags cleared, repeat the comparison.
"""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'config', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    import tifffile
    sys.path[:0] = [str(a.adapter.resolve()), str((a.engine / 'code').resolve())]
    import common
    import exclusive_pair_seat_recovery as module
    import exclusive_pair_index as adapter
    cfg = common.Config.load(a.config)
    params = dict(cfg.values['accepted_postprocessing']['field_wide_exclusive_pair_seat_recovery'].get('parameters', {}))
    params['targeting_mode'] = 'field_wide_discovery'
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    labels = tifffile.imread(folder / '45_well_issue004_recurrent_two_body_fusion_recovery.tif')
    evidence = folder / '26_latent_body_tracks/out'
    points = pd.read_csv(evidence / 'latent_track_points.csv')
    encounters = pd.read_csv(evidence / 'encounter_frames.csv')
    rng = np.random.default_rng(1)
    cases = [('real', points, encounters, params)]
    for case in range(2):
        some = encounters.sample(frac=.5, random_state=case).sort_index()
        dropped = rng.choice(points.track_id.unique(), 40, replace=False)
        fewer = points.copy()
        fewer.loc[fewer.track_id.isin(dropped), 'in_encounter'] = True  # no owner support
        fewer.loc[rng.random(len(fewer)) < .2, 'strong'] = False
        loose = dict(params, minimum_separable_frames=1, minimum_encounter_episodes=1)
        cases.append((f'subset-{case}', fewer, some, loose))
    report = dict(cases=[])
    for name, table, frames, settings in cases:
        tick = time.perf_counter()
        expected = module.discover(labels, table, frames, settings)
        middle = time.perf_counter()
        with adapter.install(module) as stats:
            assert stats['supported']
            actual = module.discover(labels, table, frames, settings)
        seconds = time.perf_counter() - middle
        for old, new in zip(expected, actual):
            pd.testing.assert_frame_equal(old, new, check_exact=True)
            assert old.to_csv(index=False) == new.to_csv(index=False)
        row = dict(name=name, pairs=len(actual[0]), eligible=int(actual[0].eligible.astype(bool).sum()),
                   original_seconds=round(middle - tick, 1), adapter_seconds=round(seconds, 1), stats=stats)
        if name == 'real':
            reference = folder / '61_exclusive_pair_seat_recovery/out/exclusive_pair_seat_audit.csv'
            row['reference_bytes_equal'] = actual[0].to_csv(index=False).encode() == reference.read_bytes()
            assert row['reference_bytes_equal']
        report['cases'].append(row)
    report['status'] = 'pass'
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
