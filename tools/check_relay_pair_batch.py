"""Compare batched owner-relay pair scoring with the original discovery, exactly.

Synthetic reference tracks (gaps, ties, zero owners, shared owners, odd and even
overlaps, long and one-frame tracks, permissive and default thresholds) run the
original discover, the engine's relay_pair_speed version and this adapter; the
audit and point tables must be identical. The real Issue 089 input then runs
through the adapter and its two tables must equal the reference run's bytes.
"""
import argparse
from contextlib import contextmanager
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
    import cross_reference_owner_relay as module
    from relay_pair_speed import faster_relay_pairs
    import relay_pair_batch

    @contextmanager
    def attached(table):
        original = module.physical.attach_owners
        module.physical.attach_owners = lambda points, labels: table.copy()
        try:
            yield
        finally:
            module.physical.attach_owners = original

    rng = np.random.default_rng(89)
    permissive = dict(minimum_overlap_same_owner_fraction=0.0, maximum_median_separation_sum_radii=50.0,
                      maximum_separation_sum_radii=50.0, minimum_successor_extension_frames=0,
                      minimum_dedicated_support_frames=1, minimum_dedicated_support_fraction=0.0,
                      minimum_dedicated_runs=1, minimum_owner_transitions=0,
                      minimum_competing_owner_count=0, minimum_competing_owner_tracks=1)
    synthetic, rows, eligible = 0, 0, 0
    paths = dict(summaries=0, counted=0)
    for case in range(160):
        tracks = []
        for track in range(int(rng.integers(1, 40))):
            span = int(rng.integers(1, 60))
            start = int(rng.integers(0, 80))
            frames = np.unique(rng.integers(start, start + span, int(rng.integers(1, span + 1))))
            owners = rng.integers(0, 6, len(frames)) * (rng.random(len(frames)) < .8)
            if case % 4 == 0:
                owners[:] = owners[0]  # long single-owner runs
            if case % 5 == 1:
                # Owners dedicated to one track pair, alternating with shared owners.
                owners = rng.choice([0, track // 2 + 500, 1, 2], len(frames), p=[.1, .5, .2, .2])
            x = np.round(rng.normal(50, 6, len(frames)), 1 if case % 3 else 6)
            y = np.round(rng.normal(50, 6, len(frames)), 1 if case % 3 else 6)
            radius = rng.choice([0.0, 0.25, 3.0, 5.5], len(frames))
            tracks.append(pd.DataFrame(dict(track_id=track * 3 + 1, frame=frames, x=x, y=y, radius_px=radius,
                                            accepted_owner=owners,
                                            physically_visible=rng.random(len(frames)) < .95)))
        table = pd.concat(tracks, ignore_index=True).sample(frac=1, random_state=case).reset_index(drop=True)
        params = dict(permissive if case % 2 else {})
        params['minimum_overlap_frames'] = [1, 2, 4, 7][case % 4]
        params['targeting_mode'] = 'field_wide_discovery'
        with attached(table):
            expected = module.discover(None, table, params)
            with faster_relay_pairs(module):
                speed = module.discover(None, table, params)
                with relay_pair_batch.install(module) as stats:
                    assert stats['supported']
                    actual = module.discover(None, table, params)
                    assert stats['batched'] == 1, stats
        for reference in (expected, speed):
            for old, new in zip(reference, actual):
                pd.testing.assert_frame_equal(old, new, check_exact=True)
                assert old.to_csv(index=False) == new.to_csv(index=False)
        synthetic += 1
        paths['summaries'] += stats['summaries']; paths['counted'] += stats['counted']
        rows += len(actual[0])
        eligible += int(actual[0]['eligible'].sum()) if len(actual[0]) else 0

    cfg = common.Config.load(a.config)
    params = dict(cfg.values['accepted_postprocessing'].get('field_wide_cross_reference_owner_relay', {}).get('parameters', {}))
    params['targeting_mode'] = 'field_wide_discovery'
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    labels = tifffile.imread(folder / '40_issue080_terminal_companion_assimilation.tif')
    points = pd.read_csv(folder / '26_latent_body_tracks/out/latent_track_points.csv')
    stage = folder / '52_cross_reference_owner_relay/out'
    with faster_relay_pairs(module), relay_pair_batch.install(module) as stats:
        tick = time.perf_counter()
        audit, scored = module.discover(labels, points, params)
        seconds = time.perf_counter() - tick
    assert stats['batched'] == 1, stats
    same_audit = audit.to_csv(index=False).encode() == (stage / 'reference_handoff_audit.csv').read_bytes()
    same_points = scored.to_csv(index=False).encode() == (stage / 'physical_owner_points.csv').read_bytes()
    report = dict(status='pass' if same_audit and same_points else 'fail', synthetic_cases=synthetic,
                  synthetic_rows=rows, synthetic_eligible=eligible, synthetic_paths=paths, real_rows=len(audit),
                  real_paths=dict(summaries=stats['summaries'], counted=stats['counted']),
                  real_audit_equal=same_audit, real_points_equal=same_points, real_seconds=round(seconds, 1))
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
