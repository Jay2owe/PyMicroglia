"""Compare boxed owner-component areas and right-censored seat partition with the original, exactly.

Synthetic frames compare the point's component area with whole-frame labelling
and physical._component_at: points on, beside and far from masks, outside the
frame, at borders, with ties and several search radii. On the real Issue 006
input every track's expected area for each of its owners (at several cut
frames) is compared, then the complete discover_and_apply outputs.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'config', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--stem', default='95_full_A1')
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    import tifffile
    from scipy import ndimage
    sys.path[:0] = [str(a.adapter.resolve()), str((a.engine / 'code').resolve())]
    import common
    import right_censored_seat_partition as module
    import right_censored_seat_partition_integration as integration
    import censored_area_boxes as adapter
    physical = module.physical
    rng = np.random.default_rng(6)
    points = 0
    for case in range(300):
        labels = np.zeros((3, int(rng.integers(5, 40)), int(rng.integers(5, 40))), np.uint16)
        for t in range(3):
            for _ in range(int(rng.integers(0, 10))):
                y, x = rng.integers(0, labels.shape[1]), rng.integers(0, labels.shape[2])
                labels[t, y:y + int(rng.integers(1, 5)), x:x + int(rng.integers(1, 5))] = rng.choice([1, 2, 7])
        stats = dict(labelled=0, reused=0)
        memo = adapter.OwnerParts(labels, module.partition.STRUCTURE, stats)
        for _ in range(40):
            t, owner = int(rng.integers(0, 3)), int(rng.choice([1, 2, 7, 9]))
            x = float(rng.uniform(-8, labels.shape[2] + 8)); y = float(rng.uniform(-8, labels.shape[1] + 8))
            if rng.random() < .3:
                x, y = round(x) + .5, round(y) - .5  # rounding ties
            radius = int(rng.choice([2, 3, 6, 40]))
            parts, _ = ndimage.label(labels[t] == owner, module.partition.STRUCTURE)
            component = physical._component_at(parts, x, y, radius)
            expected = int(np.count_nonzero(parts == component)) if component > 0 else 0
            assert memo.area_at(t, owner, x, y, radius) == expected, (case, t, owner, x, y, radius)
            points += 1
    cfg = common.Config.load(a.config)
    enabled, params = integration._configured(cfg.values)
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    labels = tifffile.imread(folder / '92_well_b3_issue005_component_continuity_duplicate_takeover.tif')
    raw = tifffile.imread(folder / '07_trimmed_evidence/out' / f'{a.stem}_registered_raw.tif')
    table = pd.read_csv(folder / '26_latent_body_tracks/out/latent_track_points.csv')
    scored = physical.attach_owners(table, labels)
    groups = {int(track): group.sort_values('frame') for track, group in scored.groupby('track_id', sort=True)}
    stats = dict(labelled=0, reused=0, fallback=0)
    memo = adapter.OwnerParts(labels, module.partition.STRUCTURE, stats)
    fast = adapter.expected_area(module, memo)
    areas = 0
    for track, group in list(groups.items())[::3]:
        owners = [int(o) for o in group.accepted_owner.astype(int).unique() if o > 0][:3]
        for owner in owners:
            for cut in (int(group.frame.max()) + 1, int(group.frame.median())):
                assert module._expected_area(labels, group, owner, cut) == fast(labels, group, owner, cut), (track, owner, cut)
                areas += 1
    tick = time.perf_counter()
    expected = module.discover_and_apply(labels, raw, table, params)
    middle = time.perf_counter()
    with adapter.install(module) as counters:
        assert counters['supported']
        actual = module.discover_and_apply(labels, raw, table, params)
    seconds = time.perf_counter() - middle
    assert np.array_equal(expected[0], actual[0])
    for old, new in zip(expected[1:], actual[1:]):
        pd.testing.assert_frame_equal(old, new, check_exact=True)
        assert old.to_csv(index=False) == new.to_csv(index=False)
    report = dict(status='pass', synthetic_points=points, real_expected_areas=areas,
                  real_transitions=len(actual[1]), real_eligible=int(actual[1].eligible.astype(bool).sum()),
                  real_applied=int(actual[1].applied.astype(bool).sum()),
                  original_seconds=round(middle - tick, 1), adapter_seconds=round(seconds, 1), counters=counters)
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
