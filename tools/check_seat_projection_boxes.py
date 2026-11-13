"""Compare boxed expected areas in projection-tolerant seat preservation with the original, exactly.

python check_seat_projection_boxes.py --engine <engine> --adapter <identity_candidate dir> --config <motion_config.json> --output <report.json>

On the real A1 input of the stage (the labels saved before it), every third
track's expected area for each of its owners at several cut frames is compared
with the original _expected_area, also on a copy of the labels (which must
take the original route), then the complete discover_and_apply outputs.
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
    p.add_argument('--stem', default='95_full_A1')
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    import tifffile
    sys.path[:0] = [str(a.adapter.resolve()), str((a.engine / 'code').resolve())]
    import common
    import projection_tolerant_seat_preservation as module
    import projection_tolerant_seat_integration as integration
    import seat_projection_boxes as adapter
    cfg = common.Config.load(a.config)
    enabled, params = integration._configured(cfg.values)
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    # As projection_tolerant_seat_integration.run supplies them.
    params.update({'labels_path': '<current-run-label-array>', 'unclaimed_path': '<current-run-unclaimed-array>',
                   'raw_path': '<current-run-raw-array>', 'output_stem': 'current_run', 'mode': 'accepted',
                   'physical_track_points_path': str(folder / '26_latent_body_tracks/out/latent_track_points.csv')})
    labels = tifffile.imread(folder / '52_well_a5_issue003_complete_flip_recovery.tif')
    raw = tifffile.imread(folder / '07_trimmed_evidence/out' / f'{a.stem}_registered_raw.tif')
    table = pd.read_csv(folder / '26_latent_body_tracks/out/latent_track_points.csv')
    scored = module.physical.attach_owners(table, labels)
    groups = {int(track): group.sort_values('frame') for track, group in scored.groupby('track_id', sort=True)}
    original_expected = module._expected_area
    areas = 0
    with adapter.install(module) as counters:
        assert counters['supported'] and module._expected_area is not original_expected
        # Inside discovery the memo serves the call's own labels; a copy falls back.
        memo = adapter.OwnerParts(labels, module.STRUCTURE, counters)
        adapter._local.memo = memo
        other = labels.copy()
        for track, group in list(groups.items())[::3]:
            owners = [int(o) for o in group.accepted_owner.astype(int).unique() if o > 0][:3]
            for owner in owners:
                for cut in (int(group.frame.max()) + 1, int(group.frame.median())):
                    want = original_expected(labels, group, owner, cut)
                    assert module._expected_area(labels, group, owner, cut) == want, (track, owner, cut)
                    assert module._expected_area(other, group, owner, cut) == want, (track, owner, cut)
                    areas += 1
        adapter._local.memo = None
        boxed_areas = dict(counters)
    assert module._expected_area is original_expected and not hasattr(module, '_seat_area')
    tick = time.perf_counter()
    expected = module.discover_and_apply(labels, raw, table, params)
    middle = time.perf_counter()
    with adapter.install(module) as counters:
        actual = module.discover_and_apply(labels, raw, table, params)
    seconds = time.perf_counter() - middle
    assert np.array_equal(expected[0], actual[0]) and expected[0].dtype == actual[0].dtype
    assert np.array_equal(expected[4], actual[4])
    for old, new in zip(expected[1:4], actual[1:4]):
        pd.testing.assert_frame_equal(old, new, check_exact=True)
        assert old.to_csv(index=False) == new.to_csv(index=False)
    report = dict(status='pass', real_expected_areas=areas, area_counters=boxed_areas,
                  transitions=len(actual[1]), eligible=int(actual[1].eligible.astype(bool).sum()),
                  applied=int(actual[1].applied.astype(bool).sum()), projections=len(actual[3]),
                  original_seconds=round(middle - tick, 1), adapter_seconds=round(seconds, 1), counters=counters)
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
