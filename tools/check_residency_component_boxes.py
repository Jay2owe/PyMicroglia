"""Compare boxed owner components in merge residency with the original, exactly.

python check_residency_component_boxes.py --engine <engine> --adapter <identity_candidate dir> --config <motion_config.json> --output <report.json>

Synthetic stacks: _component_at on frames of the memo's labels (served) and on
copies, transposes and other stacks (original route), for present, absent,
zero, negative and NumPy owners and points inside, beside and outside the
image with NaN and small radii, comparing the returned masks. Then the real A1
merge residency input with and without the adapter (with the batched disk
owners, as installed together): labels, unclaimed and the audit must be equal.
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
    import merge_residency as module
    import residency_component_boxes as adapter
    import disk_owner_batch
    from censored_area_boxes import OwnerParts
    original = module._component_at
    rng = np.random.default_rng(21)
    compared = served = 0
    for case in range(250):
        dtype = [np.uint16, np.int32, np.int64][case % 3]
        labels = np.zeros((3, int(rng.integers(5, 40)), int(rng.integers(5, 40))), dtype)
        for t in range(3):
            for _ in range(int(rng.integers(0, 10))):
                y, x = rng.integers(0, labels.shape[1]), rng.integers(0, labels.shape[2])
                labels[t, y:y + int(rng.integers(1, 6)), x:x + int(rng.integers(1, 6))] = rng.choice([1, 2, 7])
        if case % 9 == 0 and dtype != np.uint16:
            labels[1, :2, :2] = -4
        stats = dict(labelled=0, reused=0, boxed=0, served=0, fallback=0)
        fast = adapter.component_at(original, stats)
        adapter._local.memo = OwnerParts(labels, module.STRUCTURE, stats)
        other = labels.copy()
        for _ in range(40):
            t = int(rng.integers(0, 3))
            owner = rng.choice([1, 2, 7, 9, 0, -4])
            owner = int(owner) if rng.random() < .7 else np.int64(owner)
            x = float(rng.uniform(-8, labels.shape[2] + 8)); y = float(rng.uniform(-8, labels.shape[1] + 8))
            radius = float(rng.choice([0.5, 2.0, 3.7, 9.0, np.nan]))
            if rng.random() < .3:
                x, y = round(x) + .5, round(y) - .5
            for plane in (labels[t], other[t], labels[t].T.copy().T, np.ascontiguousarray(labels[t])):
                want = original(plane, owner, x, y, radius)
                got = fast(plane, owner, x, y, radius)
                same = (want is None and got is None) or (
                    want is not None and got is not None and want.dtype == got.dtype
                    and want.shape == got.shape and np.array_equal(want, got))
                assert same, (case, t, owner, x, y, radius)
                compared += 1
        served += stats['served']
        adapter._local.memo = None
    cfg = common.Config.load(a.config)
    params = cfg.values['accepted_postprocessing'].get('field_wide_merge_residency', {}).get('parameters', {})
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    labels = tifffile.imread(folder / '18_issue034_recording_start_split.tif')
    unclaimed = tifffile.imread(folder / '18_issue034_recording_start_split_unclaimed.tif')
    raw = tifffile.imread(folder / '07_trimmed_evidence/out/95_full_A1_registered_raw.tif')
    points = pd.read_csv(folder / '26_latent_body_tracks/out/latent_track_points.csv')
    tick = time.perf_counter()
    expected = module.recover_merge_residents(labels, unclaimed, raw, points, params)
    middle = time.perf_counter()
    with disk_owner_batch.install(module, None) as disk, adapter.install(module) as counters:
        assert counters['supported'] and disk['supported']
        actual = module.recover_merge_residents(labels, unclaimed, raw, points, params)
    seconds = time.perf_counter() - middle
    assert module._component_at is original
    for old, new in zip(expected[:2], actual[:2]):
        assert old.dtype == new.dtype and np.array_equal(old, new)
    pd.testing.assert_frame_equal(expected[2], actual[2], check_exact=True)
    assert expected[2].to_csv(index=False) == actual[2].to_csv(index=False)
    report = dict(status='pass', synthetic_masks=compared, synthetic_served=served,
                  real_audit_rows=len(actual[2]), real_applied=int(actual[2].applied.astype(bool).sum()) if len(actual[2]) else 0,
                  original_seconds=round(middle - tick, 1), adapter_seconds=round(seconds, 1), counters=counters, disk=disk)
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
