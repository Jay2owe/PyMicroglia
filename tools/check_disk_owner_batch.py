"""Compare batched disk owners in merge residency and persistent body identity with the original, exactly.

python check_disk_owner_batch.py --engine <engine> --adapter <identity_candidate dir> --output <report.json>

Synthetic stacks (several integer dtypes, negative labels) with points inside,
beside and far outside the image, below -2 (where the original slice wraps),
on rounding ties, with NaN, huge and negative radii and frames out of range;
then the real A1 track points on a saved label stack. Every attached table
must equal the original's exactly (values, dtypes, CSV text), and every error
must have the same type and message.
"""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    import tifffile
    sys.path[:0] = [str(a.adapter.resolve()), str((a.engine / 'code').resolve())]
    import merge_residency
    import persistent_body_identity
    import disk_owner_batch as adapter
    originals = {merge_residency: merge_residency._attach_owners,
                 persistent_body_identity: persistent_body_identity._attach_evidence}

    def call(function, module, points, labels, unclaimed=None):
        try:
            if module is persistent_body_identity:
                return 'ok', function(points, labels, unclaimed)
            return 'ok', function(points, labels)
        except Exception as error:
            return 'error', (type(error).__name__, str(error))

    def same(left, right):
        if left[0] != right[0]:
            return False
        if left[0] == 'error':
            return left[1] == right[1]
        pd.testing.assert_frame_equal(left[1], right[1], check_exact=True)
        return left[1].to_csv(index=False) == right[1].to_csv(index=False)

    rng = np.random.default_rng(11)
    cases = failures = errors = 0
    problems = []
    with adapter.install(merge_residency, persistent_body_identity) as stats:
        assert len(stats['supported']) == 2, stats
        for trial in range(400):
            dtype = [np.uint16, np.int32, np.int64, np.uint8][trial % 4]
            height, width = int(rng.integers(3, 30)), int(rng.integers(3, 30))
            labels = rng.choice([0, 0, 0, 1, 2, 7, 81], size=(3, height, width)).astype(dtype)
            if trial % 4 == 2 and trial % 8 == 2:
                labels[0, 0, :] = -3
            n = int(rng.integers(0, 40))
            points = pd.DataFrame(dict(
                frame=rng.integers(0, 3, n), x=rng.uniform(-2, width + 6, n), y=rng.uniform(-2, height + 6, n),
                radius_px=rng.uniform(-3, 14, n), state=rng.choice(['observed', 'latent_visible', 'missing'], n),
                track_id=rng.integers(1, 9, n)))
            if n and trial % 5 == 0:
                points.loc[0, 'x'] = float(round(points.loc[0, 'x'])) + .5
            if n and trial % 13 == 0:
                points.loc[0, 'x'] = np.nan
            if n and trial % 17 == 0:
                points.loc[0, 'frame'] = 5
            if n and trial % 19 == 0:
                points.loc[0, 'frame'] = -1
            if n and trial % 23 == 0:
                points.loc[0, 'radius_px'] = 1e12
            if n and trial % 29 == 0:
                points.loc[0, 'x'] = -40.0
            if trial % 31 == 0:
                points['frame'] = points['frame'].astype(float)
            unclaimed = rng.choice([0, 0, 3, 5], size=labels.shape).astype(np.uint16)
            if trial % 37 == 0:
                unclaimed = unclaimed[:, :-1]          # mismatched shape: original route
            if trial % 41 == 0:
                unclaimed = np.zeros(labels.shape, np.int64)
            for module, original in originals.items():
                new = call(getattr(module, original.__name__), module, points, labels, unclaimed)
                old = call(original, module, points, labels, unclaimed)
                cases += 1
                errors += old[0] == 'error'
                if not same(old, new):
                    failures += 1
                    problems.append(dict(trial=trial, module=module.__name__, old=str(old[1])[:200], new=str(new[1])[:200]))
        folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
        labels = tifffile.imread(folder / '52_well_a5_issue003_complete_flip_recovery.tif')
        unclaimed = tifffile.imread(folder / '52_well_a5_issue003_complete_flip_recovery_unclaimed.tif')
        points = pd.read_csv(folder / '26_latent_body_tracks/out/latent_track_points.csv')
        timing = {}
        for module, original in originals.items():
            tick = time.perf_counter()
            old = call(original, module, points, labels, unclaimed)
            middle = time.perf_counter()
            new = call(getattr(module, original.__name__), module, points, labels, unclaimed)
            timing[module.__name__] = [round(middle - tick, 2), round(time.perf_counter() - middle, 2)]
            assert old[0] == 'ok', old
            cases += 1
            if not same(old, new):
                failures += 1
                problems.append(dict(module=module.__name__, real=True))
    assert merge_residency._attach_owners is originals[merge_residency]
    assert persistent_body_identity._attach_evidence is originals[persistent_body_identity]
    assert not hasattr(merge_residency, '_disk_owner_batch')
    report = dict(status='pass' if not failures else 'fail', cases=cases, original_errors=errors,
                  real_points=len(points), seconds_original_adapted=timing, counters=stats, failures=problems[:20])
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
