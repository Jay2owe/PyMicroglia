"""Compare alias host evidence with the boxed measurements.

1. Direct: on sampled real frames, every identity ending or starting near a
   host in the next frame is scored by the original statements (whole-frame
   dilation mean and KD-tree distance) and the boxed helpers, with plain
   Boolean masks and with lazy label masks; float bit patterns must agree.
   Presence tests are compared with `value in np.unique(frame)`.
2. Whole step: consolidate_recurrent_host_aliases on a real stage input with
   the configured settings, without and with the adapter; labels and the three
   tables must be equal.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
INPUTS = ROOT / 'development/cd68_full_red_green/rigid_batch/.j/A1/inputs'


def bits(value):
    return np.float64(value).tobytes()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True, help='composed snapshot (for LabelMask)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=80)
    parser.add_argument('--skip-step', action='store_true')
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve()), str(a.snapshot.resolve())]
    import identity_aliases as module
    import alias_host_boxes as adapter
    from pymicroglia.tracking.frozen_io import LabelMask
    config = json.loads((INPUTS / 'motion_config.json').read_text(encoding='utf-8'))
    params = config['accepted_postprocessing']['recurrent_aliases']
    labels = tifffile.imread(a.labels)
    raw = tifffile.imread(INPUTS / '95_full_A1.tif')
    rng = np.random.default_rng(20261021)
    failures, cases, presence = [], 0, 0
    times = dict(original=0.0, boxed=0.0)
    helpers = adapter.helpers(module, dict(boxed=0, whole=0, presence=0))
    for t in sorted(rng.choice(len(labels) - 1, a.frames, replace=False).tolist()):
        for lazy in (False, True):
            before = module.frame_observations(labels[t], raw[t])
            after = module.frame_observations(labels[t + 1], raw[t + 1])
            if lazy:
                for rows, frame in ((before, labels[t].copy()), (after, labels[t + 1].copy())):
                    for identity, row in rows.items():
                        row['mask'] = LabelMask(frame, identity)
            for identity, old in before.items():
                for host, row in after.items():
                    if host == identity or np.linalg.norm(old['position'] - row['position']) > 60:
                        continue
                    t0 = time.perf_counter()
                    expanded = module.ndi.binary_dilation(row['mask'], iterations=2)
                    absorbed = float(np.mean(expanded[old['mask']]))
                    distance = module._mask_distance(old['mask'], row['mask'])
                    t1 = time.perf_counter()
                    boxed_absorbed = helpers['_boxed_absorbed'](row, old)
                    boxed_distance = helpers['_boxed_mask_distance'](old, row)
                    t2 = time.perf_counter()
                    times['original'] += t1 - t0; times['boxed'] += t2 - t1
                    cases += 1
                    if bits(absorbed) != bits(boxed_absorbed) or bits(distance) != bits(boxed_distance):
                        failures.append(f'{t}-{identity}-{host}-{lazy}')
        for value in list(rng.integers(0, int(labels[t].max()) + 3, 20)) + [0, 70000]:
            presence += 1
            if (int(value) in np.unique(labels[t])) != helpers['_frame_has'](labels[t], int(value)):
                failures.append(f'presence-{t}-{value}')
    report = dict(direct=dict(pairs=cases, presence=presence, seconds=times))
    if not a.skip_step:
        t0 = time.perf_counter()
        expected = module.consolidate_recurrent_host_aliases(labels, raw, params)
        t1 = time.perf_counter()
        with adapter.install(module) as stats:
            assert stats['supported']
            actual = module.consolidate_recurrent_host_aliases(labels, raw, params)
        t2 = time.perf_counter()
        same = np.array_equal(expected[0], actual[0]) and expected[0].dtype == actual[0].dtype
        for left, right in zip(expected[1:], actual[1:]):
            try:
                pd.testing.assert_frame_equal(left, right, check_exact=True)
            except AssertionError:
                same = False
        if not same:
            failures.append('whole-step')
        report['step'] = dict(equal=bool(same), original_s=round(t1 - t0, 1), adapted_s=round(t2 - t1, 1),
                              rows=[len(v) for v in expected[1:]], **{k: v for k, v in stats.items()})
    report.update(status='pass' if not failures else 'fail', failures=failures[:20])
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report, indent=1))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
