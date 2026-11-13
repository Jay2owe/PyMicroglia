"""Compare complete-flip takeover proposals with and without counted prior frames.

The snapshot's packaged engine runs complete-flip discovery (every native
adapter installed) on windows of the recorded A5 Issue 003 input; the arguments
of each _takeover_proposals call are captured, and the frozen function and the
adapter are compared on them and on variants: duplicated point rows, float
frame columns, shuffled rows, a NaN frame and an empty table. Proposal rows
(every key, value, type and order) and raised errors must be identical.
"""
import argparse
import hashlib
import io
import json
import sys
import time
from contextlib import redirect_stderr
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

import snapshot_engines


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
    if isinstance(x, float) and np.isnan(x):
        return np.isnan(y)
    return x == y


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--adapters', type=Path, required=True)
    p.add_argument('--scratch', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--windows', default='0:160,0:300')
    a = p.parse_args()
    engine_root = Path('development/cd68_full_red_green/rigid_analysis/A1/tracking/engine')
    config = Path('development/cd68_full_red_green/rigid_batch/.j/A1/inputs/motion_config.json').resolve()
    history = engine_root / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'
    sys.path[:0] = [str(a.snapshot.resolve())]
    from pymicroglia.tracking import identity_performance
    engine = a.scratch / 'motion_engine'
    if not engine.exists():
        snapshot_engines.unpack(a.snapshot, snapshot_engines.MOTION, engine)
    sys.path[:0] = [str(a.adapters.resolve()), str(engine / 'code')]
    import takeover_prior_counts as adapter
    points = pd.read_csv(history / '26_latent_body_tracks/out/latent_track_points.csv', low_memory=False)
    failures, cases = [], {}
    log = io.StringIO()
    with redirect_stderr(log), identity_performance.native(engine / 'code', a.scratch / 'native_scratch'):
        import complete_flip_recovery as flip
        import complete_flip_recovery_integration as integration
        import common
        _, params = integration._configured(common.Config.load(config).values)
        params.update({"labels_path": "<current-run-label-array>",
                       "unclaimed_path": "<current-run-unclaimed-array>",
                       "physical_track_points_path": "latent_track_points.csv",
                       "output_stem": "current_run", "mode": "accepted"})
        points = integration._read_table(history / '26_latent_body_tracks/out/latent_track_points.csv')
        original = flip._takeover_proposals
        for window in a.windows.split(','):
            start, stop = map(int, window.split(':'))
            labels = tifffile.imread(history / '51_well_a5_issue003_branched_motion_lineage_recovery.tif',
                                     key=range(start, stop))
            selected = points[(points.frame >= start) & (points.frame < stop)].copy()
            selected['frame'] -= start
            captured = []

            def record(*args):
                captured.append(args)
                return original(*args)
            flip._takeover_proposals = record
            try:
                flip.discover(labels, selected, params)
            finally:
                flip._takeover_proposals = original
            for number, (labels_arg, visible, params_arg, field_step) in enumerate(captured):
                variants = dict(recorded=visible,
                                duplicated=pd.concat([visible, visible.iloc[::7]]),
                                float_frames=visible.assign(frame=visible.frame.astype(float)),
                                shuffled=visible.sample(frac=1, random_state=4),
                                nan_frame=visible.assign(frame=visible.frame.astype(float).where(
                                    np.arange(len(visible)) != len(visible) // 2)),
                                empty=visible.iloc[0:0])
                for name, table in variants.items():
                    label = f'{window}:call{number}:{name}'
                    t0 = time.perf_counter()
                    expected = outcome(original, labels_arg, table, params_arg, field_step)
                    t1 = time.perf_counter()
                    with adapter.install(flip) as stats:
                        assert stats['supported'] == 1
                        actual = outcome(flip._takeover_proposals, labels_arg, table, params_arg, field_step)
                    t2 = time.perf_counter()
                    cases[label] = dict(kind=expected[0], proposals=len(expected[1]) if expected[0] == 'value' else None,
                                        original_seconds=round(t1 - t0, 2), adapted_seconds=round(t2 - t1, 2),
                                        prior_counts=stats['prior_counts'])
                    if not same(expected, actual):
                        failures.append(label)
            if not captured:
                failures.append(f'{window}:no calls')
        if flip._takeover_proposals is not original:
            failures.append('restore')
    report = dict(status='pass' if not failures else 'fail', cases=cases, failures=failures,
                  sources={str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                           for f in [Path(__file__), Path(adapter.__file__), Path(flip.__file__)]})
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], failures=failures,
                          recorded={k: v for k, v in cases.items() if k.endswith(':recorded')})))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
