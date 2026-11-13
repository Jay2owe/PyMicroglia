"""Compare resident-takeover runs and proposals with and without column-list rows.

Recorded A1 latent track points get their owners from a recorded accepted-history
label stack (resident_takeover.attach_owners). _runs of every visible track and
discover_proposals of the whole table are computed by the frozen module and with
the adapter installed, on the recorded table and on variants: float frames,
duplicated rows (equal frames inside a track), shuffled rows, a NaN frame, an
extra object column, a nullable integer column (itertuples fallback) and an empty
table. Rows must match in type name, fields, values and value types; proposal
tables must match cell for cell, including the point rows they hold.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


def described(value):
    if isinstance(value, tuple) and hasattr(value, '_fields'):
        return (type(value).__name__, value._fields, [described(v) for v in value])
    if isinstance(value, (list, tuple)):
        return [described(v) for v in value]
    if isinstance(value, dict):
        return {k: described(v) for k, v in value.items()}
    if isinstance(value, float) and np.isnan(value):
        return ('nan',)
    return (type(value).__name__, value)


def outcome(function, *args):
    try:
        value = function(*args)
    except Exception as error:
        return ('error', type(error).__name__, str(error))
    if isinstance(value, pd.DataFrame):
        return ('table', list(value.columns), [str(t) for t in value.dtypes],
                [described(list(row)) for row in value.itertuples(index=True)])
    return ('value', described(value))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--engine-code', type=Path, required=True)
    p.add_argument('--adapters', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frames', type=int, default=476)
    a = p.parse_args()
    history = Path('development/cd68_full_red_green/rigid_analysis/A1/tracking/engine/'
                   'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history')
    sys.path[:0] = [str(a.adapters.resolve()), str(a.engine_code.resolve())]
    import resident_takeover as module
    import resident_run_rows as adapter
    labels = tifffile.imread(history / '51_well_a5_issue003_branched_motion_lineage_recovery.tif',
                             key=range(a.frames))
    points = pd.read_csv(history / '26_latent_body_tracks/out/latent_track_points.csv', low_memory=False)
    points = points[points.frame < a.frames].reset_index(drop=True)
    attached = module.attach_owners(points, labels)
    rng = np.random.default_rng(7)
    nan_frame = attached.assign(frame=attached.frame.astype(float))
    nan_frame.loc[len(nan_frame) // 3, 'frame'] = np.nan
    variants = dict(
        recorded=attached,
        float_frames=attached.assign(frame=attached.frame.astype(float)),
        duplicated=pd.concat([attached, attached.iloc[::5]]),
        shuffled=attached.sample(frac=1, random_state=3),
        nan_frame=nan_frame,
        object_column=attached.assign(note=np.where(rng.random(len(attached)) < .5, 'a', None)),
        nullable_column=attached.assign(extra=pd.array(np.arange(len(attached)), dtype='Int64')),
        empty=attached.iloc[0:0],
        # The two median columns: missing values, object values, and absent.
        purity_missing=attached.assign(candidate_owner_purity=attached.candidate_owner_purity.where(
            rng.random(len(attached)) < .8)),
        coverage_object=attached.assign(candidate_coverage_fraction=attached.candidate_coverage_fraction.astype(object)),
        no_purity=attached.drop(columns=['candidate_owner_purity']))
    params = {}
    failures, cases = [], {}
    original_runs = module._runs
    for name, table in variants.items():
        visible = table[table['physically_visible'].astype(bool)]
        groups = [group for _, group in visible.groupby('track_id')]
        t0 = time.perf_counter()
        expected_runs = [outcome(module._runs, group) for group in groups]
        expected = outcome(module.discover_proposals, table, params)
        t1 = time.perf_counter()
        with adapter.install(module) as stats:
            assert stats['supported'] == 1
            actual_runs = [outcome(module._runs, group) for group in groups]
            actual = outcome(module.discover_proposals, table, params)
        t2 = time.perf_counter()
        cases[name] = dict(groups=len(groups), proposals=len(expected[3]) if expected[0] == 'table' else None,
                           original_seconds=round(t1 - t0, 2), adapted_seconds=round(t2 - t1, 2),
                           tracks=stats['tracks'], fallback=stats['fallback'],
                           median_columns=stats['median_columns'], median_tables=stats['median_tables'],
                           in_order=stats.get('in_order', 0), sorted=stats.get('sorted', 0))
        if expected_runs != actual_runs:
            failures.append(name + ':runs')
        if expected != actual:
            failures.append(name + ':proposals')
    if (module._runs is not original_runs or hasattr(module.discover_proposals, '__wrapped__')
            or '_median_columns' in vars(module)):
        failures.append('restore')
    if not (sum(case['in_order'] for case in cases.values()) and sum(case['sorted'] for case in cases.values())):
        failures.append('in-order and sorted routes not both used')
    if not sum(case['median_columns'] for case in cases.values()):
        failures.append('median columns never used')
    report = dict(status='pass' if not failures else 'fail', cases=cases, failures=failures,
                  sources={str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                           for f in [Path(__file__), Path(adapter.__file__), Path(module.__file__)]})
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], failures=failures, cases=cases)))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
