"""Compare three per-track sorts with their skipped or single-sort rewrites, exactly.

python tools/check_track_sort_skips.py --snapshot <combined snapshot> --scratch <empty folder> --adapters <identity_candidate> --output <report.json>

On the recorded A1 latent track points (with an owner, node and path column
for the functions that read them), as recorded, shuffled, with an offset
index, with repeated (track, frame) rows and with float frames:
- bookended _runs on every track group: the same runs (rows, index, columns,
  dtypes) as the frozen function;
- precontact by_track: the helper's dict equals the per-track sort (keys in
  order, every table);
- tracked-inventory track_points: every track read from the lazy mapping
  equals the eager table, a missing track raises the same KeyError, and
  float tracks keep the eager dict.
Each adapted function must install on its frozen module and be restored.
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import snapshot_engines

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / ('development/cd68_full_red_green/rigid_analysis/A1/tracking/engine/'
                  'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history')


def same(a, b):
    try:
        pd.testing.assert_frame_equal(a, b, check_exact=True)
    except AssertionError:
        return False
    return type(a.index) is type(b.index) and a.index.dtype == b.index.dtype


def error(function, *args):
    try:
        function(*args)
    except Exception as caught:
        return (type(caught).__name__, str(caught))
    return None


def main():
    p = argparse.ArgumentParser()
    for name in ('snapshot', 'scratch', 'adapters', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    snapshot_engines.unpack(a.snapshot, snapshot_engines.COMPLETED_FEEDBACK, a.scratch)
    sys.path[:0] = [str(a.scratch / 'frozen/reviewed/R205-A007'), str(a.scratch / 'code'), str(a.adapters.resolve())]
    import track_sort_skips as adapter
    points = pd.read_csv(HISTORY / '26_latent_body_tracks/out/latent_track_points.csv', low_memory=False)
    rng = np.random.default_rng(17)
    points = points.assign(owner=rng.integers(0, 4, len(points)), node=rng.integers(0, 9, len(points)),
                           path=rng.integers(0, 5, len(points)))
    tables = dict(recorded=points, shuffled=points.sample(frac=1, random_state=6),
                  offset_index=points.set_axis(np.arange(len(points)) * 3 + 5),
                  repeated=pd.concat([points, points.iloc[::11]]),
                  float_frames=points.assign(frame=points.frame.astype(float)))
    stats = {key: 0 for key in adapter.KEYS}
    failures, cases = [], {}

    bookended = importlib.import_module('recover_bookended_owner_with_small_remnants')
    ordered = adapter.frame_ordered(stats)
    for name, table in tables.items():
        groups = [g for _, g in table.groupby('track_id')]
        ok = True
        for g in groups:
            rows = ordered(g).reset_index(drop=True)
            original_rows = g.sort_values("frame").reset_index(drop=True)
            ok &= same(rows, original_rows)
            if not ok:
                break
        cases['runs ' + name] = dict(tracks=len(groups), equal=bool(ok))
        if not ok:
            failures.append('runs ' + name)

    tracks_helper = adapter.sorted_tracks(stats)
    for name, table in tables.items():
        expected = {int(k): q.sort_values('frame') for k, q in table.groupby('track_id')}
        actual = tracks_helper(table)
        ok = list(expected) == list(actual) and all(same(expected[k], actual[k]) for k in expected)
        cases['precontact ' + name] = dict(tracks=len(expected), equal=bool(ok))
        if not ok:
            failures.append('precontact ' + name)

    in_order = adapter.tracks_in_order(stats)
    reversed_track = points.copy()
    first = reversed_track.track_id.iloc[0]
    rows = np.flatnonzero(reversed_track.track_id.to_numpy() == first)
    reversed_track.iloc[rows, reversed_track.columns.get_loc('frame')] = reversed_track.frame.to_numpy()[rows[::-1]]
    for name, table in dict(tables, reversed_track=reversed_track,
                            float_tracks=points.assign(track_id=points.track_id.astype(float))).items():
        for key in ('track_id',):
            expected = {int(k): g.sort_values('frame') for k, g in table.groupby(key)}
            actual = in_order(table, key)
            ok = list(expected) == list(actual) and all(same(expected[k], actual[k]) for k in expected)
            cases['groups ' + name] = dict(tracks=len(expected), equal=bool(ok))
            if not ok:
                failures.append('groups ' + name)

    frames_helper = adapter.track_frames(stats)
    for name, table in dict(tables, float_tracks=points.assign(track_id=points.track_id.astype(float))).items():
        expected = {int(track): group.set_index("frame") for track, group in table.groupby("track_id")}
        actual = frames_helper(table)
        keys = list(expected)
        order = [keys[i] for i in rng.permutation(len(keys))]
        ok = all(same(expected[k], actual[k]) for k in order)
        missing = max(keys) + 1000
        ok &= error(expected.__getitem__, missing) == error(actual.__getitem__, missing)
        cases['inventory ' + name] = dict(tracks=len(keys), equal=bool(ok), type=type(actual).__name__)
        if not ok:
            failures.append('inventory ' + name)

    # Persistent host owners: whole function, original against adapted, on
    # owners that mostly follow the track (some noise, some exact ties), with
    # the default dominance and with dominance 1 (value_counts kept).
    resident = importlib.import_module('recover_persistent_resident_contacts')
    reciprocal = importlib.import_module('restore_spatial_reciprocal_soma_paths_a002')
    tracks = points.track_id.to_numpy()
    owners = (tracks % 7 + 1).astype(np.int64)
    noise = rng.random(len(points)) < .1
    owners[noise] = rng.integers(0, 9, int(noise.sum()))
    tied = np.isin(tracks, np.unique(tracks)[::13])
    owners[tied] = np.where(np.arange(int(tied.sum())) % 2, 2, 3)
    points = points.assign(physically_visible=rng.random(len(points)) < .95)
    host_tables = dict(recorded=points.assign(accepted_owner=owners),
                       shuffled=points.assign(accepted_owner=owners).sample(frac=1, random_state=8),
                       object_owner=points.assign(accepted_owner=owners.astype(object)))
    anchors = {int(t): {'owner': int(t) % 7 + 1} for t in np.unique(tracks)[::5]}
    host_originals = resident._persistent_host_owners
    host_results = {}

    # Spatial reciprocal paths on real frames: whole correction.
    import tifffile
    final = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine/m23_completed_feedback/pymicroglia_95_full_A1/out'
    window = range(200, 240)
    labels = tifffile.imread(final / '95_full_A1.tif', key=window)
    raw = tifffile.imread(HISTORY / '07_trimmed_evidence/out/95_full_A1_registered_raw.tif', key=window)
    window_points = points[(points.frame >= 200) & (points.frame < 240)].assign(frame=lambda d: d.frame - 200)
    protected = np.zeros(labels.shape, bool)

    def whole(function, *args):
        try:
            value = function(*args)
        except Exception as caught:
            return ('error', type(caught).__name__, str(caught))
        return value

    def same_value(x, y):
        if isinstance(x, tuple) and isinstance(y, tuple) and len(x) == len(y):
            return all(same_value(u, v) for u, v in zip(x, y))
        if isinstance(x, np.ndarray):
            return isinstance(y, np.ndarray) and x.dtype == y.dtype and np.array_equal(x, y)
        if isinstance(x, pd.DataFrame):
            return isinstance(y, pd.DataFrame) and same(x, y)
        return x == y and type(x) is type(y)
    reciprocal_expected = whole(reciprocal.correct, labels, raw, window_points, protected)

    # Whole runs of the bookended correction's _runs callers go through the replay;
    # here each adapted function must install and be restored.
    sys.path[len(sys.path):] = [str(a.scratch / 'frozen/reviewed/R206-A002'), str(a.scratch / 'frozen/reviewed/R207-A003')]
    modules = [bookended, importlib.import_module('precontact_soma_ownership'),
               importlib.import_module('process_inventory_tracked'), resident, reciprocal,
               importlib.import_module('shared_birth_soma_ownership'),
               importlib.import_module('silhouette_owner_continuity'),
               importlib.import_module('recover_flanked_large_contact_interruptions')]
    # Whole runs of these go through check_track_sort_corrections.py.
    modules += [importlib.import_module(s['module']) for s in adapter.SPECS[len(modules):]]
    originals = [getattr(m, s['function']) for m, s in zip(modules, adapter.SPECS)]
    with adapter.install() as install_stats:
        installed = [getattr(m, s['function']).__wrapped__ is o for m, s, o in zip(modules, adapter.SPECS, originals)]
        # The adapted _runs on real groups, through the module.
        runs_equal = all(
            len(x) == len(y) and all(same(u, v) for u, v in zip(x, y))
            for x, y in ((originals[0](g), bookended._runs(g))
                         for _, g in tables['shuffled'].groupby('track_id')))
        for name, table in host_tables.items():
            for dominance in (3.0, 1.0):
                params = resident.Params(minimum_persistent_host_dominance=dominance)
                expected = whole(host_originals, table, anchors, params)
                actual = whole(resident._persistent_host_owners, table, anchors, params)
                ok = expected == actual
                host_results[f'{name} dominance {dominance}'] = dict(
                    owners=len(expected) if isinstance(expected, dict) else expected, equal=bool(ok))
                if not ok:
                    failures.append(f'host {name} {dominance}')
        reciprocal_actual = whole(reciprocal.correct, labels, raw, window_points, protected)
        reciprocal_ok = same_value(reciprocal_expected, reciprocal_actual)
        cases['reciprocal'] = dict(equal=bool(reciprocal_ok),
                                   applications=len(reciprocal_expected[2]) if not isinstance(reciprocal_expected[0], str) else reciprocal_expected)
        if not reciprocal_ok:
            failures.append('reciprocal')
    cases['host'] = host_results
    restored = all(getattr(m, s['function']) is o and (s['helper'] is None or s['helper'] not in vars(m))
                   for m, s, o in zip(modules, adapter.SPECS, originals))
    if not (all(installed) and install_stats['versions'] == len(adapter.SPECS)
            and install_stats['host_unique'] and install_stats['host_counts']):
        failures.append('install')
    if not runs_equal:
        failures.append('adapted runs')
    if not restored:
        failures.append('restore')
    if not (stats['in_order'] and stats['sorted'] and stats['sorted_once'] and stats['per_track']
            and stats['lazy'] and stats['eager'] and stats['groups_in_order'] and stats['groups_sorted']):
        failures.append('routes')
    report = dict(status='pass' if not failures else 'fail', cases=cases, stats=stats,
                  install=install_stats, failures=failures)
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
