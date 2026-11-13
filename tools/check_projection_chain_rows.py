"""Compare terminal projection-chain completion with per-call track row memos.

discover and apply run on the accepted A1 inputs of stage 103 (accepted and
loosened settings so that proposals are applied; plain and memory-pool label
movies), with and without projection_chain_rows; audit tables, internal
rows, attached points, parent counts, labels, application tables, details
and warnings must be identical. `_point` is then compared row for row inside
a call scope on real track groups and synthetic ones with duplicate frames,
float frames, missing frames and text, NaN and out-of-range frame questions.
"""
import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_owner_presence_frames import same as same_value  # noqa: E402

LOOSE = dict(minimum_anchor_support_movie_fraction=0.3, minimum_anchor_owner_purity=0.8,
             minimum_parent_support_movie_fraction=0.05, minimum_parent_owner_purity=0.6,
             minimum_parent_anchor_presence_fraction=0.5, maximum_parent_anchor_radius_ratio=1.5,
             maximum_parent_anchor_distance_sum_radii=6.0, maximum_child_span_movie_fraction=0.3,
             maximum_right_censor_movie_fraction=0.5, minimum_child_visible_fraction=0.3,
             minimum_child_strong_fraction=0.1, maximum_child_parent_radius_ratio=3.0,
             maximum_child_parent_distance_sum_radii=8.0, minimum_outward_cosine=-1.0,
             minimum_radial_extension_parent_radii=-5.0, maximum_child_step_sum_radii=5.0,
             maximum_parent_lapse_fraction=1.0, minimum_parent_distance_margin=0.0)


def same(a, b):
    if isinstance(a, (list, tuple)) and type(a) is type(b):
        if len(a) != len(b):
            return 'length'
        for index, (x, y) in enumerate(zip(a, b)):
            problem = same(x, y)
            if problem:
                return f'[{index}] {problem}'
        return None
    if isinstance(a, pd.Series) or isinstance(b, pd.Series):
        if type(a) is not type(b) or a.name != b.name:
            return 'series type or name'
        try:
            pd.testing.assert_series_equal(a, b, check_exact=True)
        except AssertionError as error:
            return 'series ' + str(error)[:200]
        return None
    return same_value(a, b)


def outcome(function, *args):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            result = ('ok', function(*args))
        except Exception as error:
            result = ('error', (type(error).__name__, str(error)))
    return result + ([(w.category.__name__, str(w.message)) for w in caught],)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import array_storage
    import terminal_projection_chain_completion as completion
    import terminal_projection_chain_completion_integration as integration
    import projection_chain_rows as adapter
    config = json.loads((ENGINE / 'config.json').read_text(encoding='utf-8'))
    _, params = integration._configured(config)
    loose = {**params, **LOOSE}
    a.scratch.mkdir(parents=True, exist_ok=True)
    pool = array_storage.Pool(a.scratch / 'pool')
    kept = []

    def pooled(stack):
        path = a.scratch / f'stack_{len(kept)}.dat'
        mapped = np.memmap(path, dtype=stack.dtype, mode='w+', shape=stack.shape)
        mapped[:] = stack
        mapped.flush()
        working = pool.adopt(mapped, path)
        kept.append(working)
        return working

    stem = '101_well_b3_issue013_anchored_projection_raw_gap_completion'
    labels = tifffile.imread(HISTORY / (stem + '.tif'))
    unclaimed = tifffile.imread(HISTORY / (stem + '_unclaimed.tif'))
    raw = tifffile.imread(next((HISTORY / '07_trimmed_evidence/out').glob('*_registered_raw.tif')))
    points = pd.read_csv(HISTORY / '26_latent_body_tracks/out/latent_track_points.csv')
    thresholds = pd.read_csv(HISTORY / '25_raw_physical_hypotheses/out/frame_evidence_thresholds.csv')
    report = dict(failures=[], stages={}, points=0)

    def stage():
        audit, internal, attached, parents = completion.discover(labels_in, points.copy(), settings)
        applied = completion.apply(labels_in, unclaimed_in, raw, thresholds.copy(), attached, internal, settings)
        return audit, internal, attached, parents, applied

    for name, settings in (('accepted', params), ('loose', loose)):
        for form in ('plain', 'pool'):
            labels_in = labels.copy() if form == 'plain' else pooled(labels)
            unclaimed_in = unclaimed.copy() if form == 'plain' else pooled(unclaimed)
            t0 = time.perf_counter()
            expected = outcome(stage)
            t1 = time.perf_counter()
            with adapter.install(completion) as stats:
                assert stats['supported'], stats
                actual = outcome(stage)
            t2 = time.perf_counter()
            problem = same(expected, actual)
            eligible = int(expected[1][0].eligible.sum()) if expected[0] == 'ok' else None
            report['stages'][f'{name}:{form}'] = dict(
                status=expected[0], eligible=eligible, warnings=len(expected[2]),
                original_seconds=round(t1 - t0, 2), adapted_seconds=round(t2 - t1, 2), stats=dict(stats))
            print(name, form, report['stages'][f'{name}:{form}'], problem, flush=True)
            if problem:
                report['failures'].append(dict(stage=name, form=form, problem=problem))

    # _point inside a call scope: every visible frame of real tracks, plus synthetic groups.
    attached = completion.attach_owners(points, labels).sort_values(['track_id', 'frame']).reset_index(drop=True)
    groups = [group.sort_values('frame') for _, group in attached.groupby('track_id', sort=True)][:400]
    base = attached.iloc[:6].copy()
    duplicate = base.copy(); duplicate['frame'] = [5, 5, 6, 7, 7, 8]
    floats = base.copy(); floats['frame'] = [1.0, 2.7, 3.2, 4.0, 5.9, 6.0]
    missing = base.copy(); missing['frame'] = [1.0, np.nan, 3.0, 4.0, 5.0, 6.0]
    groups += [duplicate, floats, missing, base.iloc[:0]]
    questions = list(range(-2, 480, 7)) + [5, 7, 2, 2.9, np.int64(8), '3', 'x', None, float('nan'), 2 ** 70, True]
    def ask_all():
        rows = []
        for group in groups:
            for frame in questions:
                rows.append(outcome(completion._point, group, frame))
            rows.append(outcome(completion._visible, group))
            rows.append(outcome(completion._positive_owner_summary, group))
            rows.append(outcome(completion._visible, group))
        return rows

    expected = ask_all()
    with adapter.install(completion) as stats:
        # The questions are asked as discover would ask them: inside its call scope.
        scope = completion.discover.row_scope
        scope.append({})
        try:
            actual = ask_all()
        finally:
            scope.pop()
        outside = ask_all()
        seen = dict(stats)
    report['points'] = len(expected)
    for number, (left, right) in enumerate(zip(expected, outside)):
        problem = same(left, right)
        if problem:
            report['failures'].append(dict(question=number, outside_scope=True, problem=problem))
    restore_ok = True
    for number, (left, right) in enumerate(zip(expected, actual)):
        problem = same(left, right)
        if problem:
            report['failures'].append(dict(question=number, problem=problem))
    if len(expected) != len(actual):
        report['failures'].append('question count')
    report['function_stats'] = seen
    restored = all(not hasattr(getattr(completion, name), '__wrapped__')
                   for name in ('_point', '_visible', '_positive_owner_summary', 'discover', 'apply'))
    report['restored'] = restored and restore_ok
    report['status'] = 'pass' if not report['failures'] and report['restored'] and seen.get('points') else 'fail'
    report['failures'] = report['failures'][:20]
    a.output.write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
    print(json.dumps(dict(status=report['status'], points=report['points'], stats=seen, failures=report['failures'])))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
