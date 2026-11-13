"""Compare terminal projection diversion with per-call track birth/row memos.

discover and discover_and_apply run on the accepted A1 input of stage 57
(accepted and loosened settings; plain and memory-pool label movies) with and
without diversion_track_rows; proposals, labels, applications, details and
warnings must be identical. The memo's two answers are then compared with the
original expressions on real track groups and synthetic ones with duplicate,
float, missing and absent frames.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_projection_chain_rows import same, outcome  # noqa: E402

LOOSE = dict(terminal_boundary_movie_fraction=0.6, minimum_anchor_movie_fraction=0.02,
             maximum_newborn_age_movie_fraction=0.1, maximum_takeover_delay_movie_fraction=0.1,
             maximum_terminal_slack_movie_fraction=0.5, minimum_false_transfer_jump_sum_radii=0.5,
             maximum_neighbour_distance_sum_radii=6.0, minimum_qualified_two_core_frames=0)


def original_born(group):
    visible = group[group.physically_visible.astype(bool)]
    return None if visible.empty else int(visible.frame.min())


def original_at(group, frame):
    rows = group[group.frame.astype(int).eq(frame)]
    return rows.iloc[0] if len(rows) == 1 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import array_storage
    import terminal_projection_diversion as diversion
    import terminal_projection_diversion_integration as integration
    import diversion_track_rows as adapter
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

    labels = tifffile.imread(HISTORY / '56_well_b1_issue001_bracketed_established_owner_invasion.tif')
    raw = tifffile.imread(next((HISTORY / '07_trimmed_evidence/out').glob('*_registered_raw.tif')))
    points = pd.read_csv(HISTORY / '26_latent_body_tracks/out/latent_track_points.csv')
    report = dict(failures=[], stages={}, questions=0)
    for name, settings in (('accepted', params), ('loose', loose)):
        for form in ('plain', 'pool'):
            for function in ('discover', 'discover_and_apply'):
                if form == 'pool' and function == 'discover':
                    continue
                labels_in = labels.copy() if form == 'plain' else pooled(labels)
                call = lambda: getattr(diversion, function)(labels_in, raw, points.copy(), settings)
                t0 = time.perf_counter()
                expected = outcome(call)
                t1 = time.perf_counter()
                with adapter.install(diversion) as stats:
                    assert stats['supported'], stats
                    actual = outcome(call)
                t2 = time.perf_counter()
                problem = same(expected, actual)
                table = expected[1] if function == 'discover' else expected[1][1] if expected[0] == 'ok' else None
                key = f'{name}:{form}:{function}'
                report['stages'][key] = dict(
                    status=expected[0], proposals=None if table is None else len(table),
                    eligible=None if table is None or not len(table) else int(table.eligible.sum()),
                    warnings=len(expected[2]), original_seconds=round(t1 - t0, 2),
                    adapted_seconds=round(t2 - t1, 2), stats=dict(stats))
                print(key, report['stages'][key], problem, flush=True)
                if problem:
                    report['failures'].append(dict(stage=key, problem=problem))

    attached = diversion.attach_owners(points, labels).sort_values(['track_id', 'frame']).reset_index(drop=True)
    attached['physically_visible'] = attached.state.astype(str).isin(diversion.VISIBLE_STATES)
    groups = [group.sort_values('frame') for _, group in attached.groupby('track_id', sort=True)][:500]
    base = attached.iloc[:6].copy()
    duplicate = base.copy(); duplicate['frame'] = [5, 5, 6, 7, 7, 8]
    floats = base.copy(); floats['frame'] = [1.0, 2.7, 3.2, 4.0, 5.9, 6.0]
    missing = base.copy(); missing['frame'] = [1.0, np.nan, 3.0, 4.0, 5.0, 6.0]
    hidden = base.copy(); hidden['physically_visible'] = False
    groups += [duplicate, floats, missing, hidden, base.iloc[:0]]
    frames = list(range(-2, 480, 5)) + [5, 7, 2, 3]
    stats = dict(groups=0, born_reused=0, rows=0)
    memo = adapter.Memo(stats)
    for group in groups:
        for _ in range(2):
            report['questions'] += 1
            problem = same(outcome(original_born, group), outcome(memo.born, group))
            if problem:
                report['failures'].append(dict(born=len(group), problem=problem))
            for frame in frames:
                report['questions'] += 1
                problem = same(outcome(original_at, group, frame), outcome(memo.at, group, frame))
                if problem:
                    report['failures'].append(dict(at=frame, problem=problem))
    restored = not hasattr(diversion.discover, '__wrapped__') and '_diversion_rows_factory' not in vars(diversion)
    report['restored'] = restored
    report['status'] = 'pass' if not report['failures'] and restored else 'fail'
    report['failures'] = report['failures'][:20]
    a.output.write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
    print(json.dumps(dict(status=report['status'], questions=report['questions'], failures=report['failures'])))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
