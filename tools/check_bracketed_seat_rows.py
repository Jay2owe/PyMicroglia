"""Compare bracketed ownerless seat completion with once-cast gap-frame rows.

complete() runs on the accepted A1 input of stage 109 (accepted and loosened
settings, with and without dormant-track support; plain and memory-pool label
movies) with and without bracketed_seat_rows; labels, audits, frame audits,
details and warnings must be identical, and on the memory pool the audited
presence sites must be recognised as often as before (the rewrite keeps the
file's name and line numbers). The memo's rows are then compared with the
original filter for every frame and identity of real attached points, and of
synthetic tables with float, missing and duplicate frames.
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

LOOSE = dict(minimum_flank_presence_fraction=0.3, minimum_visible_gap_fraction=0.3,
             minimum_strong_gap_fraction=0.0, maximum_path_offset_field_radii=8.0,
             maximum_step_field_radii=8.0, minimum_track_owner_purity=0.6,
             minimum_identity_presence_movie_fraction=0.05)


def original_rows(attached, frame, identity, states):
    return attached[
        attached.frame.astype(int).eq(frame)
        & attached.state.astype(str).isin(states)
        & attached.physically_visible.astype(bool)
        & attached.candidate_owner.astype(int).eq(0)
        & attached.inferred_owner.astype(int).eq(identity)].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import array_storage
    import bracketed_ownerless_seat_completion as completion
    import bracketed_ownerless_seat_completion_integration as integration
    import bracketed_seat_rows as adapter
    config = json.loads((ENGINE / 'config.json').read_text(encoding='utf-8'))
    _, params = integration._configured(config)
    variants = [('accepted', params), ('loose', {**params, **LOOSE}),
                ('loose_no_dormant', {**params, **LOOSE, 'allow_dormant_track_raw_support': False})]
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

    stem = '106_well_b3_issue017_terminal_boundary_vanished_seat'
    labels = tifffile.imread(HISTORY / (stem + '.tif'))
    unclaimed = tifffile.imread(HISTORY / (stem + '_unclaimed.tif'))
    raw = tifffile.imread(next((HISTORY / '07_trimmed_evidence/out').glob('*_registered_raw.tif')))
    points = pd.read_csv(HISTORY / '26_latent_body_tracks/out/latent_track_points.csv')
    thresholds = pd.read_csv(HISTORY / '25_raw_physical_hypotheses/out/frame_evidence_thresholds.csv')
    report = dict(failures=[], stages={}, questions=0)
    for name, settings in variants:
        for form in ('plain', 'pool'):
            if form == 'pool' and name == 'loose_no_dormant':
                continue
            labels_in = labels.copy() if form == 'plain' else pooled(labels)
            unclaimed_in = unclaimed.copy() if form == 'plain' else pooled(unclaimed)
            call = lambda: completion.complete(labels_in, unclaimed_in, raw, points.copy(), thresholds.copy(), settings)
            queries = array_storage.presence_stats['queries']
            t0 = time.perf_counter()
            expected = outcome(call)
            t1 = time.perf_counter()
            original_queries = array_storage.presence_stats['queries'] - queries
            with adapter.install(completion) as stats:
                assert stats['supported'], stats
                actual = outcome(call)
            t2 = time.perf_counter()
            adapted_queries = array_storage.presence_stats['queries'] - queries - original_queries
            problem = same(expected, actual)
            if not problem and original_queries != adapted_queries:
                problem = f'presence site queries {original_queries} != {adapted_queries}'
            audit = expected[1][1] if expected[0] == 'ok' else None
            key = f'{name}:{form}'
            report['stages'][key] = dict(
                status=expected[0], gaps=None if audit is None else len(audit),
                warnings=len(expected[2]), presence_queries=[original_queries, adapted_queries],
                original_seconds=round(t1 - t0, 2), adapted_seconds=round(t2 - t1, 2), stats=dict(stats))
            print(key, report['stages'][key], problem, flush=True)
            if problem:
                report['failures'].append(dict(stage=key, problem=problem))

    attached = completion.attach_owners(points, labels)
    rng = np.random.default_rng(109)
    attached['inferred_owner'] = rng.choice(np.unique(labels)[1:30], len(attached))
    base = attached.iloc[:12].copy()
    floats = base.copy(); floats['frame'] = np.arange(12) + 0.5
    missing = base.copy(); missing['frame'] = [np.nan] + list(range(11))
    duplicate = base.copy(); duplicate['frame'] = [1] * 6 + [2] * 6
    tables = [('real', attached), ('floats', floats), ('missing', missing), ('duplicate', duplicate),
              ('empty', base.iloc[:0])]
    stats = dict(tables=0, rows=0)
    for name, table in tables:
        memo = adapter.Memo(stats)
        identities = sorted(set(map(int, table.inferred_owner.tolist())))[:30] + [0, 10 ** 6]
        frames = sorted(set(range(0, 480, 9)) | {1, 2, 3})
        for frame in frames:
            for identity in identities[:8] if name == 'real' else identities:
                report['questions'] += 1
                problem = same(outcome(original_rows, table, frame, identity, completion.VISIBLE_STATES),
                               outcome(memo.visible, table, frame, identity, completion.VISIBLE_STATES))
                if problem:
                    report['failures'].append(dict(table=name, frame=frame, identity=identity, problem=problem))
    report['memo_stats'] = stats
    restored = not hasattr(completion.complete, '__wrapped__') and '_seat_rows_factory' not in vars(completion)
    report['restored'] = restored
    report['status'] = 'pass' if not report['failures'] and restored else 'fail'
    report['failures'] = report['failures'][:20]
    a.output.write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
    print(json.dumps(dict(status=report['status'], questions=report['questions'], failures=report['failures'])))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
