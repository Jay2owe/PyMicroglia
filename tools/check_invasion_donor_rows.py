"""Compare concurrent duplicate invasion with positional donor-interval rows.

recover runs on the accepted A1 input of stage 32 (accepted and loosened
settings; plain and memory-pool movies) with and without invasion_donor_rows;
proposals, labels, applications, frame audits and warnings must be identical.
The memo's interval answer is then compared with the original filter on real
track groups and synthetic ones with float, missing, object and nullable
frames, over many intervals and lengths.
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

LOOSE = dict(minimum_resident_run_frames=3, minimum_donor_canonical_support_frames=3,
             minimum_donor_canonical_purity=0.5, maximum_intervening_blank_frames=6,
             minimum_intervening_blank_frames=0, maximum_endpoint_step_radii=20.0,
             minimum_donor_separation_sum_radii=0.0, maximum_separable_valley_ratio=1.0,
             maximum_invasion_fraction=0.3, minimum_invasion_run_frames=1)


def original_between(group, first, last, invasion):
    rows = group[group.frame.between(first, last)]
    return rows if len(rows) == int(invasion['frames']) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import array_storage
    import concurrent_duplicate_invasion as invasion
    import invasion_donor_rows as adapter
    config = json.loads((ENGINE / 'config.json').read_text(encoding='utf-8'))
    params = dict(config['accepted_postprocessing']['field_wide_concurrent_duplicate_invasion'].get('parameters', {}))
    params['targeting_mode'] = 'field_wide_discovery'
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

    stem = '31_issue055_post_split_backfill'
    labels = tifffile.imread(HISTORY / (stem + '.tif'))
    unclaimed = tifffile.imread(HISTORY / (stem + '_unclaimed.tif'))
    raw = tifffile.imread(next((HISTORY / '07_trimmed_evidence/out').glob('*_registered_raw.tif')))
    points = pd.read_csv(HISTORY / '26_latent_body_tracks/out/latent_track_points.csv')
    thresholds = pd.read_csv(HISTORY / '25_raw_physical_hypotheses/out/frame_evidence_thresholds.csv')
    report = dict(failures=[], stages={}, questions=0)
    for name, settings in (('accepted', params), ('loose', loose)):
        for form in ('plain', 'pool'):
            labels_in = labels.copy() if form == 'plain' else pooled(labels)
            unclaimed_in = unclaimed.copy() if form == 'plain' else pooled(unclaimed)
            call = lambda: invasion.recover(labels_in, unclaimed_in, raw, points.copy(), thresholds.copy(), settings)
            t0 = time.perf_counter()
            expected = outcome(call)
            t1 = time.perf_counter()
            with adapter.install(invasion) as stats:
                assert stats['supported'], stats
                actual = outcome(call)
            t2 = time.perf_counter()
            problem = same(expected, actual)
            proposals = expected[1][1] if expected[0] == 'ok' else None
            key = f'{name}:{form}'
            report['stages'][key] = dict(
                status=expected[0], proposals=None if proposals is None else len(proposals),
                eligible=None if proposals is None or not len(proposals)
                else int((proposals.discovery_status == 'eligible').sum()),
                warnings=len(expected[2]), original_seconds=round(t1 - t0, 2),
                adapted_seconds=round(t2 - t1, 2), stats=dict(stats))
            print(key, report['stages'][key], problem, flush=True)
            if problem:
                report['failures'].append(dict(stage=key, problem=problem))

    scored = invasion.physical.attach_owners(points, labels)
    groups = [group.sort_values('frame') for _, group in scored.groupby('track_id', sort=True)][:300]
    base = scored.iloc[:8].copy()
    floats = base.copy(); floats['frame'] = [1.0, 2.5, 3.0, 4.0, 4.5, 6.0, 7.0, 9.0]
    missing = base.copy(); missing['frame'] = [1.0, np.nan, 3.0, 4.0, 5.0, np.nan, 7.0, 8.0]
    texts = base.copy(); texts['frame'] = ['1', '2', '3', '4', '5', '6', '7', '8']
    objects = base.copy(); objects['frame'] = pd.Series([1, 2, 3, 4, 5, 6, 7, 8], dtype=object).values
    nullable = base.copy(); nullable['frame'] = pd.array([1, 2, None, 4, 5, 6, 7, 8], dtype='Int64')
    duplicate = base.copy(); duplicate['frame'] = [1, 2, 2, 3, 4, 4, 5, 6]
    groups += [floats, missing, texts, objects, nullable, duplicate, base.iloc[:0]]
    stats = dict(indexed=0, original=0)
    memo = adapter.Memo(stats)
    rng = np.random.default_rng(56)
    for group in groups:
        for _ in range(25):
            first = int(rng.integers(-2, 470)); last = first + int(rng.integers(-1, 12))
            frames = int(rng.integers(0, 13))
            for asked in ((first, last), (1, 8), (2, 4)):
                report['questions'] += 1
                spec = {'frames': frames}
                problem = same(outcome(original_between, group, *asked, spec), outcome(memo.between, group, *asked, spec))
                if problem:
                    report['failures'].append(dict(between=asked, frames=frames, problem=problem))
            for frames in (1, 3, 4, 8):
                report['questions'] += 1
                spec = {'frames': frames}
                problem = same(outcome(original_between, group, 1, 8, spec), outcome(memo.between, group, 1, 8, spec))
                if problem:
                    report['failures'].append(dict(between=(1, 8), frames=frames, problem=problem))
    report['memo_stats'] = stats
    restored = not hasattr(invasion.discover, '__wrapped__') and '_invasion_rows_factory' not in vars(invasion)
    report['restored'] = restored
    report['status'] = 'pass' if not report['failures'] and restored and stats['original'] and stats['indexed'] else 'fail'
    report['failures'] = report['failures'][:20]
    a.output.write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
    print(json.dumps(dict(status=report['status'], questions=report['questions'], memo=stats, failures=report['failures'])))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
