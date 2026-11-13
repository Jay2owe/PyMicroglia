"""Compare local-lineage owner bodies with a growing border against the original, exactly.

reserve_local_lineages runs on A1's saved issue023 stack (accepted and loosened
settings) behind lineage_frame_components, as the pipeline installs it, with
and without owner_body_border; every array and table of the result must be
identical. _owner_body is also asked directly for every identity of sampled
frames, seeded from the previous frame, under both settings and on frames
without boxes; bodies, details and errors must match.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'
LOOSE = dict(minimum_fragment_px=1, minimum_area_continuity_improvement=0.0,
             minimum_expanded_area_ratio=0.05, maximum_expanded_area_ratio=20.0,
             minimum_area_ratio=0.05, maximum_area_ratio=20.0, minimum_substantial_core_px=40)


def same(first, second):
    if type(first) is not type(second):
        return f'type {type(first)} != {type(second)}'
    if isinstance(first, np.ndarray):
        return None if (first.dtype == second.dtype and first.shape == second.shape
                        and np.array_equal(first, second, equal_nan=first.dtype.kind == 'f')) else 'array'
    if isinstance(first, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(first, second, check_exact=True)
        except AssertionError as error:
            return str(error)[:300]
        return None
    if isinstance(first, dict):
        if first.keys() != second.keys():
            return 'keys'
        for key in first:
            problem = same(first[key], second[key])
            if problem:
                return f'{key}: {problem}'
        return None
    if isinstance(first, (tuple, list)):
        if len(first) != len(second):
            return 'length'
        for index, (a, b) in enumerate(zip(first, second)):
            problem = same(a, b)
            if problem:
                return f'[{index}] {problem}'
        return None
    if isinstance(first, float) and np.isnan(first) and np.isnan(second):
        return None
    return None if first == second else f'{first!r} != {second!r}'


def outcome(function, *args):
    try:
        return 'ok', function(*args)
    except Exception as error:
        return 'error', (type(error).__name__, str(error))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import accepted_history as history
    import local_lineage_reservations as module
    import lineage_frame_components
    import owner_body_border as adapter
    params = history._general('R08_generalise_local_lineage_reservation', 0)
    loose = {**params, **LOOSE}
    labels = tifffile.imread(HISTORY / '12_issue023_generalized.tif')
    unclaimed = tifffile.imread(HISTORY / '12_issue023_generalized_unclaimed.tif')
    lag = tifffile.imread(HISTORY / '07_trimmed_evidence/out/95_full_A1_lag01_float.tif')
    report = dict(failures=[], stages={}, questions=0, errors=0, bodies=0)
    with lineage_frame_components.install(module) as front:
        assert front['supported'], front
        for name, settings in (('accepted', params), ('loose', loose)):
            t0 = time.perf_counter()
            expected = outcome(module.reserve_local_lineages, labels, unclaimed, lag, settings)
            t1 = time.perf_counter()
            with adapter.install(module) as stats:
                assert stats['supported'], stats
                actual = outcome(module.reserve_local_lineages, labels, unclaimed, lag, settings)
            t2 = time.perf_counter()
            if expected[0] == 'ok' and actual[0] == 'ok':
                expected = ('ok', vars(expected[1]))
                actual = ('ok', vars(actual[1]))
            problem = same(expected, actual)
            report['stages'][name] = dict(status=expected[0], original_seconds=round(t1 - t0, 2),
                                          adapted_seconds=round(t2 - t1, 2), stats=dict(stats))
            print(name, report['stages'][name], problem, flush=True)
            if problem:
                report['failures'].append(dict(stage=name, problem=problem))
        rng = np.random.default_rng(23)
        frames = sorted(rng.choice(np.arange(1, len(labels)), 25, replace=False).tolist())
        with adapter.install(module) as stats:
            for frame in frames:
                owners = [int(v) for v in np.unique(labels[frame - 1]) if v > 0]
                for owner in owners:
                    reference = labels[frame - 1] == owner
                    for settings in (params, loose):
                        for distance in (12.0, 60.0):
                            for target in (labels[frame], labels[frame].astype(np.int32)):
                                report['questions'] += 1
                                old = outcome(module._owner_body.__wrapped__, target, owner, reference, distance, settings)
                                new = outcome(module._owner_body, target, owner, reference, distance, settings)
                                report['errors'] += old[0] == 'error'
                                report['bodies'] += old[0] == 'ok' and old[1][0] is not None
                                problem = same(old, new)
                                if problem:
                                    report['failures'].append(dict(frame=frame, owner=owner, problem=problem))
            report['question_stats'] = dict(stats)
        # Frames whose numbers map has no boxes take the whole-body route.
        boxed = module._frame_components

        def unboxed(frame):
            result = boxed(frame)
            result.__dict__.pop('_boxes', None)
            return result
        module._frame_components = unboxed
        try:
            with adapter.install(module) as stats:
                for frame in frames[:10]:
                    for owner in [int(v) for v in np.unique(labels[frame - 1]) if v > 0][:20]:
                        reference = labels[frame - 1] == owner
                        report['questions'] += 1
                        problem = same(outcome(module._owner_body.__wrapped__, labels[frame], owner, reference, 60.0, loose),
                                       outcome(module._owner_body, labels[frame], owner, reference, 60.0, loose))
                        if problem:
                            report['failures'].append(dict(unboxed=True, frame=frame, owner=owner, problem=problem))
                report['unboxed_stats'] = dict(stats)
        finally:
            module._frame_components = boxed
    restored = not hasattr(module, '_owner_border') and not hasattr(module._owner_body, '__wrapped__')
    report['restored'] = restored
    used = (report['question_stats']['local_borders'] > 0 and report['unboxed_stats']['whole_borders'] > 0
            and report['unboxed_stats']['local_borders'] == 0)
    report['status'] = 'pass' if not report['failures'] and restored and used else 'fail'
    report['failures'] = report['failures'][:20]
    report['sources'] = {str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                         for f in [Path(__file__), Path(adapter.__file__), Path(module.__file__)]}
    a.output.write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'sources'}, default=str))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
