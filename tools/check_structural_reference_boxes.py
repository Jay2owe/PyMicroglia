"""Compare boxed structural-run references and the whole correction with the original.

References: every identity of several accepted stacks (plain and disk-backed)
at several history and core thresholds, against the original statements.
Correction: correct_structurally_proven_unresolved_runs on real stacks with
default and relaxed proof thresholds, original vs adapted, every output.
"""
import argparse, json, sys, time
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'


def original_references(module, source, identity_frames, params):
    references = []
    for identity, frames in identity_frames.items():
        if len(frames) < params.minimum_reference_history_frames:
            continue
        for frame_index in frames:
            for _, mask in module.identity_components(source[frame_index], identity):
                cores = module.eroded_cores(mask, 1)
                if not cores or int(cores[0][0]) <= params.maximum_existing_core_px:
                    continue
                centre = module._centroid(cores[0][1])
                references.append({
                    "identity": int(identity),
                    "imagej_frame": int(frame_index + 1),
                    "history_frames": int(len(frames)),
                    "y": float(centre[0]), "x": float(centre[1]),
                })
    return references


def same_outputs(expected, actual):
    for e, a in zip(expected, actual):
        if isinstance(e, pd.DataFrame):
            try:
                pd.testing.assert_frame_equal(e, a, check_exact=True)
            except AssertionError as error:
                return str(error)[:300]
        elif not (type(e) is type(a) and e.dtype == a.dtype and np.array_equal(e, a)):
            return 'array'
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import partial_body_transfers as module
    import model_review
    import pipeline
    import array_storage
    import structural_reference_boxes as adapter
    failures, rows = [], []
    times = dict(original=0.0, boxed=0.0)
    raw = tifffile.imread(HISTORY / '07_trimmed_evidence/out/95_full_A1_registered_raw.tif')
    base = module.StructuralFallbackParams()
    settings = [base, replace(base, minimum_reference_history_frames=1, maximum_existing_core_px=0),
                replace(base, minimum_reference_history_frames=200, maximum_existing_core_px=60)]
    stems = ('01_issue005.tif', '05_issue009.tif', '11_issue017_rebase.tif')
    for name in stems:
        stack = tifffile.imread(HISTORY / name)
        identity_frames = module._identity_frame_indices(stack)
        with ExitStack() as scope:
            pool = array_storage.Pool(a.scratch, minimum_bytes=1 << 20, temporary_ram_bytes=32 << 20)
            scope.enter_context(array_storage.install(pool))
            working = np.array(stack)
            with adapter.install(module, model_review, pipeline) as stats:
                assert stats['supported']
                assert pipeline.correct_structurally_proven_unresolved_runs is module.correct_structurally_proven_unresolved_runs
                builder = vars(module)['_structural_references']
                for number, params in enumerate(settings):
                    for label, source in (('plain', stack), ('working', working)):
                        t0 = time.perf_counter(); expected = original_references(module, source, identity_frames, params)
                        t1 = time.perf_counter(); actual = builder(source, identity_frames, params)
                        t2 = time.perf_counter()
                        times['original'] += t1 - t0; times['boxed'] += t2 - t1
                        rows.append(dict(case=f'{name}-{label}-{number}', references=len(expected)))
                        if expected != actual:
                            failures.append(f'references {name}-{label}-{number}')
                assert stats['fallback'] == 0
            assert pipeline.correct_structurally_proven_unresolved_runs is module.correct_structurally_proven_unresolved_runs
    relaxed = replace(base, strong_two_core_score=0.3, strong_core_separation_px=8.0,
                      strong_core_intensity_ratio=0.2, repeated_pair_score=0.3,
                      repeated_pair_intensity_ratio=0.3, maximum_reference_distance_px=40.0)
    for name in ('01_issue005.tif', '11_issue017_rebase.tif'):
        stack = tifffile.imread(HISTORY / name)
        for number, params in enumerate((base, relaxed)):
            objects = model_review.audit_two_core_objects(stack, raw, params.two_core)
            original = module.correct_structurally_proven_unresolved_runs
            t0 = time.perf_counter(); expected = original(stack, raw, params, objects=objects)
            with adapter.install(module, model_review, pipeline) as stats:
                t1 = time.perf_counter(); actual = module.correct_structurally_proven_unresolved_runs(
                    stack, raw, params, objects=objects)
                t2 = time.perf_counter()
            problem = same_outputs(expected, actual)
            rows.append(dict(case=f'correction-{name}-{number}', events=len(expected[1]),
                             frames=len(expected[2]), seconds=[t1 - t0, t2 - t1]))
            if problem:
                failures.append(f'correction {name}-{number}: {problem}')
    report = dict(status='pass' if not failures else 'fail', seconds=times, rows=rows, failures=failures[:20])
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report, indent=1))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
