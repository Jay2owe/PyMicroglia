"""Compare the two-core object audit with the boxed audit on real label stacks.

Each stack is audited with its registered raw stack under the default and
looser settings (more objects, two erosion steps); the tables must be equal
in every value and dtype.
"""
import argparse, json, sys, time
from dataclasses import replace
from pathlib import Path

import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
INPUTS = ROOT / 'development/cd68_full_red_green/rigid_batch/.j/A1/inputs'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--labels', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import model_review as module
    import two_core_audit_boxes as adapter
    raw = tifffile.imread(INPUTS / '95_full_A1.tif')
    original = module.audit_two_core_objects
    settings = [module.TwoCoreParams(),
                replace(module.TwoCoreParams(), minimum_object_px=20, minimum_core_px=4,
                        minimum_core_fraction=0.02, minimum_core_separation_px=3.0,
                        minimum_core_intensity_ratio=0.1, erosion_iterations=2)]
    results, failures = [], []
    for path in a.labels:
        labels = tifffile.imread(path)
        for number, params in enumerate(settings):
            t0 = time.perf_counter(); expected = original(labels, raw[:len(labels)], params)
            t1 = time.perf_counter()
            with adapter.install(module) as stats:
                actual = module.audit_two_core_objects(labels, raw[:len(labels)], params)
            t2 = time.perf_counter()
            try:
                pd.testing.assert_frame_equal(expected, actual, check_exact=True)
                same = stats['boxed'] == 1
            except AssertionError:
                same = False
            row = dict(labels=path.name, settings=number, equal=same, objects=len(expected),
                       original_s=round(t1 - t0, 1), boxed_s=round(t2 - t1, 1))
            print(row, flush=True)
            results.append(row)
            if not same:
                failures.append(f'{path.name}-{number}')
    report = dict(status='pass' if not failures else 'fail', results=results, failures=failures)
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
