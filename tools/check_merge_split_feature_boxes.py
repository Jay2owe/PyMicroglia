"""Compare Issue 011 frame features measured in boxes with the installed route.

The Issue 011 profiler module is loaded the way the accepted history loads it,
with the engine's largest-component speed wrapper installed, and its
extract_features runs on the real Issue 011 input movie and registered raw
movie with and without the boxed adapter. Tables must be identical (values,
dtypes, column order). Memory-mapped frames must take the boxed path through
plain views; negative labels, an empty frame and an unwrapped feature function
must take the original loop and still match.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'
RAW = HISTORY / '07_trimmed_evidence/out/95_full_A1_registered_raw.tif'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=120)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import accepted_history as history
    from component_feature_speed import make_component_feature
    import merge_split_feature_boxes as adapter
    labels = tifffile.imread(HISTORY / '06_issue010.tif')
    raw = tifffile.imread(RAW)
    assert labels.shape == raw.shape, (labels.shape, raw.shape)
    labels, raw = labels[:a.frames], raw[:a.frames]
    a.scratch.mkdir(parents=True, exist_ok=True)
    mapped = np.memmap(a.scratch / 'labels.dat', dtype=labels.dtype, mode='w+', shape=(8,) + labels.shape[1:])
    mapped[:] = labels[:8]
    mapped.flush()
    odd = labels[:4].astype(np.int32)
    odd[1] = 0
    odd[2][odd[2] == odd[2].max()] = -3
    cases = [('real', labels, raw, 0), ('real offset', labels[:20], raw[:20], 7),
             ('memmap', mapped, raw[:8], 0), ('negative and empty', odd, raw[:4], 0)]
    failures, seconds, results = [], dict(original=0.0, boxed=0.0), []
    for wrapped in (True, False):
        module = history._issue_module(history.MERGE_SPLITS_OPS, 'profile_merge_splits')
        if wrapped:
            module.component_feature = make_component_feature(module.component_feature)
        for name, movie, intensity, offset in cases:
            if not wrapped and name != 'real offset':
                continue
            t0 = time.perf_counter()
            expected = module.extract_features(movie, intensity, offset)
            t1 = time.perf_counter()
            with adapter.install(module) as stats:
                assert stats['supported'], stats
                actual = module.extract_features(movie, intensity, offset)
                counters = dict(stats)
            seconds['original'] += t1 - t0; seconds['boxed'] += time.perf_counter() - t1
            try:
                pd.testing.assert_frame_equal(expected, actual, check_exact=True)
                assert list(expected.dtypes) == list(actual.dtypes)
            except AssertionError as error:
                failures.append(dict(case=name, wrapped=wrapped, error=str(error)[:300]))
            results.append(dict(case=name, wrapped=wrapped, rows=len(expected), counters=counters))
        assert not hasattr(module.extract_features, '__wrapped__')
    boxed = {r['case']: r['counters'] for r in results if r['wrapped']}
    assert boxed['real']['boxed_frames'] == len(labels) and boxed['memmap']['boxed_frames'] == 8, boxed
    assert boxed['negative and empty']['original_frames'] >= 1, boxed
    report = dict(status='pass' if not failures else 'fail', seconds=seconds, results=results, failures=failures)
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
