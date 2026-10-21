"""Compare whole presence completion with and without the presence adapters.

Real accepted-history labels (windows of the Issue 017 input), the registered
raw and lag movies, and Issue 017's own parameter builder: end-of-track
searches (forward completion) and bracketed gaps (bidirectional completion)
for identities that really end or pause inside the window. The completed
movie, the recovered-pixel stack and every attempt row must be equal.
"""
import argparse, json, sys, time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=70)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import accepted_history as history
    import presence
    import presence_area_table, presence_contrast_crop, presence_mask_boxes, presence_settle_boxes
    import presence_descriptor_medians
    repair = history._issue_module(history.I017, 'm3_repair')
    labels_all = tifffile.imread(HISTORY / '10_blob_residency.tif')
    raw_all = tifffile.imread(HISTORY / '20_continuity_evidence_inputs/out/raw.tif')
    lag_all = tifffile.imread(HISTORY / '20_continuity_evidence_inputs/out/lag.tif')
    assert raw_all.shape == labels_all.shape, (raw_all.shape, labels_all.shape)
    rng = np.random.default_rng(20261021)
    failures, cases, rows = [], 0, 0
    seconds = dict(original=0.0, adapted=0.0)
    for start in (40, 200, 360):
        stop = min(len(labels_all), start + a.frames)
        labels = labels_all[start:stop]
        raw, lag = raw_all[start:stop], lag_all[start:stop]
        present = {int(v): np.flatnonzero(np.any(labels == v, axis=(1, 2))) for v in np.unique(labels)[1:]}
        endings = [dict(identity=i, start_t=int(f[-1]) + 1, end_t=len(labels) - 1)
                   for i, f in present.items() if 5 <= f[-1] < len(labels) - 2]
        gaps = []
        for i, f in present.items():
            holes = np.flatnonzero(np.diff(f) > 1)
            for h in holes[:2]:
                gaps.append(dict(identity=i, start_t=int(f[h]) + 1, end_t=int(f[h + 1]) - 1))
        for kind, intervals in (('ending', endings), ('bracketed', gaps)):
            if not intervals:
                continue
            for subset in (intervals, [intervals[k] for k in rng.choice(len(intervals), max(1, len(intervals) // 2), replace=False)]):
                params = repair._presence_params(subset, bracketed=kind == 'bracketed')
                function = 'complete_presence' if kind == 'ending' else 'complete_presence_bidirectional'
                t0 = time.perf_counter()
                expected = getattr(presence, function)(labels, raw, lag, params)
                t1 = time.perf_counter()
                with ExitStack() as stack:
                    for adapter in (presence_contrast_crop, presence_area_table, presence_mask_boxes, presence_settle_boxes,
                                    presence_descriptor_medians):
                        assert stack.enter_context(adapter.install(presence))['supported'], adapter.__name__
                    actual = getattr(presence, function)(labels, raw, lag, params)
                t2 = time.perf_counter()
                seconds['original'] += t1 - t0; seconds['adapted'] += t2 - t1
                cases += 1
                try:
                    for x, y in zip(expected[:2], actual[:2]):
                        assert x.dtype == y.dtype and np.array_equal(x, y)
                    pd.testing.assert_frame_equal(expected[2], actual[2], check_exact=True)
                    rows += len(expected[2])
                except AssertionError as error:
                    failures.append(dict(start=start, kind=kind, intervals=len(subset), error=str(error)[:300]))
    report = dict(status='pass' if not failures and rows else 'fail', cases=cases, attempt_rows=rows,
                  seconds=seconds, failures=failures[:10])
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
