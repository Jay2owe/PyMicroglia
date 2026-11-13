"""Compare dormant-release detection with and without the far-box skips, exactly.

The Issue 012 detector runs on real accepted label windows (three stacks of the
accepted history, 50 frames each) with dormancy starts built from real
neighbouring cells (prior and host within 40 px) at random frames, under the
permissive enumeration settings (every host loss, every candidate size) and at
the default radius as well as at 0, 15 and an unparseable radius. raw and lag
are random. The event and release tables must equal the original's exactly
(values, dtypes, order and CSV text), skips must occur in the enumeration,
prior-local and persistence searches, and the adapter must restore the module.
"""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--engine', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--frames', type=int, default=50)
    p.add_argument('--events', type=int, default=24)
    a = p.parse_args()
    import platform
    platform.platform = lambda *x, **k: 'Windows-dormant-far-checks'
    import numpy as np
    import pandas as pd
    import tifffile
    from scipy import ndimage as ndi
    sys.path.insert(0, str(a.engine.resolve() / 'code'))
    sys.path.insert(0, str(a.adapter.resolve()))
    import accepted_history as ah
    import dormant_component_reuse as reuse
    module = ah._issue_module(ah.DOMAIN_OWNERSHIP_OPS, 'detect_dormant_releases')
    original_detect = module.detect
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    rng = np.random.default_rng(77)
    report = dict(status='pass', cases=0, events=0, releases=0, persistence_values=set())
    timings = dict(original=0.0, adapted=0.0)
    totals = {}
    for name, start in (('07_issue011.tif', 30), ('08_issue012.tif', 200), ('10_blob_residency.tif', 400)):
        stack = tifffile.imread(folder / name)
        start = max(0, min(start, len(stack) - a.frames))
        labels = np.ascontiguousarray(stack[start:start + a.frames])
        raw = rng.normal(100, 10, labels.shape).astype('float32')
        lag = rng.normal(1, 2, labels.shape).astype('float32')
        rows = []
        while len(rows) < a.events:
            t = int(rng.integers(8, a.frames - 4))
            boxes = ndi.find_objects(labels[t])
            present = [k + 1 for k, b in enumerate(boxes) if b is not None]
            prior = int(rng.choice(present))
            py, px = [(s.start + s.stop) / 2 for s in boxes[prior - 1]]
            near = [k for k in present if k != prior
                    and abs((boxes[k - 1][0].start + boxes[k - 1][0].stop) / 2 - py) < 40
                    and abs((boxes[k - 1][1].start + boxes[k - 1][1].stop) / 2 - px) < 40]
            if near:
                rows.append(dict(action='dormancy_started', source_imagej_frame=t + 1,
                                 prior_identity=prior, host_identity=int(rng.choice(near))))
        actions = pd.DataFrame(rows)
        base = dict(minimum_component_area_px=8, minimum_takeover_established_area_px=10 ** 9,
                    enumeration_minimum_host_loss_ratio=0, minimum_candidate_area_ratio=0,
                    maximum_candidate_area_ratio=10 ** 6)
        for radius in (None, 0, 15, 'wide'):
            cfg = dict(base) if radius is None else dict(base, enumeration_maximum_radius_px=radius)
            tick = time.perf_counter()
            try:
                expected = ('ok', original_detect(labels, raw, lag, actions, 0, cfg))
            except Exception as error:
                expected = ('error', type(error).__name__, str(error))
            middle = time.perf_counter()
            with reuse.install(module) as stats:
                assert module.detect.__wrapped__ is original_detect
                try:
                    actual = ('ok', module.detect(labels, raw, lag, actions, 0, cfg))
                except Exception as error:
                    actual = ('error', type(error).__name__, str(error))
                timings['adapted'] += time.perf_counter() - middle
            timings['original'] += middle - tick
            assert module.detect is original_detect
            assert expected[0] == actual[0], (name, radius, expected[:3], actual[:3])
            if expected[0] == 'ok':
                for old, new in zip(expected[1], actual[1]):
                    pd.testing.assert_frame_equal(old, new, check_exact=True)
                    assert old.to_csv(index=False) == new.to_csv(index=False)
                report['events'] += len(expected[1][0])
                report['releases'] += len(expected[1][1])
                if len(expected[1][1]):
                    report['persistence_values'] |= set(map(int, expected[1][1].persistence_frames))
            else:
                assert expected == actual
            for key, value in stats.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    totals[key] = totals.get(key, 0) + value
            report['cases'] += 1
        print(name, report, totals, flush=True)
    assert totals['far_skips'] > 0 and totals['far_measured'] > totals['far_skips']
    assert report['releases'] > 0 and len(report['persistence_values']) > 1
    report['persistence_values'] = sorted(report['persistence_values'])
    report.update(seconds=timings, stats=totals)
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
