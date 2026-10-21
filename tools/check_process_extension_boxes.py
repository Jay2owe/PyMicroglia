"""Compare extend_approved_process_paths.correct with its boxed-gap rewrite, exactly.

Real accepted label frames and raw movie. Seeds pair a large cell with each of
its nearest neighbouring names (the neighbour's components are the candidate
branches), the seed being a piece of the large cell; edge limits of 0, a
fraction, the default and larger, with loose overlap thresholds so that many
components are measured and applied. The returned labels and the evidence,
audit and application tables must be identical.
"""
import argparse
import dataclasses
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'code', 'config', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--stem', default='95_full_A1')
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--frames', type=int, default=80)
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    import tifffile
    from scipy import ndimage as ndi
    sys.path[:0] = [str((a.engine / 'code').resolve())]
    import common
    import pipeline
    cfg = common.Config.load(a.config)
    raw = tifffile.imread(pipeline._pinned_paths(cfg, a.stem)['registered_raw'])[:a.frames]
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    labels = tifffile.imread(folder / '99_well_b3_issue013_anchored_projection_owner_relay.tif')[:a.frames]
    sys.path[:0] = [str(a.code.resolve()), str(a.adapter.resolve())]
    import extend_approved_process_paths as module
    import process_extension_boxes as adapter
    rng = np.random.default_rng(65)
    seeds = []
    for t in rng.choice(a.frames, 25, replace=False):
        frame = labels[int(t)]
        names, sizes = np.unique(frame, return_counts=True)
        large = int(names[1:][np.argmax(sizes[1:])]) if rng.random() < .3 else int(rng.choice(names[1:]))
        cell = frame == large
        ring = ndi.binary_dilation(cell, iterations=12) & ~cell & (frame > 0)
        neighbours = [int(v) for v in np.unique(frame[ring]) if v != large]
        if not neighbours:
            continue
        ys, xs = np.nonzero(cell)
        keep = np.zeros_like(cell)
        pick = rng.choice(len(ys), max(1, len(ys) // 4), replace=False)
        keep[ys[pick], xs[pick]] = True
        for old in neighbours[:3]:
            seeds.append(dict(frame=int(t), large_name=large, old_name=old, mask=keep.copy()))
    protected = np.zeros(labels.shape, bool)
    protected[rng.random(labels.shape) < .0005] = True
    stats = dict(versions=0, boxed=0, beyond=0)
    original, rewritten = adapter.adapt(module, stats)
    report = dict(status='pass', seeds=len(seeds), runs=0, evidence_rows=0, applications=0,
                  seconds=dict(original=0.0, adapted=0.0))
    for limit in (0.0, 0.5, 2.4, 10.0, 31.0):
        params = dataclasses.replace(module.Params(), maximum_edge_distance_px=limit,
                                     minimum_exact_seed_overlap=0.0, minimum_near_seed_overlap=0.0,
                                     maximum_raw_q95_ratio=1e9, maximum_component_area=5000,
                                     minimum_component_area=1)
        for current in (seeds, seeds[: len(seeds) // 2]):
            t0 = time.perf_counter()
            try:
                expected = original(labels, raw, protected, current, params)
            except AssertionError as error:
                expected = ('raised', str(error))
            t1 = time.perf_counter()
            try:
                actual = rewritten(labels, raw, protected, current, params)
            except AssertionError as error:
                actual = ('raised', str(error))
            t2 = time.perf_counter()
            report['seconds']['original'] += t1 - t0
            report['seconds']['adapted'] += t2 - t1
            if isinstance(expected[0], str) or isinstance(actual[0], str):
                assert expected == actual, (limit, expected[:2], actual[:2])
            else:
                assert expected[0].dtype == actual[0].dtype and np.array_equal(expected[0], actual[0]), limit
                for left, right in zip(expected[1:], actual[1:]):
                    pd.testing.assert_frame_equal(left, right, check_exact=True)
                report['evidence_rows'] += len(expected[1])
                report['applications'] += len(expected[3])
            report['runs'] += 1
    report['stats'] = stats
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
