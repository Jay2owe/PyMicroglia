"""Compare host-merge graph candidates with the boxed frame-table version.

Real accepted labels, with one-frame gaps made by giving a cell's pixels to a
touching neighbour (a grown host) or to background; the partition step is
replaced by a deterministic stand-in so every host that passes the distance
and growth tests yields a row. Candidate tables (every column, exact) and the
retained partition frames must be equal; optimise_host_merge_graph through the
installed adapter must equal the engine's on the same stand-in.
"""
import argparse, inspect, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'


def gapped(labels, rng, gaps):
    labels = labels.copy()
    made = 0
    for _ in range(gaps * 20):
        if made >= gaps:
            break
        t = int(rng.integers(1, len(labels) - 1))
        ids = np.unique(labels[t])[1:]
        if not len(ids):
            continue
        identity = int(rng.choice(ids))
        mask = labels[t] == identity
        ring = ndi.binary_dilation(mask, iterations=2) & ~mask
        neighbours = np.unique(labels[t][ring])
        neighbours = neighbours[neighbours != 0]
        labels[t][mask] = int(rng.choice(neighbours)) if len(neighbours) and rng.random() < .8 else 0
        made += 1
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=60)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import oscillatory_reconciliation as module
    import host_partition_reuse as adapter

    def split(labels, raw, lag, t, stop, identity, host, cfg):
        key = (t * 7919 + identity * 104729 + host * 1299709) % 1000
        if key < 100:
            return None
        return labels, [dict(t=t, imagej_frame=t + 1, area_a_px=key % 97 + 20, area_b_px=key % 89 + 30,
                             a_motion_support=key / 1000, b_motion_support=(key * 3 % 1000) / 1000,
                             a_previous_overlap=(key * 7 % 1000) / 1000, b_previous_overlap=.9,
                             a_identity_shape_change=(key * 11 % 1000) / 4000, b_identity_shape_change=.1)]

    source = inspect.getsource(module.host_merge_graph_candidates)
    assert all(source.count(old) == 1 for old, _ in adapter.BOXED)
    boxed_source = source
    for old, new in adapter.BOXED:
        boxed_source = boxed_source.replace(old, new)
    original_split = module._split_interval
    module._split_interval = split
    namespace = dict(module.host_merge_graph_candidates.__globals__)
    namespace['_frame_tables'] = adapter.boxed_frame_tables
    exec(compile(boxed_source, '<boxed-check>', 'exec'), namespace)
    boxed = namespace['host_merge_graph_candidates']
    rng = np.random.default_rng(20261020)
    stack = tifffile.imread(HISTORY / '11_issue017_rebase.tif')
    settings = [dict(), dict(minimum_identity_frames=2, minimum_identity_area_px=1.0,
                             maximum_host_distance_px=40.0, minimum_host_growth_ratio=-5.0,
                             maximum_host_growth_ratio=50.0)]
    failures, rows, seconds = [], 0, dict(original=0.0, boxed=0.0)
    try:
        for start in (0, len(stack) // 2):
            labels = gapped(stack[start:start + a.frames], rng, a.frames)
            for data in (labels, labels.astype(np.int32), np.ascontiguousarray(labels[:, ::-1])):
                for cfg in settings:
                    t0 = time.perf_counter(); before = module.host_merge_graph_candidates(data, data, data, cfg, {})
                    t1 = time.perf_counter(); after = boxed(data, data, data, cfg, {})
                    t2 = time.perf_counter()
                    seconds['original'] += t1 - t0; seconds['boxed'] += t2 - t1
                    rows += len(before[0])
                    try:
                        pd.testing.assert_frame_equal(before[0], after[0], check_exact=True)
                        assert list(before[1]) == list(after[1])
                        assert all(before[1][k] is after[1][k] for k in before[1])
                    except AssertionError as error:
                        failures.append(dict(start=start, dtype=str(data.dtype), cfg=cfg, error=str(error)[:300]))
                    with adapter.install(module) as stats:
                        installed = module.optimise_host_merge_graph(data, data, data, cfg, {})
                    expected = module.optimise_host_merge_graph(data, data, data, cfg, {})
                    try:
                        assert stats.get('boxed') == 1
                        assert np.array_equal(expected[0], installed[0]) and expected[0].dtype == installed[0].dtype
                        for x, y in zip(expected[1:], installed[1:]):
                            pd.testing.assert_frame_equal(x, y, check_exact=True)
                    except AssertionError as error:
                        failures.append(dict(start=start, dtype=str(data.dtype), cfg=cfg, optimise=str(error)[:300]))
    finally:
        module._split_interval = original_split
    report = dict(status='pass' if not failures and rows else 'fail', candidate_rows=rows,
                  seconds=seconds, failures=failures[:10])
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
