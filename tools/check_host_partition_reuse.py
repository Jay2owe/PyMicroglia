"""Check exact identity-presence ordering and private frame-only retention."""
import argparse
import json
from pathlib import Path
import platform
import sys
from types import SimpleNamespace

platform.platform = lambda *a, **k: 'Windows-host-partition-check'
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'report'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.engine.resolve()/'code'))
    sys.path.insert(0, str(a.adapter.resolve()))
    import oscillatory_reconciliation as module
    import host_partition_reuse as reuse
    rng = np.random.default_rng(419)
    for shape in ((0, 3, 5), (3, 0, 5), (1, 3, 5), (12, 37, 43)):
        values = rng.choice(np.array([-3000000, 0, 1, 2, 7000000]), size=shape)
        for data in (values, values.transpose(0, 2, 1)):
            before, after = module._frames(data), reuse.frame_index(data)
            assert list(before) == list(after)
            for identity in before:
                assert np.array_equal(before[identity], after[identity])
                assert before[identity].dtype == after[identity].dtype
    labels = np.zeros((8, 23, 29), np.uint16)
    labels[:, 2:5, 2:5] = 1
    labels[:, 2:5, 6:9] = 2
    labels[3, 2:5, 2:5] = 2
    raw = np.ones(labels.shape, np.float32)
    original_split = module._split_interval
    calls = []
    def split(data, raw, lag, t, stop, identity, host, cfg):
        result = data.copy()
        result[t, 2:5, 2:5] = identity
        calls.append(t)
        return result, [dict(t=t, imagej_frame=t+1, area_a_px=9, area_b_px=9,
                            a_motion_support=1, b_motion_support=1,
                            a_previous_overlap=1, b_previous_overlap=1,
                            a_identity_shape_change=0, b_identity_shape_change=0)]
    module._split_interval = split
    cfg = dict(minimum_identity_frames=1, minimum_identity_area_px=1,
               maximum_host_distance_px=10)
    alias = SimpleNamespace(optimise_host_merge_graph=module.optimise_host_merge_graph)
    original_candidates = module.host_merge_graph_candidates
    try:
        expected = module.optimise_host_merge_graph(labels, raw, raw, cfg, {})
        assert calls and not expected[1].empty
        with reuse.install(module, alias) as counters:
            assert counters['supported']
            actual = alias.optimise_host_merge_graph(labels, raw, raw, cfg, {})
            assert module.host_merge_graph_candidates is original_candidates
            _, public = module.host_merge_graph_candidates(labels, raw, raw, cfg, {})
            assert all(value.shape == labels.shape for value in public.values())
        assert alias.optimise_host_merge_graph is module.optimise_host_merge_graph
        assert np.array_equal(expected[0], actual[0])
        for before, after in zip(expected[1:], actual[1:]):
            pd.testing.assert_frame_equal(before, after, check_exact=True)
        assert counters['avoided_movie_bytes'] == (len(labels)-1)*counters['retained_frame_bytes']
    finally:
        module._split_interval = original_split
    report = dict(status='pass', presence_cases=8, graph_candidates=len(actual[2]), counters=counters,
                  scope='Exact ordered presence including sparse labels and transposes; selected partitions and all graph evidence; unchanged public candidate API')
    a.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
