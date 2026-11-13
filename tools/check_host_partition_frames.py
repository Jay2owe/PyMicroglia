"""Compare the host-merge graph with real partitions returned as frames, exactly.

Real accepted labels with one-frame gaps made by giving a cell to a touching
neighbour (as check_host_partition_boxes does), the registered raw and lag
frames of the same window, and the run's own host-merge and persistent-merge
settings (plus a looser variant that yields more partitions). The engine's
optimise_host_merge_graph runs unchanged, then with the deferred merge copy and
host-partition reuse installed, where an accepted partition comes back as a
FrameOverlay; plain and disk-backed working-array movies. The returned movie
and both tables must be equal, and overlays must have been used.
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
CONFIG = ROOT / 'development/cd68_full_red_green/rigid_batch/.j/A1/inputs/motion_config.json'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=40)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve()), str(Path(__file__).resolve().parent)]
    import oscillatory_reconciliation as osc
    import persistent_merges as merges
    import merge_proposal_storage
    import host_partition_reuse
    import array_storage
    from check_host_partition_boxes import gapped
    config = json.loads(CONFIG.read_text())['accepted_postprocessing']
    graph = dict(config['oscillatory_lineage']['host_merge_graph'])
    loose = dict(graph, minimum_identity_frames=2, minimum_identity_area_px=1.0, maximum_host_distance_px=40.0,
                 minimum_host_growth_ratio=-5.0, maximum_host_growth_ratio=50.0,
                 minimum_partition_motion_support=0.0, minimum_partition_previous_overlap=0.0,
                 maximum_partition_shape_change=10.0, minimum_partition_area_ratio=0.0,
                 maximum_partition_area_ratio=100.0)
    partition = config['persistent_merges']
    stack = tifffile.imread(HISTORY / '11_issue017_rebase.tif')
    rng = np.random.default_rng(20261022)
    failures, cases, overlays, selected = [], 0, 0, 0
    seconds = dict(original=0.0, adapted=0.0)
    for start in (30, len(stack) // 2):
        frames = range(start, start + a.frames)
        labels = gapped(stack[start:start + a.frames], rng, a.frames)
        raw = tifffile.imread(HISTORY / '17_seat_inputs/out/95_full_A1_registered_raw.tif', key=frames)
        lag = tifffile.imread(HISTORY / '20_continuity_evidence_inputs/out/lag.tif', key=frames)
        for params in (graph, loose):
            t0 = time.perf_counter()
            expected = osc.optimise_host_merge_graph(labels, raw, lag, params, partition)
            t1 = time.perf_counter()
            seconds['original'] += t1 - t0
            for storage in ('plain', 'working'):
                with ExitStack() as stack_scope:
                    data = labels
                    if storage == 'working':
                        pool = array_storage.Pool(a.scratch / f'{start}_{len(params)}',
                                                  minimum_bytes=1 << 10, temporary_ram_bytes=1 << 20)
                        stack_scope.enter_context(array_storage.install(pool))
                        data = np.array(labels)
                    counter = stack_scope.enter_context(merge_proposal_storage.install(merges, osc))
                    reuse = stack_scope.enter_context(host_partition_reuse.install(osc))
                    assert counter['supported'] and reuse['supported']
                    t2 = time.perf_counter()
                    actual = osc.optimise_host_merge_graph(data, raw, lag, params, partition)
                    seconds['adapted'] += time.perf_counter() - t2
                    overlays += counter['frame_overlays']
                    cases += 1
                    try:
                        assert np.array_equal(np.asarray(expected[0]), np.asarray(actual[0]))
                        assert expected[0].dtype == actual[0].dtype
                        for x, y in zip(expected[1:], actual[1:]):
                            pd.testing.assert_frame_equal(x, y, check_exact=True)
                    except AssertionError as error:
                        failures.append(dict(start=start, storage=storage, loose=params is loose,
                                             error=str(error)[:300]))
            selected += len(expected[1])
    report = dict(status='pass' if not failures and overlays else 'fail', cases=cases,
                  frame_overlays=overlays, selected_partitions=selected, seconds=seconds,
                  failures=failures[:10])
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
