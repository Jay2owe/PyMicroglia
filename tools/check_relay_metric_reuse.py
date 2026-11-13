"""Compare relay-pair graph candidates with reused union masks, exactly.

Two accepted label movies (after Issue 005 and after the Issue 017 rebase), the
run's lag movie and its lineage-graph settings, plus a variant allowing longer
overlaps (more pairs reach the metrics), scored by graph_lineage_candidates with
the run's other relay adapters installed, then also with relay_metric_reuse;
plain and disk-backed working-array movies. Tables must be equal exactly.
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
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import oscillatory_reconciliation as osc
    import relay_transition_geometry, relay_pair_reuse, relay_metric_reuse
    import array_storage
    graph = json.loads(CONFIG.read_text())['accepted_postprocessing']['oscillatory_lineage']['lineage_graph']
    settings = [graph, dict(graph, maximum_overlap_frames=9)]
    lag = tifffile.imread(HISTORY / '20_continuity_evidence_inputs/out/lag.tif')
    failures, cases, rows, distances = [], 0, 0, 0
    seconds = dict(original=0.0, plain=0.0, working=0.0)
    with ExitStack() as base:
        assert base.enter_context(relay_transition_geometry.install(osc))['supported']
        assert base.enter_context(relay_pair_reuse.install(osc, relay_transition_geometry))['supported']
        for name in ('01_issue005.tif', '11_issue017_rebase.tif'):
            labels = tifffile.imread(HISTORY / name)
            for number, params in enumerate(settings):
                t0 = time.perf_counter()
                expected = osc.graph_lineage_candidates(labels, lag, params)
                seconds['original'] += time.perf_counter() - t0
                rows += len(expected)
                for storage in ('plain', 'working'):
                    with ExitStack() as scope:
                        data = labels
                        if storage == 'working':
                            pool = array_storage.Pool(a.scratch / f'{name}_{number}', minimum_bytes=1 << 10,
                                                      temporary_ram_bytes=32 << 20)
                            scope.enter_context(array_storage.install(pool))
                            data = np.array(labels)
                        stats = scope.enter_context(relay_metric_reuse.install(osc))
                        assert stats['supported']
                        t1 = time.perf_counter()
                        actual = osc.graph_lineage_candidates(data, lag, params)
                        seconds[storage] += time.perf_counter() - t1
                        distances += stats['pair_distances']
                        cases += 1
                        try:
                            pd.testing.assert_frame_equal(expected, actual, check_exact=True)
                        except AssertionError as error:
                            failures.append(dict(movie=name, setting=number, storage=storage,
                                                 error=str(error)[:300]))
    report = dict(status='pass' if not failures and distances else 'fail', cases=cases,
                  candidate_rows=rows, pair_distances=distances, seconds=seconds, failures=failures)
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
