"""Compare lazy relay orientation and reused transition points with the original, exactly.

Random frame sets exercise every orientation branch. Real accepted-history
stacks with the recorded lag run the complete relay and lineage-graph
candidate tables with no adapters, with the existing transition adapter, and
with this adapter on top; all tables must be identical.
"""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'config', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--stem', default='95_full_A1')
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    a = p.parse_args()
    import numpy as np
    import tifffile
    sys.path[:0] = [str(a.adapter.resolve()), str((a.engine / 'code').resolve())]
    import common, pipeline
    import oscillatory_reconciliation as module
    import relay_transition_geometry, relay_pair_reuse
    rng = np.random.default_rng(41)
    orientations = 0
    for _ in range(20000):
        sets = {}
        for identity in (1, 2):
            n = int(rng.integers(0, 30))
            sets[identity] = np.unique(rng.integers(0, int(rng.integers(1, 60)), n)).astype(np.intp)
        minimum = int(rng.integers(0, 5))
        assert module._relay_orientation(1, 2, sets, minimum) == relay_pair_reuse.orientation(1, 2, sets, minimum)
        orientations += 1
    cfg = common.Config.load(a.config)
    lag = tifffile.imread(pipeline._pinned_paths(cfg, a.stem)['lag_float'])
    params = cfg.values['accepted_postprocessing']['oscillatory_lineage']
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    report = dict(orientation_cases=orientations, stacks=[], seconds={})

    def tables(labels):
        relay = module.overlapping_relay_candidates(labels, lag, params.get('relay', {}), params['minimum_substantial_core_px']) \
            if 'relay' in params else None
        graph = module.graph_lineage_candidates(labels, lag, params['lineage_graph'], params['minimum_substantial_core_px'])
        return (None if relay is None else relay.to_csv(index=False)), graph.to_csv(index=False)
    for name in ('01_issue005.tif', '10_blob_residency.tif'):
        labels = tifffile.imread(folder / name)
        tick = time.perf_counter(); plain = tables(labels); plain_s = time.perf_counter() - tick
        with relay_transition_geometry.install(module) as first:
            assert first['supported']
            tick = time.perf_counter(); existing = tables(labels); existing_s = time.perf_counter() - tick
            with relay_pair_reuse.install(module, relay_transition_geometry) as stats:
                assert stats['supported']
                tick = time.perf_counter(); actual = tables(labels); actual_s = time.perf_counter() - tick
        assert plain == existing == actual, name
        report['stacks'].append(dict(stack=name, graph_rows=plain[1].count('\n') - 1, counters=dict(stats)))
        report['seconds'][name] = dict(original=plain_s, existing_adapter=existing_s, with_reuse=actual_s)
    report['status'] = 'pass'
    print(report)
    a.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
