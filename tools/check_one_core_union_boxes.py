"""Compare one-core interaction discovery with the boxed union test, exactly.

The run's own anchor observations, observation table and neutral track movie
with the accepted Issue 017 labels, at the run's source-frame offset and owner
rule plus looser and stricter core minima (so more and fewer pairs reach the
union test), on plain and disk-backed working arrays. The events (with their
union masks), the audit domain movie and the interaction table must be equal.
"""
import argparse, dataclasses, importlib, json, sys, time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'
CONFIG = ROOT / 'development/cd68_full_red_green/rigid_batch/.j/A1/inputs/motion_config.json'


def same_events(expected, actual):
    if len(expected) != len(actual):
        return False
    for x, y in zip(expected, actual):
        for field in dataclasses.fields(x):
            a, b = getattr(x, field.name), getattr(y, field.name)
            if isinstance(a, np.ndarray):
                if not (a.dtype == np.asarray(b).dtype and np.array_equal(a, np.asarray(b))):
                    return False
            elif type(a) is not type(b) or a != b:
                return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import common, pipeline
    import accepted_history as ah
    import physical_lineage_aliases as lineage
    import physical_core_boxes
    import one_core_union_boxes
    import array_storage
    cfg = common.Config.load(CONFIG)
    stem, offset = '95_full_A1', int(cfg.source_frame_offset)
    neutral = ah.load_stack(pipeline._pinned_paths(cfg, stem)['neutral_tracks'])
    observations, table, _ = pipeline._load_anchor_inputs(stem, 'pymicroglia_95_full_A1')
    labels = tifffile.imread(HISTORY / '11_issue017_rebase.tif')
    owner = ah._general('R05_generalise_full_lineage_alias', offset)['owner_rule']
    minima = sorted({int(owner['minimum_union_core_px']), 4, 30})
    failures, cases, seconds, totals = [], 0, dict(original=0.0, plain=0.0, working=0.0), {}
    with physical_core_boxes.install(lineage) as core_stats:
        assert core_stats['supported']
        for minimum in minima:
            t0 = time.perf_counter()
            expected = lineage.discover_one_core_interactions(
                observations, table, neutral, labels, offset, minimum)
            seconds['original'] += time.perf_counter() - t0
            for storage in ('plain', 'working'):
                with ExitStack() as scope:
                    inputs = (observations, neutral, labels)
                    if storage == 'working':
                        pool = array_storage.Pool(a.scratch / f'{minimum}', minimum_bytes=1 << 10,
                                                  temporary_ram_bytes=1 << 20)
                        scope.enter_context(array_storage.install(pool))
                        inputs = tuple(np.array(value) for value in inputs)
                    stats = scope.enter_context(one_core_union_boxes.install(lineage))
                    assert stats['supported']
                    t1 = time.perf_counter()
                    actual = lineage.discover_one_core_interactions(
                        inputs[0], table, inputs[1], inputs[2], offset, minimum)
                    seconds[storage] += time.perf_counter() - t1
                    for key, value in stats.items():
                        if key != 'supported':
                            totals[key] = totals.get(key, 0) + value
                    # Compare while a working-array pool still maps the results.
                    cases += 1
                    try:
                        assert same_events(expected[0], actual[0]), 'events'
                        assert expected[1].dtype == actual[1].dtype and np.array_equal(
                            expected[1], np.asarray(actual[1])), 'domains'
                        pd.testing.assert_frame_equal(expected[2], actual[2], check_exact=True)
                    except AssertionError as error:
                        failures.append(dict(minimum=minimum, storage=storage, error=str(error)[:300]))
            totals.setdefault('events', {})[minimum] = len(expected[0])
    report = dict(status='pass' if not failures and totals.get('apart') else 'fail', cases=cases,
                  offset=offset, minima=minima, counters=totals, seconds=seconds, failures=failures)
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
