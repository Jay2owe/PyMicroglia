"""Compare core-replacement ownership with indexed owner nodes and collected point nodes, exactly.

The frozen R208-A005 preserve_core_replacement_owners runs on real accepted
label movies (current and an earlier stage as the original), the raw movie and
tracker points, and again rewritten by core_replacement_index; labels and the
audit table must be identical. A copy of the points with a repeated index
label keeps the per-row writes and must also match.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'frozen', 'config', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--stem', default='95_full_A1')
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--frames', type=int, default=300)
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    import tifffile
    sys.path[:0] = [str((a.engine / 'code').resolve())]
    import common
    import pipeline
    cfg = common.Config.load(a.config)
    raw = tifffile.imread(pipeline._pinned_paths(cfg, a.stem)['registered_raw'])[:a.frames]
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    current = tifffile.imread(folder / '99_well_b3_issue013_anchored_projection_owner_relay.tif')[:a.frames]
    original = tifffile.imread(folder / '44_well_issue001_isolated_pair_encounter_recovery.tif')[:a.frames]
    points = pd.read_csv(folder / '26_latent_body_tracks/out/latent_track_points.csv', low_memory=False)
    points = points[points.frame < a.frames].reset_index(drop=True)
    sys.path[:0] = [str(a.frozen.resolve()), str(a.adapter.resolve())]
    module = importlib.import_module('core_replacement_ownership')
    import core_replacement_index as adapter
    stats = dict(adapted=0, indexes=0, lookups=0)
    rewritten = adapter.adapt(module.preserve_core_replacement_owners, stats)
    assert rewritten is not module.preserve_core_replacement_owners and stats['adapted'] == 1
    report = dict(cases=[])
    repeated = points.copy()
    repeated.index = list(range(len(repeated) - 1)) + [0]
    for name, table, observed in (('current', points, current), ('observed-earlier', points, original),
                                  ('repeated-index', repeated, current)):
        arguments = dict(proposal_history=None, current_labels=current, original_labels=original)
        tick = time.perf_counter()
        expected = module.preserve_core_replacement_owners(observed, observed, raw, table, None, **arguments)
        middle = time.perf_counter()
        actual = rewritten(observed, observed, raw, table, None, **arguments)
        seconds = time.perf_counter() - middle
        assert expected[0].dtype == actual[0].dtype and np.array_equal(expected[0], actual[0]), name
        pd.testing.assert_frame_equal(expected[1], actual[1], check_exact=True)
        report['cases'].append(dict(name=name, audit_rows=len(actual[1]),
                                    changed_pixels=int((expected[0] != current).sum()),
                                    original_seconds=round(middle - tick, 1), adapted_seconds=round(seconds, 1)))
    report.update(status='pass', frames=a.frames, stats=stats)
    assert stats['lookups'] > 0
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
