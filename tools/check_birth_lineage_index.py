"""Compare restore_measured_birth_lineage with its one-pass rewrite, exactly.

First native frames from the count tables must equal the per-name movie scans
on real accepted movies. The complete function (all option combinations) runs
on real accepted labels, raw movie and tracker points, with fresh names
substituted part way along real cells so name births occur; the returned movie
and audit table must be identical.
"""
import argparse
import importlib.util
import itertools
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
    p.add_argument('--frames', type=int, default=80)
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    import tifffile
    sys.path[:0] = [str((a.engine / 'code').resolve()), str(a.adapter.resolve())]
    import common
    import pipeline
    import birth_lineage_index as adapter
    folder = a.frozen / 'current/r203_a003'
    sys.path.insert(0, str(folder.resolve()))
    spec = importlib.util.spec_from_file_location('measured_birth_lineage', folder / 'measured_birth_lineage.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules['measured_birth_lineage'] = module
    spec.loader.exec_module(module)
    cfg = common.Config.load(a.config)
    history = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    raw = tifffile.imread(pipeline._pinned_paths(cfg, a.stem)['registered_raw'])[:a.frames]
    observed = tifffile.imread(history / '99_well_b3_issue013_anchored_projection_owner_relay.tif')[:a.frames]
    points = pd.read_csv(history / '26_latent_body_tracks/out/latent_track_points.csv', low_memory=False)
    points = points[points.frame < a.frames].reset_index(drop=True)
    stats = {key: 0 for key in ('versions', 'tables', 'fallback')}
    first = adapter.first_frames(stats)
    for movie in (observed, tifffile.imread(history / '08_issue012.tif')[:a.frames]):
        table = first(movie)
        for name in np.unique(movie)[1:]:
            assert table[int(name)] == int(np.flatnonzero(np.any(movie == name, axis=(1, 2)))[0])
    rng = np.random.default_rng(203)
    proposals = observed.copy()
    fresh = int(observed.max()) + 1
    names = np.unique(observed)[1:]
    for name in rng.choice(names, min(len(names), max(1, len(names) * 2 // 3)), replace=False):
        frames = np.flatnonzero(np.any(observed == name, axis=(1, 2)))
        if len(frames) < 6:
            continue
        cut = int(frames[len(frames) // 2])
        later = proposals[cut:]
        later[later == name] = fresh
        fresh += 1
    current = proposals.copy()
    original, rewritten = adapter.adapt(module, stats)
    report = dict(status='pass', frames=a.frames, option_sets=0, audit_rows=0, seconds=dict(original=0.0, adapted=0.0))
    for flags in itertools.product((False, True), repeat=3):
        options = dict(zip(('established_birth_only', 'core_supported_roots', 'require_core_turnover'), flags))
        # Renamed in the measured movie too, so tracker points see a measured name birth.
        for measured, labels in ((proposals, current), (observed, current), (observed, observed)):
            t0 = time.perf_counter()
            expected = original(measured, proposals, raw, points, current_labels=labels, original_labels=observed, **options)
            t1 = time.perf_counter()
            actual = rewritten(measured, proposals, raw, points, current_labels=labels, original_labels=observed, **options)
            t2 = time.perf_counter()
            report['seconds']['original'] += t1 - t0
            report['seconds']['adapted'] += t2 - t1
            assert expected[0].dtype == actual[0].dtype and np.array_equal(expected[0], actual[0]), options
            pd.testing.assert_frame_equal(expected[1], actual[1], check_exact=True)
            report['option_sets'] += 1
            report['audit_rows'] += len(expected[1])
            report['changed_pixels'] = report.get('changed_pixels', 0) + int(np.count_nonzero(expected[0] != labels))
    assert stats['fallback'] == 0
    report['stats'] = stats
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
