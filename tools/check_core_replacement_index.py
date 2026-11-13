"""Compare core-replacement ownership with indexed owner nodes and collected point nodes, exactly.

The frozen R208-A005 preserve_core_replacement_owners runs on real accepted
label movies (current and an earlier stage as the original), the raw movie and
tracker points, and again rewritten by core_replacement_index; labels and the
audit table must be identical. A copy of the points with a repeated index
label keeps the per-row writes and must also match.

With --repair <core_replacement_evidence.py>, the runtime repair's patched code
is first swapped into the frozen function as the repair does; the adapter must
still engage and match the repaired function.
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
    p.add_argument('--repair', type=Path, help='core_replacement_evidence.py to swap in before adapting')
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
    # The adapter is handed the function first; a runtime repair may swap its code afterwards.
    stats = dict(adapted=0, indexes=0, lookups=0)
    rewritten = adapter.adapt(module.preserve_core_replacement_owners, stats)
    assert rewritten is not module.preserve_core_replacement_owners
    assert rewritten.__wrapped__ is module.preserve_core_replacement_owners
    if a.repair:
        from importlib import util as import_util
        spec = import_util.spec_from_file_location('core_replacement_evidence_repair', a.repair.resolve())
        repair = import_util.module_from_spec(spec)
        sys.modules[spec.name] = repair
        spec.loader.exec_module(repair)
        target = module.preserve_core_replacement_owners
        _, patched = repair.repaired_function(Path(module.__file__).read_text(encoding='utf8'))
        namespace = {}
        exec(compile(patched, str(Path(repair.__file__)), 'exec'), target.__globals__, namespace)
        target.__code__ = namespace['preserve_core_replacement_owners'].__code__
    report = dict(cases=[])
    repeated = points.copy()
    repeated.index = list(range(len(repeated) - 1)) + [0]
    # Point rows come from column lists; these tables vary the column types
    # and order the loop reads (a nullable radius keeps iterrows).
    float_frames = points.assign(frame=points.frame.astype(float), radius_px=points.radius_px.astype(np.float32))
    nullable = points.assign(radius_px=pd.array(points.radius_px, dtype='Float64'))
    shuffled = points.sample(frac=1, random_state=5)
    for name, table, observed in (('current', points, current), ('observed-earlier', points, original),
                                  ('repeated-index', repeated, current), ('float-columns', float_frames, current),
                                  ('nullable-radius', nullable, current), ('shuffled', shuffled, current),
                                  ('reversed', points.iloc[::-1], current)):
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
    assert stats['adapted'] == 1, stats  # one rewrite per running code object
    report.update(status='pass', frames=a.frames, stats=stats, repaired=bool(a.repair))
    assert stats['lookups'] > 0 and stats.get('column_rows', 0) > 0 and stats.get('iterrows', 0) > 0
    assert stats.get('previous_indexed', 0) > 0, stats
    # Recorded tables skip the per-track sorts; shuffled and reversed ones sort.
    assert stats.get('groups_in_order', 0) > 0 and stats.get('groups_sorted', 0) > 0, stats
    assert stats.get('disc_points', 0) > 0, stats
    # The disc search alone, against the original statements, on random node
    # windows with ties, empty and edge windows, integer and float points.
    def original_disc(nodes, f, y0, y1, x0, x1, py, px, radius, n):
        field = nodes[f, y0:y1, x0:x1]; yy, xx = np.indices(field.shape)
        distance = (yy+y0-py)**2+(xx+x0-px)**2
        hit = (field > 0) & (distance <= radius**2)
        if hit.any():
            minimum = distance[hit].min(); choices = np.unique(field[hit & (distance == minimum)])
            if len(choices) == 1: n = int(choices[0])
        return n
    nearest = adapter.nearest_node({})
    rng = np.random.default_rng(29)
    discs = 0
    for trial in range(20000):
        nodes = (rng.random((2, 14, 16)) < (.02, .1, .4)[trial % 3]) * rng.integers(1, 4, (2, 14, 16))
        nodes = nodes.astype(np.int32)
        f = int(rng.integers(0, 2))
        if trial % 2:
            py, px, radius = float(rng.uniform(-2, 16)), float(rng.uniform(-2, 18)), float(rng.uniform(0, 5))
        else:
            py, px, radius = int(rng.integers(0, 14)), int(rng.integers(0, 16)), float(rng.integers(0, 4))
        y0 = max(0, int(np.floor(py-radius))); y1 = min(14, int(np.ceil(py+radius))+1)
        x0 = max(0, int(np.floor(px-radius))); x1 = min(16, int(np.ceil(px+radius))+1)
        expected = original_disc(nodes, f, y0, y1, x0, x1, py, px, radius, 0)
        actual = nearest(nodes[f, y0:y1, x0:x1], y0, x0, py, px, radius, 0)
        assert expected == actual and type(expected) is type(actual), (trial, expected, actual)
        discs += expected != 0
    report['disc_cases'] = dict(cases=20000, found=discs)
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
