"""Compare stable dual-soma reference areas and complete correct() outputs, exactly.

The correction engine's own recover_stable_dual_soma_contacts runs on a real
accepted label movie, raw movie and tracker points (first frames only). Every
track's reference area from the original _reference_area must equal the
box-labelled memo's, also with points moved up to four pixels off their
components; then the complete correct() result (labels and tables) with the
adapter installed must equal the original's.
"""
import argparse
import json
from pathlib import Path
import sys
import time


def same(left, right):
    import numpy as np
    import pandas as pd
    if isinstance(left, pd.DataFrame):
        pd.testing.assert_frame_equal(left, right, check_exact=True)
        return True
    if isinstance(left, np.ndarray):
        return left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(map(same, left, right))
    return left == right


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'code', 'config', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--stem', default='95_full_A1')
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--frames', type=int, default=400)
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
    labels = tifffile.imread(folder / '99_well_b3_issue013_anchored_projection_owner_relay.tif')[:a.frames]
    points = pd.read_csv(folder / '26_latent_body_tracks/out/latent_track_points.csv', low_memory=False)
    points = points[points.frame < a.frames].reset_index(drop=True)
    sys.path[:0] = [str(a.code.resolve()), str(a.adapter.resolve())]
    import recover_stable_dual_soma_contacts as module
    import reference_area_boxes as adapter
    stats = dict(calls=0, labelled=0, reused=0, fallback=0)
    scored = module.physical.attach_owners(points, labels)
    for field in ('strong', 'physically_visible', 'in_encounter'):
        scored[field] = scored[field].astype(bool)
    anchors = module._references(scored, module.Params())
    moved = scored.copy()
    rng = np.random.default_rng(3)
    moved['x'] = moved['x'] + rng.uniform(-4, 4, len(moved))
    moved['y'] = moved['y'] + rng.uniform(-4, 4, len(moved))
    timings = dict(original=0.0, adapted=0.0)
    compared = 0
    nonzero = 0
    for table in (scored, moved):
        groups = {int(t): table[table.track_id.astype(int).eq(int(t))] for t in anchors}
        tick = time.perf_counter()
        expected = {t: module._reference_area(labels, groups[t], int(anchor['owner']), anchor)
                    for t, anchor in anchors.items()}
        middle = time.perf_counter()
        area = adapter.reference_area(adapter.owner_parts(module, labels, stats), module._reference_area, stats)
        actual = {t: area(labels, groups[t], int(anchor['owner']), anchor) for t, anchor in anchors.items()}
        timings['original'] += middle - tick
        timings['adapted'] += time.perf_counter() - middle
        assert expected == actual and all(type(expected[t]) is type(actual[t]) for t in expected)
        compared += len(expected)
        nonzero += sum(v > 0 for v in expected.values())
    tick = time.perf_counter()
    expected = module.correct(labels, raw, points)
    middle = time.perf_counter()
    with adapter.install(module, stats):
        assert module.correct.__wrapped__
        actual = module.correct(labels, raw, points)
    timings['correct_original'] = middle - tick
    timings['correct_adapted'] = time.perf_counter() - middle
    assert stats['calls'] == 1 and stats['fallback'] == 0
    assert same(list(expected), list(actual))
    report = dict(status='pass', frames=a.frames, anchors=len(anchors), areas_compared=compared,
                  nonzero_areas=nonzero, correct_outputs=len(expected), seconds=timings, stats=stats)
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
