"""Compare prefiltered component markers (and the clamped marker search) with the originals."""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--stack', default='61_well_b2_issue005_bracketed_mixed_owner_flash.tif')
    p.add_argument('--components', type=int, default=1500)
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    import tifffile
    from scipy import ndimage as ndi
    sys.path[:0] = [str(a.adapter.resolve()), str((a.engine / 'code').resolve())]
    import bracketed_mixed_owner_flash as module
    import bounded_owner_excursion as bounded
    from array_bounds import occupied_bounds
    import component_marker_prefilter as prefilter
    import bounded_marker_geometry as geometry
    reference = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    stack = tifffile.imread(reference / a.stack)
    points = pd.read_csv(reference / '26_latent_body_tracks/out/latent_track_points.csv')
    rng = np.random.default_rng(9)
    points['physically_visible'] = points.state.isin(['observed', 'latent_visible'])
    points['candidate_owner'] = rng.integers(0, 40, len(points))
    # Edge rows: outside the image, NaN radius, huge radius.
    extra = points.sample(300, random_state=1).copy()
    extra['x'] += rng.choice([-30, 30, 600], len(extra)); extra['radius_px'] = rng.choice([np.nan, 0.5, 80.0], len(extra))
    points = pd.concat([points, extra], ignore_index=True)
    by_frame = {int(f): g for f, g in points.groupby(points.frame.astype(int))}
    cases = []
    while len(cases) < a.components:
        t = int(rng.integers(0, len(stack)))
        ids = np.unique(stack[t]); ids = ids[ids > 0]
        if not len(ids) or t not in by_frame:
            continue
        component = stack[t] == int(rng.choice(ids))
        frame_points = by_frame[t]
        target = frame_points.iloc[int(rng.integers(0, len(frame_points)))]
        cases.append((component, frame_points, int(target.track_id),
                      (int(rng.integers(0, 471)), int(rng.integers(0, 471))),
                      None if rng.random() < 0.5 else int(rng.integers(0, 40))))
    cases.append((np.zeros((471, 471), bool), by_frame[0], -1, None, None))
    original_marker = bounded._marker
    tick = time.perf_counter(); expected = [module._component_markers(*c) for c in cases]; before = time.perf_counter() - tick
    with geometry.install(bounded) as gstats, prefilter.install(module, occupied_bounds) as stats:
        assert gstats['supported'] and stats['supported']
        tick = time.perf_counter(); actual = [module._component_markers(*c) for c in cases]; after = time.perf_counter() - tick
        # The clamped marker itself, against the frozen marker, on every row of a few cases.
        rows = [(c[0], r) for c in cases[:200] for r in c[1].itertuples(index=False)]
        assert [bounded._marker(m, r) for m, r in rows] == [original_marker(m, r) for m, r in rows]
    assert expected == actual
    report = dict(status='pass', cases=len(cases), nonempty=sum(bool(e) for e in expected),
                  marker_rows=len(rows), before_seconds=before, after_seconds=after, counters=stats)
    print(report)
    a.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
