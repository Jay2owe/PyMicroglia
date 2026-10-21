"""Compare ephemeral speckle alias discovery with per-frame presence and remembered empty frames, exactly.

The real Issue 013 input runs the original discovery and the adapter, with the
configured settings and with permissive alias limits (so more identities reach
the distance search); the audit, internal rows (including component arrays)
and attached points must be identical. A synthetic movie with frames holding
no durable owner exercises the remembered empty answer.
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
        if not (left.columns.equals(right.columns) and left.index.equals(right.index)
                and left.dtypes.equals(right.dtypes)):
            return False
        for column in left.columns:
            if left[column].dtype == object:
                # Cells can hold lists of dicts with mask arrays.
                if not all(map(same, left[column].tolist(), right[column].tolist())):
                    return False
            else:
                pd.testing.assert_series_equal(left[column], right[column], check_exact=True)
        return True
    if isinstance(left, np.ndarray):
        return type(left) is type(right) and left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return list(left) == list(right) and all(same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(map(same, left, right))
    return type(left) is type(right) and (left == right or (left != left and right != right))


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'config', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    import tifffile
    sys.path[:0] = [str(a.adapter.resolve()), str((a.engine / 'code').resolve())]
    import common
    import ephemeral_speckle_alias_retirement as module
    import ephemeral_speckle_alias_retirement_integration as integration
    import speckle_alias_presence as adapter
    cfg = common.Config.load(a.config)
    _, params = integration._configured(cfg.values)
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    labels = tifffile.imread(folder / '99_well_b3_issue013_anchored_projection_owner_relay.tif')
    points = pd.read_csv(folder / '26_latent_body_tracks/out/latent_track_points.csv')
    loose = dict(params, maximum_alias_span_movie_fraction=1.0, maximum_alias_physical_support_movie_fraction=1.0)
    # Synthetic: frames 0-2 hold only short-lived identities, so they have no durable owner.
    small = np.zeros((40, 30, 30), np.uint16)
    small[3:, 5:12, 5:12] = 1
    small[:, 20:22, 20:22] = 5
    small[:3, 2:4, 25:27] = 6
    small[10, 14:15, 14:16] = 7
    table = pd.DataFrame(dict(track_id=1, frame=np.arange(3, 40), x=8.0, y=8.0, radius_px=4.0,
                              state='observed', strong=True, in_encounter=False,
                              evidence=1.0, visibility=1.0, nearest_track_distance_px=20.0))
    cases = [('real', labels, points, params), ('real-loose', labels, points, loose),
             ('synthetic', small, table, dict(params, minimum_durable_support_movie_fraction=0.5,
                                              maximum_alias_span_movie_fraction=1.0,
                                              maximum_alias_physical_support_movie_fraction=1.0))]
    report = dict(cases=[])
    for name, stack, rows, settings in cases:
        tick = time.perf_counter()
        expected = module.discover(stack, rows, settings)
        middle = time.perf_counter()
        with adapter.install(module) as stats:
            assert stats['supported']
            actual = module.discover(stack, rows, settings)
        seconds = time.perf_counter() - middle
        assert same(expected, actual), name
        report['cases'].append(dict(name=name, aliases=len(actual[0]), eligible=int(actual[0].eligible.astype(bool).sum()) if len(actual[0]) else 0,
                                    original_seconds=round(middle - tick, 1), adapter_seconds=round(seconds, 1), stats=dict(stats)))
    report['status'] = 'pass'
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
