"""Compare point_disks.footprints with the frozen footprint loops, exactly.

python tools/check_point_disks.py <report.json>

The two loop texts (raw-soma footprint without the centre, destination soma
with it) run as written on random point tables and on edge cases: zero, tiny,
huge and overflowing radii, NaN and infinite values, negative radii, frames out
of range, points beyond every border, a missing radius column, integer
columns and an empty table. Keys (and their order), dtypes and every Boolean
frame must match, or the same error be raised.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'validation/completed_base_20261012/memory/snapshots/identity_candidate'))
import point_disks


def original(selected, raw, centre):
    footprints = {}
    for q in selected.itertuples(index=False):
        t = int(q.frame); x = float(q.x); y = float(q.y)
        radius = float(getattr(q, 'radius_px', 0.))
        if not 0 <= t < len(raw) or not np.isfinite([x, y, radius]).all() or radius < 0:
            continue
        y0=max(0,int(np.floor(y-radius))); y1=min(raw.shape[1],int(np.ceil(y+radius))+1)
        x0=max(0,int(np.floor(x-radius))); x1=min(raw.shape[2],int(np.ceil(x+radius))+1)
        if y0 >= y1 or x0 >= x1:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        coverage = footprints.setdefault(t, np.zeros(raw.shape[1:], bool))
        coverage[y0:y1, x0:x1] |= (yy-y)**2 + (xx-x)**2 <= radius**2
        if centre:
            iy, ix = int(np.floor(y+.5)), int(np.floor(x+.5))
            if 0 <= iy < raw.shape[1] and 0 <= ix < raw.shape[2]:
                coverage[iy, ix] = True
    return footprints


def outcome(function, *args):
    try:
        value = function(*args)
        return ('value', list(value), [(v.dtype.str, v.shape, v.tobytes()) for v in value.values()])
    except Exception as error:
        return ('error', type(error).__name__, str(error))


rng = np.random.default_rng(404)
raw = np.zeros((12, 37, 53), np.uint16)
tables = []
for trial in range(60):
    n = int(rng.integers(0, 400))
    tables.append(pd.DataFrame(dict(frame=rng.integers(-2, 14, n), x=rng.uniform(-8, 61, n), y=rng.uniform(-8, 45, n),
                                    radius_px=rng.choice([0., .3, .5, 1., 2.5, 4., 9.7, 30.], n))))
halves = pd.DataFrame(dict(frame=[1] * 6, x=[2.5, 3.5, 10.0, 0.5, -0.5, 52.5], y=[2.5, 3.5, 7.0, 36.5, 0.5, -0.4],
                           radius_px=[.5, 1.5, 2.0, .5, .5, .5]))
specials = pd.DataFrame(dict(frame=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2],
                             x=[np.nan, 5., np.inf, 5., 1e9, -1e9, 10., 10., 26., 26., 5.],
                             y=[5., np.nan, 5., 5., 5., 5., 10., 10., 18., 18., 5.],
                             radius_px=[2., 2., 2., -1., 3., 3., 0., 1e-9, 1e3, 1e9, np.inf]))
overflow = pd.DataFrame(dict(frame=[3, 3], x=[5., 5.], y=[5., 5.], radius_px=[2., 1e200]))
missing = tables[0].drop(columns='radius_px').head(50)
integers = pd.DataFrame(dict(frame=[1, 1, 2], x=[4, 30, 52], y=[3, 20, 36], radius_px=[2, 5, 1]))
bad_frame = pd.DataFrame(dict(frame=[1.0, np.nan], x=[3., 3.], y=[3., 3.], radius_px=[1., 1.]))
tables += [halves, specials, overflow, missing, integers, bad_frame, tables[0].iloc[:0]]
cases = equal = 0
for table in tables:
    for centre in (False, True):
        cases += 1
        equal += outcome(original, table, raw, centre) == outcome(point_disks.footprints, table, raw, centre)
# Larger than one chunk: every point's box together exceeds 2**20 pixels.
big_raw = np.zeros((2, 300, 300), np.uint8)
big = pd.DataFrame(dict(frame=[0] * 40 + [1] * 3, x=rng.uniform(0, 300, 43), y=rng.uniform(0, 300, 43),
                        radius_px=[90.] * 40 + [400., 0., 1.]))
for centre in (False, True):
    cases += 1
    equal += outcome(original, big, big_raw, centre) == outcome(point_disks.footprints, big, big_raw, centre)
passed = cases == equal
report = dict(status='passed' if passed else 'failed', cases=cases, equal=int(equal))
Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
raise SystemExit(0 if passed else 1)
