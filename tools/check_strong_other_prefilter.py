"""Compare prefiltered strong other somata with the frozen function, exactly.

python tools/check_strong_other_prefilter.py <engine code folder> <labels.tif> <points.csv> <report.json>

Real track points on the real movie: each frame's strong visible points
against components of real names (and of their dilations, so neighbours fall
inside), then edge tables (non-finite and huge coordinates, integer x and y,
float frames, unsigned tracks, float and integer components, points outside
the image, empty tables, duplicate and missing columns, object strong flags,
prepared rows, numpy and bool arguments). Every call must give the same rows
(field names, values and value types, order) or the same error.
"""
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import tifffile

code, labels_path, points_path, report_path = map(Path, sys.argv[1:5])
here = Path(__file__).resolve().parents[1] / 'validation/completed_base_20261012/memory/snapshots/identity_candidate'
sys.path[:0] = [str(code.resolve()), str(here)]
module = importlib.import_module('complete_stationary_soma_path')
import strong_other_prefilter

held = [module._strong_other_somata]
stats = dict(versions=0, prefiltered=0, empty=0, fallback=0)
seconds = dict(before=0., after=0.)


def outcome(function, key, *args, **named):
    tick = time.perf_counter()
    try:
        value = function(*args, **named)
    except Exception as error:
        value = (type(error), str(error))
    seconds[key] += time.perf_counter() - tick
    return value


def same(a, b):
    if isinstance(a, tuple):
        return a == b
    return (type(a) is type(b) and len(a) == len(b) and all(
        type(x).__name__ == type(y).__name__ and x._fields == y._fields and tuple(x) == tuple(y) and [type(v) for v in x] == [type(v) for v in y]
        for x, y in zip(a, b)))


cases = equal = 0
with strong_other_prefilter.install(stats):
    adapted = module._strong_other_somata
    installed = adapted is not held[0] and adapted.__wrapped__ is held[0]

    def compare(scored, frame, track, point, component, **named):
        global cases, equal
        cases += 1
        equal += same(outcome(held[0], 'before', scored, frame, track, point, component, **named),
                      outcome(adapted, 'after', scored, frame, track, point, component, **named))

    rng = np.random.default_rng(1172)
    labels = tifffile.imread(labels_path)
    points = pd.read_csv(points_path)
    points['physically_visible'] = points['state'].isin(['observed', 'latent_visible'])
    rows = list(points[points.strong.astype(bool)].itertuples(index=False))
    for k in rng.choice(len(rows), 2000, replace=False):
        point = rows[int(k)]
        t = int(point.frame)
        name = int(labels[t, min(max(int(round(point.y)), 0), labels.shape[1] - 1),
                          min(max(int(round(point.x)), 0), labels.shape[2] - 1)])
        component = labels[t] == name if name else labels[t] > 0
        if k % 2:
            component = ndi.binary_dilation(component, iterations=12)
        compare(points, t, int(point.track_id), point, component)
    real = dict(seconds)

    point = rows[0]
    t = int(point.frame)
    frame_rows = points.index[points.frame == t]
    wide = np.ones(labels.shape[1:], bool)
    edits = []
    for column, value in (('y', np.nan), ('x', np.inf), ('y', 1e300), ('x', -3.0), ('y', 10_000.0), ('x', 2.5), ('y', 3.5)):
        table = points.copy()
        table.loc[frame_rows[1], column] = value
        edits.append(table)
    integer = points.copy()
    integer['x'] = integer['x'].round().astype(np.int64)
    integer['y'] = integer['y'].round().astype(np.int32)
    edits.append(integer)
    floating = points.copy()
    floating['frame'] = floating['frame'].astype(float)
    edits.append(floating)
    unsigned = points.copy()
    unsigned['track_id'] = unsigned['track_id'].astype(np.uint32)
    edits.append(unsigned)
    flags = points.copy()
    flags['strong'] = flags['strong'].astype(float)
    flags['physically_visible'] = flags['physically_visible'].astype(np.int8)
    edits.append(flags)
    objects = points.copy()
    objects['strong'] = objects['strong'].astype(object)
    edits.append(objects)
    duplicate = pd.concat([points, points[['x']]], axis=1)
    edits.append(duplicate)
    edits.append(points.drop(columns=['physically_visible']))
    edits.append(points.iloc[:0])
    shuffled = points.sample(frac=1, random_state=3)
    edits.append(shuffled)
    dated = points.copy()
    dated['when'] = pd.Timestamp('2026-10-23') + pd.to_timedelta(dated['frame'], unit='h')
    dated['kind'] = dated['state'].astype('category')
    edits.append(dated)
    odd_names = points.copy()
    odd_names[7] = 1.5
    odd_names['class'] = 'x'
    edits.append(odd_names)
    for table in edits:
        for component in (wide, wide.astype(np.uint8), np.where(wide, np.nan, 0.), labels[t] == 0, labels[t].astype(np.int32)):
            compare(table, t, int(point.track_id), point, component)
    compare(points, np.int64(t), np.int64(point.track_id), point, wide)
    compare(points, True, int(point.track_id), point, wide)
    compare(points, t, float(point.track_id), point, wide)
    compare(points, t, int(point.track_id), point, wide[None])
    compare(points, t, int(point.track_id), point, [[True]])
    compare(points, t, -1, point, wide)
    compare(points, t + 10_000, int(point.track_id), point, wide)
    compare(points, t, int(point.track_id), point, wide, prepared_rows=module.prepare_strong_somata_by_frame(points))
restored = module._strong_other_somata is held[0]
checks = dict(installed=bool(installed and restored), cases=cases, equal=int(equal), stats=stats,
              real_movie_seconds={k: round(v, 3) for k, v in real.items()})
passed = checks['installed'] and cases == equal and stats['versions'] == 1 and stats['prefiltered'] > stats['empty'] > 0 and stats['fallback'] > 0
report = dict(status='passed' if passed else 'failed', checks=checks)
report_path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
raise SystemExit(0 if passed else 1)
