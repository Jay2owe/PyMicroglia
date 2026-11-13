"""Compare the owner-window and motion inventories with and without plain lookups.

python tools/check_process_inventory_lookups.py <correction engine code folder> <report.json>
Imports the frozen process_inventory_owners and process_inventory_motion and
runs both modules' scan on synthetic wells (labelled blocks that move, points
with visible, latent and dormant states, duplicated (track, frame) points,
float and integer frame columns, missing track ids in pairs and in points,
tracks absent from the points) before and after install. The tables must be
equal, with the same columns, dtypes and row order.
"""
import contextlib
import importlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

code, report_path = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(code))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'validation/completed_base_20261012/memory/snapshots/identity_candidate'))
owners = importlib.import_module('process_inventory_owners')
motion = importlib.import_module('process_inventory_motion')
import process_inventory_lookups

rng = np.random.default_rng(9)


def well(variant):
    frames, size, names = 30, 72, 12
    labels = np.zeros((frames, size, size), np.int32)
    centres = rng.integers(8, size - 8, (names, 2))
    rows = []
    for t in range(frames):
        for name in range(1, names + 1):
            cy, cx = centres[name - 1] + rng.integers(-2, 3, 2)
            centres[name - 1] = np.clip((cy, cx), 6, size - 7)
            half = 3 + name % 3
            labels[t, cy - half:cy + half, cx - half:cx + half] = name
            if name % 4 == 0:
                labels[t, cy + half:cy + half + 2, cx:cx + 2] = name + 20  # small fragment
            state = rng.choice(['observed', 'latent_visible', 'dormant'], p=[.6, .25, .15])
            rows.append(dict(track_id=name, frame=t, y=float(cy) + rng.random() - .5,
                             x=float(cx) + rng.random() - .5, radius_px=2.0 + half / 2, state=state))
            if name % 4 == 0:
                rows.append(dict(track_id=name + 100, frame=t, y=float(cy + half), x=float(cx),
                                 radius_px=1.5, state='observed'))
    points = pd.DataFrame(rows)
    if variant == 'duplicates':
        points = pd.concat([points, points.sample(40, random_state=4)], ignore_index=True)
    elif variant == 'float_frames':
        points['frame'] = points.frame.astype(float)
    elif variant == 'missing_track':
        points.loc[points.sample(10, random_state=5).index, 'track_id'] = np.nan
    pairs = []
    for t in range(1, frames):
        for name in (4, 8, 12):
            pairs.append(dict(frame=t + 1, large_track=name, small_track=name + 100, small_identity=name + 20,
                              large_identity=name, large_occupancy_lag_5=rng.random() if t % 3 else np.nan,
                              accepted_large_area=float(rng.integers(30, 80)), fragment_elongation=rng.random()))
    pairs.append(dict(frame=5, large_track=np.nan, small_track=104, small_identity=24, large_identity=4,
                      large_occupancy_lag_5=0.1, accepted_large_area=40.0, fragment_elongation=.5))
    pairs.append(dict(frame=5, large_track=4, small_track=999, small_identity=24, large_identity=4,
                      large_occupancy_lag_5=0.1, accepted_large_area=40.0, fragment_elongation=.5))
    pairs.append(dict(frame=5, large_track=4, small_track=np.inf, small_identity=24, large_identity=4,
                      large_occupancy_lag_5=0.1, accepted_large_area=40.0, fragment_elongation=.5))
    return labels, points, pd.DataFrame(pairs)


def run(scan, movies, source):
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            return scan(movies, source)
        except Exception as error:
            return type(error)


def same(a, b):
    if isinstance(a, type) or isinstance(b, type):
        return a is b
    return list(a.columns) == list(b.columns) and list(a.dtypes) == list(b.dtypes) and a.equals(b)


checks = {}
stats = {}
for variant in ('plain', 'duplicates', 'float_frames', 'missing_track'):
    movies, sources = {}, []
    for name in ('A1', 'B2'):
        labels, points, pairs = well(variant)
        movies[name] = dict(branch_labels=labels, embedded_labels=labels, points=points)
        pairs['well'] = name
        sources.append(pairs)
    source = pd.concat(sources, ignore_index=True)
    motion_source = source[np.isfinite(source.small_track.fillna(0))]
    expected = (run(owners.scan, movies, source), run(motion.scan, movies, motion_source))
    with process_inventory_lookups.install(owners, motion, stats):
        got = (run(owners.scan, movies, source), run(motion.scan, movies, motion_source))
    checks[variant] = dict(owners=bool(same(expected[0], got[0])), motion=bool(same(expected[1], got[1])),
                           owner_rows=0 if isinstance(expected[0], type) else len(expected[0]),
                           motion_rows=0 if isinstance(expected[1], type) else len(expected[1]))
restored = owners.switches.__module__ == owners.__name__ and not hasattr(owners.switches, '__wrapped__') \
    and not hasattr(motion.screen_well, '__wrapped__')
passed = (restored and all(c['owners'] and c['motion'] for c in checks.values())
          and stats['owner_windows'] > 0 and stats['motion_rows'] > 0 and stats['motion_fallback'] > 0
          and all(c['owner_rows'] > 0 for c in checks.values())
          # Repeated or missing point keys give no motion rows in the original either.
          and checks['plain']['motion_rows'] > 0 and checks['float_frames']['motion_rows'] > 0)
report = dict(status='passed' if passed else 'failed', restored=restored, checks=checks, stats=stats)
report_path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
raise SystemExit(0 if passed else 1)
