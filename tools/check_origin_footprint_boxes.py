"""Compare certify_recovery_origins and certify_raw_soma_footprints with their boxed rewrites, exactly.

Real accepted label frames and the registered raw movie, with recovered blobs
added beside, inside and away from measured cells (forward and backward
origins) and real tracker points. The returned movie and audit table must be
identical for both frozen functions, including a repeated call (fresh boxes
per call) and a call on a changed movie.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    sys.path.remove(str(path.parent))
    return module


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
    import origin_footprint_boxes as adapter
    cfg = common.Config.load(a.config)
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    raw = tifffile.imread(pipeline._pinned_paths(cfg, a.stem)['registered_raw'])[:a.frames]
    observed = tifffile.imread(folder / '99_well_b3_issue013_anchored_projection_owner_relay.tif')[:a.frames]
    points = pd.read_csv(folder / '26_latent_body_tracks/out/latent_track_points.csv', low_memory=False)
    points = points[points.frame < a.frames].reset_index(drop=True)
    rng = np.random.default_rng(182)
    proposals = observed.copy()
    rows, used = [], set()
    shape = observed.shape[1:]
    for _ in range(900):
        t = int(rng.integers(1, a.frames - 1))
        names = np.unique(observed[t])[1:]
        name = int(rng.choice(names))
        if (name, t + 1) in used:
            continue
        ys, xs = np.nonzero(observed[t] == name)
        offset = rng.choice([0, 3, 8, 20])
        cy = int(np.clip(ys.mean() + rng.integers(-offset - 1, offset + 2), 2, shape[0] - 8))
        cx = int(np.clip(xs.mean() + rng.integers(-offset - 1, offset + 2), 2, shape[1] - 8))
        size = int(rng.integers(1, 7))
        block = (slice(cy, cy + size), slice(cx, cx + size))
        free = proposals[t][block] == 0
        if not free.any():
            continue
        proposals[t][block][free] = name
        used.add((name, t + 1))
        rows.append(dict(identity=name, imagej_frame=t + 1,
                         direction='forward' if rng.random() < .5 else 'backward', outcome='recovered'))
    history = pd.DataFrame(rows)
    changed = observed.copy()
    changed[5:9][changed[5:9] == int(np.unique(observed[6])[3])] = 0
    report = dict(status='pass', frames=a.frames, proposals=len(history), functions={})
    for digest, (function, _) in adapter.VERSIONS.items():
        matches = [path for path in sorted(a.frozen.glob('*/*/*.py'))
                   if __import__('hashlib').sha256(path.read_bytes()).hexdigest() == digest]
        assert matches, function
        module = load(matches[0], 'origin_check_' + function)
        stats = {key: 0 for key in ('versions', 'frames', 'boxed', 'fallback')}
        original, rewritten = adapter.adapt(module, stats)
        seconds = dict(original=0.0, adapted=0.0)
        rows_checked = 0
        for movie in (observed, observed, changed):
            additive = np.where(movie > 0, movie, proposals)
            t0 = time.perf_counter()
            expected = original(movie, additive, raw, points, proposal_history=history)
            t1 = time.perf_counter()
            actual = rewritten(movie, additive, raw, points, proposal_history=history)
            t2 = time.perf_counter()
            seconds['original'] += t1 - t0
            seconds['adapted'] += t2 - t1
            assert expected[0].dtype == actual[0].dtype and np.array_equal(expected[0], actual[0]), function
            pd.testing.assert_frame_equal(expected[1], actual[1], check_exact=True)
            rows_checked += len(expected[1])
        assert stats['fallback'] == 0 and stats['boxed'] > 0, stats
        report['functions'][function] = dict(path=str(matches[0]), audit_rows=rows_checked, stats=stats, seconds=seconds)
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
