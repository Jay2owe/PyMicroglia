"""Compare constrain_destination_soma with its boxed-support rewrite, exactly.

Each verified frozen version runs on real accepted label frames and the
registered raw movie, with recovered blobs added beside, inside and away from
measured cells (forward and backward origins), real tracker points, and every
measured-path, source-overlap and footprint mode the version accepts. The
returned movie and audit table must be identical.
"""
import argparse
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import time


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'frozen', 'config', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--stem', default='95_full_A1')
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--frames', type=int, default=60)
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    import tifffile
    sys.path[:0] = [str((a.engine / 'code').resolve()), str(a.adapter.resolve())]
    import common
    import pipeline
    import destination_support_boxes as adapter
    cfg = common.Config.load(a.config)
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    raw = tifffile.imread(pipeline._pinned_paths(cfg, a.stem)['registered_raw'])[:a.frames]
    observed = tifffile.imread(folder / '99_well_b3_issue013_anchored_projection_owner_relay.tif')[:a.frames]
    points = pd.read_csv(folder / '26_latent_body_tracks/out/latent_track_points.csv', low_memory=False)
    points = points[points.frame < a.frames].reset_index(drop=True)
    rng = np.random.default_rng(19)
    proposals = observed.copy()
    rows = []
    used = set()
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
        observed_frames = np.flatnonzero(np.any(observed == name, axis=(1, 2)))
        direction = 'forward' if rng.random() < .5 else 'backward'
        used.add((name, t + 1))
        rows.append(dict(identity=name, imagej_frame=t + 1, direction=direction, outcome='recovered'))
        if len(observed_frames) and rng.random() < .3:
            # Remove the observation so the origin search uses neighbouring frames.
            pass
    history = pd.DataFrame(rows)
    versions = {}
    for path in sorted(a.frozen.glob('*/*/recovery_destination_soma.py')):
        digest = __import__('hashlib').sha256(path.read_bytes()).hexdigest()
        if digest in adapter.VERSIONS and digest not in versions:
            versions[digest] = path
    assert len(versions) == len(adapter.VERSIONS), sorted(versions)
    report = dict(status='pass', frames=a.frames, proposals=len(history), versions={}, seconds=dict(original=0.0, adapted=0.0))
    for number, (digest, path) in enumerate(sorted(versions.items())):
        module = load(path, f'destination_check_{number}')
        stats = {key: 0 for key in ('versions', 'frames', 'boxed', 'fallback')}
        original, rewritten = adapter.adapt(module, stats)
        parameters = inspect.signature(original).parameters
        options = [{}]
        for keyword, values in (('measured_path_mode', ('any', 'measured_interior', 'neighbour_interior', 'raw_supported_adjacent')),
                                ('source_overlap_mode', ('any', 'measured_interior', 'raw_supported_interior')),
                                ('tracker_footprint_mode', ('disk', 'centred_raw_component'))):
            if keyword in parameters:
                options = [dict(o, **{keyword: v}) for o in options for v in values]
        compared = 0
        withdrawn = 0
        for option in options:
            try:
                t0 = time.perf_counter()
                expected = original(observed, proposals, raw, points, proposal_history=history, **option)
            except ValueError as error:
                # A mode this version names but rejects must be rejected the same way.
                try:
                    rewritten(observed, proposals, raw, points, proposal_history=history, **option)
                except ValueError as again:
                    assert str(error) == str(again)
                    continue
                raise AssertionError(f'rewrite accepted {option}')
            t1 = time.perf_counter()
            actual = rewritten(observed, proposals, raw, points, proposal_history=history, **option)
            t2 = time.perf_counter()
            report['seconds']['original'] += t1 - t0
            report['seconds']['adapted'] += t2 - t1
            assert expected[0].dtype == actual[0].dtype and np.array_equal(expected[0], actual[0]), (path, option)
            pd.testing.assert_frame_equal(expected[1], actual[1], check_exact=True)
            compared += 1
            withdrawn += len(expected[1])
        assert stats['fallback'] == 0
        report['versions'][digest[:16]] = dict(path=str(path), option_sets=compared, audit_rows=withdrawn, stats=stats)
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
