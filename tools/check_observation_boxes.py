"""Compare boxed observation building with the original, exactly.

The recorded stage-m2 inputs (registered raw, lag, neutral tracks) with the
configured observation settings, plus small synthetic movies that exercise
fast pairs, single lobes, one-sided green tracks, shared tracks and parameter
extremes. Labels, the observation table and both lobe stacks must match; the
real labels and table must also equal the saved stage-m2 outputs.
"""
import argparse
import json
from pathlib import Path
import sys
import time


def outcome(function, *args):
    import numpy as np
    try:
        labels, table, gained, lost = function(*args)
    except Exception as error:
        return ('error', type(error).__name__, str(error))
    import hashlib
    digest = lambda value: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    return ('ok', np.asarray(labels).dtype.str, digest(labels), table, digest(gained), digest(lost))


def compare(e, f):
    import pandas as pd
    assert e[0] == f[0], (e[:3], f[:3])
    if e[0] == 'error':
        assert e == f
        return
    assert e[1] == f[1] and e[2] == f[2] and e[4] == f[4] and e[5] == f[5]
    pd.testing.assert_frame_equal(e[3], f[3], check_exact=True)


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'config', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--stem', default='95_full_A1')
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--synthetic', type=int, default=120)
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    import tifffile
    sys.path[:0] = [str(a.adapter.resolve()), str((a.engine / 'code').resolve())]
    import common, pipeline
    import observations as module
    import observation_boxes as boxes
    cfg = common.Config.load(a.config)
    params = cfg.values['observation']
    sources = pipeline._pinned_paths(cfg, a.stem)
    raw = tifffile.imread(sources['registered_raw'])
    lag = tifffile.imread(sources['lag_float'])
    neutral = tifffile.imread(sources['neutral_tracks'])
    rng = np.random.default_rng(2)
    cases = [(raw, lag, neutral, dict(params))]
    for number in range(a.synthetic):
        frames, h, w = int(rng.integers(2, 6)), int(rng.integers(12, 40)), int(rng.integers(12, 40))
        movie = (rng.random((frames, h, w)) * 200 * (rng.random((frames, h, w)) < 0.7)).astype(np.uint16)
        motion = rng.normal(0, 2.5, (frames - 1, h, w)).astype(np.float32)
        if number % 9 == 0:
            motion[0, 0, :3] = np.nan
        tracks = np.zeros((frames - 1, h, w), [np.uint16, np.int32, np.uint8][number % 3])
        for t in range(frames - 1):
            for track in range(1, int(rng.integers(1, 7))):
                y, x = int(rng.integers(0, h - 4)), int(rng.integers(0, w - 4))
                tracks[t, y:y + int(rng.integers(2, 6)), x:x + int(rng.integers(2, 6))] = track + int(rng.integers(0, 3))
        settings = dict(params)
        settings.update(min_component_px=int(rng.choice([1, 3, int(params['min_component_px'])])),
                        min_lobe_px=int(rng.choice([1, 2, int(params['min_lobe_px'])])),
                        lobe_attach_px=float(rng.choice([0.0, 2.0, float(params['lobe_attach_px'])])),
                        fast_pair_px=float(rng.choice([0.0, 5.0, float(params['fast_pair_px'])])),
                        green_overlap_merge_fraction=float(rng.choice([0.0, 0.3, 1.0])))
        cases.append((movie, motion, tracks, settings))
    cases.append((raw[:3], lag[:2], np.full((2,) + raw.shape[1:], -1, np.int32), dict(params)))
    original = module.build_observations
    tick = time.perf_counter(); expected = [outcome(original, *case) for case in cases]; before = time.perf_counter() - tick
    with boxes.install(module, pipeline) as stats:
        assert stats['supported'] and pipeline.build_observations is not original
        tick = time.perf_counter(); actual = [outcome(module.build_observations, *case) for case in cases]; after = time.perf_counter() - tick
    assert pipeline.build_observations is original
    for e, f in zip(expected, actual):
        compare(e, f)
    stage = a.engine / 'm2_anchor' / a.run / 'out'
    saved = tifffile.imread(stage / f'{a.stem}_observations.tif')
    import hashlib
    assert saved.dtype.str == expected[0][1] and hashlib.sha256(saved.tobytes()).hexdigest() == expected[0][2]
    pd.testing.assert_frame_equal(pd.read_csv(stage / 'observations.csv'),
                                  pd.read_csv(__import__('io').StringIO(expected[0][3].to_csv(index=False))),
                                  check_exact=True)
    report = dict(status='pass', cases=len(cases), real_rows=len(expected[0][3]),
                  synthetic_rows=sum(len(e[3]) for e in expected[1:] if e[0] == 'ok'),
                  errors=sum(e[0] == 'error' for e in expected),
                  before_seconds=before, after_seconds=after, counters=stats,
                  saved_stage_outputs_equal=True)
    print(report)
    a.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
