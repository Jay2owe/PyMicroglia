"""Compare boxed merge-bridge identity components and far-pair skipping with the original, exactly.

The recorded Issue 008 merge-bridge input (global-quarantine labels and lag)
runs detect_events with and without the adapter, at the accepted entry gap and
at wider gaps; every event field, mask and float must match. identity_objects
is also compared on random real frames, absent and present identities and
synthetic frames, and boxed pair distances with mask_distance on real pairs.
"""
import argparse
import json
from pathlib import Path
import sys
import time


def same(a, b):
    import numpy as np
    if type(a) is not type(b) and not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
        return False
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    if isinstance(a, dict):
        return list(a) == list(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return a == b or (a != a and b != b)
    return a == b


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--stem', default='95_full_A1')
    a = p.parse_args()
    import numpy as np
    import tifffile
    sys.path[:0] = [str(a.adapter.resolve()), str((a.engine / 'code').resolve())]
    import accepted_history as history
    import merge_bridge_objects as adapter
    module = history._issue_module(history.I008, 'm8_general_merge_bridge')
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    labels = tifffile.imread(folder / '08_global_quarantine/out' / f'{a.stem}.tif')
    lag = tifffile.imread(folder / '07_trimmed_evidence/out' / f'{a.stem}_lag01_float.tif')
    cfg = history._params('issue008', 'general_merge_bridge')['merge']
    rng = np.random.default_rng(8)
    frames = []
    for _ in range(300):
        t = int(rng.integers(len(labels)))
        ids = np.unique(labels[t]); ids = ids[ids > 0]
        choice = int(rng.integers(4))
        identity = None if choice == 0 else (int(rng.choice(ids)) if choice < 3 else int(rng.integers(1, 600)))
        frames.append((labels[t], identity))
    for dtype in (np.uint8, np.int32, np.int64):
        synthetic = rng.integers(0, 6, (37, 41)).astype(dtype)
        synthetic[5:9, 5:9] = 0
        frames += [(synthetic, None), (synthetic, 3), (synthetic, np.int64(2)), (np.zeros((9, 9), dtype), None)]
    settings = [dict(cfg), dict(cfg, maximum_entry_gap_px=float(cfg['maximum_entry_gap_px']) * 3),
                dict(cfg, maximum_entry_gap_px=0.0)]
    original_objects = module.identity_objects
    tick = time.perf_counter()
    expected_objects = [original_objects(f, i) for f, i in frames]
    expected_events = [module.detect_events(labels, lag, s, 7) for s in settings]
    before = time.perf_counter() - tick
    with adapter.install(module) as stats:
        assert stats['supported']
        tick = time.perf_counter()
        actual_objects = [module.identity_objects(f, i) for f, i in frames]
        actual_events = [module.detect_events(labels, lag, s, 7) for s in settings]
        after = time.perf_counter() - tick
    assert module.identity_objects is original_objects
    for number, (e, f) in enumerate(zip(expected_objects, actual_objects)):
        assert same(e, f), number
    for e, f in zip(expected_events, actual_events):
        assert same(e, f)
    # Boxed pair distances against mask_distance on real components, near and far.
    distances = 0
    for t in rng.choice(len(labels), 40, replace=False):
        rows, boxes = adapter.frame_objects(module, labels[int(t)], dict(frames=0, fallback=0))
        for _ in range(60):
            i, j = rng.integers(len(rows), size=2)
            expected = module.mask_distance(rows[i]['mask'], rows[j]['mask'])
            actual = adapter.boxed_distance(module, rows[i]['mask'], rows[j]['mask'], boxes[i], boxes[j])
            assert type(expected) is type(actual) and expected == actual, (t, i, j, expected, actual)
            distances += 1
    report = dict(status='pass', identity_object_calls=len(frames), boxed_distances=distances,
                  components=sum(len(e) for e in expected_objects),
                  events=[len(e) for e in expected_events],
                  before_seconds=before, after_seconds=after, counters=stats)
    print(report)
    a.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
