"""Compare boxed Issue 006 label observations with the original, exactly.

python check_tenure_observation_boxes.py --engine <engine> --adapter <identity_candidate dir> --output <report.json>

Every frame of a real A1 label stack with no, random and dilated-anchor
excluded masks, synthetic frames of several dtypes, and frames that must take
the original route (negative values, float frames, integer excluded masks):
each row must hold the same keys in the same order, equal masks (dtype and
shape), bit-equal positions and the same area and identity.
"""
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
    a = p.parse_args()
    import numpy as np
    import tifffile
    from scipy import ndimage as ndi
    sys.path[:0] = [str(a.adapter.resolve()), str((a.engine / 'code/corrections/contact_tenure').resolve()),
                    str((a.engine / 'code').resolve())]
    import apply_tenure as module
    import tenure_observation_boxes as adapter
    original = module._label_observations
    rows_compared = calls = 0
    timing = [0.0, 0.0]

    def compare(frame, excluded):
        nonlocal rows_compared, calls
        try:
            tick = time.perf_counter()
            want = ('ok', original(frame, excluded))
            timing[0] += time.perf_counter() - tick
        except Exception as error:
            want = ('error', (type(error).__name__, str(error)))
        try:
            tick = time.perf_counter()
            got = ('ok', module._label_observations(frame, excluded))
            timing[1] += time.perf_counter() - tick
        except Exception as error:
            got = ('error', (type(error).__name__, str(error)))
        calls += 1
        assert want[0] == got[0], (want, got)
        if want[0] == 'error':
            assert want[1] == got[1], (want, got)
            return
        assert len(want[1]) == len(got[1]), (len(want[1]), len(got[1]))
        for left, right in zip(want[1], got[1]):
            assert list(left) == list(right)
            assert type(left['mask']) is type(right['mask']) and left['mask'].dtype == right['mask'].dtype
            assert left['mask'].shape == right['mask'].shape and np.array_equal(left['mask'], right['mask'])
            assert left['position'].dtype == right['position'].dtype
            assert left['position'].tobytes() == right['position'].tobytes()
            assert left['area'] == right['area'] and type(left['area']) is type(right['area'])
            assert left['input_identity'] == right['input_identity'] and type(left['input_identity']) is type(right['input_identity'])
            rows_compared += 1

    rng = np.random.default_rng(5)
    with adapter.install(module) as stats:
        assert stats['supported']
        for case in range(200):
            dtype = [np.uint16, np.int32, np.int64, np.uint8][case % 4]
            frame = np.zeros((int(rng.integers(4, 40)), int(rng.integers(4, 40))), dtype)
            for _ in range(int(rng.integers(0, 12))):
                y, x = rng.integers(0, frame.shape[0]), rng.integers(0, frame.shape[1])
                frame[y:y + int(rng.integers(1, 7)), x:x + int(rng.integers(1, 7))] = rng.choice([1, 2, 7, 30])
            compare(frame, np.zeros(frame.shape, bool))
            compare(frame, rng.random(frame.shape) < .05)
        odd = frame.astype(np.int64); odd[0, 0] = -2
        compare(odd, np.zeros(odd.shape, bool))
        compare(frame.astype(float), np.zeros(frame.shape, bool))
        compare(frame, np.zeros(frame.shape, np.uint8))
        compare(frame, np.zeros((2, 2), bool))
        compare(np.zeros((5, 5), np.uint16), np.zeros((5, 5), bool))
        folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
        labels = tifffile.imread(folder / '52_well_a5_issue003_complete_flip_recovery.tif')
        for t in range(len(labels)):
            frame = labels[t]
            compare(frame, np.zeros(frame.shape, bool))
            if t % 4 == 0:
                anchor = ndi.binary_dilation(frame == frame.max(), iterations=3) if frame.max() else np.zeros(frame.shape, bool)
                compare(frame, anchor)
                compare(frame, rng.random(frame.shape) < .002)
    assert module._label_observations is original
    report = dict(status='pass', calls=calls, rows=rows_compared, original_seconds=round(timing[0], 2),
                  adapter_seconds=round(timing[1], 2), counters=stats)
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
