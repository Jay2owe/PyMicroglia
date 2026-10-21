"""Compare the recording-start duplicate summary with frame-local component excess, exactly.

Per-frame terms are compared with the original count on synthetic frames
(empty, one identity, split identities, touching corners, borders, very large
and negative identities, every integer width) and on every frame of real
stacks. Complete summaries are compared on the real Issue 046 input and output
and on synthetic stack pairs that differ in some frames.
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
    p.add_argument('--stem', default='95_full_A1')
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    import tifffile
    sys.path[:0] = [str(a.adapter.resolve()), str((a.engine / 'code').resolve())]
    import recording_start_duplicate_release as module
    import start_duplicate_excess as adapter
    rng = np.random.default_rng(46)
    terms = 0
    for case in range(600):
        dtype = [np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64][case % 6]
        frame = np.zeros((int(rng.integers(1, 40)), int(rng.integers(1, 40))), dtype)
        top = np.iinfo(dtype).max
        identities = [1, 2, 5, min(top, 9_000_001), min(top, 1 << 25)]
        if np.iinfo(dtype).min < 0 and case % 7 == 0:
            identities.append(-3)
        for _ in range(int(rng.integers(0, 12))):
            y, x = rng.integers(0, frame.shape[0]), rng.integers(0, frame.shape[1])
            frame[y:y + int(rng.integers(1, 4)), x:x + int(rng.integers(1, 4))] = rng.choice(identities)
        expected = module.excess_components(frame[None])
        actual = adapter.frame_excess(frame, module.STRUCTURE)
        assert actual is None or actual == expected, (case, actual, expected)
        if actual is None:
            assert frame.min() < 0
        terms += 1
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    labels = tifffile.imread(folder / '27_issue045_atomic_two_seat_recovery.tif')
    candidate = tifffile.imread(folder / '28_issue046_recording_start_duplicate_release.tif')
    unclaimed = tifffile.imread(folder / '27_issue045_atomic_two_seat_recovery_unclaimed.tif')
    raw = tifffile.imread(folder / '07_trimmed_evidence/out' / f'{a.stem}_registered_raw.tif')
    for t in range(len(labels)):
        assert adapter.frame_excess(labels[t], module.STRUCTURE) == module.excess_components(labels[t:t + 1])
        terms += 1
    audit = pd.DataFrame(dict(eligible=[True, True, False], applied=[True, False, False]))
    params = dict(example=1)
    pairs = [('issue046', labels, candidate, unclaimed, raw)]
    for case in range(12):
        small = np.zeros((6, 30, 30), np.int32)
        for t in range(6):
            for identity in (1, 2, 3):
                y, x = rng.integers(0, 26, 2)
                small[t, y:y + 4, x:x + 4] = identity
        trial = small.copy()
        for t in rng.choice(6, int(rng.integers(0, 4)), replace=False):
            y, x = rng.integers(0, 28, 2)
            trial[t, y:y + 2, x:x + 2] = rng.choice([0, 1, 2, 4])
        pairs.append((f'synthetic-{case}', small, trial, np.zeros_like(small), rng.normal(0, 1, small.shape)))
    summaries = []
    for name, before, after, spare, signal in pairs:
        tick = time.perf_counter()
        expected = module.summarize(before, after, spare, signal, audit, params)
        middle = time.perf_counter()
        with adapter.install(module) as stats:
            assert stats['supported']
            actual = module.summarize(before, after, spare, signal, audit, params)
        assert expected == actual, name
        summaries.append(dict(name=name, frames=stats['frames'], original_seconds=round(middle - tick, 2),
                              adapter_seconds=round(time.perf_counter() - middle, 2),
                              new_components=actual['new_same_frame_identity_components']))
    report = dict(status='pass', frame_terms=terms, summaries=summaries)
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report)[:600])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
