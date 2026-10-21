"""Compare seat-identity overlap counts with the original per-component statement.

Real frames of two recorded stacks, labelled into components as seat identity
labels them, against seat masks taken from neighbouring frames' identities and
random masks, plus synthetic frames, empty frames, non-boolean and mismatched
masks (which keep the original statement): every seat's overlap with every
component and the selected seats at several minimum overlaps must match.
The full recompiled step is checked by the real _seat_identity replay.
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
    p.add_argument('--frames', type=int, default=150)
    a = p.parse_args()
    import numpy as np
    import tifffile
    from scipy import ndimage as ndi
    sys.path[:0] = [str(a.adapter.resolve()), str((a.engine / 'code').resolve())]
    import seat_overlap_counts as adapter
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    stacks = [tifffile.imread(folder / name) for name in ('08_issue012.tif', '10_blob_residency.tif')]
    structure = ndi.generate_binary_structure(2, 2)
    rng = np.random.default_rng(13)
    cases = []
    for _ in range(a.frames):
        stack = stacks[int(rng.integers(2))]
        t = int(rng.integers(1, len(stack)))
        numbered, count = ndi.label(stack[t] > 0, structure=structure)
        seats = [{'mask': stack[t - 1] == identity} for identity in np.unique(stack[t - 1])[1:]]
        seats += [{'mask': rng.random(stack[t].shape) < 0.01} for _ in range(3)]
        cases.append((numbered, count, seats))
    for number in range(150):
        shape = (int(rng.integers(1, 25)), int(rng.integers(1, 25)))
        frame = rng.random(shape) < float(rng.choice([0.0, 0.2, 0.6]))
        numbered, count = ndi.label(frame, structure=structure)
        seats = [{'mask': rng.random(shape) < 0.3} for _ in range(int(rng.integers(0, 6)))]
        if number % 10 == 0 and seats:
            seats[0] = {'mask': (rng.random(shape) < 0.3).astype(np.uint8)}
        cases.append((numbered, count, seats))
    stats = dict(counted=0, fallback=0)
    overlaps = adapter.seat_overlaps(stats)
    selections = 0
    tick = time.perf_counter()
    expected = [[[int(np.count_nonzero(seat['mask'] & (numbered == n))) for n in range(1, count + 1)] for seat in seats]
                for numbered, count, seats in cases]
    before = time.perf_counter() - tick
    tick = time.perf_counter()
    actual = [[[int(row[n]) for n in range(1, count + 1)] for row in overlaps(numbered, count, seats)]
              for numbered, count, seats in cases]
    after = time.perf_counter() - tick
    for number, (e, f) in enumerate(zip(expected, actual)):
        assert e == f, number
        for minimum in (0, 1, 4, 30):
            for n in range(len(e[0]) if e else 0):
                assert [s for s, row in enumerate(e) if row[n] >= minimum] == [s for s, row in enumerate(f) if row[n] >= minimum]
                selections += 1
    # Soma cores and crumb shedding on real raw frames and labels, through the
    # history loader, against the module's own functions.
    import hashlib
    import accepted_history as history
    raw = tifffile.imread(folder / '07_trimmed_evidence/out/95_full_A1_registered_raw.tif')
    module_stats = None
    with adapter.install_loader(history) as loaded:
        module = history._issue_module(history.I013, 'm1_seat_identity')
        assert loaded and loaded[0]['supported'], loaded
        module_stats = loaded[0]
        replaced_cores, replaced_shed = module.soma_cores, module.shed_crumbs
    original_cores, original_shed = module.soma_cores, module.shed_crumbs
    assert replaced_cores is not original_cores and replaced_shed is not original_shed

    def cores_key(found):
        return [(hashlib.sha256(np.packbits(c['mask']).tobytes()).hexdigest(), c['mask'].shape,
                 np.float64(c['row']).tobytes(), np.float64(c['column']).tobytes(),
                 np.float64(c['peak']).tobytes(), c['area'], c['dim']) for c in found]
    core_cases = shed_cases = 0
    for _ in range(a.frames):
        stack = stacks[int(rng.integers(2))]
        t = int(rng.integers(len(stack)))
        assigned = stack[t] > 0
        for percentile, minimum in ((99.0, 6), (97.5, 2), (99.5, 20)):
            assert cores_key(original_cores(raw[t], assigned, percentile, minimum)) ==                 cores_key(replaced_cores(raw[t], assigned, percentile, minimum))
            core_cases += 1
        for body, piece in ((40, 2), (25, 1), (80, 5)):
            e_labels, e_rows = original_shed(stack[t], body, piece)
            f_labels, f_rows = replaced_shed(stack[t], body, piece)
            assert e_labels.dtype == f_labels.dtype and np.array_equal(e_labels, f_labels) and e_rows == f_rows
            shed_cases += 1
    # Frames with crumbs: real frames with each cell's pixels scattered as
    # small pieces near it and inside neighbours, and dense random frames, so
    # pieces are handed on (also to identities labelled later).
    shed_rows = 0
    for number in range(a.frames):
        stack = stacks[int(rng.integers(2))]
        frame = stack[int(rng.integers(len(stack)))].copy()
        for identity in rng.choice(np.unique(frame)[1:], 25):
            ys, xs = np.nonzero(frame == identity)
            for k in rng.choice(len(ys), min(len(ys), 6), replace=False):
                y = min(frame.shape[0] - 1, max(0, int(ys[k]) + int(rng.integers(-6, 7))))
                x = min(frame.shape[1] - 1, max(0, int(xs[k]) + int(rng.integers(-6, 7))))
                frame[y:y + int(rng.integers(1, 3)), x:x + int(rng.integers(1, 3))] = identity
        if number % 3 == 0:
            frame = frame.astype(np.int32)
        for body, piece in ((40, 1), (25, 2), (6, 1)):
            e_labels, e_rows = original_shed(frame, body, piece)
            f_labels, f_rows = replaced_shed(frame, body, piece)
            assert e_labels.dtype == f_labels.dtype and np.array_equal(e_labels, f_labels) and e_rows == f_rows
            shed_cases += 1; shed_rows += len(e_rows)
    for number in range(200):
        shape = (int(rng.integers(2, 25)), int(rng.integers(2, 25)))
        frame = (rng.integers(0, 7, shape) * (rng.random(shape) < .6)).astype([np.uint16, np.int32][number % 2])
        for body, piece in ((3, 1), (5, 1), (2, 1)):
            e_labels, e_rows = original_shed(frame, body, piece)
            f_labels, f_rows = replaced_shed(frame, body, piece)
            assert e_labels.dtype == f_labels.dtype and np.array_equal(e_labels, f_labels) and e_rows == f_rows
            shed_cases += 1; shed_rows += len(e_rows)
    assert shed_rows, 'no crumb was handed on'
    report = dict(shed_rows=shed_rows,status='pass', frames=len(cases), real_frames=a.frames, selections=selections,
                  soma_core_cases=core_cases, shed_cases=shed_cases,
                  before_seconds=before, after_seconds=after, counters=stats, module_counters=module_stats)
    print(report)
    a.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
