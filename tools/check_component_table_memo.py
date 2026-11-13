"""Compare component_table_memo with label_value_components, exactly.

Every frame of several A1 accepted-history stacks is asked in stage order (so
most later frames are memo hits), as plain arrays, read-only memory maps and
memory-pool working arrays, and every answer must equal a fresh computation in
type, dtype, shape, flags that callers can see (writeable, C order) and values.
Returned arrays are then edited and asked again, a frame is changed in one pixel
beside its unchanged twin, every frame is asked again with all checksums
equal (each lookup finds another frame's entry), and synthetic frames cover signed values
with negatives, uint8, uint32, empty frames, single pixels, non-contiguous
views, three-dimensional and float inputs (computed every time), and a memo too
small to hold one movie.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'
STACKS = ('08_issue012', '11_issue017_rebase', '90_well_b3_issue002_gap_tolerant_reciprocal_seat_exchange',
          '93_well_b3_issue006_right_censored_seat_partition', '94_well_b3_issue007_right_censored_reciprocal_exchange',
          '105_well_b3_issue016_body_scale_ownerless_allocation')


def outcome(function, frame):
    try:
        return ('ok', function(frame))
    except Exception as error:
        return ('error', (type(error).__name__, str(error)))


def same(a, b):
    if a[0] != b[0]:
        return 'outcome'
    if a[0] == 'error':
        return None if a[1] == b[1] else 'error'
    x, y = a[1], b[1]
    if type(x) is not type(y) or len(x) != len(y):
        return 'shape of result'
    for i, (p, q) in enumerate(zip(x, y)):
        if type(p) is not type(q) or p.dtype != q.dtype or p.shape != q.shape:
            return f'[{i}] type'
        if p.flags.writeable != q.flags.writeable or p.flags.c_contiguous != q.flags.c_contiguous:
            return f'[{i}] flags'
        if not np.array_equal(p, q):
            return f'[{i}] values'
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import tifffile
    import array_storage
    import component_accounting as accounting
    import plain_component_frames
    import component_table_memo as memo
    a.scratch.mkdir(parents=True, exist_ok=True)
    pool = array_storage.Pool(a.scratch / 'pool')
    report = dict(failures=[], cases=0, seconds=dict(original=0.0, memo=0.0))

    def fail(**item):
        report['failures'].append(item)

    with plain_component_frames.install(accounting):
        reference = accounting.label_value_components
        movies = []
        for index, stem in enumerate(STACKS):
            movie = tifffile.imread(HISTORY / (stem + '.tif'))
            if index % 3 == 1:
                path = a.scratch / f'movie_{index}.dat'
                mapped = np.memmap(path, dtype=movie.dtype, mode='w+', shape=movie.shape)
                mapped[:] = movie
                mapped.flush()
                movie = pool.adopt(mapped, path)
                movie.flags.writeable = False
            elif index % 3 == 2:
                path = a.scratch / f'movie_{index}.dat'
                mapped = np.memmap(path, dtype=movie.dtype, mode='w+', shape=movie.shape)
                mapped[:] = movie
                mapped.flush()
                movie = np.memmap(path, dtype=movie.dtype, mode='r', shape=movie.shape)
            movies.append((stem, movie))
        with memo.install(accounting) as stats:
            for stem, movie in movies:
                for t in range(len(movie)):
                    frame = movie[t]
                    t0 = time.perf_counter()
                    expected = outcome(reference, frame)
                    t1 = time.perf_counter()
                    actual = outcome(accounting.label_value_components, frame)
                    report['seconds']['original'] += t1 - t0
                    report['seconds']['memo'] += time.perf_counter() - t1
                    report['cases'] += 1
                    problem = same(expected, actual)
                    if problem:
                        fail(stack=stem, t=t, problem=problem)
            stage_stats = dict(stats)
            # Callers that edit what they receive cannot reach the memo.
            frame = np.asarray(movies[0][1][100]).copy()
            first = accounting.label_value_components(frame)
            for part in first:
                part[...] = 7
            again = accounting.label_value_components(frame)
            problem = same(outcome(reference, frame), ('ok', again))
            report['cases'] += 1
            if problem:
                fail(case='edited result', problem=problem)
            # A changed pixel is a different frame.
            changed = frame.copy()
            ys, xs = np.nonzero(changed)
            changed[ys[0], xs[0]] = 0
            changed[0, 0] = changed.max() + 1
            for candidate in (changed, frame, changed):
                problem = same(outcome(reference, candidate), outcome(accounting.label_value_components, candidate))
                report['cases'] += 1
                if problem:
                    fail(case='changed pixel', problem=problem)
            rng = np.random.default_rng(31)
            synthetic = {
                'signed_negatives': rng.integers(-3, 4, (40, 50)).astype(np.int16),
                'uint8': rng.integers(0, 3, (30, 30)).astype(np.uint8),
                'uint32': (rng.integers(0, 3, (25, 35)) * 90001).astype(np.uint32),
                'int64': rng.integers(0, 5, (20, 20)).astype(np.int64),
                'empty': np.zeros((30, 40), np.uint16),
                'single_pixel': np.pad(np.array([[5]], np.uint16), 9),
                'noncontiguous': rng.integers(0, 4, (60, 80)).astype(np.uint16)[::2, 1::3],
                'fortran': np.asfortranarray(rng.integers(0, 4, (30, 20)).astype(np.uint16)),
                'three_dimensional': rng.integers(0, 3, (3, 10, 10)).astype(np.uint16),
                'float': rng.integers(0, 3, (20, 20)).astype(np.float32),
                'bool': rng.integers(0, 2, (20, 20)).astype(bool),
            }
            for name, frame in synthetic.items():
                for repeat in range(2):
                    problem = same(outcome(reference, frame), outcome(accounting.label_value_components, frame))
                    report['cases'] += 1
                    if problem:
                        fail(case=name, repeat=repeat, problem=problem)
            final_stats = dict(stats)
        restored = accounting.label_value_components is reference
    # A memo far too small for one movie only repeats the work.
    with plain_component_frames.install(accounting):
        reference = accounting.label_value_components
        with memo.install(accounting, ram_bytes=2 << 20) as small:
            for stem, movie in movies[:2]:
                for t in range(0, len(movie), 3):
                    problem = same(outcome(reference, movie[t]), outcome(accounting.label_value_components, movie[t]))
                    report['cases'] += 1
                    if problem:
                        fail(case='small memo', stack=stem, t=t, problem=problem)
            for stem, movie in movies[:2]:
                for t in range(0, len(movie), 3):
                    problem = same(outcome(reference, movie[t]), outcome(accounting.label_value_components, movie[t]))
                    report['cases'] += 1
                    if problem:
                        fail(case='small memo repeat', stack=stem, t=t, problem=problem)
            small_stats = dict(small)
    # Every frame's checksum equal: each lookup finds another frame's entry.
    with plain_component_frames.install(accounting):
        reference = accounting.label_value_components
        checksum = memo._checksum
        memo._checksum = lambda view: 0
        try:
            with memo.install(accounting) as clashing:
                for stem, movie in movies[:2]:
                    for t in range(0, len(movie), 5):
                        for repeat in range(2):
                            problem = same(outcome(reference, movie[t]), outcome(accounting.label_value_components, movie[t]))
                            report['cases'] += 1
                            if problem:
                                fail(case='equal checksums', stack=stem, t=t, problem=problem)
                clash_stats = dict(clashing)
        finally:
            memo._checksum = checksum
    report['seconds'] = {k: round(v, 2) for k, v in report['seconds'].items()}
    report['stats'] = dict(stages=stage_stats, final=final_stats, small=small_stats, clashing=clash_stats)
    report['restored'] = restored
    report['status'] = 'pass' if (not report['failures'] and restored and stage_stats['hits']
                                  and final_stats['uncached'] and small_stats['evicted']
                                  and clash_stats['collisions'] and clash_stats['hits']) else 'fail'
    report['failures'] = report['failures'][:30]
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report, indent=1))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
