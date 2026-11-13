"""Check that narrowed label digests tell frames apart exactly as the full-width digests do.

python check_narrow_labels.py --adapter <identity_candidate dir> --output <report.json>

For every integer dtype, frames with boundary values (each narrower type's
maximum and one above it, negatives, empty and zero frames, strided and
working-storage views), checks that the owner-sampling frame token and the
observation-cache label key are equal for two frames exactly when the frames
are equal, as the full-width SHA-256 ones are, and that narrowing round-trips.
The caches act only on which keys are equal, so equal relations give the same
hits, misses and results.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

DTYPES = [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]
EDGES = [0, 1, 255, 256, 417, 65535, 65536, 2 ** 32 - 1, 2 ** 32, -1, -129, 2 ** 63 - 1]


def frames(dtype, rng):
    info = np.iinfo(dtype)
    base = rng.integers(0, 6, (9, 11)).astype(dtype)
    yield 'small', base
    yield 'zero', np.zeros((9, 11), dtype)
    yield 'empty', np.zeros((0, 11), dtype)
    for edge in EDGES:
        if info.min <= edge <= info.max:
            value = base.copy(); value[4, 5] = edge
            yield 'edge%d' % edge, value
            other = value.copy(); other[0, 0] = edge
            yield 'edge%d-moved' % edge, other
    wide = rng.integers(0, 6, (9, 22)).astype(dtype)
    yield 'strided', wide[:, ::2]
    yield 'fortran', np.asfortranarray(base)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.adapter)
    import correction_owner_sampling as sampling
    import observation_cache as cache
    import narrow_labels
    import hashlib

    def old_token(plane):
        plain = np.ascontiguousarray(np.asarray(plane))
        return plain.shape, plain.dtype.str, hashlib.sha256(memoryview(plain).cast('B')).digest()

    def safe(function):
        def call(value):
            try:
                return function(value)
            except Exception as error:  # an error is compared like a value
                return ('error', type(error).__name__, str(error))
        return call

    old_token, new_token = safe(old_token), safe(sampling._frame_token)
    old_key, new_key = safe(cache._digest), safe(cache._label_digest)
    rng = np.random.default_rng(11)
    failures, pairs, narrowed = [], 0, 0
    for dtype in DTYPES:
        named = list(frames(dtype, rng))
        named.append(('memmap', np.memmap(Path(args.output).with_suffix('.%s.bin' % np.dtype(dtype).name),
                                          dtype=dtype, mode='w+', shape=(9, 11))))
        named[-1][1][:] = named[0][1]
        for name, value in named:
            if old_token(value) != new_token(value) and 'error' in (old_token(value)[0], new_token(value)[0]):
                failures.append(dict(case='%s/%s' % (np.dtype(dtype).name, name), problem='different error'))
            if not value.size:
                continue
            small, code = narrow_labels.narrow(np.ascontiguousarray(value))
            if code is not None:
                narrowed += 1
                if not (np.array_equal(small.astype(value.dtype), value) and small.dtype.itemsize < value.dtype.itemsize):
                    failures.append(dict(case='%s/%s' % (np.dtype(dtype).name, name), problem='round trip'))
        for (left_name, left), (right_name, right) in itertools.product(named, repeat=2):
            pairs += 1
            equal = left.shape == right.shape and np.array_equal(left, right)
            relations = dict(
                old_token=old_token(left) == old_token(right),
                new_token=new_token(left) == new_token(right),
                old_key=old_key(left) == old_key(right),
                new_key=new_key(left) == new_key(right))
            if any(value != equal for value in relations.values()):
                failures.append(dict(case='%s/%s~%s' % (np.dtype(dtype).name, left_name, right_name),
                                     equal=equal, **relations))
        del named
    # Frames of different dtypes never share a token or key, before or after.
    mixed = [np.full((3, 4), 7, dtype) for dtype in DTYPES]
    for left, right in itertools.combinations(mixed, 2):
        pairs += 1
        if new_token(left) == new_token(right) or new_key(left) == new_key(right):
            failures.append(dict(case='%s~%s' % (left.dtype, right.dtype), problem='dtypes share a digest'))
    # Soma-path outline parts: the same calls with narrowing and with the full-width
    # key give the same reuse pattern and the same parts.
    import spatial_path_views as views
    from scipy import ndimage

    def parts_run(narrowing, copy_budget=None):
        saved = views._narrow_labels.narrow
        if not narrowing:
            views._narrow_labels.narrow = lambda array: (array, None)
        try:
            stats = dict(parts_direct=0, parts_reused=0, parts_measured=0)
            memo = views.PartsMemo() if copy_budget is None else views.PartsMemo(copy_budget=copy_budget)
            call = views.parts(stats, memo, ndimage)
            eight, cross = np.ones((3, 3), bool), ndimage.generate_binary_structure(2, 1)
            pattern, results = [], []
            for frame, observed, raw in calls:
                before = dict(stats)
                results.append(call(frame, observed, eight, cross, raw))
                pattern.append(next(k for k in stats if stats[k] != before[k]))
            return pattern, results
        finally:
            views._narrow_labels.narrow = saved

    calls = []
    base = np.zeros((24, 26), np.int64)
    base[2:9, 3:12] = 5; base[12:20, 14:22] = 300; base[3:7, 18:24] = 7
    raw = rng.random(base.shape).astype(np.float32)
    for dtype in (np.int64, np.int32, np.uint16, np.uint8):
        frame = np.clip(base, 0, np.iinfo(dtype).max).astype(dtype)
        for observed in (frame, frame.copy(), (frame > 0).astype(dtype), frame.astype(np.int64)):
            calls += [(frame, observed, raw), (frame.copy(), observed.copy(), raw.copy())]
        moved = frame.copy(); moved[0, 0] = 5
        calls += [(moved, frame, raw), (frame, moved, raw), (frame, frame, raw * 2)]
    big = base.copy(); big[1, 1] = 70000
    calls += [(big, big, raw), (big.copy(), big, raw), (base, base, raw)]
    # Raw frames whose bytes and values disagree: NaN equals itself by bytes, -0.0 differs from 0.0.
    odd = raw.copy(); odd[0, 0] = np.nan
    signed = raw.copy(); signed[0, 1] = 0.0
    negative = signed.copy(); negative[0, 1] = -0.0
    calls += [(base, base, odd), (base, base, odd.copy()), (base, base, signed), (base, base, negative)]
    original, narrowed_parts = parts_run(False, 0), parts_run(True)
    digest_only = parts_run(True, 0)
    if digest_only[0] != narrowed_parts[0]:
        failures.append(dict(case='spatial parts', problem='copy reuse pattern', digest=digest_only[0],
                             copies=narrowed_parts[0]))
    pairs += len(calls)
    if original[0] != narrowed_parts[0]:
        failures.append(dict(case='spatial parts', problem='reuse pattern', original=original[0],
                             narrowed=narrowed_parts[0]))
    for index, (left, right) in enumerate(zip(original[1], narrowed_parts[1])):
        if repr(left) != repr(right):
            failures.append(dict(case='spatial parts call %d' % index, problem='parts differ'))
    report = dict(status='pass' if not failures else 'fail', pairs=pairs, narrowed_frames=narrowed,
                  spatial_reuse_pattern=narrowed_parts[0], failures=failures[:40])
    Path(args.output).write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
    for path in Path(args.output).parent.glob(Path(args.output).stem + '.*.bin'):
        try:
            path.unlink()
        except OSError:
            pass
    print(json.dumps(dict(status=report['status'], pairs=pairs, narrowed_frames=narrowed, failures=len(failures))))


if __name__ == '__main__':
    main()
