"""Check the issue008 global-quarantine adapter against the original module, value for value.

python check_quarantine_follow.py --adapter <identity_candidate dir> --engine <A1 engine dir>
    [--history <accepted_history mid dir>] --output <report.json>

Compares the original and adapted following_component, detect_events and presence on
synthetic frames (every integer dtype, absent, zero, negative and out-of-range identities,
empty previous masks, label values large enough to skip the overlap histogram, non-ndarray
frames), and, with --history, the whole run on the saved trimmed A1 inputs.
"""
import argparse
import json
import sys
import tempfile
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def outcome(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            value = call()
        except Exception as error:  # the error itself is compared
            return ('error', type(error).__name__, str(error)), [str(w.message) for w in caught]
    return ('value', value), [str(w.message) for w in caught]


def same(left, right):
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (type(left) is type(right) and left.dtype == right.dtype and left.shape == right.shape
                and np.array_equal(left, right, equal_nan=left.dtype.kind in 'fc'))
    if isinstance(left, dict):
        return (isinstance(right, dict) and list(left) == list(right)
                and all(same(left[k], right[k]) for k in left))
    if isinstance(left, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(same(a, b) for a, b in zip(left, right)))
    if isinstance(left, float) and isinstance(right, float) and left != left:
        return right != right
    return type(left) is type(right) and left == right


def frames(rng):
    base = np.zeros((40, 48), np.int64)
    base[2:9, 3:12] = 5
    base[5:14, 10:20] = 7          # touches 5
    base[20:26, 20:23] = 7          # second component of 7
    base[30:38, 30:44] = 300
    base[0, 47] = 5                 # one-pixel component at the edge
    base[15:18, 0:3] = 2 ** 20
    noise = rng.integers(0, 4, base.shape)
    yield 'blocks', base
    yield 'noise', noise
    yield 'empty', np.zeros((12, 9), np.int64)
    yield 'single', np.full((6, 7), 3, np.int64)


DTYPES = [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64]


@contextmanager
def adapted(adapter, module):
    with adapter.install(module) as stats:
        if not stats['supported']:
            raise SystemExit('adapter refused the module source')
        yield stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--engine', required=True)
    parser.add_argument('--history')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.engine) / 'code'))
    sys.path.insert(0, args.adapter)
    import accepted_history as history
    import quarantine_overlap_batch as adapter
    module = history._issue_module(history.OWNERSHIP_QUARANTINE_OPS, 'global_quarantine')
    rng = np.random.default_rng(8)
    failures, counts = [], dict(follow=0, follow_errors=0, events=0, presence=0, run=0)

    def compare(kind, name, original, rewritten):
        left, left_warnings = original
        right, right_warnings = rewritten
        counts[kind] += 1
        if left[0] == 'error':
            counts[kind + '_errors'] = counts.get(kind + '_errors', 0) + 1
        if not same(left, right) or left_warnings != right_warnings:
            failures.append(dict(kind=kind, case=name, original=repr(left)[:300], rewritten=repr(right)[:300],
                                 warnings=[left_warnings, right_warnings]))

    previous_masks = {}
    for name, frame in frames(rng):
        previous_masks[name] = [frame == 5, frame == 7, np.zeros(frame.shape, bool), frame > 0,
                                np.roll(frame == 300, 4, axis=1)]
    for name, base in frames(rng):
        for dtype in DTYPES:
            info = np.iinfo(dtype)
            frame = np.clip(base, info.min, info.max).astype(dtype)
            values = sorted(set(map(int, np.unique(frame))))
            observed = values + [0, 1, 6, -1, info.max, info.min, int(info.max) + 1, 2 ** 70]
            views = [('plain', frame), ('strided', np.asfortranarray(frame)),
                     ('subclass', frame.view(type('Subclass', (np.ndarray,), {})))]
            for view_name, view in views:
                for identity in observed:
                    for index, previous in enumerate(previous_masks[name]):
                        case = '%s/%s/%s/id=%s/prev=%d' % (name, np.dtype(dtype).name, view_name, identity, index)
                        original = outcome(lambda: module.following_component(view, identity, previous))
                        with adapted(adapter, module):
                            rewritten = outcome(lambda: module.following_component(view, identity, previous))
                        compare('follow', case, original, rewritten)
            # a NumPy integer identity is not a Python int, so it takes the original path
            original = outcome(lambda: module.following_component(frame, np.int64(values[-1]), previous_masks[name][0]))
            with adapted(adapter, module):
                rewritten = outcome(lambda: module.following_component(frame, np.int64(values[-1]), previous_masks[name][0]))
            compare('follow', '%s/%s/numpy-identity' % (name, np.dtype(dtype).name), original, rewritten)

    cfg = dict(large_area_px=4, minimum_overlap_px=2, minimum_iou=0.05, maximum_centroid_shift_px=40.0,
               history_frames=3, minimum_owner_history_pixels=1, minimum_history_margin_pixels=0,
               minimum_prior_identity_frames=1, maximum_replacement_history_pixels=10 ** 6,
               minimum_destination_persistence_frames=1, maximum_old_identity_current_area_px=10 ** 6)
    for size, top in [(6, 9), (10, 40), (8, 5000), (5, 70000)]:
        for dtype in [np.uint16, np.uint32, np.int32, np.int64, np.uint64]:
            if top > np.iinfo(dtype).max:
                continue
            movie = np.zeros((size, 30, 34), dtype)
            identities = rng.choice(np.arange(1, top + 1), size=6, replace=False)
            for t in range(size):
                for k, identity in enumerate(identities):
                    y, x = 3 + 4 * k % 24 + t % 3, 2 + (5 * k + t) % 26
                    movie[t, y:y + 4, x:x + 5] = identity
                if t > size // 2:  # an identity takes over another's territory
                    movie[t][movie[t] == identities[0]] = identities[1]
            for offset in (0, 3):
                original = outcome(lambda: module.detect_events(movie, cfg, offset))
                with adapted(adapter, module):
                    rewritten = outcome(lambda: module.detect_events(movie, cfg, offset))
                compare('events', 'size=%d/top=%d/%s/offset=%d' % (size, top, np.dtype(dtype).name, offset),
                        original, rewritten)
            original = outcome(lambda: {k: v for k, v in _presence_original(movie).items()})
            rewritten = outcome(lambda: adapter.presence(movie))
            compare('presence', 'size=%d/top=%d/%s' % (size, top, np.dtype(dtype).name), original, rewritten)
    for dtype in DTYPES:
        movie = rng.integers(-3 if np.iinfo(dtype).min < 0 else 0, 5, (4, 6, 7)).astype(dtype)
        compare('presence', 'signed/%s' % np.dtype(dtype).name, outcome(lambda: _presence_original(movie)),
                outcome(lambda: adapter.presence(movie)))

    run = None
    if args.history:
        from common import load_stack
        hist = Path(args.history)
        params = history._params('issue008', 'global_quarantine')
        params.update({'stem': '95_full_A1', 'source_frame_offset': 0, 'frame_interval_min': 10.0})
        results = {}
        for label in ('original', 'adapted'):
            with tempfile.TemporaryDirectory() as tmp:
                out = type('Out', (), {'out': Path(tmp)})()
                start = time.perf_counter()
                if label == 'adapted':
                    with adapted(adapter, module) as stats:
                        summary = module.run(hist / '07_trimmed_evidence' / 'out', params, out)
                    stats = dict(stats)
                else:
                    summary, stats = module.run(hist / '07_trimmed_evidence' / 'out', params, out), None
                results[label] = dict(seconds=round(time.perf_counter() - start, 2), stats=stats,
                                      summary=summary['summary'],
                                      labels=load_stack(Path(tmp) / '95_full_A1.tif'),
                                      events=(Path(tmp) / 'quarantine_events.csv').read_bytes())
        saved = hist / '08_global_quarantine' / 'out'
        run = dict(seconds={k: v['seconds'] for k, v in results.items()}, adapted_stats=results['adapted']['stats'],
                   labels_equal=bool(np.array_equal(results['original']['labels'], results['adapted']['labels'])),
                   events_equal=results['original']['events'] == results['adapted']['events'],
                   summary_equal=results['original']['summary'] == results['adapted']['summary'],
                   matches_saved=bool(np.array_equal(load_stack(saved / '95_full_A1.tif'), results['adapted']['labels'])
                                      and (saved / 'quarantine_events.csv').read_bytes() == results['adapted']['events']))
        counts['run'] = 1
        if not (run['labels_equal'] and run['events_equal'] and run['summary_equal'] and run['matches_saved']):
            failures.append(dict(kind='run', case='A1', detail=run))
    report = dict(status='pass' if not failures else 'fail', counts=counts, failures=failures[:40], a1_run=run)
    Path(args.output).write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
    print(json.dumps(dict(status=report['status'], counts=counts, failures=len(failures), a1_run=run), default=str))


def _presence_original(labels):
    result = {}
    for t, frame in enumerate(labels):
        for identity in np.unique(frame):
            if identity:
                result.setdefault(int(identity), []).append(t)
    return {identity: np.asarray(frames) for identity, frames in result.items()}


if __name__ == '__main__':
    main()
