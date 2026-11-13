"""Check threaded latent-track evidence images against the original expand_track_points.

python check_threaded_evidence.py --adapter <identity_candidate dir> --engine <A1 engine dir> --output <report.json>

Runs A1's latent-track expansion (with the production visibility rewrite) once as
it is and once with the threaded evidence adapter, from the saved detector tables,
and compares the point and track tables with each other and with the saved stage
output. Also checks a frame whose image raises (same error, same frame), calls
that ask for other frames or parameter objects, and a one-frame movie.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def outcome(call):
    try:
        return ('value', call())
    except Exception as error:  # the error itself is compared
        return ('error', type(error).__name__, str(error))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--engine', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--frames', type=int, default=0, help='limit the movie for a quick check (0 = all)')
    args = parser.parse_args()
    engine = Path(args.engine)
    sys.path.insert(0, str(engine / 'code'))
    sys.path.insert(0, args.adapter)
    import latent_body_tracks as linker
    import raw_physical_hypotheses as detector
    from relative_latent_visibility import relative_latent_visibility
    from physical_tracking_speed import physical_tracking_speed
    from common import load_stack
    import threaded_evidence_images as threaded

    history = engine / 'm22_accepted_history' / 'pymicroglia_95_full_A1' / 'mid' / 'accepted_history'
    params = json.loads((history / '26_latent_body_tracks' / 'out' / 'producer_metrics.json').read_text())['parameters']
    raw = load_stack(history / '07_trimmed_evidence' / 'out' / '95_full_A1_registered_raw.tif')
    motion_full = load_stack(engine.parents[3] / 'rigid_batch' / '.j' / 'A1' / 'inputs' / '95_full_A1_motion_evidence.tif')
    hypotheses = pd.read_csv(history / '25_raw_physical_hypotheses' / 'out' / 'raw_physical_hypotheses.csv')
    thresholds = pd.read_csv(history / '25_raw_physical_hypotheses' / 'out' / 'frame_evidence_thresholds.csv')
    if args.frames:
        raw = raw[:args.frames]
        hypotheses = hypotheses[hypotheses.frame < args.frames]
        thresholds = thresholds[thresholds.frame < args.frames]
    motion = motion_full[2:2 + len(raw)]
    failures = []
    # Both are the production rewrites active around the identity stage.
    with relative_latent_visibility(linker), physical_tracking_speed(linker):
        start = time.perf_counter()
        tracks = linker.link_observations(hypotheses, raw, motion, params)
        link_seconds = time.perf_counter() - start
        start = time.perf_counter()
        before = linker.expand_track_points(copy.deepcopy(tracks), raw, thresholds, params)
        original_seconds = time.perf_counter() - start
        with threaded.install(linker, detector) as stats:
            start = time.perf_counter()
            after = linker.expand_track_points(copy.deepcopy(tracks), raw, thresholds, params)
            adapted_seconds = time.perf_counter() - start
        stats = dict(stats)
        for name, left, right in zip(('points', 'tracks'), before, after):
            if not (left.equals(right) and list(left.columns) == list(right.columns)
                    and left.dtypes.equals(right.dtypes) and left.index.equals(right.index)
                    and left.to_csv(index=False) == right.to_csv(index=False)):
                failures.append(dict(case='A1 ' + name, problem='tables differ'))
        saved = None
        if not args.frames:
            saved_points = (history / '26_latent_body_tracks' / 'out' / 'latent_track_points.csv').read_text()
            saved_tracks = (history / '26_latent_body_tracks' / 'out' / 'physical_tracks.csv').read_text()
            # Informational: this standalone linking is not the production call (its
            # track count can differ), so only original against threaded is judged.
            saved = dict(points=after[0].to_csv(index=False) == saved_points,
                         tracks=after[1].to_csv(index=False) == saved_tracks,
                         saved_track_rows=len(saved_tracks.splitlines()) - 1, track_rows=int(len(after[1])))
        if stats['served'] != len(raw) or stats['unprepared_calls'] or stats['fallback_calls']:
            failures.append(dict(case='A1 stats', detail=stats))

        # A frame whose image raises: the same error from the same frame.
        small = raw[:6]
        real = detector.evidence_image

        def failing(frame, given):
            if float(np.asarray(frame).mean()) == float(np.asarray(small[3]).mean()):
                raise FloatingPointError('frame 3 failed')
            return real(frame, given)
        detector.evidence_image = failing
        try:
            left = outcome(lambda: linker.expand_track_points(copy.deepcopy(tracks), small, thresholds, params))
            with threaded.install(linker, detector) as error_stats:
                right = outcome(lambda: linker.expand_track_points(copy.deepcopy(tracks), small, thresholds, params))
        finally:
            detector.evidence_image = real
        if left[:2] != right[:2] or left[0] != 'error' or (left[0] == 'error' and left != right):
            failures.append(dict(case='raising frame', original=repr(left)[:200], adapted=repr(right)[:200]))
        if error_stats['failed_preparations'] != 1:
            failures.append(dict(case='raising frame stats', detail=dict(error_stats)))

    # Calls for other frames or parameter objects take evidence_image as before.
    def stand_in(tracks, movie, thresholds, given):
        other = dict(given)
        images = [detector.evidence_image(movie[0], other)]
        images += [detector.evidence_image(frame, given) for frame in movie]
        images += [detector.evidence_image(movie[-1].copy(), given), detector.evidence_image(movie[::-1][0], given)]
        return images
    saved_expand = linker.expand_track_points
    linker.expand_track_points = stand_in
    try:
        for movie in (raw[:5], raw[:1], raw[:0], np.ascontiguousarray(raw[:4, ::2, ::3]).astype(np.float64)):
            if not len(movie):
                continue
            left = stand_in(None, movie, None, params)
            with threaded.install(linker, detector) as call_stats:
                right = linker.expand_track_points(None, movie, None, params)
            if len(left) != len(right) or not all(np.array_equal(a, b) and a.dtype == b.dtype for a, b in zip(left, right)):
                failures.append(dict(case='stand-in %s' % (movie.shape,), problem='images differ'))
            if call_stats['served'] != len(movie) or call_stats['unprepared_calls'] != 3:
                failures.append(dict(case='stand-in %s stats' % (movie.shape,), detail=dict(call_stats)))
    finally:
        linker.expand_track_points = saved_expand
    report = dict(status='pass' if not failures else 'fail', frames=int(len(raw)), tracks=len(tracks),
                  seconds=dict(link=round(link_seconds, 2), original=round(original_seconds, 2),
                               adapted=round(adapted_seconds, 2)),
                  adapted_stats=stats, matches_saved=saved, failures=failures)
    Path(args.output).write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
    print(json.dumps(report, default=str)[:1500])


if __name__ == '__main__':
    main()
