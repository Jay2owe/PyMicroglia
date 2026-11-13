"""Check owner boxes from find_objects against the per-owner mask scan in censored_area_boxes.OwnerParts.

python check_owner_part_boxes.py --adapter <identity_candidate dir> --labels <label stack .tif> --output <report.json>

Compares get() and area_at() of OwnerParts as it is with the same class forced
onto the mask scan, on a real label movie (as stored, as 64-bit and as a memmap)
and on frames with negative or very large values, for present, absent, zero,
negative and out-of-range owners, NumPy and Python integers, and points inside,
near and far from each owner.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--frames', type=int, default=60)
    args = parser.parse_args()
    sys.path.insert(0, args.adapter)
    import censored_area_boxes as boxes

    class Scanned(boxes.OwnerParts):
        def _boxes(self, frame):
            return None

    structure = np.ones((3, 3), bool)
    rng = np.random.default_rng(3)
    stored = tifffile.imread(args.labels)[:args.frames]
    odd = stored.astype(np.int64).copy()
    odd[0, :5, :5] = -2
    odd[1, 10:14, 10:14] = 1 << 21
    tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    mapped = np.memmap(Path(tmp.name) / 'labels.bin', dtype=stored.dtype, mode='w+', shape=stored.shape)
    mapped[:] = stored
    movies = {'stored %s' % stored.dtype: stored, 'int64': stored.astype(np.int64),
              'memmap': mapped, 'negative and large': odd}
    failures, compared, boxed = [], 0, 0
    for name, movie in movies.items():
        stats_new, stats_old = dict(labelled=0, reused=0), dict(labelled=0, reused=0)
        new, old = boxes.OwnerParts(movie, structure, stats_new), Scanned(movie, structure, stats_old)
        height, width = movie.shape[1:]
        for frame in range(len(movie)):
            present = [int(v) for v in np.unique(movie[frame]) if v > 0]
            owners = present[:: max(1, len(present) // 25)] + [0, -1, 9999, int(movie[frame].max()) + 1]
            owners += [np.int64(present[0])] if present else []
            for owner in owners:
                if isinstance(owner, int) and owner > 0 and owner in present:
                    yy, xx = np.nonzero(movie[frame] == owner)
                    k = rng.integers(len(yy))
                    points = [(float(xx[k]), float(yy[k])), (xx[k] + 4.6, yy[k] - 3.2), (xx[k] + 9.0, yy[k] + 9.0)]
                else:
                    points = [(float(rng.random() * width), float(rng.random() * height))]
                points += [(-3.0, 2.0), (width + 2.5, height - 1.0)]
                left, right = new.get(frame, owner), old.get(frame, owner)
                same = (left[0] == right[0] and (left[1] is None) == (right[1] is None)
                        and (left[1] is None or (np.array_equal(left[1], right[1]) and left[1].dtype == right[1].dtype
                                                 and np.array_equal(left[2], right[2]))))
                if not same:
                    failures.append(dict(movie=name, frame=frame, owner=str(owner), problem='parts differ'))
                for x, y in points:
                    a, b = new.area_at(frame, owner, x, y, 6), old.area_at(frame, owner, x, y, 6)
                    compared += 1
                    if a != b or type(a) is not type(b):
                        failures.append(dict(movie=name, frame=frame, owner=str(owner), x=x, y=y, new=a, old=b))
        boxed += stats_new.get('boxed', 0)
        if stats_new['labelled'] != stats_old['labelled'] or stats_new['reused'] != stats_old['reused']:
            failures.append(dict(movie=name, problem='counts differ', new=stats_new, old=stats_old))
    del mapped
    report = dict(status='pass' if not failures else 'fail', areas_compared=compared, boxed_parts=boxed,
                  failures=failures[:40])
    Path(args.output).write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
    print(json.dumps(dict(status=report['status'], areas_compared=compared, boxed_parts=boxed, failures=len(failures))))


if __name__ == '__main__':
    main()
