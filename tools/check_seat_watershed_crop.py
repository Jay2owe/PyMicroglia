"""Compare seat-identity watersheds flooded in the mask's box with the whole frame.

Accepted label frames and the registered raw supply both seat call shapes:
every connected component split by markers placed on its brightest pixels and
on random pixels (partition_component), and each frame's assigned pixels split
by markers on every identity's brightest pixel (soma cores). Raw values are
also coarsened into wide plateaus and flattened to a constant, so ties and
age order decide most pixels. Markers outside the mask, masks touching the
frame edge, disjoint masks and unsigned or float images are included. Each
result must have the same dtype and values as skimage's own call.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'
RAW = HISTORY / '07_trimmed_evidence/out/95_full_A1_registered_raw.tif'


def cases(labels, raw, rng):
    images = dict(raw=-raw.astype(np.int32), plateau=-(raw.astype(np.int32) // 64),
                  flat=np.zeros(raw.shape, np.int32), smooth=-ndimage.gaussian_filter(raw.astype(np.float32), 1.5))
    numbered, count = ndimage.label(labels > 0)
    for number, box in enumerate(ndimage.find_objects(numbered), 1):
        component = numbered == number
        values = np.where(component, raw.astype(np.int64), int(raw.min()) - 1)
        markers = np.zeros(raw.shape, np.int32)
        for k, flat in enumerate(np.argsort(values, axis=None, kind='stable')[::-1][:3], 1):
            markers.flat[flat] = k
        spots = np.flatnonzero(component)
        markers.flat[rng.choice(spots, min(2, spots.size), replace=False)] = 4
        markers[max(box[0].start - 1, 0), max(box[1].start - 1, 0)] = 5  # outside the mask
        for name, image in images.items():
            yield f'component-{number}-{name}', image, markers, component
    assigned = labels > 0
    markers = np.zeros(raw.shape, np.int32)
    for k, identity in enumerate(np.unique(labels[assigned]), 1):
        where = np.flatnonzero(labels == identity)
        markers.flat[where[np.argmax(raw.flat[where])]] = k
    for name, image in images.items():
        yield f'assigned-{name}', image, markers, assigned
        yield f'assigned-uint8-{name}', image, (markers % 200).astype(np.uint8), assigned
    pair = (numbered == 1) | (numbered == count)
    yield 'disjoint', images['plateau'], markers, pair
    edge = np.zeros(raw.shape, bool); edge[:5, :] = True; edge[:, -3:] = True
    yield 'edge', images['flat'], markers, edge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=12)
    a = parser.parse_args()
    sys.path[:0] = [str(a.adapters.resolve())]
    import seat_watershed_crop
    from skimage.segmentation import _watershed
    original = _watershed.watershed
    assert seat_watershed_crop.supported(original)
    stats = dict(cropped=0, original=0)
    fast = seat_watershed_crop.cropped(original, stats)
    movie = tifffile.imread(HISTORY / '08_issue012.tif')
    picks = np.linspace(0, len(movie) - 1, a.frames).round().astype(int)
    rng = np.random.default_rng(13)
    failures, total, seconds = [], 0, dict(whole=0.0, cropped=0.0)
    for t in picks:
        raw = tifffile.imread(RAW, key=int(t))
        for label, image, markers, mask in cases(movie[t], raw, rng):
            t0 = time.perf_counter()
            expected = original(image, markers, mask=mask)
            t1 = time.perf_counter()
            actual = fast(image, markers, mask=mask)
            seconds['whole'] += t1 - t0; seconds['cropped'] += time.perf_counter() - t1
            total += 1
            if expected.dtype != actual.dtype or not np.array_equal(expected, actual):
                failures.append(f'{t}:{label}')
    report = dict(status='pass' if not failures else 'fail', frames=picks.tolist(), calls=total,
                  stats=stats, seconds=seconds, failures=failures[:10])
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
