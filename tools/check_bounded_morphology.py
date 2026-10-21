"""Compare bounded binary erosion and dilation with scipy's, exactly.

Every name's mask in sampled frames of a real accepted label movie, plus edge
cases (masks on the frame border, empty, whole-frame, single pixels, random
blobs), is eroded and dilated by scipy and by the bounded adapter with each
structure the corrections use. Values, dtype, shape and type must be equal.
"""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--frames', type=int, default=60)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    import tifffile
    from scipy import ndimage as ndi
    sys.path[:0] = [str(a.adapter.resolve())]
    import bounded_morphology as adapter
    erosion, dilation = ndi.binary_erosion, ndi.binary_dilation
    structures = [None, np.ones((3, 3), bool), np.ones((3, 3), int), ndi.generate_binary_structure(2, 1),
                  np.ones((5, 5), bool), np.ones((7, 7), bool), ndi.generate_binary_structure(2, 2)]
    hollow = np.ones((3, 3), bool)
    hollow[1, 1] = False
    with tifffile.TiffFile(a.labels) as handle:
        total = len(handle.pages)
        picks = np.linspace(0, total - 1, min(a.frames, total)).astype(int)
        frames = [handle.pages[int(t)].asarray() for t in picks]
    rng = np.random.default_rng(11)
    masks = []
    for frame in frames:
        for name in np.unique(frame)[1:]:
            masks.append(frame == name)
    shape = frames[0].shape
    for _ in range(300):
        mask = np.zeros(shape, bool)
        y, x = rng.integers(0, shape[0]), rng.integers(0, shape[1])
        h, w = rng.integers(1, 30, 2)
        mask[max(y - h, 0):y + h, max(x - w, 0):x + w] = rng.random((len(range(max(y - h, 0), min(y + h, shape[0]))),
                                                                   len(range(max(x - w, 0), min(x + w, shape[1]))))) < .7
        masks.append(mask)
    edge = np.zeros(shape, bool)
    edge[0, 5:9] = edge[-1, -3:] = edge[10:20, 0] = True
    corner = np.zeros(shape, bool)
    corner[-2:, -2:] = True
    single = np.zeros(shape, bool)
    single[200, 200] = True
    masks += [np.zeros(shape, bool), np.ones(shape, bool), edge, corner, single,
              np.zeros((5, 7), bool), np.ones((5, 7), bool)]
    stats = {}
    compared = 0
    timings = dict(original=0.0, bounded=0.0)
    with adapter.install(stats):
        bounded_erosion, bounded_dilation = ndi.binary_erosion, ndi.binary_dilation
        assert bounded_erosion.__wrapped__ is erosion and bounded_dilation.__wrapped__ is dilation
    assert ndi.binary_erosion is erosion and ndi.binary_dilation is dilation
    for mask in masks:
        for structure in structures:
            for original, bounded in ((erosion, bounded_erosion), (dilation, bounded_dilation)):
                tick = time.perf_counter()
                expected = original(mask, structure=structure)
                middle = time.perf_counter()
                actual = bounded(mask, structure=structure)
                timings['original'] += middle - tick
                timings['bounded'] += time.perf_counter() - middle
                assert type(expected) is type(actual) and expected.dtype == actual.dtype
                assert expected.shape == actual.shape and np.array_equal(expected, actual)
                assert actual.flags.c_contiguous == expected.flags.c_contiguous
                compared += 1
        # A hollow structure erodes outside the mask: the original call is kept.
        assert np.array_equal(erosion(mask, hollow), bounded_erosion(mask, hollow))
        assert np.array_equal(dilation(mask, hollow), bounded_dilation(mask, hollow))
        assert np.array_equal(erosion(mask, iterations=2), bounded_erosion(mask, iterations=2))
        assert np.array_equal(dilation(mask, border_value=1), bounded_dilation(mask, border_value=1))
    report = dict(status='pass', frames=len(frames), masks=len(masks), compared=compared,
                  seconds=timings, stats=stats)
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
