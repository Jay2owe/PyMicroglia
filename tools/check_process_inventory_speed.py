"""Compare the process-inventory edge, occupancy and pair measurements with and without the adapter, exactly.

On real label frames (three windows) every connected outline of every name,
including outlines on the frame border, gives the same edge coordinates in the
same order. sample_occupancy runs on outline pixels with shifts that move them
partly and wholly outside the frame, on plain frames and read-only memmaps, for
present and absent names. measure (bridge, with the raw movie) and interface
run on every pair of nearby outlines through the installed adapter, and their
result dicts must equal the original's.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'code', 'config', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--stem', default='95_full_A1')
    p.add_argument('--starts', default='0,230,470')
    p.add_argument('--frames', type=int, default=3)
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    import tifffile
    from scipy import ndimage as ndi
    sys.path[:0] = [str((a.engine / 'code').resolve())]
    import common
    import pipeline
    cfg = common.Config.load(a.config)
    raw_movie = tifffile.imread(pipeline._pinned_paths(cfg, a.stem)['registered_raw'])
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    movie = tifffile.imread(folder / '99_well_b3_issue013_anchored_projection_owner_relay.tif')
    sys.path[:0] = [str(a.code.resolve()), str(a.adapter.resolve())]
    import process_inventory_bridge as bridge
    import process_inventory_interface_v2 as face
    import process_inventory_motion as motion
    import process_inventory_speed as adapter
    structure = np.ones((3, 3), bool)
    stats = {}
    edge = adapter.edge(stats | {'edges_boxed': 0, 'edges_empty': 0, 'edges_direct': 0})
    report = dict(status='pass', outlines=0, occupancy=0, pairs=0)
    timings = dict(original=0.0, adapted=0.0)
    scratch = Path(tempfile.mkdtemp(prefix='inventory-'))
    rng = np.random.default_rng(29)
    maps = []
    for start in (int(s) for s in a.starts.split(',')):
        for t in range(max(0, min(start, len(movie) - a.frames)), min(len(movie), start + a.frames)):
            frame = movie[t].copy()
            # A border outline: copy one outline onto the frame's corner rows.
            frame[:3, :5] = frame.max() + 1
            path = scratch / f'{t}.dat'
            frame.tofile(path)
            mapped = np.memmap(path, dtype=frame.dtype, mode='r', shape=frame.shape)
            maps.append(mapped)
            masks = []
            for name in np.unique(frame[frame > 0]):
                parts, count = ndi.label(frame == name, structure)
                for k in range(1, count + 1):
                    mask = parts == k
                    expected = np.argwhere(mask & ~ndi.binary_erosion(mask, structure=structure))
                    actual = edge(mask, ndi, structure)
                    assert actual.dtype == expected.dtype and np.array_equal(actual, expected), (t, name, k)
                    report['outlines'] += 1
                    masks.append((int(name), mask))
            original = motion.sample_occupancy
            fast = adapter.occupancy(original, {'occupancy_gathered': 0, 'occupancy_direct': 0})
            for name, mask in masks[::3]:
                yy, xx = np.where(mask)
                for dy, dx in ((0, 0), (3, -2), (-5, 4), (frame.shape[0] - 2, 0), (0, -frame.shape[1]), (900, 900)):
                    for image in (frame, mapped):
                        for target in (name, name + 1, 0):
                            assert original(image, target, yy, xx, dy, dx) == fast(image, target, yy, xx, dy, dx)
                            report['occupancy'] += 1
            # Nearby outline pairs for measure and interface.
            centres = {}
            for name, mask in masks:
                yy, xx = np.nonzero(mask)
                centres.setdefault(name, (yy[len(yy) // 2], xx[len(xx) // 2]))
            names = list(centres)
            raw = raw_movie[t]
            for i, left in enumerate(names):
                for right in names[i + 1:]:
                    (ly, lx), (ry, rx) = centres[left], centres[right]
                    if abs(ly - ry) + abs(lx - rx) > 40:
                        continue
                    pair = pd.Series(dict(large_identity=left, small_identity=right))
                    lp = pd.Series(dict(y=float(ly), x=float(lx)))
                    sp = pd.Series(dict(y=float(ry), x=float(rx)))
                    pair = next(pd.DataFrame([pair]).itertuples(index=False))
                    tick = time.perf_counter()
                    expected = (bridge.measure(frame, raw, pair, lp, sp), face.interface(frame, left, right, lp, sp))
                    middle = time.perf_counter()
                    with adapter.install(stats):
                        assert bridge.measure.__wrapped__ and face.interface.__wrapped__ and motion.sample_occupancy.__wrapped__
                        inner = time.perf_counter()
                        actual = (bridge.measure(frame, raw, pair, lp, sp), face.interface(frame, left, right, lp, sp))
                        timings['adapted'] += time.perf_counter() - inner
                    timings['original'] += middle - tick
                    assert getattr(bridge.measure, '__wrapped__', None) is None
                    assert expected == actual, (t, left, right, expected, actual)
                    report['pairs'] += 1
        print(start, report, flush=True)
    assert report['pairs'] > 0 and stats.get('edges_boxed', 0) > 0 and stats.get('versions') == 3 * report['pairs']
    report.update(seconds=timings, stats=stats)
    for mapped in maps:
        mapped._mmap.close()
    import shutil
    shutil.rmtree(scratch, ignore_errors=True)
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
