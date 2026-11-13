"""Compare density_reciprocal_overlap.correct with and without reused frame objects and array neighbour counts, exactly.

The accepted label movie (windows of real frames) runs as accepted and with
injected reciprocal swaps (two touching outlines exchanged for a run of
frames, so swaps are proposed, applied and some reverted by the recurrence
rule), with default and permissive parameters, with the module's own object
function and with the grouped-geometry one, and as read-only disk-backed
memmaps. Labels, audit tables (values, dtypes, order) and summaries must equal
the original's. Neighbour counts are also compared directly, value and type,
on every frame.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

from check_reference_area_boxes import same


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'code', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--starts', default='0,220,416')
    p.add_argument('--frames', type=int, default=60)
    a = p.parse_args()
    import numpy as np
    import tifffile
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    movie = tifffile.imread(folder / '99_well_b3_issue013_anchored_projection_owner_relay.tif')
    sys.path[:0] = [str(a.code.resolve()), str(a.adapter.resolve())]
    import density_reciprocal_overlap as density
    import grouped_cell_geometry
    import density_overlap_reuse as adapter
    from contextlib import nullcontext

    class Cyclic:
        _objects = None
    original = density.correct
    scratch = Path(tempfile.mkdtemp(prefix='density-reuse-'))
    report = dict(status='pass', cases=[], calls=0)
    timings = dict(original=0.0, adapted=0.0)
    totals = {}
    rng = np.random.default_rng(17)
    permissive = dict(minimum_overlap_px=4, minimum_cross_coverage=0.2, minimum_margin_fraction=-0.5,
                      minimum_local_neighbours=0, minimum_recurrent_exchanges=2,
                      maximum_recurrence_gap_movie_fraction=0.05)
    neighbour_frames = 0
    maps = []
    for start in (int(s) for s in a.starts.split(',')):
        start = max(0, min(start, len(movie) - a.frames))
        labels = movie[start:start + a.frames].copy()
        swapped = labels.copy()
        names = [int(v) for v in np.unique(labels[0]) if v]
        for k in range(0, min(len(names) - 1, 16), 2):
            left, right = names[k], names[k + 1]
            for t in range(5 + k, min(len(swapped), 5 + k + 4)):
                lm, rm = swapped[t] == left, swapped[t] == right
                swapped[t][lm], swapped[t][rm] = right, left
        for kind, stack in (('accepted', labels), ('swapped', swapped)):
            path = scratch / f'{start}_{kind}.dat'
            stack.tofile(path)
            mapped = np.memmap(path, dtype=stack.dtype, mode='r', shape=stack.shape)
            maps.append(mapped)
            for storage, data in (('plain', stack), ('memmap', mapped)):
                for objects in ('module', 'grouped'):
                    for label, params in (('default', None), ('permissive', permissive)):
                        context = (grouped_cell_geometry.grouped_geometry(density, Cyclic) if objects == 'grouped'
                                   else nullcontext())
                        with context:
                            tick = time.perf_counter()
                            expected = original(data, params)
                            middle = time.perf_counter()
                            stats = {}
                            with adapter.install(stats):
                                assert density.correct.__wrapped__ is original
                                actual = density.correct(data, params)
                            timings['original'] += middle - tick
                            timings['adapted'] += time.perf_counter() - middle
                        assert density.correct is original
                        assert same(list(expected), list(actual)), (start, kind, storage, objects, label)
                        report['cases'].append(dict(start=start, labels=kind, input=storage, objects=objects,
                                                    params=label, audit_rows=len(expected[1]),
                                                    applied=expected[2]['applied_exchanges'], stats=stats))
                        for key, value in stats.items():
                            totals[key] = totals.get(key, 0) + value
                        report['calls'] += 1
            # Neighbour counts, value and type, frame by frame.
            counts = adapter.neighbour_counts({'neighbours_arrays': 0, 'neighbours_direct': 0})
            for t in range(len(stack)):
                old = density._objects(stack[t])
                expected = {}
                for identity, item in old.items():
                    expected[identity] = sum(
                        other != identity and np.hypot(
                            item["y"] - candidate["y"], item["x"] - candidate["x"]
                        ) <= 4.0 * max(item["radius"] + candidate["radius"], 1.0)
                        for other, candidate in old.items())
                actual = counts(old, 4.0)
                assert list(expected) == list(actual)
                assert all(type(expected[k]) is type(actual[k]) and expected[k] == actual[k] for k in expected), t
                neighbour_frames += 1
        print(start, report['calls'], flush=True)
    assert totals.get('objects_reused', 0) > 0 and totals.get('neighbours_arrays', 0) > 0
    assert sum(case['applied'] for case in report['cases']) > 0
    report.update(seconds=timings, stats=totals, neighbour_frames=neighbour_frames)
    for mapped in maps:
        mapped._mmap.close()
    import shutil
    shutil.rmtree(scratch, ignore_errors=True)
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], calls=report['calls'], seconds=timings, stats=totals,
                          applied=sum(case['applied'] for case in report['cases']))))


if __name__ == '__main__':
    main()
