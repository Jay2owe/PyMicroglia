"""Compare count-table presence frames and area medians with the original, exactly.

Five recorded accepted-history stacks and synthetic stacks with uint16, int32
and int64 identities, negative and very large values, empty frames and
single-frame identities: identity frames (keys, order and frames), the maximum
identity area and the minimum persistent area (with and without the movie
ratio, and invalid ratios) must match bit for bit or raise the same error.
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
    a = p.parse_args()
    import numpy as np
    import tifffile
    sys.path[:0] = [str(a.adapter.resolve()), str((a.engine / 'code').resolve())]
    import presence
    import presence_area_table as table
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    stacks = [tifffile.imread(folder / name) for name in
              ('01_issue005.tif', '04_issue008.tif', '07_issue011.tif', '08_issue012.tif', '10_blob_residency.tif')]
    rng = np.random.default_rng(8)
    for number in range(120):
        shape = (int(rng.integers(1, 9)), int(rng.integers(1, 30)), int(rng.integers(1, 30)))
        dtype = [np.uint16, np.int32, np.int64][number % 3]
        stack = rng.integers(0, 12, shape).astype(dtype)
        stack[rng.random(shape) < 0.5] = 0
        if number % 10 == 1 and dtype != np.uint16:
            stack[0, 0, 0] = -3
        if number % 10 == 2 and dtype == np.int64:
            stack[0, 0, 0] = 1 << 40
        if number % 10 == 3:
            stack[:] = 0
        stacks.append(stack)
    settings = [dict(presence.DEFAULTS)]
    for ratio in (0.5, 1.7, 0.0, -1.0):
        settings.append(dict(presence.DEFAULTS, minimum_persistent_area_ratio_to_movie_median=ratio,
                             minimum_persistent_area_floor_px=10, maximum_identity_area_ratio=2.5))

    def outcome(function, *args):
        try:
            value = function(*args)
        except Exception as error:
            return ('error', type(error).__name__, str(error))
        if isinstance(value, dict):
            return ('dict', [(k, v) for k, v in value.items()])
        return ('float', type(value).__name__, np.float64(value).tobytes())

    def evaluate():
        rows = []
        for stack in stacks:
            rows.append(outcome(presence._identity_frames, stack))
            for params in settings:
                rows.append(outcome(presence._maximum_identity_area, stack, params))
                rows.append(outcome(presence.resolve_minimum_persistent_area, stack, params))
        return rows
    tick = time.perf_counter(); expected = evaluate(); before = time.perf_counter() - tick
    with table.install(presence) as stats:
        assert stats['supported']
        tick = time.perf_counter(); actual = evaluate(); after = time.perf_counter() - tick
    for number, (e, f) in enumerate(zip(expected, actual)):
        assert e == f, number
    report = dict(status='pass', stacks=len(stacks), real_stacks=5, comparisons=len(expected),
                  errors=sum(e[0] == 'error' for e in expected),
                  before_seconds=before, after_seconds=after, counters=stats)
    print(report)
    a.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
