"""Paired map preparation measurements on bounded, reproducible label movies."""
import argparse
import json
from pathlib import Path
import time
import tracemalloc
import numpy as np
from pymicroglia.figure_tables.territory_values import cumulative_occupancy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipts = []
    for shape in [(48, 64, 64), (96, 256, 256), (192, 256, 256)]:
        rng = np.random.default_rng(20261019)
        labels = rng.integers(0, 4, shape, dtype=np.uint16)
        hours = np.cumsum(rng.uniform(.01, 1.7, shape[0]))
        reference = cumulative_occupancy(labels, hours)[-1].copy()
        row = {'shape': shape, 'samples': [], 'exact': True}
        for trial in range(4):
            for mode in (['before', 'after'] if trial % 2 == 0 else ['after', 'before']):
                start = time.perf_counter()
                result = cumulative_occupancy(labels, hours, final_only=mode == 'after')
                elapsed = time.perf_counter() - start
                np.testing.assert_array_equal(result if mode == 'after' else result[-1], reference)
                del result
                row['samples'].append({'mode': mode, 'trial': trial, 'seconds': elapsed})
        row['peak_allocated_bytes'] = {}
        for mode in ['before', 'after']:
            tracemalloc.start()
            result = cumulative_occupancy(labels, hours, final_only=mode == 'after')
            row['peak_allocated_bytes'][mode] = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            del result
        receipts.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(receipts, indent=2))


if __name__ == '__main__':
    main()
