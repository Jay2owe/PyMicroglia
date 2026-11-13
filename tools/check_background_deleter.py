"""Check that released working files are closed and deleted off the main thread, and that nothing reads differently.

A pool allocates file-backed movies, fills, reads and releases them, and
allocates again while deletions are pending. Values read back must equal the
written values; after drain_deletions and after Pool.close no working file
remains; adopted outside files are still deleted at once; and the release
time on the calling thread is reported against the synchronous route.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--movies', type=int, default=12)
    p.add_argument('--megabytes', type=int, default=200)
    a = p.parse_args()
    sys.path.insert(0, str(a.snapshot.resolve()))
    import array_storage
    shape = (a.megabytes, 1024, 512)  # uint16: about megabytes MiB
    folder = Path(tempfile.mkdtemp(prefix='deleter-'))
    pool = array_storage.Pool(folder / 'working', minimum_bytes=1024**2, temporary_ram_bytes=0, released_bytes=0)
    report = dict(status='pass')
    release = 0.0
    for k in range(a.movies):
        movie = pool.allocate(shape, np.uint16)
        movie[::7] = k + 1
        assert int(np.asarray(movie)[7 * (k % 3), 3, 5]) == k + 1 and int(np.asarray(movie)[1, 0, 0]) == 0
        tick = time.perf_counter()
        del movie
        gc.collect()
        release += time.perf_counter() - tick
    array_storage.drain_deletions()
    assert not list((folder / 'working').glob('*.dat'))
    report['deferred_release_seconds'] = release
    # The synchronous route, for comparison (adopted-style deletion on this thread).
    sync = 0.0
    for k in range(a.movies):
        path = folder / f'sync_{k}.dat'
        mapped = np.memmap(path, dtype=np.uint16, mode='w+', shape=shape)
        mapped[::7] = k + 1
        mapping = mapped._mmap
        del mapped
        tick = time.perf_counter()
        mapping.close()
        path.unlink()
        sync += time.perf_counter() - tick
    report['synchronous_release_seconds'] = sync
    # Adopted outside files are deleted at once.
    outside = folder / 'outside.dat'
    adopted = np.memmap(outside, dtype=np.uint8, mode='w+', shape=(4, 4))
    held = pool.adopt(adopted, outside)
    del adopted, held
    gc.collect()
    assert not outside.exists()
    pool.close()
    assert not list((folder / 'working').glob('*.dat'))
    import shutil
    shutil.rmtree(folder, ignore_errors=True)
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
