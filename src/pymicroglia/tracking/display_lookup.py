"""Review outline rendering with longer frame runs and a grey lookup table.

The buffered renderer drew two output frames per run plus one halo frame on
each side, so every frame's outlines and grey display were computed twice.
Longer runs keep the same halo and the same calls to the engine's outline
overlay. For 8- and 16-bit movies the grey value of a pixel depends only on
its raw value, so the display statement is evaluated once for every possible
value, in the same thread with the same scale, and indexed.
"""
from pathlib import Path
import tempfile, threading, weakref

import numpy as np

BLOCK_BYTES = 64 * 1024 ** 2


def lookup_renderer(common, scratch, *, block_bytes=BLOCK_BYTES):
    from .display_scale import positive_display_high
    original = common.outline_overlay
    original_display = common.display_raw
    lock = threading.RLock()

    def render(raw, labels, thick=2, inferred=None, unresolved=None):
        high = positive_display_high(raw)
        handle = tempfile.NamedTemporaryFile(prefix='rgb-', suffix='.dat', dir=scratch, delete=False)
        path = Path(handle.name); handle.close()
        output = np.memmap(path, mode='w+', dtype=np.uint8, shape=(*raw.shape, 3))

        def cleanup(mapping, filename):
            mapping.close(); filename.unlink(missing_ok=True)
        weakref.finalize(output, cleanup, output._mmap, path)
        table = None
        if raw.dtype in (np.dtype(np.uint8), np.dtype(np.uint16)):
            values = np.arange(np.iinfo(raw.dtype).max + 1, dtype=raw.dtype)
            table = (np.clip(values.astype(np.float32) / max(high, 1.), 0, 1) ** .7 * 180).astype(np.uint8)

        def display(chunk):
            if table is not None and chunk.dtype == raw.dtype:
                return np.repeat(table[np.asarray(chunk)][..., None], 3, axis=-1)
            grey = np.clip(chunk.astype(np.float32) / max(high, 1.), 0, 1) ** .7
            return np.repeat((grey[..., None] * 180).astype(np.uint8), 3, axis=-1)
        halo = max(int(thick) - 1, 1)
        with lock:
            common.display_raw = display
            try:
                frame_bytes = max(1, int(np.prod(raw.shape[1:])) * 16)
                frames = max(1, int(block_bytes) // frame_bytes)
                for t in range(0, len(raw), frames):
                    stop = min(len(raw), t + frames)
                    lo, hi = max(0, t - halo), min(len(raw), stop + halo)
                    chunk = original(raw[lo:hi], labels[lo:hi], thick=thick,
                        inferred=None if inferred is None else inferred[lo:hi],
                        unresolved=None if unresolved is None else unresolved[lo:hi])
                    output[t:stop] = chunk[t - lo:stop - lo]
            finally:
                common.display_raw = original_display
        return output
    return render
