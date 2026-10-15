"""Exact correction audit with one reusable snapshot and frame-sized scratch."""
import hashlib

import numpy as np


def array_sha256(array):
    """Hash the same C-order bytes as ndarray.tobytes without a movie copy."""
    digest = hashlib.sha256()
    if array.flags.c_contiguous and array.size:
        digest.update(memoryview(array).cast('B'))
    elif array.ndim:
        for frame in array:
            digest.update(frame.tobytes(order='C'))
    else:
        digest.update(array.tobytes(order='C'))
    return digest.hexdigest()


class CorrectionAudit:
    """Snapshot inputs so in-place edits and later reversals remain auditable."""

    def __init__(self, labels):
        self.previous = labels.copy()
        self.inferred = np.zeros(labels.shape, dtype=bool)

    def record(self, labels, ledger):
        if labels.shape != self.previous.shape:
            raise ValueError('Correction changed the movie shape')
        changed = 0
        for t in range(len(labels)):
            delta = labels[t] != self.previous[t]
            self.inferred[t] |= delta
            changed += int(np.count_nonzero(delta))
        result = dict(changed_pixels=changed,
                      label_array_sha256=array_sha256(labels),
                      ledger_array_sha256=array_sha256(ledger))
        # A new numeric representation must not be narrowed to the old dtype.
        if labels.dtype != self.previous.dtype:
            self.previous = labels.copy()
        else:
            np.copyto(self.previous, labels)
        return result
