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


def frame_sha256(array, states=None, first=0):
    """array_sha256 fed one frame at a time, resuming at frame first.

    SHA-256 reads its bytes in order, so frame-by-frame updates give the
    digest of the whole C-order movie. states[t] holds the hash of frames
    before t; when frames before first are byte-identical to the movie the
    states were filled from, the hash resumes there. Returns the digest and
    the states for this movie, or None for empty or zero-dimensional arrays.
    """
    if not array.ndim or not array.size:
        return array_sha256(array), None
    if states is None or len(states) != len(array) or not 0 <= first < len(array):
        states, first = [None] * len(array), 0
    digest = states[first].copy() if first else hashlib.sha256()
    for t in range(first, len(array)):
        states[t] = digest.copy()
        frame = array[t]
        digest.update(memoryview(frame).cast('B') if frame.flags.c_contiguous
                      else frame.tobytes(order='C'))
    return digest.hexdigest(), states


class CorrectionAudit:
    """Snapshot inputs so in-place edits and later reversals remain auditable."""

    def __init__(self, labels):
        self.previous = labels.copy()
        self.inferred = np.zeros(labels.shape, dtype=bool)
        self.label_sha256 = None
        self.label_states = None

    def _ledger_sha256(self, ledger):
        held = getattr(self, 'ledger_held', None)
        first = 0
        if (held is not None and ledger.ndim and ledger.dtype == held[0].dtype
                and ledger.shape == held[0].shape):
            # The first frame that differs; integer values equal means bytes equal.
            first = next((t for t in range(len(ledger)) if np.any(ledger[t] != held[0][t])), None)
            if first is None:
                return held[1]
        states = held[2] if held is not None and first else None
        digest, states = frame_sha256(ledger, states, first if states is not None else 0)
        if ledger.dtype.kind in 'biu' and ledger.ndim and ledger.nbytes <= 512 * 1024**2:
            if held is not None and held[0].dtype == ledger.dtype and held[0].shape == ledger.shape:
                np.copyto(held[0], ledger)
                self.ledger_held = (held[0], digest, states)
            else:
                self.ledger_held = (np.array(ledger, copy=True), digest, states)
        else:
            self.ledger_held = None
        return digest

    def record(self, labels, ledger):
        if labels.shape != self.previous.shape:
            raise ValueError('Correction changed the movie shape')
        changed = 0
        first = None
        moved = []
        for t in range(len(labels)):
            delta = labels[t] != self.previous[t]
            count = int(np.count_nonzero(delta))
            if count:
                # An unchanged frame adds nothing to inferred and keeps its
                # snapshot values; the snapshot is only ever compared with !=.
                self.inferred[t] |= delta
                moved.append(t)
                if first is None:
                    first = t
            changed += count
        same_kind = labels.dtype == self.previous.dtype and labels.dtype.kind in 'biu'
        if changed == 0 and self.label_sha256 is not None and same_kind:
            label_sha256 = self.label_sha256
        else:
            # Frames before the first changed one hold the bytes last hashed.
            resume = same_kind and self.label_sha256 is not None and first is not None
            label_sha256, self.label_states = frame_sha256(
                labels, self.label_states if resume else None, first if resume else 0)
        self.label_sha256 = label_sha256
        result = dict(changed_pixels=changed,
                      label_array_sha256=label_sha256,
                      ledger_array_sha256=self._ledger_sha256(ledger))
        # A new numeric representation must not be narrowed to the old dtype.
        if labels.dtype != self.previous.dtype:
            self.previous = labels.copy()
        else:
            for t in moved:
                np.copyto(self.previous[t], labels[t])
        return result
