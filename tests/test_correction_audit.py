"""Audit storage changes preserve exact history, including undone edits."""
import hashlib

import numpy as np
import pytest

from pymicroglia.tracking.correction_audit import CorrectionAudit, array_sha256


@pytest.mark.parametrize('layout', ['C', 'F', 'reverse', 'empty', 'scalar'])
@pytest.mark.parametrize('dtype', ['uint16', '>u4', 'float64'])
def test_hash_matches_original_bytes(layout, dtype):
    a = np.arange(60).reshape(3, 4, 5).astype(dtype)
    if layout == 'F': a = np.asfortranarray(a)
    if layout == 'reverse': a = a[::-1, :, ::-1]
    if layout == 'empty': a = a[:0]
    if layout == 'scalar': a = a[0, :1, :1].reshape(())
    assert array_sha256(a) == hashlib.sha256(a.tobytes()).hexdigest()


def test_reusable_snapshot_tracks_edits_reversals_and_dtype_growth():
    original = np.zeros((3, 4, 5), np.uint16)
    audit = CorrectionAudit(original)
    snapshot = audit.previous
    original[0, 0, 0] = 8
    assert audit.record(original, original)['changed_pixels'] == 1
    assert audit.previous is snapshot
    original[0, 0, 0] = 0
    assert audit.record(original, original)['changed_pixels'] == 1
    assert audit.inferred.sum() == 1
    assert audit.record(original, original)['changed_pixels'] == 0
    grown = original.astype(np.uint32)
    grown[2, 2, 2] = 100000
    assert audit.record(grown, grown)['changed_pixels'] == 1
    assert audit.previous.dtype == grown.dtype
    assert audit.record(grown, grown)['changed_pixels'] == 0
    assert audit.inferred.sum() == 2
    with pytest.raises(ValueError, match='shape'):
        audit.record(grown[:1], grown)
