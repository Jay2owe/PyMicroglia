"""Compare the composed correction audit's ledger digests with a fresh SHA-256 of every ledger, exactly.

A sequence of records changes and keeps the label movie and the unclaimed
identity ledger: unchanged ledgers (same object, equal copies, read-only and
disk-backed views), one changed pixel, a reverted change, a new dtype, a new
shape, a float ledger (never kept), and a ledger
edited in place after it was recorded. Every record's ledger digest and label
digest must equal hashlib's digest of the array's C-order bytes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np


def plain_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.snapshot.resolve()))
    from pymicroglia.tracking import correction_audit
    rng = np.random.default_rng(7)
    labels = rng.integers(0, 50, (12, 64, 64)).astype(np.uint16)
    ledger = np.where(rng.random(labels.shape) < .05, rng.integers(1, 300, labels.shape), 0).astype(np.uint16)
    scratch = Path(tempfile.mkdtemp(prefix='audit-ledger-'))
    mapped = np.memmap(scratch / 'ledger.dat', dtype=ledger.dtype, mode='w+', shape=ledger.shape)
    mapped[:] = ledger
    steps = []

    def step(name, labels, ledger):
        steps.append((name, labels, ledger))
    step('first', labels, ledger)
    step('same object', labels, ledger)
    step('equal copy', labels.copy(), ledger.copy())
    readonly = ledger.copy(); readonly.flags.writeable = False
    step('read-only', labels, readonly)
    step('memmap', labels, mapped)
    changed = ledger.copy(); changed[3, 10, 10] += 1
    step('one pixel', labels, changed)
    step('reverted', labels, ledger)
    step('in place before', labels, ledger)
    edited = ledger  # edited in place after being recorded
    step('in place after', labels, 'edit')
    step('new dtype', labels, ledger.astype(np.int32))
    step('back to uint16', labels, ledger.copy())
    step('new shape', labels, ledger[:, :32])
    step('float', labels, ledger.astype(np.float32))
    step('float again', labels, ledger.astype(np.float32))
    step('non-contiguous', labels, np.asfortranarray(ledger))
    step('labels change', np.where(labels == 3, 4, labels).astype(labels.dtype), ledger)
    audit = correction_audit.CorrectionAudit(labels)
    report = dict(status='pass', steps=[])
    for name, l, g in steps:
        if isinstance(g, str):
            edited[0, 0, 0] += 7
            g = edited
        result = audit.record(l, g)
        assert result['ledger_array_sha256'] == plain_sha256(g), name
        assert result['label_array_sha256'] == plain_sha256(l), name
        report['steps'].append(dict(step=name, kept=getattr(audit, 'ledger_held', None) is not None))
    assert sum(not s['kept'] for s in report['steps']) == 2  # the two float ledgers
    del mapped
    import shutil
    shutil.rmtree(scratch, ignore_errors=True)
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], steps=len(report['steps']))))


if __name__ == '__main__':
    main()
