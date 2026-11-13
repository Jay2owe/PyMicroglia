"""Compare resumed correction-audit hashes with hashes of the whole movie.

Runs the candidate CorrectionAudit beside a reference that hashes every label
and ledger movie in full (the original rule), over a sequence of edits made
to accepted label and ledger movies: single pixels early, middle and last,
scattered frames, undone edits, unchanged records, a dtype change, Fortran and
reversed views and an empty movie. The original audit class (--reference)
runs the same records for timing; every record must return the same hashes
as the whole-movie digest.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
HISTORY = (ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
           / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history')


def whole(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    a = parser.parse_args()
    sys.path.insert(0, str(a.snapshot.resolve()))
    from pymicroglia.tracking import correction_audit
    assert Path(correction_audit.__file__).resolve().is_relative_to(a.snapshot.resolve())
    labels = tifffile.imread(HISTORY / '08_issue012.tif')
    ledger = tifffile.imread(HISTORY / '08_issue012_unclaimed.tif')
    import importlib.util
    spec = importlib.util.spec_from_file_location('reference_audit', a.reference)
    reference = importlib.util.module_from_spec(spec); spec.loader.exec_module(reference)
    audit = correction_audit.CorrectionAudit(labels)
    before = reference.CorrectionAudit(labels)
    rng = np.random.default_rng(21)
    T = len(labels)
    edits = [('none', []), ('last', [T - 1]), ('first', [0]), ('middle', [T // 2]),
             ('scattered', sorted(rng.choice(T, 5, replace=False).tolist())), ('none', []),
             ('undo', None), ('ledger-only', 'ledger'), ('both-late', [T - 3, T - 2]), ('none', [])]
    failures, seconds, kept = [], dict(states=0.0, original=0.0), []
    for name, frames in edits:
        if frames is None:
            labels[:] = kept[-1][0]; ledger[:] = kept[-1][1]
        elif frames == 'ledger':
            ledger[T - 5, 3, 4] += 1
        else:
            kept.append((labels.copy(), ledger.copy()))
            for t in frames:
                labels[t, rng.integers(labels.shape[1]), rng.integers(labels.shape[2])] += 7
                ledger[t, 10, 10] += 1
        t0 = time.perf_counter()
        result = audit.record(labels, ledger)
        t1 = time.perf_counter()
        old = before.record(labels, ledger)
        seconds['states'] += t1 - t0; seconds['original'] += time.perf_counter() - t1
        expected_label, expected_ledger = whole(labels), whole(ledger)
        if result != old or (result['label_array_sha256'] != expected_label
                or result['ledger_array_sha256'] != expected_ledger):
            failures.append(name)
    # A dtype change, then views that are not C-contiguous.
    grown = labels.astype(np.uint32); grown[T - 1, 0, 0] = 70000
    for name, movie in [('grown', grown), ('fortran', np.asfortranarray(grown)),
                        ('reversed', grown[:, ::-1, :])]:
        if name == 'reversed':
            movie[T // 3, 1, 1] += 1
        result = audit.record(movie, movie)
        if (result['label_array_sha256'] != whole(movie)
                or result['ledger_array_sha256'] != whole(movie)):
            failures.append(name)
    small = correction_audit.CorrectionAudit(np.zeros((0, 3, 3), np.uint16))
    if small.record(np.zeros((0, 3, 3), np.uint16), np.zeros(()))['label_array_sha256'] != whole(np.zeros((0, 3, 3), np.uint16)):
        failures.append('empty')
    report = dict(status='pass' if not failures else 'fail', records=len(edits) + 4,
                  seconds=seconds, failures=failures)
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
