"""Compare the correction audit before and after copying only changed frames.

python tools/check_correction_audit_changed_frames.py <before.py> <after.py> <report.json>
Both audits record the same sequences of stage outputs: unchanged stages, edits
in a few frames, reversals, whole-movie edits, a dtype change, float movies
with NaN and signed zeros, and in-place edits of the recorded array. Every
result dict and the inferred mask after every stage must be equal.
"""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


before, after = load(sys.argv[1], 'audit_before'), load(sys.argv[2], 'audit_after')
rng = np.random.default_rng(11)


def sequence(kind):
    shape = (12, 40, 50)
    if kind == 'float':
        labels = rng.integers(0, 6, shape).astype(np.float32)
        labels[3, 5, 5] = np.nan
    else:
        labels = rng.integers(0, 6, shape).astype(np.uint16)
    ledger = rng.integers(0, 3, shape).astype(np.uint16)
    yield labels, labels.copy(), ledger.copy()
    original = labels.copy()
    for step in range(40):
        choice = step % 8
        if choice == 0:
            pass  # unchanged stage
        elif choice == 1:
            t = int(rng.integers(len(labels)))
            labels[t, rng.integers(40), rng.integers(50)] += 1
        elif choice == 2:
            labels[:] = original  # reversal to the input
        elif choice == 3:
            labels[rng.integers(len(labels), size=3)] = 0
        elif choice == 4:
            labels += 1  # whole-movie edit
        elif choice == 5 and kind == 'float':
            labels[labels == 0] = -0.0
            labels[1, 1, 1] = np.nan
        elif choice == 6:
            ledger[int(rng.integers(len(ledger)))] ^= 1
        elif choice == 7 and kind == 'dtype':
            labels = labels.astype(np.int32)
        yield labels, labels, ledger
    # A stage that edits the recorded array in place after recording.
    labels[0] = 7
    yield labels, labels, ledger


checks = {}
for kind in ('integer', 'float', 'dtype'):
    mismatches = compared = 0
    runs = sequence(kind)
    init, _, _ = next(runs)
    audit_b, audit_a = before.CorrectionAudit(init.copy()), after.CorrectionAudit(init.copy())
    for _, stage, ledger in runs:
        # The same stage array, as the sequence hands each audit.
        rb = audit_b.record(stage, ledger)
        ra = audit_a.record(stage, ledger)
        compared += 1
        mismatches += not (rb == ra and np.array_equal(audit_b.inferred, audit_a.inferred))
    checks[kind] = dict(stages=compared, mismatched=mismatches)
passed = all(c['mismatched'] == 0 and c['stages'] > 30 for c in checks.values())
report = dict(status='passed' if passed else 'failed', checks=checks)
Path(sys.argv[3]).write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
raise SystemExit(0 if passed else 1)
