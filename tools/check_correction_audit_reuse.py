"""Correction audit records: unchanged movies reuse their digest, every record equal to the original.

Runs the previous and the candidate correction_audit.py side by side over
sequences of unchanged and edited movies (uint16, int32, bool, float32 with
NaN and signed zeros, a dtype widening, non-contiguous views) and requires the
same records, inferred masks and previous copies after every step.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    old, new = load(a.original, 'audit_original'), load(a.candidate, 'audit_candidate')
    rng = np.random.default_rng(114)
    cases = 0
    reused = 0
    for dtype in (np.uint16, np.int32, np.bool_, np.float32):
        for view in ('contiguous', 'strided'):
            base = (rng.integers(0, 5, (12, 40, 30)) if dtype is not np.float32
                    else rng.normal(size=(12, 40, 30))).astype(dtype)
            if dtype is np.float32:
                base[0, 0, :3] = (np.nan, 0.0, -0.0)
            ledger = rng.integers(0, 3, base.shape).astype(np.uint16)
            left, right = old.CorrectionAudit(base), new.CorrectionAudit(base)
            labels = base.copy()
            for step in range(40):
                action = rng.integers(0, 5)
                if action == 1:
                    at = (rng.integers(0, 12), rng.integers(0, 40), rng.integers(0, 30))
                    labels[at] = (not labels[at]) if dtype is np.bool_ else labels[at] + 1
                elif action == 2 and dtype is np.float32:
                    labels[0, 0, 1] = -labels[0, 0, 1]  # 0.0 <-> -0.0: equal values, different bytes
                elif action == 3 and dtype in (np.uint16,) and step == 20:
                    labels = labels.astype(np.int32)  # a widened representation
                elif action == 4:
                    ledger[rng.integers(0, 12)] ^= 1
                shown = labels if view == 'contiguous' else np.asfortranarray(labels)
                expected, actual = left.record(shown, ledger), right.record(shown, ledger)
                assert expected == actual, (dtype, view, step, expected, actual)
                assert np.array_equal(left.inferred, right.inferred)
                assert left.previous.dtype == right.previous.dtype and np.array_equal(
                    left.previous, right.previous, equal_nan=True)
                reused += int(expected['changed_pixels'] == 0 and right.label_sha256 is not None)
                cases += 1
    report = dict(status='pass', records=cases, unchanged_records=reused,
                  sources={str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                           for f in (Path(__file__), a.original, a.candidate)})
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'sources'}))


if __name__ == '__main__':
    main()
