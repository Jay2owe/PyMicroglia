"""Compose parallel identity repairs without inventing a winner on disagreement.

A proposal is one before/after ownership transfer across the whole movie. When
overlapping proposals disagree, retain their parent ownership throughout those
proposals. Keep independent and agreeing proposals. Labels and the unclaimed
ledger are one ownership state, never merged independently.
"""
from __future__ import annotations

from contextlib import contextmanager
import ast
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
from .checkpoints import reconcile_repair

VERSION = '2026-10-21-parallel-repair-composition'


def compose_identity_repairs(baseline, baseline_unclaimed, alias_labels,
                            alias_unclaimed, continuity_labels, continuity_unclaimed):
    """Return coherent labels, ledger and an audit; temporary arrays are per frame."""
    arrays = (baseline, baseline_unclaimed, alias_labels, alias_unclaimed,
              continuity_labels, continuity_unclaimed)
    if any(a.shape != baseline.shape for a in arrays) or baseline.ndim != 3:
        raise ValueError('Parallel repairs must use the same movie coordinates')
    nodes, parents, details, bad = {}, [], [], set()

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def proposals(t, branch, labels, ledger, *, collect):
        changed = (labels[t] != baseline[t]) | (ledger[t] != baseline_unclaimed[t])
        codes = np.full(changed.shape, -1, np.int32)
        if not np.any(changed):
            return codes
        states = np.column_stack((baseline[t][changed], baseline_unclaimed[t][changed],
                                  labels[t][changed], ledger[t][changed]))
        unique, inverse, counts = np.unique(states, axis=0, return_inverse=True, return_counts=True)
        indices = []
        for state, count in zip(unique, counts):
            key = (branch, *map(int, state))
            if key not in nodes:
                if not collect: raise AssertionError('Proposal changed during composition')
                nodes[key] = len(parents)
                parents.append(len(parents))
                details.append(dict(branch=branch, before_identity=key[1], before_unclaimed=key[2],
                    after_identity=key[3], after_unclaimed=key[4], pixels=0,
                    first_frame=t+1, last_frame=t+1))
            i = nodes[key]
            if collect:
                details[i]['pixels'] += int(count)
                details[i]['last_frame'] = t+1
            indices.append(i)
        codes[changed] = np.asarray(indices, dtype=np.int32)[inverse]
        return codes

    for t in range(len(baseline)):
        foreground = (baseline[t] > 0) | (baseline_unclaimed[t] > 0)
        for labels, ledger in ((baseline, baseline_unclaimed), (alias_labels, alias_unclaimed),
                               (continuity_labels, continuity_unclaimed)):
            if np.any((labels[t] > 0) & (ledger[t] > 0)) or not np.array_equal(
                    (labels[t] > 0) | (ledger[t] > 0), foreground):
                raise ValueError('Parallel proposal has incoherent foreground ownership')
        a = proposals(t, 'alias', alias_labels, alias_unclaimed, collect=True)
        b = proposals(t, 'continuity', continuity_labels, continuity_unclaimed, collect=True)
        overlap = (a >= 0) & (b >= 0)
        for left, right in np.unique(np.column_stack((a[overlap], b[overlap])), axis=0):
            left, right = find(int(left)), find(int(right))
            parents[right] = left
        conflict = overlap & ((alias_labels[t] != continuity_labels[t]) |
                              (alias_unclaimed[t] != continuity_unclaimed[t]))
        bad.update(map(int, np.unique(a[conflict])))
    rejected_roots = {find(i) for i in bad}
    accepted = np.asarray([find(i) not in rejected_roots for i in range(len(parents))], dtype=bool)
    labels, unclaimed = baseline.copy(), baseline_unclaimed.copy()
    for t in range(len(baseline)):
        for branch, proposed, ledger in (('alias', alias_labels, alias_unclaimed),
                                        ('continuity', continuity_labels, continuity_unclaimed)):
            codes = proposals(t, branch, proposed, ledger, collect=False)
            changed = codes >= 0
            if not np.any(changed): continue
            keep = np.zeros(changed.shape, dtype=bool)
            keep[changed] = accepted[codes[changed]]
            labels[t][keep], unclaimed[t][keep] = proposed[t][keep], ledger[t][keep]
    for i, row in enumerate(details):
        row['status'] = 'applied' if accepted[i] else 'unresolved_conflicting_proposals_parent_retained'
    columns = ['branch', 'before_identity', 'before_unclaimed', 'after_identity',
               'after_unclaimed', 'pixels', 'first_frame', 'last_frame', 'status']
    return labels, unclaimed, pd.DataFrame(details, columns=columns)


def repaired_function(source):
    source = source.replace('\r\n', '\n')
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == 'run_accepted_history')
    original = ast.get_source_segment(source, node)
    start = '        alias_delta = alias.labels != issue015_labels\n'
    end = '        _save(issue017_checkpoint, labels, interval)'
    if 'compose_identity_repairs(' in original:
        return original, original
    if original.count(start) != 1 or original.count(end) != 1:
        raise ValueError('Unrecognised parallel repair composition; no implicit patch')
    first, last = original.index(start), original.index(end)
    replacement = '''        labels, unclaimed, composition_audit = compose_identity_repairs(
            issue015_labels, issue015_unclaimed, alias.labels, alias.unclaimed,
            continuity_labels, continuity_unclaimed)
        composition_audit.to_csv(output_root / "parallel_repair_composition.csv", index=False)
'''
    return original, original[:first] + replacement + original[last:]


@contextmanager
def install(history, checkpoints, root):
    """Patch the frozen function in place so timing/direct-import wrappers survive."""
    original, patched = repaired_function(Path(history.__file__).read_text(encoding='utf8'))
    if original == patched:
        yield None
        return
    sha = lambda b: hashlib.sha256(b).hexdigest()
    signature = dict(version=VERSION, repair_sha256=sha(Path(__file__).read_bytes()),
        original_function_sha256=sha(original.encode()), patched_function_sha256=sha(patched.encode()))
    root = Path(root)
    record = root/'parallel_repair_composition.json'
    reconcile_repair(root, record, signature, 'mid/accepted_history/11_issue017_rebase.tif', 'composition repair')
    record.write_text(json.dumps(signature, indent=2))
    target = inspect.unwrap(history.run_accepted_history)
    old_code, old_call = target.__code__, checkpoints.Stages.call
    namespace = {}
    exec(compile(patched, str(Path(__file__)), 'exec'), history.__dict__, namespace)
    prior_helper = history.__dict__.get('compose_identity_repairs')
    history.compose_identity_repairs = compose_identity_repairs
    target.__code__ = namespace['run_accepted_history'].__code__

    def call(self, method, *args, **kwargs):
        if method != 'stage_m22_accepted_history':
            return old_call(self, method, *args, **kwargs)
        base = self.signature
        self.signature = {**base, 'composition_repair': signature}
        try:
            return old_call(self, method, *args, **kwargs)
        finally:
            self.signature = base

    checkpoints.Stages.call = call
    try:
        yield signature
    finally:
        checkpoints.Stages.call = old_call
        target.__code__ = old_code
        if prior_helper is None: history.__dict__.pop('compose_identity_repairs', None)
        else: history.compose_identity_repairs = prior_helper
