"""Compose unchanged recovery operations with isolated historical imports.

Operation specifications select scientific methods, never movies or identities.
The caller supplies the observed/proposed native arrays and their histories.
"""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import hashlib
import importlib
import sys

import numpy as np
import pandas as pd


@contextmanager
def frozen_method(spec):
    """Temporarily isolate original local dependencies; restore the import state."""
    folder = Path(spec['code_dir'])
    names = set(spec['code_sha256'])
    for name, digest in spec['code_sha256'].items():
        assert hashlib.sha256((folder/(name+'.py')).read_bytes()).hexdigest() == digest
    previous = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    old_path = sys.path.copy()
    try:
        sys.path.insert(0, str(folder))
        importlib.invalidate_caches()
        method = getattr(importlib.import_module(spec['module']), spec['function'])
        yield method
    finally:
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(previous)
        sys.path[:] = old_path


def apply(labels, ledger, raw, points, restored, observed, saved, params, history, operations):
    assert labels.shape == restored.shape == observed.shape == saved.shape == raw.shape
    assert not params.get('carve_hosts', False)
    assert not params.get('forced_identity_ids') and not params.get('forced_intervals')
    for spec in operations:
        # Original producers read these afresh for each operation. Copies retain
        # that isolation while reusing the exact already-loaded input bytes.
        before = saved.copy()
        kwargs = {'proposal_history': history.copy(deep=True)} if spec['uses_history'] else {}
        with frozen_method(spec) as method:
            after, audit = method(observed.copy(), saved.copy(), raw.copy(),
                                  points.copy(deep=True), deepcopy(params), **kwargs)
        candidate = labels.copy()
        eligible = (labels == saved) & (restored == saved)
        candidate[eligible] = after[eligible]
        changed = candidate != labels
        new_ledger = ledger.copy()
        new_ledger[candidate > 0] = 0
        removed = changed & (candidate == 0)
        new_ledger[removed] = labels[removed]
        applications = []
        for t in range(len(labels)):
            for source, target in np.unique(np.stack((labels[t][changed[t]], candidate[t][changed[t]]), axis=1), axis=0):
                applications.append(dict(frame=t+1, source=int(source), target=int(target),
                    changed_pixels=int(((labels[t] == source) & (candidate[t] == target) & changed[t]).sum())))
        tables = dict(presence_audit=audit, presence_before_audit=pd.DataFrame(),
            birth_applications=pd.DataFrame(applications, columns=['frame', 'source', 'target', 'changed_pixels']),
            birth_decision_audit=pd.DataFrame(columns=['decision']))
        labels, ledger = candidate, new_ledger
        yield spec['stage'], labels, ledger, tables, dict(presence_before=before, presence_after=after)
